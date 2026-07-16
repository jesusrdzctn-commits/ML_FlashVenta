#!/usr/bin/env python3
"""
PepsiCo Financial Forecasting - Modelo Certero para Series Financieras
Sistema conservador optimizado para precisión en pronósticos de ventas
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Importar dependencias
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    try:
        from fbprophet import Prophet
        HAS_PROPHET = True
    except ImportError:
        HAS_PROPHET = False

class PepsicoFinancialForecast:
    """Sistema certero para pronósticos financieros con enfoque conservador."""
    
    def __init__(self):
        self.xgb_model = None
        self.prophet_model = None
        self.is_trained = False
        
        # PARÁMETROS XGBoost optimizados
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
        
        # PARÁMETROS Prophet optimizados
        self.prophet_params = {
            'growth': 'linear',
            'yearly_seasonality': False,
            'weekly_seasonality': False,  # Custom
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'seasonality_prior_scale': 0.03,
            'holidays_prior_scale': 0.01,
            'changepoint_prior_scale': 0.002,
            'changepoint_range': 0.8,
            'interval_width': 0.80,
            'mcmc_samples': 0,
            'uncertainty_samples': 100
        }
        
        # FACTORES CONSERVADORES POR DÍA DE SEMANA (basados en medianas históricas)
        # NOTA: Factores recalibrados para mayor precisión financiera
        self.dow_factors = {
            0: 1.05,   # Lunes: +5% (más conservador)
            1: 1.03,   # Martes: +3% 
            2: 1.08,   # Miércoles: +8% (día fuerte pero moderado)
            3: 1.02,   # Jueves: +2%
            4: 0.98,   # Viernes: -2%
            5: 0.75,   # Sábado: -25% vs entre semana
            6: 0.30    # Domingo base: -70% vs entre semana
        }
        
        # FACTORES ESPECÍFICOS PARA DOMINGOS (recalibrados conservadoramente)
        self.sunday_factors = {
            'working': 0.45,       # Domingo laboral: 45% vs entre semana (más conservador)
            'non_working': 0.15    # Domingo no laboral: 15% vs entre semana (muy bajo)
        }
        
        # INTERVALOS DE CONFIANZA CONSERVADORES PARA DOMINGOS
        self.sunday_confidence_intervals = {
            'working': {'lower': 0.35, 'upper': 0.55},     # Rango estrecho y conservador
            'non_working': {'lower': 0.10, 'upper': 0.20}  # Muy conservador para no laborales
        }
        
        # VOLATILIDAD POR DÍA DE SEMANA
        self.dow_volatility = {
            0: 0.081, 1: 0.073, 2: 0.067, 3: 0.053, 4: 0.062,
            5: 0.052,  # Sábado: baja volatilidad
            6: 0.627   # Domingo: ALTÍSIMA volatilidad
        }
        
        # FACTORES ESTACIONALES MENSUALES
        self.monthly_factors = {
            1: 0.98, 2: 0.95, 3: 1.02, 4: 1.01, 5: 1.04, 6: 1.02,
            7: 1.06, 8: 1.04, 9: 1.00, 10: 1.08, 11: 1.12, 12: 1.15
        }
        
    def load_all_data(self):
        """Carga todos los datos."""
        logger.info("Cargando datos...")
        
        input_dir = Path("Input")
        if not input_dir.exists():
            raise FileNotFoundError("Directorio Input/ no encontrado")
        
        csv_files = list(input_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No se encontraron archivos CSV en Input/")
        
        all_data = []
        for file in sorted(csv_files):
            df = pd.read_csv(file)
            all_data.append(df)
            logger.info(f"Cargado: {file.name} ({len(df):,} filas)")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total datos combinados: {len(combined_data):,} filas")
        
        return combined_data
    
    def preprocess_data(self, df):
        """Preprocesamiento con análisis específico de fin de semana."""
        logger.info("Preprocesando con análisis de fin de semana...")
        
        # Validar columnas
        required_cols = ['Dia Fecha', 'Canal de Venta', 'Unidad Negocio', 'Venta Pesos']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")
        
        # Limpiar datos
        df = df.copy()
        df['Dia Fecha'] = pd.to_datetime(df['Dia Fecha'], format='%m/%d/%Y', errors='coerce')
        df['Venta Pesos'] = pd.to_numeric(df['Venta Pesos'], errors='coerce').fillna(0)
        df['Canal de Venta'] = df['Canal de Venta'].astype(str).str.strip()
        df['Unidad Negocio'] = df['Unidad Negocio'].astype(str).str.strip()
        
        # Filtrar datos válidos
        df = df.dropna(subset=['Dia Fecha'])
        df = df[df['Venta Pesos'] >= 0]
        df = df[~df['Canal de Venta'].isin(['nan', 'NaN', ''])]
        df = df[~df['Unidad Negocio'].isin(['nan', 'NaN', ''])]
        
        # Crear serie diaria total
        daily_total = df.groupby('Dia Fecha')['Venta Pesos'].sum().sort_index()
        
        # ANÁLISIS ESPECÍFICO DE OUTLIERS POR DÍA DE SEMANA
        weekend_outliers = 0
        weekday_outliers = 0
        
        for dow in range(7):
            dow_data = daily_total[daily_total.index.dayofweek == dow]
            if len(dow_data) > 5:  # Suficientes datos
                Q1 = dow_data.quantile(0.25)
                Q3 = dow_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Límites específicos por día
                if dow >= 5:  # Fin de semana - límites más amplios por alta volatilidad
                    lower_bound = Q1 - 2.0 * IQR  # Más permisivo
                    upper_bound = Q3 + 2.0 * IQR
                else:  # Entre semana - límites estándar
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                # Corregir outliers
                outliers = (dow_data < lower_bound) | (dow_data > upper_bound)
                if outliers.sum() > 0:
                    if dow >= 5:
                        weekend_outliers += outliers.sum()
                    else:
                        weekday_outliers += outliers.sum()
                    
                    # Reemplazar con mediana del mismo día de semana
                    for idx in dow_data[outliers].index:
                        daily_total[idx] = dow_data.median()
        
        if weekend_outliers > 0 or weekday_outliers > 0:
            logger.info(f"Outliers corregidos: {weekday_outliers} entre semana, {weekend_outliers} fin de semana")
        
        # Interpolar faltantes
        if daily_total.isna().sum() > 0:
            daily_total = daily_total.interpolate(method='linear')
        
        # Crear datos por segmento
        daily_segments = df.groupby(['Dia Fecha', 'Canal de Venta', 'Unidad Negocio'])['Venta Pesos'].sum().reset_index()
        
        logger.info(f"Datos procesados: {len(daily_total)} días, {len(daily_segments)} registros")
        logger.info(f"Rango: {daily_total.index.min().strftime('%Y-%m-%d')} a {daily_total.index.max().strftime('%Y-%m-%d')}")
        
        return daily_total, daily_segments
    
    def analyze_weekend_patterns(self, daily_total):
        """Analiza patrones específicos de fin de semana con validación estadística."""
        recent_90 = daily_total.tail(90)
        
        # Calcular estadísticas por día de semana
        dow_stats = {}
        for dow in range(7):
            dow_data = recent_90[recent_90.index.dayofweek == dow]
            if len(dow_data) > 0:
                dow_stats[dow] = {
                    'mean': dow_data.mean(),
                    'median': dow_data.median(),
                    'std': dow_data.std(),
                    'cv': dow_data.std() / dow_data.mean() if dow_data.mean() > 0 else 0,
                    'count': len(dow_data)
                }
        
        logger.info("Patrones por día de semana detectados:")
        dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        for dow, stats in dow_stats.items():
            logger.info(f"  {dias[dow]}: ${stats['mean']:,.0f} (CV: {stats['cv']:.3f})")
        
        # VALIDACIÓN ESTADÍSTICA DE DOMINGOS
        if 6 in dow_stats:
            sunday_stats = dow_stats[6]
            logger.info("\n🔍 VALIDACIÓN ESTADÍSTICA DE DOMINGOS:")
            
            if sunday_stats['count'] < 5:
                logger.warning(f"⚠️  MUESTRA LIMITADA: Solo {sunday_stats['count']} domingos")
                logger.warning("   Factores de domingo tienen BAJA CONFIANZA")
            elif sunday_stats['count'] < 8:
                logger.warning(f"⚠️  MUESTRA PEQUEÑA: {sunday_stats['count']} domingos")
                logger.warning("   Usar factores con PRECAUCIÓN")
            else:
                logger.info(f"✅ MUESTRA ADECUADA: {sunday_stats['count']} domingos")
            
            if sunday_stats['cv'] > 0.5:
                logger.warning("🔴 VOLATILIDAD EXTREMA en domingos - Predicciones imprecisas")
            elif sunday_stats['cv'] > 0.3:
                logger.warning("🟡 ALTA VOLATILIDAD en domingos - Usar intervalos amplios")
            else:
                logger.info("🟢 VOLATILIDAD ACEPTABLE en domingos")
        
        return dow_stats
    
    def train_models(self, daily_total):
        """Entrena ambos modelos con features específicas de fin de semana."""
        logger.info("Entrenando modelos con features de fin de semana...")
        
        # Entrenar XGBoost
        xgb_success = False
        if HAS_XGB and len(daily_total) >= 60:
            X_train, y_train = [], []
            
            for i in range(21, len(daily_total)):
                date = daily_total.index[i]
                dow = date.dayofweek
                
                # Features específicas de fin de semana
                features = {
                    'dow': dow,
                    'is_weekend': int(dow >= 5),
                    'is_saturday': int(dow == 5),
                    'is_sunday': int(dow == 6),
                    'is_weekday': int(dow < 5),
                    'day': date.day,
                    'month': date.month,
                    'is_month_end': int(date.day >= 28),
                    'seasonal': self.monthly_factors.get(date.month, 1.0),
                    'dow_factor': self.dow_factors.get(dow, 1.0),
                    'dow_volatility': self.dow_volatility.get(dow, 0.1),
                    
                    # Lags específicos
                    'lag_1': daily_total.iloc[i-1],
                    'lag_7': daily_total.iloc[i-7],
                    'lag_14': daily_total.iloc[i-14],
                    
                    # Rolling features
                    'rolling_7': daily_total.iloc[i-7:i].mean(),
                    'rolling_14': daily_total.iloc[i-14:i].mean(),
                    
                    # Features específicas de fin de semana
                    'weekend_ratio': 0.0,  # Se calculará después
                    'prev_weekend_avg': 0.0,
                }
                
                # Calcular ratio fin de semana vs entre semana
                if i >= 14:
                    recent_weekend = daily_total.iloc[i-14:i][daily_total.iloc[i-14:i].index.dayofweek >= 5]
                    recent_weekday = daily_total.iloc[i-14:i][daily_total.iloc[i-14:i].index.dayofweek < 5]
                    
                    if len(recent_weekend) > 0 and len(recent_weekday) > 0:
                        features['weekend_ratio'] = recent_weekend.mean() / recent_weekday.mean()
                        features['prev_weekend_avg'] = recent_weekend.mean()
                
                X_train.append(features)
                y_train.append(daily_total.iloc[i])
            
            if len(X_train) >= 20:
                X_train_df = pd.DataFrame(X_train)
                self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
                self.xgb_model.fit(X_train_df, y_train)
                xgb_success = True
                logger.info(f"XGBoost entrenado con {len(X_train)} samples")
        
        # Entrenar Prophet (habilitado para mayor precisión)
        prophet_success = False
        if HAS_PROPHET:  # Rehabilitado para mejor estacionalidad
            prophet_data = pd.DataFrame({
                'ds': daily_total.index,
                'y': daily_total.values
            })
            
            # Regresores específicos de fin de semana
            prophet_data['is_weekend'] = (prophet_data['ds'].dt.dayofweek >= 5).astype(int)
            prophet_data['is_saturday'] = (prophet_data['ds'].dt.dayofweek == 5).astype(int)
            prophet_data['is_sunday'] = (prophet_data['ds'].dt.dayofweek == 6).astype(int)
            prophet_data['is_month_end'] = (prophet_data['ds'].dt.day >= 28).astype(int)
            
            self.prophet_model = Prophet(**self.prophet_params)
            
            # Estacionalidad semanal custom
            self.prophet_model.add_seasonality(
                name='weekly_custom',
                period=7,
                fourier_order=3,
                prior_scale=0.02
            )
            
            # Regresores específicos de fin de semana
            self.prophet_model.add_regressor('is_weekend', prior_scale=0.05)
            self.prophet_model.add_regressor('is_saturday', prior_scale=0.03)
            self.prophet_model.add_regressor('is_sunday', prior_scale=0.08)  # Mayor peso por alta volatilidad
            self.prophet_model.add_regressor('is_month_end', prior_scale=0.015)
            
            self.prophet_model.fit(prophet_data)
            prophet_success = True
            logger.info(f"Prophet entrenado con {len(prophet_data)} observaciones")
        else:
            logger.info("Prophet no disponible - usando XGB + estadístico robusto")
        
        return xgb_success, prophet_success
    
    def predict_with_weekend_calibration(self, daily_total, forecast_dates, sunday_scenario='working'):
        """Predicción con calibración específica de fin de semana."""
        logger.info(f"Generando predicciones con calibración de fin de semana para {len(forecast_dates)} días...")
        if sunday_scenario == 'working':
            logger.info("📅 ESCENARIO: Domingos LABORALES")
        else:
            logger.info("📅 ESCENARIO: Domingos NO LABORALES")
        
        predictions = pd.Series(index=forecast_dates, dtype=float)
        
        for date in forecast_dates:
            dow = date.dayofweek
            is_weekend = dow >= 5
            
            # Predicciones base
            xgb_pred = None
            prophet_pred = None
            
            # XGBoost prediction
            if self.xgb_model is not None:
                features = {
                    'dow': dow,
                    'is_weekend': int(is_weekend),
                    'is_saturday': int(dow == 5),
                    'is_sunday': int(dow == 6),
                    'is_weekday': int(dow < 5),
                    'day': date.day,
                    'month': date.month,
                    'is_month_end': int(date.day >= 28),
                    'seasonal': self.monthly_factors.get(date.month, 1.0),
                    'dow_factor': self.dow_factors.get(dow, 1.0),
                    'dow_volatility': self.dow_volatility.get(dow, 0.1),
                    'lag_1': daily_total.iloc[-1],
                    'lag_7': daily_total.iloc[-7] if len(daily_total) >= 7 else daily_total.mean(),
                    'lag_14': daily_total.iloc[-14] if len(daily_total) >= 14 else daily_total.mean(),
                    'rolling_7': daily_total.tail(7).mean(),
                    'rolling_14': daily_total.tail(14).mean() if len(daily_total) >= 14 else daily_total.mean(),
                    'weekend_ratio': 0.5,  # Default
                    'prev_weekend_avg': daily_total.tail(30).mean() * 0.6,  # Aproximación
                }
                
                X_pred = pd.DataFrame([features])
                xgb_pred = self.xgb_model.predict(X_pred)[0]
            
            # Prophet prediction
            if self.prophet_model is not None:
                future_single = pd.DataFrame({
                    'ds': [date],
                    'is_weekend': [int(is_weekend)],
                    'is_saturday': [int(dow == 5)],
                    'is_sunday': [int(dow == 6)],
                    'is_month_end': [int(date.day >= 28)]
                })
                
                forecast_single = self.prophet_model.predict(future_single)
                prophet_pred = forecast_single['yhat'].iloc[0]
            
            # Predicción estadística ULTRA CONSERVADORA
            recent_same_dow = daily_total[daily_total.index.dayofweek == dow].tail(6)
            if len(recent_same_dow) >= 2:
                # Usar el percentil 40 (más conservador que mediana)
                stat_pred = recent_same_dow.quantile(0.4)
            else:
                # Fallback muy conservador
                recent_avg = daily_total.tail(14).median()  # Usar mediana de 2 semanas
                stat_pred = recent_avg * self.dow_factors.get(dow, 1.0) * 0.9  # 10% más conservador
            
            # ENSEMBLE CONSERVADOR PARA SERIES FINANCIERAS
            if is_weekend:
                # Fin de semana: MÁXIMO peso estadístico (más conservador)
                if dow == 6:  # Domingo - altamente volátil
                    weights = {'stat': 0.80, 'xgb': 0.15, 'prophet': 0.05}
                else:  # Sábado - moderadamente estable
                    weights = {'stat': 0.70, 'xgb': 0.25, 'prophet': 0.05}
            else:
                # Entre semana: equilibrio conservador
                weights = {'stat': 0.50, 'xgb': 0.35, 'prophet': 0.15}
            
            # Calcular ensemble
            ensemble_pred = 0
            weight_sum = 0
            
            if xgb_pred is not None:
                ensemble_pred += weights['xgb'] * xgb_pred
                weight_sum += weights['xgb']
            
            if prophet_pred is not None:
                ensemble_pred += weights['prophet'] * prophet_pred
                weight_sum += weights['prophet']
            
            ensemble_pred += weights['stat'] * stat_pred
            weight_sum += weights['stat']
            
            if weight_sum > 0:
                ensemble_pred = ensemble_pred / weight_sum
            else:
                ensemble_pred = stat_pred
            
            # CALIBRACIÓN ESPECÍFICA POR DÍA DE SEMANA
            # Aplicar factor específico para domingos según escenario
            if dow == 6:  # Domingo
                sunday_factor = self.sunday_factors.get(sunday_scenario, 0.388)
                
                # Aplicar intervalos de confianza para mayor robustez
                confidence_interval = self.sunday_confidence_intervals.get(sunday_scenario, {'lower': 0.2, 'upper': 0.8})
                base_pred = ensemble_pred * sunday_factor
                
                # Ajustar con intervalos de confianza basados en volatilidad histórica
                volatility_adjustment = self.dow_volatility.get(6, 0.5)
                if volatility_adjustment > 0.5:  # Alta volatilidad
                    # Ampliar rango de predicción
                    lower_bound = base_pred * confidence_interval['lower'] / sunday_factor
                    upper_bound = base_pred * confidence_interval['upper'] / sunday_factor
                    calibrated_pred = np.clip(base_pred, lower_bound, upper_bound)
                else:
                    calibrated_pred = base_pred
                
                logger.info(f"    Aplicando factor domingo {sunday_scenario}: {sunday_factor:.3f}")
                logger.info(f"    Rango confianza: ${calibrated_pred * confidence_interval['lower'] / sunday_factor:,.0f} - ${calibrated_pred * confidence_interval['upper'] / sunday_factor:,.0f}")
            else:
                # Aplicar factor normal para otros días
                calibrated_pred = ensemble_pred * self.dow_factors.get(dow, 1.0)
            
            # Límites específicos por volatilidad del día
            recent_avg = daily_total.tail(14).mean()
            volatility = self.dow_volatility.get(dow, 0.1)
            
            if is_weekend:
                if dow == 6:  # Domingo - límites muy amplios por alta volatilidad
                    lower_bound = recent_avg * 0.2  # Puede ser muy bajo
                    upper_bound = recent_avg * 0.8  # No muy alto
                else:  # Sábado - límites moderados
                    lower_bound = recent_avg * 0.6
                    upper_bound = recent_avg * 1.0
            else:
                # Entre semana - límites normales
                lower_bound = recent_avg * 0.8
                upper_bound = recent_avg * 1.2
            
            # Aplicar límites
            calibrated_pred = np.clip(calibrated_pred, lower_bound, upper_bound)
            
            # Asegurar no negativo
            predictions[date] = max(0, calibrated_pred)
            
            # Log detallado para fin de semana
            if is_weekend:
                dia_name = 'Sábado' if dow == 5 else 'Domingo'
                logger.info(f"  {dia_name} {date.strftime('%Y-%m-%d')}: ${predictions[date]:,.0f} (factor: {self.dow_factors.get(dow):.3f})")
        
        logger.info(f"Total predicho: ${predictions.sum():,.0f}")
        
        return predictions
    
    def generate_segments(self, daily_segments, predictions):
        """Genera pronósticos por segmento."""
        logger.info("Calculando shares por segmento...")
        
        # Usar últimos 60 días para shares
        max_date = daily_segments['Dia Fecha'].max()
        cutoff_date = max_date - timedelta(days=60)
        recent = daily_segments[daily_segments['Dia Fecha'] >= cutoff_date]
        
        if len(recent) < 500:
            recent = daily_segments.tail(2000)
        
        # Calcular shares
        segment_totals = recent.groupby(['Canal de Venta', 'Unidad Negocio'])['Venta Pesos'].sum()
        total_sales = segment_totals.sum()
        
        if total_sales <= 0:
            raise ValueError("Total de ventas históricas es cero")
        
        shares = segment_totals / total_sales
        min_share = 0.0001
        shares = shares.clip(lower=min_share)
        shares = shares / shares.sum()
        
        logger.info(f"Calculando shares con {len(recent):,} registros (60 días)")
        
        # Generar forecast por segmento
        forecast_data = []
        for date, total_pred in predictions.items():
            for (canal, unidad), share in shares.items():
                segment_pred = total_pred * share
                segment_pred = max(segment_pred, 50)  # Mínimo más bajo para fines de semana
                
                forecast_data.append({
                    'fecha': date.strftime('%Y-%m-%d'),
                    'Canal de Venta': canal,
                    'Unidad Negocio': unidad,
                    'prediccion': round(segment_pred, 2)
                })
        
        forecast_df = pd.DataFrame(forecast_data)
        logger.info(f"Forecast generado: {len(forecast_df)} registros, {len(shares)} segmentos")
        
        return forecast_df
    
    def save_forecast(self, forecast_df, scenario_suffix=""):
        """Guarda el forecast."""
        os.makedirs('Output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_weekend_opt_{timestamp}{scenario_suffix}.csv"
        filepath = Path("Output") / filename
        
        forecast_df.to_csv(filepath, index=False, float_format="%.2f")
        
        logger.info(f"Forecast guardado: {filename}")
        logger.info(f"Total: ${forecast_df['prediccion'].sum():,.2f}")
        
        return filepath
    
    def show_daily_totals_comparison(self, predictions_working, predictions_non_working, forecast_dates):
        """Muestra comparación de totales diarios entre escenarios."""
        logger.info("\n" + "="*60)
        logger.info("📅 TOTALES DIARIOS POR ESCENARIO")
        logger.info("="*60)
        
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        total_working = 0
        total_non_working = 0
        
        for date in forecast_dates:
            working_pred = predictions_working[date]
            non_working_pred = predictions_non_working[date]
            
            total_working += working_pred
            total_non_working += non_working_pred
            
            day_name = dias_semana[date.dayofweek]
            date_str = date.strftime('%Y-%m-%d')
            
            # Diferencia porcentual
            diff_pct = ((working_pred / non_working_pred) - 1) * 100 if non_working_pred > 0 else 0
            
            if date.dayofweek == 6:  # Domingo
                logger.info(f"🔴 {date_str} ({day_name}):")
                logger.info(f"   💼 Laboral:     ${working_pred:>15,.0f}")
                logger.info(f"   🏠 No Laboral:  ${non_working_pred:>15,.0f}")
                logger.info(f"   📊 Diferencia:  {diff_pct:>15.1f}%")
            else:
                logger.info(f"📅 {date_str} ({day_name}): ${working_pred:,.0f}")
        
        logger.info("-" * 60)
        logger.info(f"💰 TOTAL ESCENARIO LABORAL:     ${total_working:,.0f}")
        logger.info(f"💰 TOTAL ESCENARIO NO LABORAL:  ${total_non_working:,.0f}")
        
        total_diff_pct = ((total_working / total_non_working) - 1) * 100 if total_non_working > 0 else 0
        logger.info(f"📊 DIFERENCIA TOTAL:            {total_diff_pct:+.1f}%")
        
        # Mostrar solo domingos si los hay
        sunday_dates = [d for d in forecast_dates if d.dayofweek == 6]
        if sunday_dates:
            sunday_working = sum(predictions_working[d] for d in sunday_dates)
            sunday_non_working = sum(predictions_non_working[d] for d in sunday_dates)
            sunday_diff_pct = ((sunday_working / sunday_non_working) - 1) * 100 if sunday_non_working > 0 else 0
            
            logger.info("-" * 60)
            logger.info(f"🔴 IMPACTO SOLO DOMINGOS:")
            logger.info(f"   💼 Domingos Laborales:   ${sunday_working:,.0f}")
            logger.info(f"   🏠 Domingos No Laborales: ${sunday_non_working:,.0f}")
            logger.info(f"   📊 Diferencia Domingos:   {sunday_diff_pct:+.1f}%")
        
        logger.info("="*60)
    
    def show_daily_breakdown(self, predictions, forecast_dates, sunday_types):
        """Muestra desglose diario personalizado."""
        logger.info("\n" + "="*60)
        logger.info("📅 DESGLOSE DIARIO PERSONALIZADO")
        logger.info("="*60)
        
        dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        total_pred = 0
        
        for date in forecast_dates:
            pred = predictions[date]
            total_pred += pred
            
            day_name = dias_semana[date.dayofweek]
            date_str = date.strftime('%Y-%m-%d')
            
            if date.dayofweek == 6:  # Domingo
                if date in sunday_types:
                    sunday_type = "LABORAL" if sunday_types[date] == 'working' else "NO LABORAL"
                    logger.info(f"🔴 {date_str} ({day_name} {sunday_type}): ${pred:,.0f}")
                else:
                    logger.info(f"🔴 {date_str} ({day_name}): ${pred:,.0f}")
            else:
                logger.info(f"📅 {date_str} ({day_name}): ${pred:,.0f}")
        
        logger.info("-" * 60)
        logger.info(f"💰 TOTAL PREDICCIÓN PERSONALIZADA: ${total_pred:,.0f}")
        logger.info("="*60)
    
    def predict_with_custom_sundays(self, daily_total, forecast_dates, sunday_types):
        """Predicción personalizada con tipos específicos de domingo por fecha."""
        logger.info(f"Generando predicción personalizada para {len(forecast_dates)} días...")
        
        predictions = pd.Series(index=forecast_dates, dtype=float)
        
        for date in forecast_dates:
            dow = date.dayofweek
            is_weekend = dow >= 5
            
            # Determinar escenario de domingo específico para esta fecha
            if dow == 6 and date in sunday_types:
                sunday_scenario = sunday_types[date]
                scenario_name = "LABORAL" if sunday_scenario == 'working' else "NO LABORAL"
                logger.info(f"📅 {date.strftime('%Y-%m-%d')} (Domingo {scenario_name})")
            else:
                sunday_scenario = 'working'  # Default para no domingos
            
            # Usar la lógica existente pero con escenario personalizado
            predictions[date] = self._predict_single_day(daily_total, date, sunday_scenario)
        
        total_pred = predictions.sum()
        logger.info(f"💰 Total predicho personalizado: ${total_pred:,.0f}")
        
        return predictions
    
    def _predict_single_day(self, daily_total, date, sunday_scenario='working'):
        """Predicción para un día específico (método auxiliar)."""
        dow = date.dayofweek
        is_weekend = dow >= 5
        
        # Predicción estadística ULTRA CONSERVADORA
        recent_same_dow = daily_total[daily_total.index.dayofweek == dow].tail(6)
        if len(recent_same_dow) >= 2:
            stat_pred = recent_same_dow.quantile(0.4)
        else:
            recent_avg = daily_total.tail(14).median()
            stat_pred = recent_avg * self.dow_factors.get(dow, 1.0) * 0.9
        
        # Para este modelo simplificado, usar solo predicción estadística
        base_pred = stat_pred
        
        # Aplicar factor específico para domingos
        if dow == 6:  # Domingo
            sunday_factor = self.sunday_factors.get(sunday_scenario, 0.30)
            calibrated_pred = base_pred * sunday_factor
        else:
            # Aplicar factor normal para otros días
            calibrated_pred = base_pred * self.dow_factors.get(dow, 1.0)
        
        # Límites conservadores
        recent_avg = daily_total.tail(14).median()
        if is_weekend:
            if dow == 6:  # Domingo
                lower_bound = recent_avg * 0.05  # Muy bajo
                upper_bound = recent_avg * 0.50  # Moderado
            else:  # Sábado
                lower_bound = recent_avg * 0.50
                upper_bound = recent_avg * 0.85
        else:
            # Entre semana - límites normales pero conservadores
            lower_bound = recent_avg * 0.80
            upper_bound = recent_avg * 1.15
        
        # Aplicar límites
        final_pred = np.clip(calibrated_pred, lower_bound, upper_bound)
        
        return max(0, final_pred)
    
    def run(self, days=3, input_file=None):
        """Ejecuta el sistema optimizado para fin de semana."""
        logger.info("=== PEPSICO WEEKEND OPTIMIZED FORECAST ===")
        
        try:
            # 1. Cargar y preprocesar
            if input_file:
                logger.info(f"Cargando archivo específico: {input_file}")
                data = pd.read_csv(input_file)
            else:
                data = self.load_all_data()
            
            daily_total, daily_segments = self.preprocess_data(data)
            
            # 2. Analizar patrones de fin de semana
            dow_stats = self.analyze_weekend_patterns(daily_total)
            
            # 3. Entrenar modelos
            xgb_success, prophet_success = self.train_models(daily_total)
            
            if not xgb_success and not prophet_success:
                logger.warning("Solo predicción estadística disponible")
            
            # 4. Generar fechas
            last_date = daily_total.index.max()
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='D')
            
            logger.info(f"Pronóstico: {forecast_dates[0].strftime('%Y-%m-%d')} a {forecast_dates[-1].strftime('%Y-%m-%d')}")
            
            # Identificar días de fin de semana y domingos específicamente
            weekend_days = [date for date in forecast_dates if date.dayofweek >= 5]
            sunday_days = [date for date in forecast_dates if date.dayofweek == 6]
            
            if weekend_days:
                dias_weekend = [('Sábado' if d.dayofweek == 5 else 'Domingo') for d in weekend_days]
                logger.info(f"⚠️ DÍAS FIN DE SEMANA DETECTADOS: {', '.join(dias_weekend)}")
                logger.info("Aplicando calibración específica de fin de semana...")
            
            # 5. Generar predicciones - preguntar tipo de domingos si los hay
            if sunday_days:
                logger.info(f"🔄 DOMINGOS DETECTADOS: {len(sunday_days)} día(s)")
                
                # ADVERTENCIA DE CONFIANZA BASADA EN MUESTRA
                sunday_count = len(daily_total[daily_total.index.dayofweek == 6])
                if sunday_count < 5:
                    logger.warning("⚠️  ADVERTENCIA: Muestra de domingos limitada - Predicciones con BAJA confianza")
                elif sunday_count < 8:
                    logger.warning("⚠️  PRECAUCIÓN: Muestra de domingos pequeña - Usar con reservas")
                else:
                    logger.info("✅ Muestra de domingos adecuada para predicción")
                
                # Preguntar al usuario qué tipo de domingo es cada uno
                sunday_types = get_sunday_type_from_user(sunday_days)
                if sunday_types is None:
                    logger.error("Operación cancelada por el usuario")
                    return None, None
                
                logger.info("\n" + "="*50)
                logger.info("GENERANDO PREDICCIÓN PERSONALIZADA")
                logger.info("="*50)
                
                # Generar predicción única con tipos específicos de domingo
                predictions = self.predict_with_custom_sundays(daily_total, forecast_dates, sunday_types)
                
                # Mostrar desglose diario personalizado
                self.show_daily_breakdown(predictions, forecast_dates, sunday_types)
                
                # Generar segmentos
                forecast_df = self.generate_segments(daily_segments, predictions)
                
                # Guardar archivo único personalizado
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.save_forecast(forecast_df, f"_PERSONALIZADO_{timestamp}")
                
                logger.info("=== COMPLETADO: PREDICCIÓN PERSONALIZADA ===")
                
                return output_file, forecast_df
                
            else:
                # Sin domingos: predicción normal
                predictions = self.predict_with_weekend_calibration(daily_total, forecast_dates)
            
                # 6. Validación (solo para escenario sin domingos)
                recent_avg = daily_total.tail(7).mean()
                predicted_avg = predictions.mean()
                error_pct = abs(predicted_avg / recent_avg - 1) * 100 if recent_avg > 0 else 0
                
                logger.info(f"Validación Weekend Optimized:")
                logger.info(f"  Error estimado: {error_pct:.1f}%")
                
                if weekend_days:
                    weekend_pred_avg = predictions[[d for d in forecast_dates if d.dayofweek >= 5]].mean()
                    weekday_pred_avg = predictions[[d for d in forecast_dates if d.dayofweek < 5]].mean()
                    if len([d for d in forecast_dates if d.dayofweek < 5]) > 0:
                        logger.info(f"  Fin de semana vs Entre semana: {(weekend_pred_avg/weekday_pred_avg-1)*100:+.1f}%")
                
                # 7. Generar segmentos y guardar (escenario normal sin domingos)
                forecast_df = self.generate_segments(daily_segments, predictions)
                output_file = self.save_forecast(forecast_df)
                
                logger.info("=== COMPLETADO EXITOSAMENTE ===")
                
                return output_file, forecast_df
            
        except Exception as e:
            logger.error(f"Error: {e}")
            raise

def get_sunday_type_from_user(sunday_dates):
    """Pregunta al usuario qué tipo de domingo es cada uno"""
    sunday_types = {}
    
    print(f"\n🔍 DETECTADOS {len(sunday_dates)} DOMINGO(S) EN EL PRONÓSTICO:")
    for date in sunday_dates:
        print(f"   📅 {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})")
    
    print("\n💼 TIPOS DE DOMINGO:")
    print("   1 = LABORAL (tiendas abiertas, actividad comercial normal)")
    print("   2 = NO LABORAL (tiendas cerradas, actividad mínima)")
    print()
    
    for date in sunday_dates:
        while True:
            try:
                response = input(f"¿El domingo {date.strftime('%Y-%m-%d')} es? (1=Laboral, 2=No Laboral): ").strip()
                
                if response == '1':
                    sunday_types[date] = 'working'
                    print(f"   ✅ {date.strftime('%Y-%m-%d')}: LABORAL")
                    break
                elif response == '2':
                    sunday_types[date] = 'non_working'
                    print(f"   ✅ {date.strftime('%Y-%m-%d')}: NO LABORAL")
                    break
                else:
                    print("   ❌ Por favor ingresa 1 o 2")
            except KeyboardInterrupt:
                print("\n❌ Operación cancelada")
                return None
    
    return sunday_types

def main():
    parser = argparse.ArgumentParser(description='PepsiCo Financial Forecast - Modelo Certero')
    parser.add_argument('--days', type=int, default=None, help='Días a pronosticar')
    parser.add_argument('--input', type=str, default=None, help='Archivo CSV específico')
    parser.add_argument('--auto', action='store_true', help='Modo automático (sin preguntas)')
    
    args = parser.parse_args()
    
    # Modo interactivo por defecto
    if args.days is None and not args.auto:
        print("=== PEPSICO FINANCIAL FORECAST ===")
        print("💰 Modelo Certero para Series Financieras")
        print()
        
        while True:
            try:
                days_input = input("¿Cuántos días deseas pronosticar? (1-15): ").strip()
                if not days_input:
                    print("Por favor ingresa un número.")
                    continue
                    
                days = int(days_input)
                if 1 <= days <= 15:  # Limitado a 15 días para mayor precisión
                    break
                else:
                    print("❌ Error: días debe estar entre 1 y 15 (mayor precisión)")
            except ValueError:
                print("❌ Error: Por favor ingresa un número válido")
        
        args.days = days
        print(f"✅ Configurado para pronosticar {days} días")
    elif args.days is None:
        args.days = 7  # Default para modo auto
    
    if args.days < 1 or args.days > 31:
        print("Error: días debe estar entre 1 y 31")
        return 1
    
    try:
        forecaster = PepsicoFinancialForecast()
        result = forecaster.run(days=args.days, input_file=args.input)
        
        # Manejar resultado personalizado
        if result is not None:
            output_file, forecast_df = result
            
            print(f"\n🎯 PEPSICO FINANCIAL FORECAST COMPLETADO")
            print(f"💰 Modelo Certero para Series Financieras")
            print(f"📁 Archivo: {os.path.basename(output_file)}")
            print(f"📊 Registros: {len(forecast_df):,}")
            print(f"💰 Total: ${forecast_df['prediccion'].sum():,.2f}")
            print(f"🎯 Segmentos: {forecast_df[['Canal de Venta', 'Unidad Negocio']].drop_duplicates().shape[0]}")
            
            print(f"\n🔬 Método: Predicción Estadística Conservadora")
            print(f"📅 Factores domingo: Laboral (45%) vs No Laboral (15%)")
            print(f"✅ Predicción personalizada basada en tu clasificación de domingos")
        else:
            print(f"\n❌ Operación cancelada por el usuario")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
