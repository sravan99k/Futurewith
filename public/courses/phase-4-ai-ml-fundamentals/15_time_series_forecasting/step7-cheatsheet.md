# Time Series Forecasting - Quick Reference Cheatsheet

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Preprocessing](#preprocessing)
3. [Exploratory Analysis](#exploratory-analysis)
4. [Traditional Methods](#traditional-methods)
5. [Modern Methods](#modern-methods)
6. [Model Evaluation](#model-evaluation)
7. [Production Deployment](#production-deployment)

## Basic Concepts {#basic-concepts}

### Time Series Components

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load time series data
df = pd.read_csv('time_series.csv', parse_dates=['date'], index_col='date')

# Decompose time series
decomposition = seasonal_decompose(df['value'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
```

### Common Time Series Patterns

```python
# Trend Analysis
def analyze_trend(series):
    """Analyze trend in time series"""
    from scipy import stats

    # Linear trend
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)

    return {
        'slope': slope,
        'r_squared': r_value**2,
        'p_value': p_value,
        'trend_strength': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
    }

# Seasonality Detection
def detect_seasonality(series, periods=[7, 30, 365]):
    """Detect seasonal patterns"""
    from statsmodels.tsa.stattools import acf

    autocorr = acf(series.values, nlags=min(len(series)//2, 1000))

    seasonality_info = {}
    for period in periods:
        if period < len(autocorr):
            peak_at_period = autocorr[period]
            seasonality_info[f'period_{period}'] = {
                'autocorrelation': peak_at_period,
                'is_seasonal': abs(peak_at_period) > 0.3
            }

    return seasonality_info
```

## Preprocessing {#preprocessing}

### Handling Missing Values

```python
def handle_missing_values(series, method='interpolate'):
    """Handle missing values in time series"""

    if method == 'forward_fill':
        return series.fillna(method='ffill')

    elif method == 'backward_fill':
        return series.fillna(method='bfill')

    elif method == 'interpolate':
        return series.interpolate(method='time')

    elif method == 'linear_interpolate':
        return series.interpolate()

    elif method == 'spline_interpolate':
        return series.interpolate(method='spline', order=3)

    elif method == 'seasonal_interpolate':
        # Use seasonal pattern for interpolation
        return series.interpolate(method='time').fillna(series.groupby(series.index.dayofyear).transform('mean'))

    elif method == 'knn':
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        values = series.values.reshape(-1, 1)
        imputed = imputer.fit_transform(values)
        return pd.Series(imputed.flatten(), index=series.index)

# Usage
cleaned_series = handle_missing_values(raw_series, method='interpolate')
```

### Resampling and Aggregation

```python
# Resample to different frequencies
def resample_time_series(df, freq='D', agg_func='mean'):
    """Resample time series to different frequency"""

    aggregation_map = {
        'mean': 'mean',
        'sum': 'sum',
        'min': 'min',
        'max': 'max',
        'count': 'count',
        'std': 'std'
    }

    return df.resample(freq).agg(aggregation_map[agg_func])

# Common resampling operations
daily_data = df.resample('D').mean()      # Daily averages
weekly_data = df.resample('W').sum()      # Weekly totals
monthly_data = df.resample('M').mean()    # Monthly averages
hourly_data = df.resample('H').max()      # Hourly maxima
```

### Detrending and Differencing

```python
from scipy import signal

def detrend_series(series, method='linear'):
    """Remove trend from time series"""

    if method == 'linear':
        # Linear detrending
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        trend_line = np.polyval(coeffs, x)
        return series - pd.Series(trend_line, index=series.index)

    elif method == 'polynomial':
        # Polynomial detrending
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 2)
        trend_line = np.polyval(coeffs, x)
        return series - pd.Series(trend_line, index=series.index)

    elif method == 'hp_filter':
        # Hodrick-Prescott filter
        from statsmodels.tsa.filters.hp_filter import hpfilter
        cycle, trend = hpfilter(series.values, lamb=1600)
        return pd.Series(cycle, index=series.index)

def difference_series(series, order=1):
    """Apply differencing to make series stationary"""

    if order == 1:
        return series.diff()
    elif order == 2:
        return series.diff().diff()
    else:
        result = series.copy()
        for i in range(order):
            result = result.diff()
        return result
```

### Outlier Detection and Treatment

```python
def detect_outliers(series, method='iqr', threshold=1.5):
    """Detect outliers in time series"""

    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers

    elif method == 'z_score':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold
        return outliers

    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1, random_state=42)
        outliers = model.fit_predict(series.values.reshape(-1, 1)) == -1
        return pd.Series(outliers, index=series.index)

    elif method == 'rolling_stats':
        # Rolling window outlier detection
        rolling_mean = series.rolling(window=24).mean()
        rolling_std = series.rolling(window=24).std()
        outliers = np.abs(series - rolling_mean) > (threshold * rolling_std)
        return outliers

def treat_outliers(series, outliers, method='remove'):
    """Treat detected outliers"""

    if method == 'remove':
        return series[~outliers]

    elif method == 'interpolate':
        return series.copy().mask(outliers).interpolate()

    elif method == 'winsorize':
        from scipy.stats import mstats
        return pd.Series(mstats.winsorize(series.values, limits=0.05), index=series.index)

    elif method == 'cap_floor':
        Q1 = series.quantile(0.05)
        Q3 = series.quantile(0.95)
        return series.clip(lower=Q1, upper=Q3)
```

## Exploratory Analysis {#exploratory-analysis}

### Statistical Summary

```python
def time_series_summary(series):
    """Generate comprehensive time series summary"""

    summary = {
        'basic_stats': series.describe(),
        'missing_values': series.isnull().sum(),
        'duplicates': series.duplicated().sum(),
        'data_types': series.dtypes,
        'index_info': {
            'start': series.index.min(),
            'end': series.index.max(),
            'frequency': pd.infer_freq(series.index),
            'length': len(series)
        }
    }

    # Stationarity tests
    from statsmodels.tsa.stattools import adfuller

    adf_result = adfuller(series.dropna())
    summary['stationarity'] = {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'is_stationary': adf_result[1] < 0.05
    }

    # Normality test
    from scipy.stats import jarque_bera
    jb_stat, jb_pvalue = jarque_bera(series.dropna())
    summary['normality'] = {
        'jarque_bera_statistic': jb_stat,
        'p_value': jb_pvalue,
        'is_normal': jb_pvalue > 0.05
    }

    return summary

# Usage
summary = time_series_summary(time_series_data)
print(summary)
```

### Visualization Functions

```python
def plot_time_series_analysis(series, decomposition=None):
    """Comprehensive time series visualization"""

    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle('Time Series Analysis', fontsize=16)

    # Original series
    axes[0, 0].plot(series.index, series.values)
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Rolling statistics
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()

    axes[0, 1].plot(series.index, series.values, label='Original')
    axes[0, 1].plot(rolling_mean.index, rolling_mean.values, label='Rolling Mean')
    axes[0, 1].plot(rolling_std.index, rolling_std.values, label='Rolling Std')
    axes[0, 1].set_title('Rolling Statistics (12 periods)')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Histogram
    axes[1, 0].hist(series.dropna(), bins=30, alpha=0.7)
    axes[1, 0].set_title('Distribution')

    # Box plot by period
    if hasattr(series.index, 'month'):
        monthly_data = [series[series.index.month == month].values for month in range(1, 13)]
        axes[1, 1].boxplot(monthly_data, labels=range(1, 13))
        axes[1, 1].set_title('Monthly Distribution')

    # Autocorrelation
    from statsmodels.tsa.stattools import acf
    autocorr = acf(series.values, nlags=40)
    axes[2, 0].plot(range(len(autocorr)), autocorr)
    axes[2, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2, 0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('Autocorrelation Function')
    axes[2, 0].set_xlabel('Lag')

    # Partial autocorrelation
    from statsmodels.tsa.stattools import pacf
    partial_autocorr = pacf(series.values, nlags=40)
    axes[2, 1].plot(range(len(partial_autocorr)), partial_autocorr)
    axes[2, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].set_title('Partial Autocorrelation Function')
    axes[2, 1].set_xlabel('Lag')

    # Q-Q plot
    from scipy import stats
    stats.probplot(series.dropna(), dist="norm", plot=axes[3, 0])
    axes[3, 0].set_title('Q-Q Plot')

    # Residuals (if decomposition provided)
    if decomposition is not None:
        axes[3, 1].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3, 1].set_title('Residuals')
        axes[3, 1].tick_params(axis='x', rotation=45)
    else:
        axes[3, 1].axis('off')

    plt.tight_layout()
    return fig
```

## Traditional Methods {#traditional-methods}

### ARIMA Models

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def auto_arima(series, max_p=5, max_d=2, max_q=5, seasonal=False, m=12):
    """Automatic ARIMA model selection"""

    import itertools
    from sklearn.metrics import mean_absolute_error

    p = d = q = range(0, max_p + 1)
    if seasonal:
        P = D = Q = range(0, max_p + 1)
        s = [m]
    else:
        P = D = Q = [0]
        s = [0]

    # Generate all combinations
    if seasonal:
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = list(itertools.product(P, D, Q, s))
    else:
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(0, 0, 0, 0)]

    best_aic = float('inf')
    best_params = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                if seasonal:
                    model = SARIMAX(series, order=param, seasonal_order=param_seasonal)
                else:
                    model = ARIMA(series, order=param)

                fitted_model = model.fit(disp=False)
                aic = fitted_model.aic

                if aic < best_aic:
                    best_aic = aic
                    best_params = (param, param_seasonal)

            except Exception as e:
                continue

    return best_params, best_aic

def fit_arima_model(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    """Fit ARIMA/SARIMA model"""

    if seasonal_order[3] > 0:  # Seasonal model
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    else:  # Non-seasonal model
        model = ARIMA(series, order=order)

    fitted_model = model.fit(disp=False)
    return fitted_model

def forecast_arima(model, steps=12, alpha=0.05):
    """Generate forecast with confidence intervals"""

    forecast = model.forecast(steps=steps)
    forecast_ci = model.get_forecast(steps=steps).conf_int(alpha=alpha)

    return {
        'forecast': forecast,
        'lower_ci': forecast_ci.iloc[:, 0],
        'upper_ci': forecast_ci.iloc[:, 1]
    }
```

### Exponential Smoothing

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_exponential_smoothing(series, trend='add', seasonal='add', seasonal_periods=12):
    """Fit various exponential smoothing models"""

    if seasonal == 'none':
        # Simple Exponential Smoothing
        model = ExponentialSmoothing(series, trend=trend)
    else:
        # Holt-Winters Exponential Smoothing
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )

    fitted_model = model.fit()
    return fitted_model

def exponential_smoothing_forecast(model, steps=12):
    """Generate forecast from exponential smoothing model"""

    forecast = model.forecast(steps=steps)

    return {
        'forecast': forecast,
        'fitted_values': model.fittedvalues,
        'residuals': model.resid
    }
```

### Seasonal Decomposition

```python
def seasonal_decomposition_analysis(series, model='additive', period=12):
    """Perform seasonal decomposition"""

    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(series, model=model, period=period)

    # Analyze components
    analysis = {
        'trend_strength': calculate_trend_strength(series, decomposition.trend),
        'seasonal_strength': calculate_seasonal_strength(series, decomposition.seasonal),
        'residual_variance': decomposition.resid.var()
    }

    return decomposition, analysis

def calculate_trend_strength(series, trend_component):
    """Calculate strength of trend component"""

    detrended = series - trend_component
    trend_var = trend_component.var()
    detrended_var = detrended.var()

    if trend_var + detrended_var > 0:
        return trend_var / (trend_var + detrended_var)
    return 0

def calculate_seasonal_strength(series, seasonal_component):
    """Calculate strength of seasonal component"""

    deseasonalized = series - seasonal_component
    seasonal_var = seasonal_component.var()
    deseasonalized_var = deseasonalized.var()

    if seasonal_var + deseasonalized_var > 0:
        return seasonal_var / (seasonal_var + deseasonalized_var)
    return 0
```

## Modern Methods {#modern-methods}

### Facebook Prophet

```python
try:
    import prophet

    def prophet_forecast(series, periods=365, changepoint_prior_scale=0.05):
        """Forecast using Facebook Prophet"""

        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        # Initialize and fit model
        model = prophet.Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(df_prophet)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)

        # Generate forecast
        forecast = model.predict(future)

        return model, forecast

    def prophet_components_plot(model, forecast):
        """Plot Prophet forecast components"""

        fig = model.plot_components(forecast)
        return fig

except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
```

### LSTM Neural Networks

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(series, look_back=60, test_size=0.2):
    """Prepare data for LSTM model"""

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into train and test
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test, scaler

def build_lstm_model(input_shape):
    """Build LSTM model architecture"""

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def lstm_forecast(model, last_sequence, scaler, steps=30):
    """Generate LSTM forecast"""

    predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(steps):
        # Make prediction
        pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(pred[0, 0])

        # Update sequence
        current_sequence = np.append(current_sequence[1:], pred[0, 0])

    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions.flatten()
```

### XGBoost for Time Series

```python
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def create_features(series, lags=[1, 2, 3, 7, 14, 30], rolling_windows=[7, 14, 30]):
    """Create features for ML models"""

    df = pd.DataFrame({'value': series.values}, index=series.index)

    # Lag features
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Rolling statistics
    for window in rolling_windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()

    # Time-based features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear

    # Remove rows with NaN values
    df = df.dropna()

    return df

def ml_forecast(series, test_size=0.2, model_type='xgboost'):
    """Forecast using machine learning models"""

    # Create features
    df = create_features(series)

    # Split data
    split_index = int(len(df) * (1 - test_size))
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    # Prepare features and target
    feature_columns = [col for col in df.columns if col != 'value']
    X_train = train_data[feature_columns]
    y_train = train_data['value']
    X_test = test_data[feature_columns]
    y_test = test_data['value']

    # Train model
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    return {
        'model': model,
        'predictions': predictions,
        'actual': y_test.values,
        'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions))
    }
```

## Model Evaluation {#model-evaluation}

### Cross-Validation for Time Series

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def time_series_cv_score(model, series, cv_splits=5, test_size=0.2):
    """Time series cross-validation"""

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []

    for train_index, test_index in tscv.split(series):
        train_data = series.iloc[train_index]
        test_data = series.iloc[test_index]

        # Fit model on training data
        model.fit(train_data)

        # Make predictions on test data
        predictions = model.predict(test_data)

        # Calculate metrics
        mae = mean_absolute_error(test_data.values, predictions)
        mse = mean_squared_error(test_data.values, predictions)
        rmse = np.sqrt(mse)

        scores.append({
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        })

    # Calculate average scores
    avg_scores = {
        'mae': np.mean([score['mae'] for score in scores]),
        'mse': np.mean([score['mse'] for score in scores]),
        'rmse': np.mean([score['rmse'] for score in scores])
    }

    return avg_scores, scores

def evaluate_forecast(forecast, actual, model_name=''):
    """Comprehensive forecast evaluation"""

    # Basic metrics
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100

    # Advanced metrics
    from sklearn.metrics import r2_score
    r2 = r2_score(actual, forecast)

    # Direction accuracy
    actual_diff = np.diff(actual)
    forecast_diff = np.diff(forecast)
    direction_accuracy = np.mean(np.sign(actual_diff) == np.sign(forecast_diff)) * 100

    metrics = {
        'model': model_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }

    return metrics

def plot_forecast_comparison(actual, forecast, forecast_index, confidence_intervals=None):
    """Plot actual vs forecast comparison"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time series plot
    axes[0].plot(forecast_index, actual, label='Actual', linewidth=2)
    axes[0].plot(forecast_index, forecast, label='Forecast', linewidth=2)

    if confidence_intervals is not None:
        axes[0].fill_between(
            forecast_index,
            confidence_intervals['lower'],
            confidence_intervals['upper'],
            alpha=0.3,
            label='Confidence Interval'
        )

    axes[0].set_title('Time Series Forecast Comparison')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals plot
    residuals = actual - forecast
    axes[1].scatter(forecast, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_title('Residuals vs Forecast')
    axes[1].set_xlabel('Forecast')
    axes[1].set_ylabel('Residuals')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

## Production Deployment {#production-deployment}

### Model Pipeline

```python
class TimeSeriesPipeline:
    """Complete time series forecasting pipeline"""

    def __init__(self, model_type='arima'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False

    def preprocess(self, series):
        """Preprocess time series data"""

        # Handle missing values
        series = series.interpolate(method='time')

        # Remove outliers
        outliers = detect_outliers(series)
        series = series[~outliers]

        # Make stationary if needed
        if self.model_type in ['arima', 'sarima']:
            # Check stationarity
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(series.dropna())
            if adf_result[1] > 0.05:
                series = series.diff().dropna()

        return series

    def fit(self, series):
        """Fit the forecasting model"""

        # Preprocess
        processed_series = self.preprocess(series)

        # Fit model based on type
        if self.model_type == 'arima':
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(processed_series, order=(1, 1, 1))
            self.model = self.model.fit()

        elif self.model_type == 'prophet':
            # Implementation for Prophet
            pass

        elif self.model_type == 'lstm':
            # Implementation for LSTM
            pass

        elif self.model_type == 'xgboost':
            # Create features and fit XGBoost
            df = create_features(processed_series)
            feature_columns = [col for col in df.columns if col != 'value']
            self.feature_columns = feature_columns

            X = df[feature_columns]
            y = df['value']

            self.model = xgb.XGBRegressor(random_state=42)
            self.model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, steps=12):
        """Generate forecast"""

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if self.model_type == 'arima':
            forecast = self.model.forecast(steps=steps)
            return forecast

        elif self.model_type == 'xgboost':
            # This is a simplified version - in practice, you'd need to handle
            # the rolling nature of features
            # Implementation depends on specific use case
            pass

    def save_model(self, filepath):
        """Save fitted model"""

        import pickle

        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath):
        """Load fitted model"""

        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        pipeline = cls(model_type=model_data['model_type'])
        pipeline.model = model_data['model']
        pipeline.scaler = model_data['scaler']
        pipeline.feature_columns = model_data['feature_columns']
        pipeline.is_fitted = model_data['is_fitted']

        return pipeline
```

### Real-time Monitoring

```python
class ForecastMonitor:
    """Monitor forecast performance in real-time"""

    def __init__(self, threshold_mae=None, threshold_mape=None):
        self.threshold_mae = threshold_mae
        self.threshold_mape = threshold_mape
        self.metrics_history = []
        self.alert_history = []

    def evaluate_prediction(self, actual_value, predicted_value, timestamp):
        """Evaluate a single prediction"""

        mae = abs(actual_value - predicted_value)
        mape = abs((actual_value - predicted_value) / actual_value) * 100

        metric_record = {
            'timestamp': timestamp,
            'actual': actual_value,
            'predicted': predicted_value,
            'mae': mae,
            'mape': mape
        }

        self.metrics_history.append(metric_record)

        # Check for alerts
        alerts = []
        if self.threshold_mae and mae > self.threshold_mae:
            alerts.append(f"MAE threshold exceeded: {mae:.2f} > {self.threshold_mae}")

        if self.threshold_mape and mape > self.threshold_mape:
            alerts.append(f"MAPE threshold exceeded: {mape:.2f}% > {self.threshold_mape}%")

        if alerts:
            alert_record = {
                'timestamp': timestamp,
                'alerts': alerts,
                'mae': mae,
                'mape': mape
            }
            self.alert_history.append(alert_record)

        return alerts

    def get_performance_summary(self, window_size=100):
        """Get performance summary for recent predictions"""

        recent_metrics = self.metrics_history[-window_size:] if len(self.metrics_history) > window_size else self.metrics_history

        if not recent_metrics:
            return {}

        mae_values = [m['mae'] for m in recent_metrics]
        mape_values = [m['mape'] for m in recent_metrics]

        summary = {
            'count': len(recent_metrics),
            'avg_mae': np.mean(mae_values),
            'avg_mape': np.mean(mape_values),
            'max_mae': np.max(mae_values),
            'max_mape': np.max(mape_values),
            'alerts_count': len(self.alert_history)
        }

        return summary

    def plot_performance(self):
        """Plot performance metrics over time"""

        if not self.metrics_history:
            return None

        timestamps = [m['timestamp'] for m in self.metrics_history]
        mae_values = [m['mae'] for m in self.metrics_history]
        mape_values = [m['mape'] for m in self.metrics_history]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # MAE plot
        axes[0].plot(timestamps, mae_values, 'b-', linewidth=1)
        if self.threshold_mae:
            axes[0].axhline(y=self.threshold_mae, color='r', linestyle='--', label=f'Threshold: {self.threshold_mae}')
        axes[0].set_title('Mean Absolute Error Over Time')
        axes[0].set_ylabel('MAE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAPE plot
        axes[1].plot(timestamps, mape_values, 'g-', linewidth=1)
        if self.threshold_mape:
            axes[1].axhline(y=self.threshold_mape, color='r', linestyle='--', label=f'Threshold: {self.threshold_mape}%')
        axes[1].set_title('Mean Absolute Percentage Error Over Time')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('MAPE (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
```

This comprehensive cheatsheet covers all essential aspects of time series forecasting, from basic concepts to production deployment. Use it as a quick reference guide for your forecasting projects!
