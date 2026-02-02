# Time Series Forecasting - Interview Preparation Guide

## Table of Contents

1. [Fundamental Concepts](#fundamental-concepts)
2. [Statistical Foundations](#statistical-foundations)
3. [Classical Methods](#classical-methods)
4. [Modern Approaches](#modern-approaches)
5. [Model Selection](#model-selection)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Real-world Scenarios](#real-world-scenarios)
8. [Coding Challenges](#coding-challenges)
9. [Advanced Topics](#advanced-topics)

## Fundamental Concepts {#fundamental-concepts}

### Q1: What is a time series and what are its key components?

**Answer:**
A time series is a sequence of data points collected or recorded at successive time intervals. The key components are:

1. **Trend**: Long-term movement or direction in the data (upward, downward, or horizontal)
2. **Seasonality**: Regular, predictable patterns that repeat at fixed intervals (daily, weekly, monthly, yearly)
3. **Cyclical**: Patterns that occur at irregular intervals, often related to economic cycles
4. **Irregular/Random**: Unpredictable variations that don't follow a pattern

**Mathematical representation:**

```
Y(t) = Trend(t) + Seasonal(t) + Cyclical(t) + Irregular(t)
```

**Code Example:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample time series with components
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')

# Create components
trend = np.linspace(100, 150, len(dates))  # Upward trend
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # Yearly seasonality
cyclical = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / (365*4))  # 4-year cycle
irregular = np.random.normal(0, 3, len(dates))  # Random noise

# Combine components
time_series = trend + seasonal + cyclical + irregular
series = pd.Series(time_series, index=dates)

# Decompose
decomposition = seasonal_decompose(series, model='additive')
decomposition.plot()
plt.title('Time Series Decomposition')
plt.show()
```

### Q2: What is stationarity and why is it important?

**Answer:**
**Stationarity** means that the statistical properties of a time series (mean, variance, and covariance) remain constant over time.

**Types of Stationarity:**

1. **Strict Stationarity**: Joint probability distribution is invariant to time shifts
2. **Weak Stationarity**: Constant mean, constant variance, and autocovariance depends only on lag

**Why it's important:**

- Many time series models (ARIMA, VAR) assume stationarity
- Statistical inference is valid only for stationary processes
- Non-stationary data can lead to spurious correlations

**Testing Stationarity:**

```python
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(timeseries):
    """Test stationarity using ADF and KPSS tests"""

    # Augmented Dickey-Fuller test
    adf_result = adfuller(timeseries.dropna())
    print(f'ADF Test Results:')
    print(f'ADF Statistic: {adf_result[0]:.6f}')
    print(f'p-value: {adf_result[1]:.6f}')
    print(f'Is Stationary (ADF): {adf_result[1] < 0.05}')

    # KPSS test
    kpss_result = kpss(timeseries.dropna())
    print(f'\\nKPSS Test Results:')
    print(f'KPSS Statistic: {kpss_result[0]:.6f}')
    print(f'p-value: {kpss_result[1]:.6f}')
    print(f'Is Stationary (KPSS): {kpss_result[1] > 0.05}')

    return {
        'adf_stationary': adf_result[1] < 0.05,
        'kpss_stationary': kpss_result[1] > 0.05
    }

# Example usage
test_stationarity(your_time_series)
```

### Q3: How do you handle non-stationary data?

**Answer:**

**1. Differencing:**

```python
# First-order differencing
first_diff = series.diff().dropna()

# Second-order differencing
second_diff = series.diff().diff().dropna()

# Seasonal differencing
seasonal_diff = series.diff(12).dropna()
```

**2. Transformation:**

```python
import numpy as np

# Log transformation (for exponential growth)
log_series = np.log(series)

# Square root transformation
sqrt_series = np.sqrt(series)

# Box-Cox transformation
from scipy.stats import boxcox
transformed_series, lambda_param = boxcox(series)
```

**3. Detrending:**

```python
from scipy import signal

# Linear detrending
x = np.arange(len(series))
coeffs = np.polyfit(x, series.values, 1)
trend_line = np.polyval(coeffs, x)
detrended = series - pd.Series(trend_line, index=series.index)

# Polynomial detrending
coeffs = np.polyfit(x, series.values, 2)
trend_line = np.polyval(coeffs, x)
detrended = series - pd.Series(trend_line, index=series.index)
```

## Statistical Foundations {#statistical-foundations}

### Q4: Explain autocorrelation and partial autocorrelation

**Answer:**

**Autocorrelation Function (ACF):**

- Measures correlation between observations at different time lags
- Helps identify MA (Moving Average) components
- Formula: ρ(h) = Cov(X*t, X*{t+h}) / Var(X_t)

**Partial Autocorrelation Function (PACF):**

- Measures correlation between X*t and X*{t+h} after removing the effect of intermediate lags
- Helps identify AR (Autoregressive) components
- Calculated using Yule-Walker equations

**Code Example:**

```python
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt

def plot_acf_pacf(series, lags=40):
    """Plot ACF and PACF for order identification"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Calculate ACF and PACF
    acf_values = acf(series.values, nlags=lags, alpha=0.05)
    pacf_values = pacf(series.values, nlags=lags, alpha=0.05)

    # Plot ACF
    axes[0].stem(range(len(acf_values[0])), acf_values[0])
    axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')

    # Plot PACF
    axes[1].stem(range(len(pacf_values[0])), pacf_values[0])
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')

    plt.tight_layout()
    plt.show()

    return acf_values, pacf_values

# Usage
acf_vals, pacf_vals = plot_acf_pacf(your_time_series)
```

### Q5: What are white noise and random walk processes?

**Answer:**

**White Noise:**

- A sequence of independent, identically distributed random variables
- Mean = 0, Variance = σ², No autocorrelation
- Used as a baseline for model comparisons

**Random Walk:**

- X*t = X*{t-1} + ε_t, where ε_t is white noise
- Non-stationary (variance increases with time)
- Cumulative sum of white noise

**Code Examples:**

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_processes(n=1000):
    """Generate white noise and random walk"""

    # White noise
    white_noise = np.random.normal(0, 1, n)

    # Random walk
    random_walk = np.cumsum(white_noise)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].plot(white_noise)
    axes[0].set_title('White Noise Process')
    axes[0].set_ylabel('Value')

    axes[1].plot(random_walk)
    axes[1].set_title('Random Walk Process')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

    return white_noise, random_walk

# Test stationarity
from statsmodels.tsa.stattools import adfuller

white_noise, random_walk = generate_processes(1000)

print("White Noise Stationarity:")
print(f"ADF p-value: {adfuller(white_noise)[1]:.6f}")

print("\\nRandom Walk Stationarity:")
print(f"ADF p-value: {adfuller(random_walk)[1]:.6f}")
```

## Classical Methods {#classical-methods}

### Q6: Explain ARIMA models and how to select parameters

**Answer:**

**ARIMA(p,d,q) Components:**

- **AR (p)**: Autoregressive terms - relationship between current and past values
- **I (d)**: Integration - differencing order to achieve stationarity
- **MA (q)**: Moving average terms - relationship between current and past forecast errors

**Parameter Selection:**

```python
from statsmodels.tsa.arima.model import ARIMA
import itertools

def auto_arima_selection(series, max_p=5, max_d=2, max_q=5):
    """Automatic ARIMA parameter selection using AIC"""

    # Generate parameter combinations
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)

    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_params = None

    for param in pdq:
        try:
            model = ARIMA(series, order=param)
            fitted_model = model.fit()
            aic = fitted_model.aic

            if aic < best_aic:
                best_aic = aic
                best_params = param

        except Exception as e:
            continue

    return best_params, best_aic

# Usage
best_order, best_aic = auto_arima_selection(your_time_series)
print(f"Best ARIMA order: {best_order}, AIC: {best_aic:.2f}")
```

**SARIMA for Seasonal Data:**

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """Fit SARIMA model"""

    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    fitted_model = model.fit()

    return fitted_model

# Forecasting
def forecast_arima(model, steps=12):
    """Generate forecast with confidence intervals"""

    forecast = model.forecast(steps=steps)
    forecast_ci = model.get_forecast(steps=steps).conf_int()

    return forecast, forecast_ci
```

### Q7: What is exponential smoothing and when to use it?

**Answer:**

**Simple Exponential Smoothing:**

- Used for data with no trend or seasonality
- Formula: S*t = α * X_t + (1-α) * S*{t-1}
- α (alpha) is the smoothing parameter (0 < α < 1)

**Holt's Linear Trend:**

- Handles linear trend but no seasonality
- Formula includes trend component

**Holt-Winters (Triple Exponential Smoothing):**

- Handles trend and seasonality
- Three smoothing parameters: α (level), β (trend), γ (seasonality)

**Code Example:**

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def exponential_smoothing_forecast(series, trend='add', seasonal='add', seasonal_periods=12):
    """Fit exponential smoothing models"""

    if seasonal == 'none':
        # Simple or Holt's method
        model = ExponentialSmoothing(series, trend=trend)
    else:
        # Holt-Winters method
        model = ExponentialSmoothing(
            series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )

    fitted_model = model.fit()

    # Generate forecast
    forecast = fitted_model.forecast(steps=12)

    return fitted_model, forecast

# Compare different methods
def compare_exponential_smoothing(series):
    """Compare different exponential smoothing methods"""

    methods = [
        ('add', 'none'),      # Simple exponential smoothing
        ('add', 'none'),      # Holt's linear trend
        ('add', 'add'),       # Holt-Winters additive
        ('add', 'mul'),       # Holt-Winters multiplicative
    ]

    results = {}

    for trend, seasonal in methods:
        try:
            if seasonal == 'none':
                model = ExponentialSmoothing(series, trend=trend)
            else:
                model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=12)

            fitted_model = model.fit()
            aic = fitted_model.aic

            results[f'{trend}_{seasonal}'] = {
                'model': fitted_model,
                'aic': aic,
                'forecast': fitted_model.forecast(steps=12)
            }

        except Exception as e:
            print(f"Error fitting {trend}_{seasonal}: {e}")

    return results
```

## Modern Approaches {#modern-approaches}

### Q8: Explain deep learning approaches for time series forecasting

**Answer:**

**LSTM (Long Short-Term Memory):**

- Handles long-term dependencies
- Gates: input, forget, output
- Suitable for complex patterns

**CNN for Time Series:**

- Convolution layers extract local patterns
- 1D convolutions for temporal features
- Often combined with other architectures

**Transformer Models:**

- Attention mechanism
- Parallel processing
- Good for long sequences

**Code Example - LSTM:**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(series, look_back=60):
    """Prepare data for LSTM"""

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(input_shape):
    """Build LSTM model"""

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

def train_lstm_model(X_train, y_train, X_val, y_val):
    """Train LSTM model"""

    model = build_lstm_model((X_train.shape[1], 1))

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    return model, history
```

**Prophet (Facebook's time series forecasting tool):**

```python
try:
    import prophet

    def prophet_forecast(series, periods=365):
        """Forecast using Prophet"""

        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        # Initialize and fit model
        model = prophet.Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(df)

        # Create future dataframe and forecast
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return model, forecast

    def prophet_components_plot(model, forecast):
        """Plot Prophet forecast components"""

        fig = model.plot_components(forecast)
        return fig

except ImportError:
    print("Prophet not installed. Install with: pip install prophet")
```

### Q9: What are ensemble methods for time series?

**Answer:**

**Why Ensemble Methods?**

- Combine multiple models to improve accuracy
- Reduce overfitting
- Capture different aspects of the data

**Common Ensemble Techniques:**

**1. Simple Averaging:**

```python
def simple_ensemble_forecast(forecasts, weights=None):
    """Simple average ensemble"""

    if weights is None:
        weights = np.ones(len(forecasts)) / len(forecasts)

    ensemble_forecast = np.average(forecasts, axis=0, weights=weights)
    return ensemble_forecast
```

**2. Weighted Ensemble:**

```python
def weighted_ensemble_forecast(forecasts, weights):
    """Weighted ensemble based on validation performance"""

    # Normalize weights
    weights = weights / np.sum(weights)

    ensemble_forecast = np.average(forecasts, axis=0, weights=weights)
    return ensemble_forecast
```

**3. Stacking:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def stacking_ensemble(models, X_meta, y_meta):
    """Stacking ensemble using meta-learner"""

    # Generate meta-features using out-of-fold predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((len(y_meta), len(models)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta)):
        for i, model in enumerate(models):
            # Train on fold
            model.fit(X_meta[train_idx], y_meta[train_idx])
            # Predict on validation fold
            meta_features[val_idx, i] = model.predict(X_meta[val_idx])

    # Train meta-learner
    meta_learner = LinearRegression()
    meta_learner.fit(meta_features, y_meta)

    return meta_learner
```

**4. Dynamic Ensemble Selection:**

```python
def dynamic_ensemble_selection(forecasts, recent_errors, threshold=0.1):
    """Select models based on recent performance"""

    # Calculate recent performance for each model
    performance_scores = []
    for i, errors in enumerate(recent_errors):
        performance_scores.append(np.mean(errors))

    # Select models below threshold
    selected_models = [i for i, score in enumerate(performance_scores)
                      if score < threshold]

    if not selected_models:
        # If no model passes threshold, use all
        selected_models = range(len(forecasts))

    # Ensemble selected models
    selected_forecasts = [forecasts[i] for i in selected_models]
    ensemble_forecast = np.mean(selected_forecasts, axis=0)

    return ensemble_forecast, selected_models
```

## Model Selection {#model-selection}

### Q10: How do you choose between different forecasting methods?

**Answer:**

**Selection Criteria:**

1. **Data characteristics** (trend, seasonality, stationarity)
2. **Forecast horizon** (short vs long-term)
3. **Data availability** (volume, quality)
4. **Computational resources**
5. **Interpretability requirements**
6. **Business constraints**

**Decision Framework:**

```python
def select_forecasting_method(series, forecast_horizon, requirements):
    """Select appropriate forecasting method"""

    # Analyze data characteristics
    from statsmodels.tsa.stattools import adfuller

    is_stationary = adfuller(series.dropna())[1] < 0.05
    has_seasonality = detect_seasonality(series)['has_seasonal']
    length = len(series)

    # Decision logic
    if length < 50:
        return "Simple moving average or exponential smoothing"

    elif forecast_horizon > 50:
        return "Machine learning or deep learning methods"

    elif not is_stationary:
        if has_seasonality:
            return "SARIMA or Prophet"
        else:
            return "ARIMA with differencing"

    elif has_seasonality:
        if length > 200:
            return "SARIMA, Prophet, or LSTM"
        else:
            return "SARIMA or Exponential Smoothing"

    else:
        return "ARIMA, Exponential Smoothing, or Linear Regression"

def detect_seasonality(series):
    """Detect seasonality in time series"""

    from statsmodels.tsa.stattools import acf

    autocorr = acf(series.values, nlags=min(len(series)//2, 100))

    # Check for peaks at seasonal lags
    seasonal_lags = [7, 30, 365]  # Weekly, monthly, yearly

    has_seasonal = False
    seasonal_strength = 0

    for lag in seasonal_lags:
        if lag < len(autocorr):
            peak_strength = abs(autocorr[lag])
            if peak_strength > 0.3:  # Threshold for seasonality
                has_seasonal = True
                seasonal_strength = max(seasonal_strength, peak_strength)

    return {
        'has_seasonal': has_seasonal,
        'seasonal_strength': seasonal_strength
    }
```

## Evaluation Metrics {#evaluation-metrics}

### Q11: What are the appropriate evaluation metrics for time series?

**Answer:**

**Scale-Dependent Metrics:**

```python
def calculate_scale_dependent_metrics(actual, predicted):
    """Calculate MAE, MSE, RMSE"""

    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }
```

**Percentage-Based Metrics:**

```python
def calculate_percentage_metrics(actual, predicted):
    """Calculate MAPE, SMAPE"""

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # Symmetric Mean Absolute Percentage Error
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape = np.mean(np.abs(actual - predicted) / denominator) * 100

    return {
        'mape': mape,
        'smape': smape
    }
```

**Correlation-Based Metrics:**

```python
def calculate_correlation_metrics(actual, predicted):
    """Calculate correlation-based metrics"""

    from scipy.stats import pearsonr

    correlation, p_value = pearsonr(actual, predicted)

    # Coefficient of determination (R²)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'correlation': correlation,
        'p_value': p_value,
        'r2': r2
    }
```

**Direction Accuracy:**

```python
def calculate_direction_accuracy(actual, predicted):
    """Calculate accuracy of directional predictions"""

    actual_diff = np.diff(actual)
    predicted_diff = np.diff(predicted)

    direction_accuracy = np.mean(
        np.sign(actual_diff) == np.sign(predicted_diff)
    ) * 100

    return direction_accuracy
```

**Cross-Validation for Time Series:**

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv_evaluation(model, series, cv_splits=5, test_size=0.2):
    """Time series cross-validation evaluation"""

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    scores = []

    for train_index, test_index in tscv.split(series):
        train_data = series.iloc[train_index]
        test_data = series.iloc[test_index]

        # Fit model and make predictions
        model.fit(train_data)
        predictions = model.predict(test_data)

        # Calculate metrics
        mae = np.mean(np.abs(test_data.values - predictions))
        rmse = np.sqrt(np.mean((test_data.values - predictions) ** 2))

        scores.append({'mae': mae, 'rmse': rmse})

    # Calculate average scores
    avg_mae = np.mean([score['mae'] for score in scores])
    avg_rmse = np.mean([score['rmse'] for score in scores])

    return {
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'scores': scores
    }
```

## Real-world Scenarios {#real-world-scenarios}

### Q12: How would you forecast sales for a retail company?

**Answer:**

**Approach:**

```python
class SalesForecasting:
    """Sales forecasting for retail company"""

    def __init__(self):
        self.models = {}
        self.feature_importance = {}

    def prepare_features(self, sales_data, external_data=None):
        """Prepare features for sales forecasting"""

        df = pd.DataFrame({'sales': sales_data.values}, index=sales_data.index)

        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['day_of_year'] = df.index.dayofyear

        # Lag features
        for lag in [1, 7, 30, 365]:
            df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

        # Rolling statistics
        for window in [7, 30, 90]:
            df[f'sales_rolling_mean_{window}'] = df['sales'].rolling(window=window).mean()
            df[f'sales_rolling_std_{window}'] = df['sales'].rolling(window=window).std()

        # Seasonal features
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

        # External features (if available)
        if external_data is not None:
            for col in external_data.columns:
                df[col] = external_data[col]

        return df.dropna()

    def build_ensemble_model(self, features_df):
        """Build ensemble model for sales forecasting"""

        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression

        X = features_df.drop('sales', axis=1)
        y = features_df['sales']

        # Split data
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Individual models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }

        # Train models and evaluate
        model_predictions = {}
        model_scores = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = np.mean(np.abs(y_test.values - predictions))
            model_predictions[name] = predictions
            model_scores[name] = mae

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))

        # Create ensemble
        weights = [1/score for score in model_scores.values()]
        weights = np.array(weights) / np.sum(weights)

        ensemble_predictions = np.zeros(len(y_test))
        for i, (name, predictions) in enumerate(model_predictions.items()):
            ensemble_predictions += weights[i] * predictions

        ensemble_mae = np.mean(np.abs(y_test.values - ensemble_predictions))

        self.models = {
            'individual': models,
            'ensemble': ensemble_predictions,
            'weights': weights,
            'scores': model_scores,
            'feature_importance': self.feature_importance
        }

        return self.models, (y_test, ensemble_predictions)

    def forecast_future_sales(self, steps=30):
        """Generate future sales forecast"""

        # This is a simplified version
        # In practice, you'd need to handle the rolling nature of features

        ensemble_predictions = []
        for step in range(steps):
            # Use last known values and model to predict next value
            # This would require a more sophisticated implementation
            prediction = np.mean(list(self.models['individual'].values())[0].predict(
                self.last_features.reshape(1, -1)
            ))
            ensemble_predictions.append(prediction)

        return ensemble_predictions
```

### Q13: How do you handle multiple related time series?

**Answer:**

**Vector Autoregression (VAR):**

```python
from statsmodels.tsa.vector_ar.var_model import VAR

def var_forecasting(multivariate_data, lags=2, forecast_steps=12):
    """Vector Autoregression for multiple related series"""

    # Ensure data is stationary
    stationary_data = multivariate_data.diff().dropna()

    # Fit VAR model
    model = VAR(stationary_data)
    fitted_model = model.fit(lags)

    # Generate forecast
    forecast = fitted_model.forecast(stationary_data.values, steps=forecast_steps)

    # Convert back to levels (if differenced)
    last_values = multivariate_data.iloc[-1].values
    forecast_levels = []

    for i in range(forecast_steps):
        if i == 0:
            next_values = last_values + forecast[i]
        else:
            next_values = forecast_levels[-1] + forecast[i]
        forecast_levels.append(next_values)

    return fitted_model, np.array(forecast_levels)

def hierarchical_forecasting(top_level, bottom_level_series):
    """Hierarchical forecasting approach"""

    # Top-down approach
    top_forecast = forecast_series(top_level)
    top_down_forecast = distribute_top_forecast(top_forecast, bottom_level_series)

    # Bottom-up approach
    bottom_forecasts = {}
    for series in bottom_level_series:
        bottom_forecasts[series.name] = forecast_series(series)

    # Optimal combination
    optimal_forecast = optimal_combination(top_forecast, bottom_forecasts)

    return optimal_forecast
```

## Coding Challenges {#coding-challenges}

### Challenge 1: Build a Complete Forecasting Pipeline

```python
class TimeSeriesForecastingPipeline:
    """Complete time series forecasting pipeline"""

    def __init__(self, models=['arima', 'exponential_smoothing', 'prophet']):
        self.models = {}
        self.model_performance = {}
        self.selected_model = None

    def preprocess(self, series):
        """Complete preprocessing pipeline"""

        # 1. Handle missing values
        series = series.interpolate(method='time')

        # 2. Detect and handle outliers
        outliers = self.detect_outliers(series)
        series = series[~outliers]

        # 3. Make stationary if needed
        stationarity = self.test_stationarity(series)
        if not stationarity['adf_stationary']:
            series = series.diff().dropna()

        return series

    def detect_outliers(self, series, method='iqr', threshold=1.5):
        """Detect outliers using IQR method"""

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers

    def test_stationarity(self, series):
        """Test stationarity using ADF test"""

        from statsmodels.tsa.stattools import adfuller

        adf_result = adfuller(series.dropna())

        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'adf_stationary': adf_result[1] < 0.05
        }

    def fit_models(self, series):
        """Fit multiple forecasting models"""

        results = {}

        # ARIMA
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(series, order=(1, 1, 1))
            fitted_model = model.fit()
            results['arima'] = fitted_model
        except Exception as e:
            print(f"ARIMA fitting failed: {e}")

        # Exponential Smoothing
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            results['exponential_smoothing'] = fitted_model
        except Exception as e:
            print(f"Exponential Smoothing fitting failed: {e}")

        # Prophet
        try:
            import prophet
            df = pd.DataFrame({'ds': series.index, 'y': series.values})
            model = prophet.Prophet()
            model.fit(df)
            results['prophet'] = model
        except Exception as e:
            print(f"Prophet fitting failed: {e}")

        self.models = results
        return results

    def evaluate_models(self, series, test_size=0.2):
        """Evaluate model performance using time series cross-validation"""

        split_index = int(len(series) * (1 - test_size))
        train_data = series.iloc[:split_index]
        test_data = series.iloc[split_index:]

        performance = {}

        for name, model in self.models.items():
            try:
                # Generate predictions
                if name == 'prophet':
                    future = model.make_future_dataframe(periods=len(test_data))
                    forecast = model.predict(future)
                    predictions = forecast['yhat'].iloc[-len(test_data):].values
                else:
                    predictions = model.forecast(steps=len(test_data))

                # Calculate metrics
                mae = np.mean(np.abs(test_data.values - predictions))
                rmse = np.sqrt(np.mean((test_data.values - predictions) ** 2))

                performance[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'predictions': predictions
                }

            except Exception as e:
                print(f"Error evaluating {name}: {e}")

        self.model_performance = performance
        return performance

    def select_best_model(self):
        """Select best model based on performance"""

        if not self.model_performance:
            raise ValueError("No model performance data available")

        # Select based on RMSE
        best_model = min(self.model_performance.keys(),
                        key=lambda x: self.model_performance[x]['rmse'])

        self.selected_model = best_model
        return best_model, self.model_performance[best_model]

    def forecast(self, steps=12):
        """Generate forecast using selected model"""

        if not self.selected_model or self.selected_model not in self.models:
            raise ValueError("No model selected or model not available")

        model = self.models[self.selected_model]

        try:
            if self.selected_model == 'prophet':
                future = model.make_future_dataframe(periods=steps)
                forecast = model.predict(future)
                return {
                    'forecast': forecast['yhat'].iloc[-steps:].values,
                    'lower_ci': forecast['yhat_lower'].iloc[-steps:].values,
                    'upper_ci': forecast['yhat_upper'].iloc[-steps:].values
                }
            else:
                forecast = model.forecast(steps=steps)
                return {
                    'forecast': forecast,
                    'model_name': self.selected_model
                }
        except Exception as e:
            print(f"Forecasting failed: {e}")
            return None

# Usage Example
def run_complete_pipeline(time_series_data):
    """Run complete forecasting pipeline"""

    pipeline = TimeSeriesForecastingPipeline()

    # Preprocess
    clean_data = pipeline.preprocess(time_series_data)
    print(f"Cleaned data length: {len(clean_data)}")

    # Fit models
    fitted_models = pipeline.fit_models(clean_data)
    print(f"Fitted models: {list(fitted_models.keys())}")

    # Evaluate
    performance = pipeline.evaluate_models(clean_data)
    print("Model Performance:")
    for model, metrics in performance.items():
        print(f"  {model}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")

    # Select best model
    best_model, best_performance = pipeline.select_best_model()
    print(f"Best model: {best_model}")

    # Generate forecast
    forecast_result = pipeline.forecast(steps=24)
    print(f"Generated {len(forecast_result['forecast'])} forecast steps")

    return pipeline, forecast_result
```

### Challenge 2: Implement Custom Evaluation Metrics

```python
class TimeSeriesMetrics:
    """Custom evaluation metrics for time series"""

    @staticmethod
    def directional_accuracy(actual, predicted):
        """Calculate directional accuracy"""

        actual_diff = np.diff(actual)
        predicted_diff = np.diff(predicted)

        correct_direction = np.sum(
            (actual_diff > 0) == (predicted_diff > 0)
        )

        return correct_direction / len(actual_diff)

    @staticmethod
    def pinball_loss(actual, predicted, quantile=0.5):
        """Pinball loss for quantile forecasting"""

        errors = actual - predicted
        loss = np.mean(
            np.maximum(quantile * errors, (quantile - 1) * errors)
        )

        return loss

    @staticmethod
    def coverage(actual, lower_bound, upper_bound):
        """Calculate prediction interval coverage"""

        within_bounds = np.sum(
            (actual >= lower_bound) & (actual <= upper_bound)
        )

        return within_bounds / len(actual)

    @staticmethod
    def relative_mae(actual, predicted, baseline):
        """Relative MAE compared to baseline"""

        mae_model = np.mean(np.abs(actual - predicted))
        mae_baseline = np.mean(np.abs(actual - baseline))

        return (mae_baseline - mae_model) / mae_baseline

    @staticmethod
    def comprehensive_evaluation(actual, predicted, lower_ci=None, upper_ci=None, baseline=None):
        """Comprehensive evaluation metrics"""

        metrics = {}

        # Basic metrics
        metrics['mae'] = np.mean(np.abs(actual - predicted))
        metrics['rmse'] = np.sqrt(np.mean((actual - predicted) ** 2))
        metrics['mape'] = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Advanced metrics
        metrics['directional_accuracy'] = TimeSeriesMetrics.directional_accuracy(actual, predicted)
        metrics['r2'] = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))

        # Prediction interval metrics
        if lower_ci is not None and upper_ci is not None:
            metrics['coverage'] = TimeSeriesMetrics.coverage(actual, lower_ci, upper_ci)
            metrics['interval_width'] = np.mean(upper_ci - lower_ci)

        # Relative performance
        if baseline is not None:
            metrics['relative_mae'] = TimeSeriesMetrics.relative_mae(actual, predicted, baseline)

        return metrics

    @staticmethod
    def plot_evaluation(actual, predicted, lower_ci=None, upper_ci=None):
        """Plot comprehensive evaluation"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Time series comparison
        axes[0, 0].plot(actual, label='Actual', linewidth=2)
        axes[0, 0].plot(predicted, label='Predicted', linewidth=2)

        if lower_ci is not None and upper_ci is not None:
            axes[0, 0].fill_between(range(len(actual)), lower_ci, upper_ci, alpha=0.3, label='CI')

        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Residuals plot
        residuals = actual - predicted
        axes[0, 1].scatter(predicted, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')

        # Histogram of residuals
        axes[1, 1].hist(residuals, bins=20, alpha=0.7)
        axes[1, 1].axvline(x=0, color='r', linestyle='--')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residual Value')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        return fig
```

This comprehensive interview preparation guide covers all essential aspects of time series forecasting. Study these concepts thoroughly to excel in technical interviews for data science and machine learning roles!
