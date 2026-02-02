# Time Series Forecasting Practice Exercises

## Table of Contents

1. [Foundational Exercises](#foundational-exercises)
2. [Statistical Methods Practice](#statistical-methods-practice)
3. [Machine Learning Approaches](#machine-learning-approaches)
4. [Deep Learning Implementation](#deep-learning-implementation)
5. [Advanced Topics Exercises](#advanced-topics-exercises)
6. [Industry Application Challenges](#industry-application-challenges)
7. [Real-World Projects](#real-world-projects)
8. [Coding Challenges](#coding-challenges)

---

## Foundational Exercises

### Exercise 1: Time Series Data Exploration

**Objective:** Analyze time series properties and identify patterns.

**Dataset:** Generate synthetic time series data with trend, seasonality, and noise.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_time_series(n_periods=365, freq='D'):
    """
    Generate synthetic time series with trend, seasonality, and noise
    """
    # Create date range
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq=freq)

    # Trend component
    trend = np.linspace(100, 150, n_periods)

    # Seasonal component (yearly pattern)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_periods) / 365.25) + \
               10 * np.sin(4 * np.pi * np.arange(n_periods) / 365.25)

    # Weekly pattern for daily data
    if freq == 'D':
        weekly = 5 * np.sin(2 * np.pi * np.arange(n_periods) / 7)
    else:
        weekly = 0

    # Random noise
    noise = np.random.normal(0, 5, n_periods)

    # Combine components
    time_series = trend + seasonal + weekly + noise

    return pd.Series(time_series, index=dates, name='value')

# Your task:
# 1. Generate the synthetic time series
# 2. Plot the time series and its components
# 3. Analyze basic statistics
# 4. Identify trends and seasonality visually

# SOLUTION:
def exercise_1_solution():
    # Generate data
    ts_data = generate_synthetic_time_series()

    # Plot original series
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original series
    ts_data.plot(ax=axes[0, 0], title='Original Time Series')

    # Moving average (30-day)
    moving_avg = ts_data.rolling(window=30).mean()
    moving_avg.plot(ax=axes[0, 1], title='30-day Moving Average')

    # Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts_data, model='additive', period=365)

    decomposition.trend.plot(ax=axes[1, 0], title='Trend Component')
    decomposition.seasonal.plot(ax=axes[1, 1], title='Seasonal Component')

    plt.tight_layout()
    plt.show()

    # Basic statistics
    print("Basic Statistics:")
    print(f"Mean: {ts_data.mean():.2f}")
    print(f"Std: {ts_data.std():.2f}")
    print(f"Min: {ts_data.min():.2f}")
    print(f"Max: {ts_data.max():.2f}")

    return ts_data

# Run the exercise
ts_data = exercise_1_solution()
```

### Exercise 2: Stationarity Testing

**Objective:** Test for stationarity and apply transformations.

```python
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import jarque_bera

def stationarity_analysis():
    """
    Comprehensive stationarity analysis
    """
    # Generate non-stationary series
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Non-stationary: random walk with drift
    non_stationary = np.cumsum(np.random.normal(0.1, 1, 1000)) + 100
    non_stationary_series = pd.Series(non_stationary, index=dates)

    # Stationary: stationary AR(1) process
    stationary = np.zeros(1000)
    for t in range(1, 1000):
        stationary[t] = 0.5 * stationary[t-1] + np.random.normal(0, 1)
    stationary_series = pd.Series(stationary, index=dates)

    def test_stationarity(series, name):
        print(f"\n=== {name} ===")

        # ADF Test
        adf_result = adfuller(series.dropna())
        print(f"ADF Test:")
        print(f"  Statistic: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  Critical Values: {adf_result[4]}")
        print(f"  Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")

        # KPSS Test
        kpss_result = kpss(series.dropna())
        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_result[0]:.6f}")
        print(f"  p-value: {kpss_result[1]:.6f}")
        print(f"  Critical Values: {kpss_result[3]}")
        print(f"  Result: {'Stationary' if kpss_result[1] > 0.05 else 'Non-stationary'}")

        return adf_result[1] < 0.05

    # Test both series
    ns_stationary = test_stationarity(non_stationary_series, "Non-stationary Series")
    s_stationary = test_stationarity(stationary_series, "Stationary Series")

    # Apply transformations to non-stationary series
    print("\n=== TRANSFORMATIONS ===")

    # First differencing
    diff_series = non_stationary_series.diff().dropna()
    test_stationarity(diff_series, "First Differenced")

    # Log transformation (for positive data)
    log_series = np.log(non_stationary_series)
    test_stationarity(log_series, "Log Transformed")

    # Log differencing
    log_diff = log_series.diff().dropna()
    test_stationarity(log_diff, "Log Differenced")

    return non_stationary_series, stationary_series

# Exercise tasks:
# 1. Implement the stationarity_analysis function
# 2. Create your own non-stationary series
# 3. Test different transformations
# 4. Compare results

def exercise_2_implementation():
    """
    Student implementation of stationarity testing
    """
    # TODO: Implement your own stationarity test
    # TODO: Create a series with trend and seasonality
    # TODO: Test various transformations

    # Solution structure:
    print("Implement stationarity testing for custom series...")

    # Your code here:
    # 1. Create a series with trend, seasonality, and noise
    # 2. Test for stationarity using ADF and KPSS
    # 3. Apply transformations
    # 4. Re-test until stationarity is achieved

    pass

# Run exercises
exercise_2_implementation()
```

### Exercise 3: ACF and PACF Analysis

**Objective:** Identify ARIMA model orders using autocorrelation analysis.

```python
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def acf_pacf_analysis():
    """
    ACF and PACF analysis for ARIMA identification
    """
    # Generate AR(2) process
    np.random.seed(42)
    n = 1000
    ar_params = [0.6, -0.3]  # AR(2) parameters
    ma_params = []  # No MA component

    # Generate AR(2) process
    ar_process = np.zeros(n)
    for t in range(2, n):
        ar_process[t] = (ar_params[0] * ar_process[t-1] +
                        ar_params[1] * ar_process[t-2] +
                        np.random.normal(0, 1))

    ar_series = pd.Series(ar_process, name='AR(2)')

    # Generate MA(2) process
    ma_process = np.zeros(n)
    for t in range(2, n):
        ma_process[t] = (np.random.normal(0, 1) +
                        ma_params[0] * np.random.normal(0, 1, 1)[0] +
                        ma_params[1] * np.random.normal(0, 1, 1)[0])

    ma_series = pd.Series(ma_process, name='MA(2)')

    # Plot ACF and PACF
    def plot_acf_pacf(series, title):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        plot_acf(series.dropna(), ax=axes[0], lags=40, title=f'{title} - ACF')
        plot_pacf(series.dropna(), ax=axes[1], lags=40, title=f'{title} - PACF')

        plt.tight_layout()
        plt.show()

        print(f"\n{title} Analysis:")
        print("Expected pattern: ACF gradual decay, PACF cutoff after lag 2 (for AR)")

    plot_acf_pacf(ar_series, "AR(2) Process")
    plot_acf_pacf(ma_series, "MA(2) Process")

    return ar_series, ma_series

# Exercise implementation:
def exercise_3_implementation():
    """
    Student practice with ACF/PACF interpretation
    """
    print("ACF/PACF Analysis Exercise")
    print("=" * 50)

    # TODO: 1. Generate your own ARMA process
    # TODO: 2. Plot ACF and PACF
    # TODO: 3. Identify the correct model order
    # TODO: 4. Compare with actual parameters used

    # Example structure:
    # 1. Create ARMA(1,1) process
    np.random.seed(123)
    n = 500

    # Your implementation here
    print("Implement ARMA(1,1) generation and analysis...")

    # Hint: ARMA(1,1) should show gradual decay in both ACF and PACF
    # with PACF cutting off after lag 1

# Run the exercise
exercise_3_implementation()
```

---

## Statistical Methods Practice

### Exercise 4: Moving Averages Implementation

**Objective:** Implement and compare different types of moving averages.

```python
def moving_average_exercises():
    """
    Practice with different moving average methods
    """
    # Generate sample data with trend and noise
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    trend = np.linspace(100, 120, 365)
    noise = np.random.normal(0, 2, 365)
    data = trend + noise

    ts_data = pd.Series(data, index=dates, name='daily_sales')

    def simple_moving_average(series, window):
        """Calculate Simple Moving Average"""
        return series.rolling(window=window).mean()

    def exponential_moving_average(series, alpha):
        """Calculate Exponential Moving Average"""
        return series.ewm(alpha=alpha).mean()

    def weighted_moving_average(series, weights):
        """Calculate Weighted Moving Average"""
        weights = np.array(weights)
        return series.rolling(window=len(weights)).apply(
            lambda x: np.sum(x * weights) / np.sum(weights)
        )

    # Calculate different moving averages
    sma_7 = simple_moving_average(ts_data, 7)
    sma_30 = simple_moving_average(ts_data, 30)
    ema_7 = exponential_moving_average(ts_data, 0.3)
    wma_7 = weighted_moving_average(ts_data, [0.5, 0.3, 0.2])

    # Plot results
    plt.figure(figsize=(15, 8))
    ts_data.plot(label='Original', alpha=0.7)
    sma_7.plot(label='SMA (7-day)', linewidth=2)
    sma_30.plot(label='SMA (30-day)', linewidth=2)
    ema_7.plot(label='EMA (α=0.3)', linewidth=2)
    wma_7.plot(label='WMA (weights: 0.5,0.3,0.2)', linewidth=2)

    plt.title('Different Moving Averages Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate forecast accuracy
    actual = ts_data.iloc[30:].values
    forecasts = sma_30.iloc[30:].dropna().values

    mae = np.mean(np.abs(actual - forecasts))
    rmse = np.sqrt(np.mean((actual - forecasts)**2))

    print(f"30-day SMA Forecast Accuracy:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return ts_data, sma_30, ema_7, wma_7

# Exercise tasks:
# 1. Implement the moving average functions
# 2. Test different window sizes
# 3. Compare SMA vs EMA vs WMA
# 4. Evaluate forecasting performance

def exercise_4_implementation():
    """
    Student implementation of moving averages
    """
    print("Moving Averages Exercise")
    print("=" * 40)

    # TODO: Implement your own moving average calculations
    # TODO: Test on real data (stock prices, sales data, etc.)
    # TODO: Compare different methods and analyze performance

    # Example: Create a centered moving average
    def centered_moving_average(series, window):
        """Centered moving average (trailing + leading)"""
        return series.rolling(window=window, center=True).mean()

    # Your implementation here
    pass

# Run exercise
exercise_4_implementation()
```

### Exercise 5: Exponential Smoothing Methods

**Objective:** Implement different exponential smoothing models.

```python
def exponential_smoothing_practice():
    """
    Practice with exponential smoothing methods
    """
    # Generate data with trend and seasonality
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365, freq='D')

    # Trend component
    trend = 100 + 0.1 * np.arange(365)

    # Seasonal component (weekly pattern)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 7)

    # Noise
    noise = np.random.normal(0, 3, 365)

    data = trend + seasonal + noise
    ts_data = pd.Series(data, index=dates, name='demand')

    def simple_exponential_smoothing(data, alpha):
        """
        Simple Exponential Smoothing (no trend or seasonality)
        """
        values = data.values
        n = len(values)
        result = np.zeros(n)

        # Initialize
        result[0] = values[0]

        # Calculate SES
        for t in range(1, n):
            result[t] = alpha * values[t] + (1 - alpha) * result[t-1]

        return pd.Series(result, index=data.index)

    def holt_linear_method(data, alpha, beta):
        """
        Holt's Linear Trend Method
        """
        values = data.values
        n = len(values)

        # Initialize level and trend
        level = np.zeros(n)
        trend = np.zeros(n)
        result = np.zeros(n)

        level[0] = values[0]
        trend[0] = values[1] - values[0]

        for t in range(1, n):
            level[t] = alpha * values[t] + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
            result[t] = level[t] + trend[t]

        return pd.Series(result, index=data.index)

    def holt_winters_additive(data, alpha, beta, gamma, season_length):
        """
        Holt-Winters Additive Method
        """
        values = data.values
        n = len(values)

        # Initialize components
        level = np.zeros(n)
        trend = np.zeros(n)
        season = np.zeros(n)
        result = np.zeros(n)

        # Initialize with first season
        level[0] = values[0]
        trend[0] = (values[season_length] - values[0]) / season_length

        for i in range(season_length):
            season[i] = values[i] - level[0]

        # Calculate components
        for t in range(season_length, n):
            level[t] = alpha * (values[t] - season[t-season_length]) + \
                      (1 - alpha) * (level[t-1] + trend[t-1])

            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]

            season[t] = gamma * (values[t] - level[t]) + \
                       (1 - gamma) * season[t-season_length]

            result[t] = level[t] + trend[t] + season[t-season_length]

        return pd.Series(result, index=data.index)

    # Implement different methods
    ses = simple_exponential_smoothing(ts_data, alpha=0.3)
    holt = holt_linear_method(ts_data, alpha=0.3, beta=0.1)
    holt_winters = holt_winters_additive(ts_data, alpha=0.3, beta=0.1,
                                       gamma=0.1, season_length=7)

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    ts_data.plot(label='Original', alpha=0.7)
    ses.plot(label='Simple ES', linewidth=2)
    plt.title('Simple Exponential Smoothing')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    ts_data.plot(label='Original', alpha=0.7)
    holt.plot(label='Holt Linear', linewidth=2)
    holt_winters.plot(label='Holt-Winters', linewidth=2)
    plt.title('Holt Linear vs Holt-Winters')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Forecast next 30 days
    def forecast_holt_winters(last_level, last_trend, last_season,
                            alpha, beta, gamma, season_length, steps):
        """Forecast using Holt-Winters method"""
        forecasts = []

        for h in range(1, steps + 1):
            season_index = (h - 1) % season_length
            forecast = last_level + h * last_trend + last_season[-(season_length - season_index):][season_index] if season_index < len(last_season[-(season_length - season_index):]) else last_season[-season_length + season_index]
            forecasts.append(forecast)

        return forecasts

    # Calculate forecast accuracy for the last 30 days
    test_size = 30
    train_data = ts_data[:-test_size]
    test_data = ts_data[-test_size:]

    # Fit models on training data
    ses_fit = simple_exponential_smoothing(train_data, alpha=0.3)
    holt_fit = holt_linear_method(train_data, alpha=0.3, beta=0.1)
    hw_fit = holt_winters_additive(train_data, alpha=0.3, beta=0.1,
                                 gamma=0.1, season_length=7)

    # Evaluate
    mae_ses = np.mean(np.abs(test_data.values - ses_fit.tail(test_size).values))
    mae_holt = np.mean(np.abs(test_data.values - holt_fit.tail(test_size).values))
    mae_hw = np.mean(np.abs(test_data.values - hw_fit.tail(test_size).values))

    print(f"Forecast Accuracy (MAE):")
    print(f"Simple ES: {mae_ses:.2f}")
    print(f"Holt Linear: {mae_holt:.2f}")
    print(f"Holt-Winters: {mae_hw:.2f}")

    return ts_data, ses, holt, holt_winters

# Exercise 5: Student Implementation
def exercise_5_implementation():
    """
    Student practice with exponential smoothing
    """
    print("Exponential Smoothing Exercise")
    print("=" * 40)

    # TODO: 1. Implement Holt-Winters multiplicative method
    # TODO: 2. Add parameter optimization (grid search for alpha, beta, gamma)
    # TODO: 3. Test on real seasonal data
    # TODO: 4. Compare different seasonal periods

    def holt_winters_multiplicative(data, alpha, beta, gamma, season_length):
        """Holt-Winters Multiplicative Method (your task)"""
        # TODO: Implement multiplicative version
        # Hint: Seasonal component is multiplicative rather than additive
        pass

    # Your implementation here
    pass

# Run exercises
exercise_5_implementation()
```

### Exercise 6: ARIMA Model Implementation

**Objective:** Build and evaluate ARIMA models.

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools

def arima_modeling_exercise():
    """
    Complete ARIMA modeling exercise
    """
    # Generate ARIMA data
    np.random.seed(42)

    # Generate ARIMA(1,1,1) process
    from statsmodels.tsa.arima_process import ArmaProcess

    # AR and MA parameters
    ar_params = np.array([1, -0.6])  # (1 - 0.6L)
    ma_params = np.array([1, 0.3])   # (1 + 0.3L)

    # Generate series
    arima_process = ArmaProcess(ar_params, ma_params)
    simulated_data = arima_process.generate_sample(nsample=500)

    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    ts_data = pd.Series(simulated_data, index=dates, name='simulated')

    # Step 1: Check stationarity
    print("=== STEP 1: STATIONARITY CHECK ===")

    def check_stationarity(series):
        result = adfuller(series.dropna())
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")

        is_stationary = result[1] <= 0.05
        print(f"Result: {'Stationary' if is_stationary else 'Non-stationary'}")
        return is_stationary

    check_stationarity(ts_data)

    # Step 2: Difference if needed
    print("\n=== STEP 2: DIFFERENCING ===")
    diff_data = ts_data.diff().dropna()
    check_stationarity(diff_data)

    # Step 3: Grid search for ARIMA parameters
    print("\n=== STEP 3: MODEL SELECTION ===")

    # Define parameter ranges
    p_values = range(0, 4)
    d_values = range(0, 2)
    q_values = range(0, 4)

    best_aic = float('inf')
    best_params = None

    results = []

    print("Testing ARIMA models...")
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(ts_data, order=(p, d, q))
            fitted_model = model.fit()

            aic = fitted_model.aic
            results.append({
                'order': (p, d, q),
                'aic': aic,
                'bic': fitted_model.bic,
                'params': len(fitted_model.params)
            })

            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)

        except Exception as e:
            continue

    # Sort results by AIC
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')

    print("\nTop 5 models by AIC:")
    print(results_df.head())

    # Step 4: Fit best model
    print(f"\n=== STEP 4: BEST MODEL ===")
    print(f"Best ARIMA order: {best_params}")
    print(f"Best AIC: {best_aic:.2f}")

    best_model = ARIMA(ts_data, order=best_params)
    fitted_best = best_model.fit()

    print("\nModel Summary:")
    print(fitted_best.summary())

    # Step 5: Residual analysis
    print("\n=== STEP 5: RESIDUAL ANALYSIS ===")

    residuals = fitted_best.resid

    # Plot residuals
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    residuals.plot(ax=axes[0, 0], title='Residuals')
    residuals.plot(kind='kde', ax=axes[0, 1], title='Residual Density')

    # ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals.dropna(), ax=axes[1, 0], lags=40)
    residuals.hist(bins=30, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    # Ljung-Box test for residual autocorrelation
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    print("\nLjung-Box Test (residuals should be white noise):")
    print(lb_test)

    # Step 6: Forecasting
    print("\n=== STEP 6: FORECASTING ===")

    forecast = fitted_best.forecast(steps=30)
    forecast_ci = fitted_best.get_forecast(steps=30).conf_int()

    # Plot forecast
    plt.figure(figsize=(15, 6))

    # Plot last 100 observations
    ts_data.tail(100).plot(label='Historical Data', color='blue')

    # Plot forecast
    forecast_index = pd.date_range(ts_data.index[-1] + timedelta(days=1),
                                 periods=30, freq='D')
    plt.plot(forecast_index, forecast, label='Forecast', color='red', linewidth=2)

    # Plot confidence intervals
    plt.fill_between(forecast_index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1],
                    alpha=0.3, color='red', label='95% Confidence Interval')

    plt.title('ARIMA Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return ts_data, fitted_best, results_df

# Exercise 6: Student Tasks
def exercise_6_implementation():
    """
    Student ARIMA modeling practice
    """
    print("ARIMA Modeling Exercise")
    print("=" * 40)

    # TODO: 1. Use real data (stock prices, economic indicators, etc.)
    # TODO: 2. Implement automatic order selection
    # TODO: 3. Add seasonal ARIMA (SARIMA) testing
    # TODO: 4. Compare with machine learning approaches

    def automatic_arima_selection(data, max_p=3, max_d=2, max_q=3):
        """
        Automatic ARIMA order selection using information criteria
        """
        best_aic = float('inf')
        best_order = None
        results = []

        # Your implementation here
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()

                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)

                        results.append({
                            'order': (p, d, q),
                            'aic': fitted.aic,
                            'bic': fitted.bic
                        })
                    except:
                        continue

        return best_order, best_aic, results

    # Your implementation here
    pass

# Run the exercise
exercise_6_implementation()
```

---

## Machine Learning Approaches

### Exercise 7: Feature Engineering for Time Series

**Objective:** Create comprehensive features for ML-based forecasting.

```python
def time_series_feature_engineering_exercise():
    """
    Comprehensive feature engineering for time series
    """
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')

    # Create complex time series with multiple patterns
    trend = np.linspace(100, 150, 1000)
    seasonal_daily = 10 * np.sin(2 * np.pi * np.arange(1000) / 7)
    seasonal_yearly = 20 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
    noise = np.random.normal(0, 3, 1000)

    ts_data = pd.Series(trend + seasonal_daily + seasonal_yearly + noise,
                       index=dates, name='target')

    def create_lag_features(data, lags=[1, 2, 3, 7, 14, 30]):
        """
        Create lag features
        """
        df = pd.DataFrame({'target': data})

        for lag in lags:
            df[f'lag_{lag}'] = data.shift(lag)

        return df

    def create_rolling_features(data, windows=[3, 7, 14, 30]):
        """
        Create rolling window statistics
        """
        df = pd.DataFrame({'target': data})

        for window in windows:
            df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window=window).std()
            df[f'rolling_min_{window}'] = data.rolling(window=window).min()
            df[f'rolling_max_{window}'] = data.rolling(window=window).max()
            df[f'rolling_median_{window}'] = data.rolling(window=window).median()

            # Rate of change features
            df[f'rolling_pct_change_{window}'] = data.pct_change(periods=window)

            # Momentum features
            df[f'rolling_momentum_{window}'] = data - data.shift(window)

        return df

    def create_time_features(data):
        """
        Create time-based features
        """
        df = pd.DataFrame({'target': data})
        df.index = pd.to_datetime(df.index)

        # Basic time components
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        df['weekofyear'] = df.index.isocalendar().week

        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # Business day indicator
        df['is_business_day'] = (~df.index.to_series().dt.dayofweek.isin([5, 6])).astype(int)

        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        return df

    def create_difference_features(data, orders=[1, 2, 7]):
        """
        Create differenced features
        """
        df = pd.DataFrame({'target': data})

        for order in orders:
            df[f'diff_{order}'] = data.diff(order)

            # Second order differences
            if order == 1:
                df['diff2_1'] = df[f'diff_{order}'].diff()

        return df

    def create_technical_indicators(data, window=14):
        """
        Create technical analysis indicators
        """
        df = pd.DataFrame({'target': data})

        # Simple and exponential moving averages
        df['sma'] = data.rolling(window=window).mean()
        df['ema'] = data.ewm(span=window).mean()

        # Moving average ratios
        df['price_sma_ratio'] = data / df['sma']
        df['price_ema_ratio'] = data / df['ema']

        # Volatility measures
        df['volatility'] = data.rolling(window=window).std()

        # RSI calculation
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (data - df['bb_lower']) / df['bb_width']

        # MACD
        ema_12 = data.ewm(span=12).mean()
        ema_26 = data.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def create_fourier_features(data, periods=[7, 30, 365], k=3):
        """
        Create Fourier features for seasonality
        """
        df = pd.DataFrame({'target': data})

        n = len(data)

        for period in periods:
            for i in range(1, k + 1):
                df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * np.arange(n) / period)
                df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * np.arange(n) / period)

        return df

    def create_interaction_features(df):
        """
        Create interaction features between existing features
        """
        # Select numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('target') if 'target' in numeric_cols else None

        # Create some interaction features
        if len(numeric_cols) >= 2:
            # Lag with rolling features
            for col in numeric_cols:
                if 'lag_' in col:
                    lag_num = int(col.split('_')[1])
                    if lag_num >= 7:
                        # Interaction with 7-day average
                        if 'rolling_mean_7' in df.columns:
                            df[f'{col}_x_rolling_mean_7'] = df[col] * df['rolling_mean_7']

        return df

    # Create all features
    print("Creating comprehensive features...")

    # Combine all feature creation functions
    df = ts_data.to_frame()

    # Add all feature types
    df = create_lag_features(df['target']).join(df.drop('target', axis=1))
    df = create_rolling_features(df['target']).join(df.drop('target', axis=1))
    df = create_time_features(df['target']).join(df.drop('target', axis=1))
    df = create_difference_features(df['target']).join(df.drop('target', axis=1))
    df = create_technical_indicators(df['target']).join(df.drop('target', axis=1))
    df = create_fourier_features(df['target']).join(df.drop('target', axis=1))

    # Add interactions
    df = create_interaction_features(df)

    # Remove target column if duplicated
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop rows with NaN values
    df_clean = df.dropna()

    print(f"Original shape: {df.shape}")
    print(f"After removing NaN: {df_clean.shape}")
    print(f"\nFeature types created:")
    feature_types = {
        'Lag features': len([col for col in df_clean.columns if 'lag_' in col]),
        'Rolling features': len([col for col in df_clean.columns if 'rolling_' in col]),
        'Time features': len([col for col in df_clean.columns if col in ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 'weekofyear', 'is_weekend', 'is_business_day', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos']]),
        'Difference features': len([col for col in df_clean.columns if 'diff' in col]),
        'Technical indicators': len([col for col in df_clean.columns if col in ['sma', 'ema', 'price_sma_ratio', 'price_ema_ratio', 'volatility', 'rsi', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'macd', 'macd_signal', 'macd_histogram']]),
        'Fourier features': len([col for col in df_clean.columns if 'fourier_' in col]),
        'Interaction features': len([col for col in df_clean.columns if '_x_' in col])
    }

    for feature_type, count in feature_types.items():
        print(f"  {feature_type}: {count}")

    # Display sample of features
    print(f"\nSample features (first 10 columns):")
    print(df_clean.columns[:10].tolist())

    # Plot feature importance using Random Forest
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X = df_clean.drop('target', axis=1)
    y = df_clean['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))

    # Model performance with all features
    from sklearn.metrics import mean_absolute_error, r2_score

    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRandom Forest Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

    return df_clean, feature_importance

# Exercise 7: Student Tasks
def exercise_7_implementation():
    """
    Student feature engineering practice
    """
    print("Feature Engineering Exercise")
    print("=" * 40)

    # TODO: 1. Create advanced features (change point detection, regime indicators)
    # TODO: 2. Implement feature selection techniques
    # TODO: 3. Test on real time series data
    # TODO: 4. Compare different feature sets

    def create_regime_features(data, window=30):
        """
        Create regime-based features (your task)
        """
        # TODO: Detect different market regimes using clustering
        # TODO: Create regime indicator features
        # TODO: Add regime-specific statistical measures
        pass

    def create_change_point_features(data, method='bayesian', window=50):
        """
        Create change point features (your task)
        """
        # TODO: Implement change point detection
        # TODO: Create features based on change points
        # TODO: Add variance change indicators
        pass

    # Your implementation here
    pass

# Run exercise
exercise_7_implementation()
```

### Exercise 8: Ensemble Methods for Time Series

**Objective:** Build ensemble models combining different approaches.

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

def ensemble_time_series_exercise():
    """
    Ensemble methods for time series forecasting
    """
    # Prepare data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=800, freq='D')

    # Complex time series with trend, seasonality, and noise
    trend = np.linspace(100, 130, 800)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(800) / 365.25)
    noise = np.random.normal(0, 5, 800)

    ts_data = pd.Series(trend + seasonal + noise, index=dates, name='demand')

    # Create features
    def create_features(data, lookback=30):
        """Create features for ML models"""
        df = data.to_frame()

        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'lag_{lag}'] = data.shift(lag)

        # Rolling features
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = data.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window=window).std()

        # Time features
        df.index = pd.to_datetime(df.index)
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear

        return df.dropna()

    # Prepare dataset
    feature_data = create_features(ts_data)

    # Split into train and test
    train_size = int(len(feature_data) * 0.8)
    train_data = feature_data.iloc[:train_size]
    test_data = feature_data.iloc[train_size:]

    X_train = train_data.drop('demand', axis=1)
    y_train = train_data['demand']
    X_test = test_data.drop('demand', axis=1)
    y_test = test_data['demand']

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Define base models
    def create_base_models():
        """Create ensemble of base models"""
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
        }
        return models

    def simple_ensemble_predictions(models, X_train, y_train, X_test):
        """Simple average ensemble"""
        predictions = {}

        # Train and predict with each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_test)

        # Simple average
        ensemble_pred = np.mean(list(predictions.values()), axis=0)

        return ensemble_pred, predictions

    def weighted_ensemble_predictions(models, X_train, y_train, X_test, X_val, y_val):
        """Weighted ensemble based on validation performance"""
        predictions = {}
        weights = {}

        # Train models and get validation performance
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_pred)

            predictions[name] = model.predict(X_test)
            # Weight inversely proportional to error
            weights[name] = 1.0 / (val_mae + 1e-8)

        # Normalize weights
        total_weight = sum(weights.values())
        for name in weights:
            weights[name] /= total_weight

        print(f"Ensemble weights: {weights}")

        # Weighted average
        ensemble_pred = np.zeros(len(X_test))
        for name, weight in weights.items():
            ensemble_pred += weight * predictions[name]

        return ensemble_pred, predictions, weights

    def stacking_ensemble(X_train, y_train, X_test, base_models):
        """Stacking ensemble with meta-learner"""
        from sklearn.model_selection import KFold

        # Prepare base model predictions for stacking
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        base_predictions = np.zeros((len(X_train), len(base_models)))

        # Generate out-of-fold predictions
        for i, (name, model) in enumerate(base_models.items()):
            print(f"Stacking fold for {name}...")
            for train_idx, val_idx in kfold.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                base_predictions[val_idx, i] = model_copy.predict(X_val_fold)

        # Train meta-learner
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(base_predictions, y_train)

        # Generate final predictions
        final_predictions = np.zeros((len(X_test), len(base_models)))
        for i, (name, model) in enumerate(base_models.items()):
            model.fit(X_train, y_train)
            final_predictions[:, i] = model.predict(X_test)

        ensemble_pred = meta_model.predict(final_predictions)

        return ensemble_pred, final_predictions

    # Train simple ensemble
    print("=" * 50)
    print("SIMPLE ENSEMBLE")
    print("=" * 50)

    models = create_base_models()
    simple_ensemble, individual_preds = simple_ensemble_predictions(
        models, X_train, y_train, X_test
    )

    # Evaluate simple ensemble
    simple_mae = mean_absolute_error(y_test, simple_ensemble)
    simple_rmse = np.sqrt(mean_squared_error(y_test, simple_ensemble))

    print(f"Simple Ensemble Performance:")
    print(f"MAE: {simple_mae:.2f}")
    print(f"RMSE: {simple_rmse:.2f}")

    # Evaluate individual models
    print("\nIndividual Model Performance:")
    individual_results = {}
    for name, pred in individual_preds.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        individual_results[name] = {'MAE': mae, 'RMSE': rmse}
        print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Weighted ensemble
    print("\n" + "=" * 50)
    print("WEIGHTED ENSEMBLE")
    print("=" * 50)

    # Use part of training data for validation
    val_size = int(len(X_train) * 0.2)
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]

    weighted_ensemble, _, weights = weighted_ensemble_predictions(
        models, X_train_fit, y_train_fit, X_test, X_val, y_val
    )

    weighted_mae = mean_absolute_error(y_test, weighted_ensemble)
    weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_ensemble))

    print(f"Weighted Ensemble Performance:")
    print(f"MAE: {weighted_mae:.2f}")
    print(f"RMSE: {weighted_rmse:.2f}")

    # Stacking ensemble
    print("\n" + "=" * 50)
    print("STACKING ENSEMBLE")
    print("=" * 50)

    stacking_ensemble, _ = stacking_ensemble(X_train, y_train, X_test, models)

    stacking_mae = mean_absolute_error(y_test, stacking_ensemble)
    stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_ensemble))

    print(f"Stacking Ensemble Performance:")
    print(f"MAE: {stacking_mae:.2f}")
    print(f"RMSE: {stacking_rmse:.2f}")

    # Compare all methods
    print("\n" + "=" * 50)
    print("ENSEMBLE COMPARISON")
    print("=" * 50)

    comparison = pd.DataFrame({
        'Method': ['Simple Average', 'Weighted Average', 'Stacking'],
        'MAE': [simple_mae, weighted_mae, stacking_mae],
        'RMSE': [simple_rmse, weighted_rmse, stacking_rmse]
    })

    print(comparison.to_string(index=False))

    # Plot predictions
    plt.figure(figsize=(15, 10))

    # Plot actual vs predicted for different methods
    plt.subplot(2, 2, 1)
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(simple_ensemble, label='Simple Ensemble', alpha=0.7)
    plt.title('Simple Ensemble')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(weighted_ensemble, label='Weighted Ensemble', alpha=0.7, color='green')
    plt.title('Weighted Ensemble')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(y_test.values, label='Actual', linewidth=2)
    plt.plot(stacking_ensemble, label='Stacking Ensemble', alpha=0.7, color='red')
    plt.title('Stacking Ensemble')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot model contributions (for weighted ensemble)
    plt.subplot(2, 2, 4)
    model_names = list(individual_preds.keys())
    contributions = [weights[name] for name in model_names]

    plt.pie(contributions, labels=model_names, autopct='%1.1f%%')
    plt.title('Ensemble Weights Distribution')

    plt.tight_layout()
    plt.show()

    return {
        'simple_ensemble': simple_ensemble,
        'weighted_ensemble': weighted_ensemble,
        'stacking_ensemble': stacking_ensemble,
        'individual_predictions': individual_preds,
        'weights': weights,
        'comparison': comparison
    }

# Exercise 8: Student Tasks
def exercise_8_implementation():
    """
    Student ensemble methods practice
    """
    print("Ensemble Methods Exercise")
    print("=" * 40)

    # TODO: 1. Implement voting regressor
    # TODO: 2. Add dynamic ensemble selection
    # TODO: 3. Create hierarchical ensemble
    # TODO: 4. Implement online learning for ensembles

    def dynamic_ensemble_selection(models, X_train, y_train, X_test,
                                 validation_data):
        """
        Dynamic ensemble selection based on local performance
        """
        # TODO: Implement dynamic selection
        # Hint: Use nearest neighbors to select best models for each test point
        pass

    def online_ensemble_update(models, new_data, new_targets,
                             update_frequency=10):
        """
        Online learning for ensemble models
        """
        # TODO: Implement incremental learning
        # Hint: Use partial_fit for sklearn models
        pass

    # Your implementation here
    pass

# Run exercise
exercise_8_implementation()
```

---

## Deep Learning Implementation

### Exercise 9: LSTM Implementation from Scratch

**Objective:** Build and train LSTM networks for time series forecasting.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

def lstm_from_scratch_exercise():
    """
    Comprehensive LSTM implementation and training
    """
    # Generate complex time series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=2000, freq='H')

    # Multi-component time series
    trend = np.linspace(100, 120, 2000)
    seasonal_daily = 10 * np.sin(2 * np.pi * np.arange(2000) / 24)  # Daily pattern
    seasonal_weekly = 5 * np.sin(2 * np.pi * np.arange(2000) / (24*7))  # Weekly pattern
    noise = np.random.normal(0, 2, 2000)

    ts_data = pd.Series(trend + seasonal_daily + seasonal_weekly + noise,
                       index=dates, name='demand')

    def create_sequences(data, seq_length, target_col='demand'):
        """
        Create sequences for LSTM training
        """
        sequences = []
        targets = []

        for i in range(seq_length, len(data)):
            # Input sequence
            sequences.append(data.iloc[i-seq_length:i].values)
            # Target (next value)
            targets.append(data.iloc[i][target_col])

        return np.array(sequences), np.array(targets)

    def prepare_lstm_data(data, seq_length=24, test_size=0.2):
        """
        Prepare data for LSTM training
        """
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        scaled_series = pd.Series(scaled_data.flatten(), index=data.index)

        # Create sequences
        X, y = create_sequences(scaled_series, seq_length)

        # Split into train and test
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test, scaler, seq_length

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, seq_length = prepare_lstm_data(ts_data)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Sequence length: {seq_length}")

    def create_simple_lstm(input_shape):
        """
        Simple LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_stacked_lstm(input_shape):
        """
        Stacked LSTM with more layers
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(50, return_sequences=True),
            Dropout(0.3),
            LSTM(25, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def create_bidirectional_lstm(input_shape):
        """
        Bidirectional LSTM
        """
        from tensorflow.keras.layers import Bidirectional

        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(50)),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_attention_lstm(input_shape, attention_dim=32):
        """
        LSTM with attention mechanism
        """
        # Custom attention layer
        class AttentionLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(AttentionLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(name='attention_weight',
                                       shape=(input_shape[-1], input_shape[-1]),
                                       initializer='random_normal',
                                       trainable=True)
                self.b = self.add_weight(name='attention_bias',
                                       shape=(input_shape[-1],),
                                       initializer='zeros',
                                       trainable=True)
                super(AttentionLayer, self).build(input_shape)

            def call(self, inputs):
                # Compute attention scores
                attention_scores = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

                # Normalize attention scores
                attention_weights = tf.nn.softmax(attention_scores, axis=1)

                # Apply attention weights
                weighted_input = inputs * attention_weights
                output = tf.reduce_sum(weighted_input, axis=1)

                return output

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            AttentionLayer(),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    # Create different LSTM models
    input_shape = (X_train.shape[1], X_train.shape[2])

    models = {
        'Simple LSTM': create_simple_lstm(input_shape),
        'Stacked LSTM': create_stacked_lstm(input_shape),
        'Bidirectional LSTM': create_bidirectional_lstm(input_shape),
        'Attention LSTM': create_attention_lstm(input_shape)
    }

    # Training configurations
    callbacks = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
    ]

    # Train and evaluate models
    results = {}

    print("\n" + "=" * 50)
    print("TRAINING LSTM MODELS")
    print("=" * 50)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform predictions
        y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate metrics
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

        results[name] = {
            'model': model,
            'history': history,
            'predictions': y_pred_original,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

        print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Plot results
    plt.figure(figsize=(20, 15))

    # Plot 1: Model performance comparison
    plt.subplot(3, 2, 1)
    model_names = list(results.keys())
    mae_values = [results[name]['mae'] for name in model_names]
    rmse_values = [results[name]['rmse'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
    plt.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.8)

    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2-5: Predictions vs Actual for each model
    colors = ['red', 'green', 'blue', 'orange']
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(3, 2, i+2)

        # Plot sample of predictions
        n_plot = min(200, len(y_test_original))
        indices = np.arange(n_plot)

        plt.plot(indices, y_test_original[:n_plot], label='Actual',
                linewidth=2, color='black')
        plt.plot(indices, result['predictions'][:n_plot],
                label='Predicted', alpha=0.7, color=colors[i])

        plt.title(f'{name} Predictions')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 6: Training history for best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    best_history = results[best_model_name]['history']

    plt.subplot(3, 2, 6)
    plt.plot(best_history.history['loss'], label='Training Loss')
    plt.plot(best_history.history['val_loss'], label='Validation Loss')
    plt.title(f'{best_model_name} - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Feature importance analysis for attention model
    if 'Attention LSTM' in results:
        attention_model = results['Attention LSTM']['model']

        # Get attention weights (simplified approach)
        print("\n" + "=" * 50)
        print("ATTENTION ANALYSIS")
        print("=" * 50)

        # Use a sample sequence to analyze attention
        sample_sequence = X_train[0:1]  # First training sequence

        # Get attention layer output (simplified)
        # Note: This is a simplified approach - in practice, you'd need to
        # extract attention weights from the custom layer

        # Create synthetic attention weights for demonstration
        seq_len = sample_sequence.shape[1]
        synthetic_attention = np.random.exponential(0.5, seq_len)
        synthetic_attention = synthetic_attention / np.sum(synthetic_attention)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(range(seq_len), synthetic_attention)
        plt.title('Attention Weights Distribution')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(ts_data.iloc[:seq_len].values)
        plt.title('Corresponding Time Series Values')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Summary results
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)

    summary_df = pd.DataFrame({
        'Model': model_names,
        'MAE': mae_values,
        'RMSE': rmse_values,
        'MAPE': [results[name]['mape'] for name in model_names]
    })

    print(summary_df.to_string(index=False))

    return results, summary_df

# Exercise 9: Student Tasks
def exercise_9_implementation():
    """
    Student LSTM implementation practice
    """
    print("LSTM Implementation Exercise")
    print("=" * 40)

    # TODO: 1. Implement CNN-LSTM hybrid model
    # TODO: 2. Add regularization techniques (dropout, batch normalization)
    # TODO: 3. Implement sequence-to-sequence LSTM
    # TODO: 4. Add multi-step forecasting capability

    def create_cnn_lstm_model(input_shape):
        """
        CNN-LSTM hybrid model (your task)
        """
        # TODO: Add 1D CNN layers before LSTM
        # TODO: This can capture local patterns in time series
        pass

    def create_seq2seq_lstm(input_shape, output_steps):
        """
        Sequence-to-sequence LSTM (your task)
        """
        # TODO: Encoder-decoder architecture
        # TODO: For multi-step forecasting
        pass

    # Your implementation here
    pass

# Run exercise
exercise_9_implementation()
```

### Exercise 10: Transformer Models for Time Series

**Objective:** Implement transformer architecture for time series forecasting.

```python
def transformer_time_series_exercise():
    """
    Transformer models for time series forecasting
    """
    # Generate complex multivariate time series
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='H')

    # Create multivariate time series
    n_series = 5
    time_series_data = {}

    for i in range(n_series):
        # Different patterns for each series
        trend = np.linspace(100 + i*10, 120 + i*10, 1000)
        seasonal = (5 + i) * np.sin(2 * np.pi * np.arange(1000) / (24*7))
        noise = np.random.normal(0, 2 + i*0.5, 1000)

        time_series_data[f'series_{i}'] = trend + seasonal + noise

    # Create DataFrame
    ts_df = pd.DataFrame(time_series_data, index=dates)

    # Add external features
    ts_df['temperature'] = 20 + 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25) + np.random.normal(0, 2, 1000)
    ts_df['humidity'] = 50 + 20 * np.sin(2 * np.pi * np.arange(1000) / 365.25) + np.random.normal(0, 5, 1000)

    def create_transformer_sequences(data, seq_length=24, pred_length=1, target_cols=None):
        """
        Create sequences for transformer training
        """
        if target_cols is None:
            target_cols = [col for col in data.columns if col.startswith('series_')]

        X, y = [], []

        for i in range(seq_length, len(data) - pred_length + 1):
            # Input sequence
            X.append(data.iloc[i-seq_length:i].values)
            # Target (next values)
            if pred_length == 1:
                y.append(data.iloc[i][target_cols].values)
            else:
                y.append(data.iloc[i:i+pred_length][target_cols].values.flatten())

        return np.array(X), np.array(y)

    def create_positional_encoding(seq_len, d_model):
        """
        Create positional encodings for transformer
        """
        angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(10000,
                                    (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def create_time_series_transformer(input_dim, seq_len, pred_len, d_model=64, num_heads=4, num_layers=2, d_ff=128):
        """
        Create transformer model for time series
        """
        inputs = tf.keras.Input(shape=(seq_len, input_dim))

        # Add positional encoding
        pos_encoding = create_positional_encoding(seq_len, input_dim)
        x = inputs + pos_encoding

        # Multi-head attention layers
        for _ in range(num_layers):
            # Multi-head self-attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )(x, x)

            # Add & Norm
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization()(x)

            # Feed Forward Network
            ffn_output = tf.keras.Sequential([
                tf.keras.layers.Dense(d_ff, activation='relu'),
                tf.keras.layers.Dense(input_dim)
            ])(x)

            # Add & Norm
            x = tf.keras.layers.Add()([x, ffn_output])
            x = tf.keras.layers.LayerNormalization()(x)

        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Dense layers for prediction
        x = tf.keras.layers.Dense(d_model, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(pred_len * input_dim)(x)

        # Reshape to expected output format
        outputs = tf.keras.layers.Reshape((pred_len, input_dim))(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    # Prepare data
    print("Preparing transformer data...")

    target_cols = [col for col in ts_df.columns if col.startswith('series_')]
    input_dim = len(ts_df.columns)  # All columns as input

    X, y = create_transformer_sequences(ts_df, seq_length=48, pred_length=6, target_cols=target_cols)

    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Scale data
    from sklearn.preprocessing import StandardScaler

    # Scale inputs
    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = input_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    # Create and train transformer model
    print("Creating transformer model...")

    transformer_model = create_time_series_transformer(
        input_dim=len(target_cols),
        seq_len=48,
        pred_len=6,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128
    )

    print(transformer_model.summary())

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint('best_transformer.h5', save_best_only=True)
    ]

    # Train model
    print("Training transformer...")

    history = transformer_model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred_scaled = transformer_model.predict(X_test_scaled)

    # Inverse transform predictions
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])).reshape(y_pred_scaled.shape)
    y_test_original = target_scaler.inverse_transform(y_test_scaled.reshape(-1, y_test_scaled.shape[-1])).reshape(y_test_scaled.shape)

    # Calculate metrics for each target series
    results = {}
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(y_test_original[:, :, i], y_pred[:, :, i])
        rmse = np.sqrt(mean_squared_error(y_test_original[:, :, i], y_pred[:, :, i]))

        results[col] = {'MAE': mae, 'RMSE': rmse}
        print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Visualize results
    plt.figure(figsize=(20, 12))

    # Plot 1: Training history
    plt.subplot(3, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Transformer Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2-6: Predictions for each series
    for i, col in enumerate(target_cols):
        plt.subplot(3, 3, i+2)

        # Plot one step ahead predictions
        n_plot = min(100, len(y_test_original))
        indices = np.arange(n_plot)

        plt.plot(indices, y_test_original[:n_plot, 0, i],
                label='Actual', linewidth=2, color='black')
        plt.plot(indices, y_pred[:n_plot, 0, i],
                label='Predicted', alpha=0.7, color='red')

        plt.title(f'{col} - One Step Ahead')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 7: Multi-step forecast example
    plt.subplot(3, 3, 8)
    sample_idx = 0
    actual_steps = y_test_original[sample_idx, :, 0]
    predicted_steps = y_pred[sample_idx, :, 0]

    steps = np.arange(len(actual_steps))
    plt.plot(steps, actual_steps, label='Actual', linewidth=2, marker='o')
    plt.plot(steps, predicted_steps, label='Predicted', linewidth=2, marker='s')
    plt.title('Multi-step Forecast Example')
    plt.xlabel('Forecast Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 8: Attention visualization (synthetic)
    plt.subplot(3, 3, 9)
    # Create synthetic attention weights for visualization
    synthetic_attention = np.random.exponential(0.3, (48, 48))
    synthetic_attention = synthetic_attention / np.sum(synthetic_attention, axis=1, keepdims=True)

    im = plt.imshow(synthetic_attention, cmap='viridis', aspect='auto')
    plt.title('Attention Pattern (Synthetic)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.colorbar(im)

    plt.tight_layout()
    plt.show()

    # Attention analysis
    print("\n" + "=" * 50)
    print("ATTENTION ANALYSIS")
    print("=" * 50)

    # For a real implementation, you would extract attention weights
    # Here we show what attention analysis would look like

    def analyze_attention_patterns(attention_weights, seq_len):
        """
        Analyze attention patterns
        """
        # Average attention over heads
        avg_attention = np.mean(attention_weights, axis=0)  # Assuming shape [heads, seq_len, seq_len]

        # Find most attended positions
        attended_positions = np.argsort(np.mean(avg_attention, axis=0))[-10:]

        print("Top 10 most attended positions:")
        print(attended_positions)

        # Attention entropy (measure of focus)
        attention_entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-8), axis=1)
        print(f"Average attention entropy: {np.mean(attention_entropy):.3f}")

        return avg_attention, attended_positions

    # Compare with other models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    # Create simple baseline models for comparison
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    # Flatten data for traditional ML models
    X_train_flat = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
    X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], -1)
    y_train_flat = y_train_scaled.reshape(y_train_scaled.shape[0], -1)
    y_test_flat = y_test_scaled.reshape(y_test_scaled.shape[0], -1)

    # Linear regression baseline
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train_flat)
    lr_pred = lr.predict(X_test_flat)

    # Random forest baseline
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train_flat)
    rf_pred = rf.predict(X_test_flat)

    # Compare results
    comparison = pd.DataFrame({
        'Model': ['Transformer', 'Random Forest', 'Linear Regression'],
        'MAE': [
            np.mean([results[col]['MAE'] for col in target_cols]),
            mean_absolute_error(y_test_flat, rf_pred),
            mean_absolute_error(y_test_flat, lr_pred)
        ],
        'RMSE': [
            np.mean([results[col]['RMSE'] for col in target_cols]),
            np.sqrt(mean_squared_error(y_test_flat, rf_pred)),
            np.sqrt(mean_squared_error(y_test_flat, lr_pred))
        ]
    })

    print(comparison.to_string(index=False))

    return {
        'model': transformer_model,
        'history': history,
        'results': results,
        'predictions': y_pred,
        'comparison': comparison
    }

# Exercise 10: Student Tasks
def exercise_10_implementation():
    """
    Student transformer implementation practice
    """
    print("Transformer Time Series Exercise")
    print("=" * 40)

    # TODO: 1. Implement causal self-attention (for autoregressive generation)
    # TODO: 2. Add convolutional position encoding
    # TODO: 3. Implement multi-scale transformer
    # TODO: 4. Add memory-efficient attention mechanisms

    def create_causal_transformer(input_dim, seq_len, pred_len):
        """
        Causal transformer for autoregressive generation (your task)
        """
        # TODO: Implement causal masking
        # TODO: Ensure model can't see future tokens
        pass

    def create_multiscale_transformer(input_dim, seq_len, pred_len):
        """
        Multi-scale transformer with different attention windows (your task)
        """
        # TODO: Combine local and global attention
        # TODO: Different scales for different time horizons
        pass

    # Your implementation here
    pass

# Run exercise
exercise_10_implementation()
```

---

## Advanced Topics Exercises

### Exercise 11: Multivariate Time Series Forecasting

**Objective:** Handle multiple related time series simultaneously.

```python
def multivariate_forecasting_exercise():
    """
    Multivariate time series forecasting with VAR and LSTM
    """
    # Generate multivariate economic indicators
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', periods=500, freq='M')

    n_series = 4
    series_names = ['GDP', 'Inflation', 'Unemployment', 'Interest_Rate']

    # Create correlated multivariate series
    correlation_matrix = np.array([
        [1.0, -0.3, -0.7, 0.5],   # GDP
        [-0.3, 1.0, 0.4, -0.2],   # Inflation
        [-0.7, 0.4, 1.0, -0.6],   # Unemployment
        [0.5, -0.2, -0.6, 1.0]    # Interest Rate
    ])

    # Generate base processes
    base_processes = np.random.multivariate_normal(
        mean=[100, 2, 5, 3],
        cov=[[10, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 0.5]],
        size=len(dates)
    )

    # Add correlations
    multivariate_data = np.dot(base_processes, np.linalg.cholesky(correlation_matrix).T)

    # Create DataFrame
    multivariate_df = pd.DataFrame(multivariate_data, columns=series_names, index=dates)

    # Add trend and seasonality
    for i, col in enumerate(series_names):
        # Add trend
        trend = np.linspace(0, 5, len(dates))
        multivariate_df[col] += trend

        # Add seasonality for some series
        if col in ['Inflation', 'Unemployment']:
            seasonal = 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
            multivariate_df[col] += seasonal

    print("Multivariate Time Series Data:")
    print(multivariate_df.head())
    print("\nCorrelation Matrix:")
    print(multivariate_df.corr())

    def vector_autoregression_analysis(data, max_lags=5):
        """
        Vector Autoregression analysis
        """
        from statsmodels.tsa.vector_ar.var_model import VAR

        # Fit VAR model
        model = VAR(data)
        lag_order = model.select_order(maxlags=max_lags)

        # Choose best lag order (using AIC)
        selected_lag = lag_order.aic

        # Fit final model
        var_model = model.fit(selected_lag)

        return var_model, lag_order

    def create_multivariate_lstm_sequences(data, seq_length, target_cols, pred_length=1):
        """
        Create sequences for multivariate LSTM
        """
        X, y = [], []

        for i in range(seq_length, len(data) - pred_length + 1):
            # Input sequence (all features)
            X.append(data.iloc[i-seq_length:i].values)

            # Target (selected target series)
            if pred_length == 1:
                y.append(data.iloc[i][target_cols].values)
            else:
                y.append(data.iloc[i:i+pred_length][target_cols].values.flatten())

        return np.array(X), np.array(y)

    # Split data
    train_size = int(len(multivariate_df) * 0.8)
    train_data = multivariate_df.iloc[:train_size]
    test_data = multivariate_df.iloc[train_size:]

    print(f"\nTraining data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Method 1: Vector Autoregression (VAR)
    print("\n" + "=" * 50)
    print("VECTOR AUTOREGRESSION")
    print("=" * 50)

    var_model, lag_order = vector_autoregression_analysis(train_data)

    print("VAR Model Summary:")
    print(f"Selected lag order: {lag_order.aic}")
    print("\nLag Order Selection:")
    print(lag_order.summary())

    # VAR forecast
    var_forecast = var_model.forecast(train_data.values[-var_model.k_ar:], steps=len(test_data))
    var_forecast_df = pd.DataFrame(var_forecast, columns=series_names, index=test_data.index)

    # Evaluate VAR
    var_results = {}
    for col in series_names:
        mae = mean_absolute_error(test_data[col], var_forecast_df[col])
        rmse = np.sqrt(mean_squared_error(test_data[col], var_forecast_df[col]))
        var_results[col] = {'MAE': mae, 'RMSE': rmse}
        print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Method 2: Multivariate LSTM
    print("\n" + "=" * 50)
    print("MULTIVARIATE LSTM")
    print("=" * 50)

    # Prepare LSTM data
    seq_length = 12  # 12 months
    target_cols = series_names  # Predict all series

    X_train, y_train = create_multivariate_lstm_sequences(train_data, seq_length, target_cols)
    X_test, y_test = create_multivariate_lstm_sequences(
        pd.concat([train_data.tail(seq_length), test_data]),
        seq_length, target_cols
    )

    # Remove the initial part that overlaps with training
    X_test = X_test[len(test_data):]
    y_test = y_test[len(test_data):]

    print(f"LSTM training shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"LSTM test shapes: X_test={X_test.shape}, y_test={y_test.shape}")

    # Create multivariate LSTM model
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate

    def create_multivariate_lstm(input_shape, output_shape):
        """
        Multivariate LSTM for forecasting multiple series
        """
        inputs = Input(shape=input_shape)

        # Shared LSTM layers
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32)(x)

        # Output layer for each target series
        outputs = []
        for i in range(output_shape[-1] if len(output_shape) > 1 else output_shape[0]):
            # Each series gets its own output head
            if len(output_shape) == 2:  # Multi-step
                series_output = Dense(output_shape[0])(x)
            else:  # Single step
                series_output = Dense(1)(x)
            outputs.append(series_output)

        # Combine outputs
        if len(outputs) > 1:
            output = Concatenate()(outputs)
        else:
            output = outputs[0]

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

    # Create and train model
    lstm_model = create_multivariate_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        output_shape=y_train.shape[1:]
    )

    print(lstm_model.summary())

    # Train LSTM
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
    ]

    history = lstm_model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=100,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # LSTM predictions
    lstm_pred = lstm_model.predict(X_test)

    # Reshape predictions if needed
    if lstm_pred.shape[1] != len(series_names):
        lstm_pred = lstm_pred.reshape(len(series_names), -1).T

    # Create DataFrame for LSTM predictions
    if len(lstm_pred.shape) == 2:
        lstm_forecast_df = pd.DataFrame(lstm_pred, columns=series_names, index=test_data.index)
    else:
        lstm_forecast_df = pd.DataFrame(lstm_pred, index=test_data.index)

    # Evaluate LSTM
    lstm_results = {}
    for col in series_names:
        if col in lstm_forecast_df.columns:
            mae = mean_absolute_error(test_data[col], lstm_forecast_df[col])
            rmse = np.sqrt(mean_squared_error(test_data[col], lstm_forecast_df[col]))
            lstm_results[col] = {'MAE': mae, 'RMSE': rmse}
            print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Method 3: Individual models for comparison
    print("\n" + "=" * 50)
    print("INDIVIDUAL UNIVARIATE MODELS")
    print("=" * 50)

    individual_results = {}
    individual_forecasts = {}

    for col in series_names:
        print(f"\nTraining individual model for {col}...")

        # Create simple univariate LSTM for this series
        univariate_data = train_data[col]

        # Create sequences for univariate
        X_uni, y_uni = [], []
        for i in range(seq_length, len(univariate_data)):
            X_uni.append(univariate_data.iloc[i-seq_length:i].values)
            y_uni.append(univariate_data.iloc[i])

        X_uni = np.array(X_uni)
        y_uni = np.array(y_uni)

        # Train univariate model
        uni_model = Sequential([
            LSTM(32, input_shape=(seq_length, 1)),
            Dense(1)
        ])
        uni_model.compile(optimizer='adam', loss='mae')

        X_uni_reshaped = X_uni.reshape(X_uni.shape[0], X_uni.shape[1], 1)

        # Split univariate data
        uni_train_size = int(len(X_uni) * 0.8)
        X_uni_train, X_uni_test = X_uni_reshaped[:uni_train_size], X_uni_reshaped[uni_train_size:]
        y_uni_train, y_uni_test = y_uni[:uni_train_size], y_uni[uni_train_size:]

        # Train
        uni_model.fit(X_uni_train, y_uni_train, epochs=50, verbose=0, batch_size=16)

        # Predict
        uni_pred = uni_model.predict(X_uni_test, verbose=0).flatten()

        # Evaluate
        mae = mean_absolute_error(y_uni_test, uni_pred)
        rmse = np.sqrt(mean_squared_error(y_uni_test, uni_pred))

        individual_results[col] = {'MAE': mae, 'RMSE': rmse}
        individual_forecasts[col] = uni_pred

        print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}")

    # Compare all methods
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)

    comparison_df = pd.DataFrame({
        'Series': series_names,
        'VAR_MAE': [var_results[col]['MAE'] for col in series_names],
        'VAR_RMSE': [var_results[col]['RMSE'] for col in series_names],
        'LSTM_MAE': [lstm_results[col]['MAE'] for col in series_names],
        'LSTM_RMSE': [lstm_results[col]['RMSE'] for col in series_names],
        'Individual_MAE': [individual_results[col]['MAE'] for col in series_names],
        'Individual_RMSE': [individual_results[col]['RMSE'] for col in series_names]
    })

    print(comparison_df.to_string(index=False))

    # Visualize results
    plt.figure(figsize=(20, 15))

    # Plot 1: Actual vs VAR predictions
    plt.subplot(3, 3, 1)
    for i, col in enumerate(series_names):
        plt.plot(test_data.index, test_data[col], label=f'{col} Actual', linewidth=2)
        plt.plot(test_data.index, var_forecast_df[col], label=f'{col} VAR', alpha=0.7)
    plt.title('VAR Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Actual vs LSTM predictions
    plt.subplot(3, 3, 2)
    for i, col in enumerate(series_names):
        if col in lstm_forecast_df.columns:
            plt.plot(test_data.index, test_data[col], label=f'{col} Actual', linewidth=2)
            plt.plot(test_data.index, lstm_forecast_df[col], label=f'{col} LSTM', alpha=0.7)
    plt.title('LSTM Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3-6: Individual series comparisons
    for i, col in enumerate(series_names):
        plt.subplot(3, 3, i+3)

        plt.plot(test_data.index, test_data[col], label='Actual', linewidth=2, color='black')
        plt.plot(test_data.index, var_forecast_df[col], label='VAR', alpha=0.7)
        if col in lstm_forecast_df.columns:
            plt.plot(test_data.index, lstm_forecast_df[col], label='LSTM', alpha=0.7)

        plt.title(f'{col} - All Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 7: Correlation analysis
    plt.subplot(3, 3, 7)
    correlation_matrix = multivariate_df.corr()
    im = plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(series_names)), series_names, rotation=45)
    plt.yticks(range(len(series_names)), series_names)
    plt.title('Series Correlation Matrix')

    # Add correlation values
    for i in range(len(series_names)):
        for j in range(len(series_names)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

    # Plot 8: Prediction error distribution
    plt.subplot(3, 3, 8)
    var_errors = []
    lstm_errors = []

    for col in series_names:
        var_error = np.abs(test_data[col] - var_forecast_df[col])
        var_errors.extend(var_error.values)

        if col in lstm_forecast_df.columns:
            lstm_error = np.abs(test_data[col] - lstm_forecast_df[col])
            lstm_errors.extend(lstm_error.values)

    plt.hist(var_errors, alpha=0.5, label='VAR', bins=20)
    if lstm_errors:
        plt.hist(lstm_errors, alpha=0.5, label='LSTM', bins=20)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.legend()

    # Plot 9: Training history
    plt.subplot(3, 3, 9)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'var_model': var_model,
        'lstm_model': lstm_model,
        'var_forecast': var_forecast_df,
        'lstm_forecast': lstm_forecast_df,
        'comparison': comparison_df,
        'results': {
            'VAR': var_results,
            'LSTM': lstm_results,
            'Individual': individual_results
        }
    }

# Exercise 11: Student Tasks
def exercise_11_implementation():
    """
    Student multivariate forecasting practice
    """
    print("Multivariate Forecasting Exercise")
    print("=" * 40)

    # TODO: 1. Implement Vector Error Correction Model (VECM)
    # TODO: 2. Add common factor models
    # TODO: 3. Implement dynamic factor models
    # TODO: 4. Add regime-switching multivariate models

    def vec_model_with_regimes(data, n_regimes=2):
        """
        Regime-switching VAR model (your task)
        """
        # TODO: Detect regime changes
        # TODO: Fit separate VAR models for each regime
        # TODO: Implement regime probability forecasting
        pass

    def dynamic_factor_model(data, n_factors=2):
        """
        Dynamic factor model (your task)
        """
        # TODO: Extract common factors
        # TODO: Model factor dynamics
        # TODO: Reconstruct individual series from factors
        pass

    # Your implementation here
    pass

# Run exercise
exercise_11_implementation()
```

This comprehensive practice exercise covers all the essential aspects of time series forecasting, from foundational concepts to advanced deep learning techniques. Each exercise builds upon the previous ones and provides hands-on experience with real implementations.

---

_Continue to the projects section to apply these techniques to real-world problems._
