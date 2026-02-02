# Time Series Forecasting Fundamentals Theory

## Table of Contents

1. [Introduction to Time Series](#introduction-to-time-series)
2. [Time Series Components](#time-series-components)
3. [Stationarity and Transformations](#stationarity-and-transformations)
4. [Traditional Forecasting Methods](#traditional-forecasting-methods)
5. [Modern Deep Learning Approaches](#modern-deep-learning-approaches)
6. [Model Selection and Validation](#model-selection-and-validation)
7. [Feature Engineering for Time Series](#feature-engineering-for-time-series)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Advanced Topics](#advanced-topics)
10. [Industry Applications](#industry-applications)

---

## Introduction to Time Series

### What is Time Series Forecasting?

Time series forecasting is the process of predicting future values based on previously observed values in a chronological sequence. Unlike other machine learning problems, time series data has a natural temporal ordering that must be preserved.

**Key Characteristics:**

- **Temporal Dependence**: Observations are not independent
- **Trend**: Long-term direction of change
- **Seasonality**: Regular, predictable patterns
- **Autocorrelation**: Values correlate with past values
- **Non-stationarity**: Statistical properties change over time

### Types of Time Series Problems

#### 1. **Univariate Time Series**

- Single variable tracked over time
- Example: Daily stock prices, monthly sales
- **Use Case**: Simple trend forecasting

#### 2. **Multivariate Time Series**

- Multiple variables tracked over time
- Example: Sales, marketing spend, economic indicators
- **Use Case**: Complex system modeling

#### 3. **Univariate Forecasting**

- Predicting one future value
- Example: Tomorrow's temperature
- **Models**: ARIMA, Exponential Smoothing

#### 4. **Multivariate Forecasting**

- Predicting multiple future values
- Example: Forecasting all product categories
- **Models**: Vector Autoregression (VAR), LSTM

#### 5. **One-step-ahead vs Multi-step Forecasting**

- **One-step**: Predict next immediate value
- **Multi-step**: Predict several future values
- **Challenge**: Uncertainty compounds over time

---

## Time Series Components

### 1. **Trend Component**

Long-term direction or movement in the data.

**Types of Trends:**

- **Increasing Trend**: Upward movement
- **Decreasing Trend**: Downward movement
- **Stationary Trend**: No clear long-term direction

**Mathematical Representation:**

```
Trend(t) = a + bt
where: a = intercept, b = slope
```

**Detection Methods:**

- Moving averages
- Linear regression
- Mann-Kendall test
- Augmented Dickey-Fuller test

### 2. **Seasonal Component**

Regular, predictable patterns that repeat at fixed intervals.

**Common Seasonal Periods:**

- **Hourly**: 24 hours
- **Daily**: 7 days (weekly pattern)
- **Monthly**: 12 months (yearly pattern)
- **Quarterly**: 4 quarters (yearly pattern)

**Mathematical Representation:**

```
Seasonality(t) = Σ(sin(2πkt/T) + cos(2πkt/T))
where: k = harmonic number, T = period
```

### 3. **Cyclical Component**

Longer-term patterns that are not fixed in length.

- Business cycles (3-10 years)
- Economic cycles
- No fixed period length

### 4. **Irregular Component (Noise)**

Random, unpredictable fluctuations.

- White noise
- Model residuals
- External shocks

### Time Series Decomposition

**Additive Model:**

```
Y(t) = Trend(t) + Seasonal(t) + Cyclical(t) + Irregular(t)
```

**Multiplicative Model:**

```
Y(t) = Trend(t) × Seasonal(t) × Cyclical(t) × Irregular(t)
```

**Classical Decomposition Process:**

1. Estimate trend using moving averages
2. Remove trend to get detrended series
3. Extract seasonal component
4. Calculate residuals

```python
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Perform seasonal decomposition
decomposition = seasonal_decompose(ts_data, model='additive', period=12)

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

---

## Stationarity and Transformations

### What is Stationarity?

A stationary time series has statistical properties that don't change over time:

- **Constant mean** over time
- **Constant variance** over time
- **Constant covariance** between values at different lags

### Types of Stationarity

#### 1. **Strict Stationarity**

Joint probability distribution remains constant for all time shifts.

#### 2. **Weak Stationarity (Wide-Sense)**

- Constant mean: E[X(t)] = μ
- Constant variance: Var[X(t)] = σ²
- Covariance depends only on lag: Cov[X(t), X(t+k)] = γ(k)

### Why Stationarity Matters?

- Most forecasting models assume stationarity
- Non-stationary series lead to spurious regressions
- Stationarity enables reliable parameter estimation

### Tests for Stationarity

#### 1. **Augmented Dickey-Fuller (ADF) Test**

```
H0: Series has a unit root (non-stationary)
H1: Series is stationary
```

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')

    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is non-stationary")
```

#### 2. **KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)**

```
H0: Series is stationary
H1: Series is non-stationary
```

### Transformations to Achieve Stationarity

#### 1. **Differencing**

First-order differencing:

```
ΔY(t) = Y(t) - Y(t-1)
```

Second-order differencing:

```
Δ²Y(t) = ΔY(t) - ΔY(t-1)
```

**When to use:**

- Linear trends
- Constant variance
- Remove serial correlation

```python
# First-order differencing
diff_series = ts_data.diff().dropna()

# Second-order differencing
diff2_series = ts_data.diff().diff().dropna()
```

#### 2. **Logarithmic Transformation**

```
Z(t) = log(Y(t))
```

**Benefits:**

- Stabilizes variance
- Makes multiplicative models additive
- Normalizes skewed distributions

```python
log_series = np.log(ts_data)
```

#### 3. **Box-Cox Transformation**

```
Z(t) = (Y(t)^λ - 1) / λ    if λ ≠ 0
Z(t) = log(Y(t))           if λ = 0
```

**Automatic Selection:**

```python
from scipy.stats import boxcox
transformed_data, lambda_param = boxcox(original_data)
```

#### 4. **Seasonal Differencing**

```
Δs Y(t) = Y(t) - Y(t-s)
where s = seasonal period
```

**Example:**

```python
# Monthly data with yearly seasonality
seasonal_diff = ts_data.diff(12).dropna()
```

---

## Traditional Forecasting Methods

### 1. **Moving Averages**

#### Simple Moving Average (SMA)

```
SMA(n) = (Y(t) + Y(t-1) + ... + Y(t-n+1)) / n
```

**Use Cases:**

- Trend estimation
- Simple forecasting baseline
- Noise reduction

```python
def simple_moving_average(data, window):
    return data.rolling(window=window).mean()

# Example usage
sma_30 = simple_moving_average(ts_data, 30)
forecast = sma_30.iloc[-1]  # Use last value as forecast
```

#### Exponential Moving Average (EMA)

```
EMA(t) = α × Y(t) + (1-α) × EMA(t-1)
where α = smoothing parameter (0 < α < 1)
```

**Advantages:**

- More weight to recent observations
- Smooth response to changes
- Memory of all past observations

```python
def exponential_moving_average(data, alpha):
    ema = [data.iloc[0]]  # Initialize with first value

    for i in range(1, len(data)):
        ema.append(alpha * data.iloc[i] + (1 - alpha) * ema[i-1])

    return pd.Series(ema, index=data.index)

ema_0_3 = exponential_moving_average(ts_data, 0.3)
```

### 2. **Exponential Smoothing Methods**

#### Simple Exponential Smoothing (SES)

```
Forecast(t+1|t) = α × Y(t) + (1-α) × Forecast(t|t-1)
```

**When to use:**

- No trend or seasonality
- Short-term forecasting
- Stable patterns

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(ts_data, trend=None, seasonal=None)
fit_model = model.fit(smoothing_level=0.3)
forecast = fit_model.forecast(steps=12)
```

#### Holt's Linear Trend Method

```
Level: L(t) = α × Y(t) + (1-α) × (L(t-1) + T(t-1))
Trend: T(t) = β × (L(t) - L(t-1)) + (1-β) × T(t-1)
Forecast: F(t+h|t) = L(t) + h × T(t)
```

**When to use:**

- Linear trend present
- No seasonality
- Medium-term forecasting

```python
model = ExponentialSmoothing(ts_data, trend='add', seasonal=None)
fit_model = model.fit(smoothing_level=0.3, smoothing_trend=0.1)
```

#### Holt-Winters Exponential Smoothing

```
Level: L(t) = α × (Y(t) - S(t-s)) + (1-α) × (L(t-1) + T(t-1))
Trend: T(t) = β × (L(t) - L(t-1)) + (1-β) × T(t-1)
Seasonal: S(t) = γ × (Y(t) - L(t)) + (1-γ) × S(t-s)
Forecast: F(t+h|t) = L(t) + h × T(t) + S(t+h-s)
```

**When to use:**

- Trend and seasonality present
- Seasonal patterns
- Medium to long-term forecasting

```python
# Additive seasonality
model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12)
fit_model = model.fit()

# Multiplicative seasonality (better for changing seasonal amplitude)
model = ExponentialSmoothing(ts_data, trend='add', seasonal='mul', seasonal_periods=12)
fit_model = model.fit()
```

### 3. **ARIMA Models**

#### ARIMA(p, d, q) Components

- **AR(p)**: Autoregressive terms - past values
- **I(d)**: Integrated (differenced) - stationarity
- **MA(q)**: Moving average terms - past forecast errors

#### Autoregressive (AR) Models

```
Y(t) = c + φ₁ × Y(t-1) + φ₂ × Y(t-2) + ... + φₚ × Y(t-p) + ε(t)
```

**Characteristics:**

- Linear relationship with past values
- Captures persistence and momentum
- Stationary when roots of characteristic equation lie outside unit circle

#### Moving Average (MA) Models

```
Y(t) = μ + ε(t) + θ₁ × ε(t-1) + θ₂ × ε(t-2) + ... + θₑ × ε(t-q)
```

**Characteristics:**

- Depends on past forecast errors
- Captures short-term dependencies
- Always stationary

#### ARMA Models

```
Y(t) = c + φ₁ × Y(t-1) + ... + φₚ × Y(t-p) + ε(t) + θ₁ × ε(t-1) + ... + θₑ × ε(t-q)
```

**Stationarity Conditions:**

- AR part must be stationary
- MA part is always stationary

#### ARIMA Model Selection

**Box-Jenkins Methodology:**

1. **Identification**: Determine p, d, q values
2. **Estimation**: Fit the model
3. **Diagnostic Checking**: Validate residuals
4. **Forecasting**: Generate predictions

**Identification Steps:**

```python
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

# 1. Check stationarity and determine d
check_stationarity(ts_data)

# 2. Plot ACF and PACF to determine p and q
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Autocorrelation Function (ACF)
sm.graphics.tsa.plot_acf(ts_data.dropna(), ax=axes[0], lags=40)

# Partial Autocorrelation Function (PACF)
sm.graphics.tsa.plot_pacf(ts_data.dropna(), ax=axes[1], lags=40)

plt.tight_layout()
plt.show()
```

**ACF/PACF Interpretation:**

- **ACF gradual decay + PACF cutoff after lag p**: AR(p) model
- **ACF cutoff after lag q + PACF gradual decay**: MA(q) model
- **Both gradual decay**: ARMA(p,q) model

**Model Selection Criteria:**

```python
# Try different ARIMA orders and compare AIC/BIC
orders = [(0,1,1), (1,1,0), (1,1,1), (2,1,0), (2,1,1), (2,1,2)]
aic_values = []

for order in orders:
    model = ARIMA(ts_data, order=order)
    fitted_model = model.fit()
    aic_values.append((order, fitted_model.aic))

# Select model with lowest AIC
best_order = min(aic_values, key=lambda x: x[1])[0]
print(f"Best ARIMA order: {best_order}")

# Fit best model
final_model = ARIMA(ts_data, order=best_order)
fitted_final = final_model.fit()
```

#### Seasonal ARIMA (SARIMA)

```
ARIMA(p,d,q)(P,D,Q)s
```

Where:

- (p,d,q): Non-seasonal orders
- (P,D,Q): Seasonal orders
- s: Seasonal period

**SARIMA Model:**

```
Φ(B^s) φ(B) (1-B^s)^D (1-B)^d Y(t) = Θ(B^s) θ(B) ε(t)
```

**Where:**

- Φ(B^s): Seasonal AR terms
- φ(B): Non-seasonal AR terms
- Θ(B^s): Seasonal MA terms
- θ(B): Non-seasonal MA terms

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(1,1,1)(1,1,1,12) for monthly data
model = SARIMAX(ts_data,
                order=(1, 1, 1),           # Non-seasonal
                seasonal_order=(1, 1, 1, 12)) # Seasonal
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=24)
```

---

## Modern Deep Learning Approaches

### 1. **Long Short-Term Memory (LSTM) Networks**

#### LSTM Architecture

LSTMs are designed to address the vanishing gradient problem in traditional RNNs.

**LSTM Cell Structure:**

```
Forget Gate:    f(t) = σ(W(f) × [h(t-1), x(t)] + b(f))
Input Gate:     i(t) = σ(W(i) × [h(t-1), x(t)] + b(i))
Candidate:      Ĉ(t) = tanh(W(C) × [h(t-1), x(t)] + b(C))
Cell State:     C(t) = f(t) × C(t-1) + i(t) × Ĉ(t)
Output Gate:    o(t) = σ(W(o) × [h(t-1), x(t)] + b(o))
Hidden State:   h(t) = o(t) × tanh(C(t))
```

**Advantages:**

- Handles long-term dependencies
- Addresses vanishing gradient problem
- Selective memory (forget/input/output gates)
- Captures complex temporal patterns

#### LSTM for Time Series Forecasting

**Architecture Options:**

1. **Vanilla LSTM**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare data for LSTM
def prepare_lstm_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Usage
n_steps = 60  # Use 60 time steps to predict next value
X, y = prepare_lstm_data(scaled_data, n_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)  # 3D for LSTM

model = create_lstm_model((n_steps, 1))
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
```

2. **Stacked LSTM**

```python
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(n_steps, 1)),
    Dropout(0.3),
    LSTM(50, return_sequences=True),
    Dropout(0.3),
    LSTM(25, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])
```

3. **Bidirectional LSTM**

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Bidirectional(LSTM(50, return_sequences=True), input_shape=(n_steps, 1)),
    Dropout(0.2),
    Bidirectional(LSTM(50)),
    Dropout(0.2),
    Dense(1)
])
```

#### Attention-Enhanced LSTMs

```python
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim):
        super(TemporalAttention, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.attention_dim),
                               initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.attention_dim,),
                               initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(self.attention_dim,),
                               initializer='random_normal', trainable=True)

    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, features)
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        return output

# LSTM with Attention
def create_attention_lstm(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    attention_out = TemporalAttention(32)(lstm_out)
    output = Dense(1)(attention_out)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model
```

### 2. **Transformer Models for Time Series**

#### Multi-Head Attention Mechanism

```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
```

**Advantages:**

- Parallel processing (no sequential bottleneck)
- Long-range dependencies
- Interpretable attention patterns
- State-of-the-art performance

#### Time Series Transformer Implementation

```python
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout

class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, d_model, num_heads, d_ff, input_dim, seq_len):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Input projection
        self.input_projection = Dense(d_model)

        # Positional encoding
        self.pos_encoding = self.positional_encoding(seq_len, d_model)

        # Multi-head attention
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation='relu'),
            Dense(d_model)
        ])

        # Layer normalization and dropout
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)

        # Output projection
        self.output_projection = Dense(input_dim)

    def get_angles(self, pos, i, d_model):
        angles = 1 / np.power(10000., (2 * (i//2)) / np.float32(d_model))
        return pos * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, training=None):
        seq_len = tf.shape(inputs)[1]

        # Input projection
        x = self.input_projection(inputs)

        # Add positional encoding
        x += self.pos_encoding[:, :seq_len, :]

        # Multi-head attention
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        # Output projection
        output = self.output_projection(out2[:, -1, :])  # Use last timestep

        return output

# Create and compile model
model = TimeSeriesTransformer(
    d_model=128,
    num_heads=8,
    d_ff=512,
    input_dim=1,
    seq_len=60
)

model.compile(optimizer='adam', loss='mse')
```

### 3. **Convolutional Neural Networks (CNN) for Time Series**

#### 1D CNN Architecture

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

def create_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=50, kernel_size=3, activation='relu'),
        Conv1D(filters=50, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model
```

#### Hybrid CNN-LSTM Model

```python
def create_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(filters=50, kernel_size=3, activation='relu'),
        Conv1D(filters=50, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        LSTM(100, return_sequences=True),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model
```

### 4. **Prophet (Facebook's Time Series Library)**

#### Prophet Advantages

- Handles missing data and outliers
- Automatic trend detection
- Built-in seasonality modeling
- Holiday effects
- Confidence intervals
- Interpretable components

#### Prophet Model Structure

```
Y(t) = g(t) + s(t) + h(t) + ε(t)
```

Where:

- g(t): Trend component (piecewise linear/logistic)
- s(t): Seasonal component ( fourier series)
- h(t): Holiday component
- ε(t): Error term

#### Prophet Implementation

```python
from prophet import Prophet
import pandas as pd

# Prepare data for Prophet
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365, freq='D'),
    'y': ts_data.values  # Your time series data
})

# Initialize and fit model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',  # or 'additive'
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    n_changepoints=25
)

model.fit(df)

# Create future dates
future = model.make_future_dataframe(periods=30)  # 30 days ahead

# Generate forecast
forecast = model.predict(future)

# Plot components
fig = model.plot_components(forecast)
```

#### Prophet with Custom Seasonality

```python
# Add custom seasonalities
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)

# Add holidays
holidays = pd.DataFrame({
    'holiday': 'special_day',
    'ds': pd.to_datetime(['2021-01-01', '2021-12-25']),
    'lower_window': 0,
    'upper_window': 1
})

model = Prophet(holidays=holidays)
```

---

## Model Selection and Validation

### Time Series Cross-Validation

#### 1. **Time Series Split**

Maintains temporal order - no future data leaks into past.

```python
from sklearn.model_selection import TimeSeriesSplit

# Initialize time series cross-validator
tscv = TimeSeriesSplit(n_splits=5)

# Example usage
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train model and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Calculate metrics...
```

#### 2. **Walk-Forward Validation**

Rolling window approach that simulates real-world forecasting.

```python
def walk_forward_validation(data, n_train, n_test, step_size=1):
    """
    Walk-forward validation for time series

    Parameters:
    - data: Time series data
    - n_train: Size of training set
    - n_test: Size of test set
    - step_size: How often to move the window
    """
    predictions = []
    actuals = []

    for i in range(n_train, len(data) - n_test + 1, step_size):
        # Train on data up to point i
        train_end = i
        train_data = data[:train_end]
        test_data = data[train_end:train_end + n_test]

        # Fit model
        model.fit(train_data)

        # Predict
        pred = model.predict(test_data)

        predictions.extend(pred)
        actuals.extend(test_data)

    return np.array(predictions), np.array(actuals)
```

#### 3. **Expanding Window Validation**

Training set grows with each iteration.

```python
def expanding_window_validation(data, min_train_size, n_test, step_size=1):
    """
    Expanding window validation
    Training set expands with each iteration
    """
    predictions = []
    actuals = []

    for i in range(min_train_size, len(data) - n_test + 1, step_size):
        train_data = data[:i]  # All data up to point i
        test_data = data[i:i + n_test]

        model.fit(train_data)
        pred = model.predict(test_data)

        predictions.extend(pred)
        actuals.extend(test_data)

    return np.array(predictions), np.array(actuals)
```

### Model Comparison Framework

#### 1. **Information Criteria**

```python
# AIC (Akaike Information Criterion)
# Lower is better: AIC = 2k - 2ln(L)
# Where k = number of parameters, L = likelihood

# BIC (Bayesian Information Criterion)
# Lower is better: BIC = kln(n) - 2ln(L)
# Where n = sample size

def compare_arima_models(data, max_p=3, max_d=2, max_q=3):
    """
    Compare ARIMA models using AIC and BIC
    """
    results = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()

                    results.append({
                        'order': (p, d, q),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic,
                        'llf': fitted_model.llf
                    })
                except:
                    continue

    results_df = pd.DataFrame(results)
    best_aic = results_df.loc[results_df['aic'].idxmin()]
    best_bic = results_df.loc[results_df['bic'].idxmin()]

    return results_df, best_aic, best_bic
```

#### 2. **Forecast Accuracy Comparison**

```python
def compare_forecasting_methods(test_data, methods_results):
    """
    Compare different forecasting methods
    """
    metrics = {}

    for method_name, predictions in methods_results.items():
        metrics[method_name] = {
            'MAE': mean_absolute_error(test_data, predictions),
            'RMSE': np.sqrt(mean_squared_error(test_data, predictions)),
            'MAPE': np.mean(np.abs((test_data - predictions) / test_data)) * 100,
            'MASE': mean_absolute_scaled_error(test_data, predictions)
        }

    metrics_df = pd.DataFrame(metrics).T
    return metrics_df
```

---

## Feature Engineering for Time Series

### 1. **Lag Features**

Create features using past values of the time series.

```python
def create_lag_features(data, lags):
    """
    Create lag features for time series
    """
    df = pd.DataFrame({'value': data})

    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # Remove rows with NaN values
    df = df.dropna()
    return df

# Usage
lags = [1, 2, 3, 7, 14, 30]  # Various lag periods
lag_features = create_lag_features(ts_data, lags)
```

### 2. **Rolling Window Features**

Calculate statistics over rolling windows.

```python
def create_rolling_features(data, windows):
    """
    Create rolling window features
    """
    df = pd.DataFrame({'value': data})

    for window in windows:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()

    return df.dropna()

# Usage
windows = [7, 14, 30, 90]  # 1 week, 2 weeks, 1 month, 3 months
rolling_features = create_rolling_features(ts_data, windows)
```

### 3. **Exponential Moving Average Features**

```python
def create_ema_features(data, alphas):
    """
    Create exponential moving average features
    """
    df = pd.DataFrame({'value': data})

    for alpha in alphas:
        ema = data.ewm(alpha=alpha).mean()
        df[f'ema_{alpha}'] = ema

    return df
```

### 4. **Time-based Features**

Extract time components from datetime index.

```python
def create_time_features(data):
    """
    Create time-based features
    """
    df = pd.DataFrame({'value': data})
    df.index = pd.to_datetime(df.index)

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    df['weekofyear'] = df.index.isocalendar().week

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    return df
```

### 5. **Difference Features**

Create differenced features to capture changes.

```python
def create_difference_features(data, orders=[1, 2]):
    """
    Create difference features
    """
    df = pd.DataFrame({'value': data})

    for order in orders:
        diff_col = f'diff_{order}'
        df[diff_col] = df['value'].diff(order)

    return df.dropna()
```

### 6. **Fourier Features**

Capture seasonal patterns using Fourier terms.

```python
def create_fourier_features(data, periods, k):
    """
    Create Fourier features for seasonality
    """
    df = pd.DataFrame({'value': data})

    for period in periods:
        for i in range(1, k + 1):
            df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * np.arange(len(data)) / period)
            df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * np.arange(len(data)) / period)

    return df
```

### 7. **Technical Indicators**

For financial time series or similar data.

```python
def create_technical_indicators(data, window=14):
    """
    Create common technical indicators
    """
    df = pd.DataFrame({'value': data})

    # Simple Moving Average
    df['sma'] = df['value'].rolling(window=window).mean()

    # Exponential Moving Average
    df['ema'] = df['value'].ewm(span=window).mean()

    # Relative Strength Index (RSI)
    delta = df['value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df['value'].rolling(window=window).mean()
    rolling_std = df['value'].rolling(window=window).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # MACD
    ema_12 = df['value'].ewm(span=12).mean()
    ema_26 = df['value'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df.dropna()
```

### 8. **Advanced Feature Engineering**

#### Change Point Detection

```python
def detect_change_points(data, window=30):
    """
    Detect change points using rolling statistics
    """
    df = pd.DataFrame({'value': data})
    df['rolling_mean'] = df['value'].rolling(window=window).mean()
    df['rolling_std'] = df['value'].rolling(window=window).std()

    # Change in mean
    df['mean_change'] = df['rolling_mean'].diff()

    # Change in variance
    df['var_change'] = df['rolling_std'].diff()

    return df.dropna()
```

#### Regime Detection

```python
def detect_regimes(data, n_regimes=3):
    """
    Detect different regimes in time series using clustering
    """
    from sklearn.cluster import KMeans

    # Create feature matrix
    features = []
    window = 30
    for i in range(window, len(data)):
        features.append(data[i-window:i])

    features = np.array(features)

    # Cluster into regimes
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    regimes = kmeans.fit_predict(features)

    # Add regime labels to original series
    regime_series = pd.Series(index=data.index, dtype=int)
    regime_series.iloc[window:] = regimes

    return regime_series
```

---

## Evaluation Metrics

### 1. **Scale-Dependent Metrics**

#### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|y_i - ŷ_i|
```

**Advantages:**

- Same scale as original data
- Intuitive interpretation
- Robust to outliers

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

#### Root Mean Squared Error (RMSE)

```
RMSE = √((1/n) × Σ(y_i - ŷ_i)²)
```

**Advantages:**

- Penalizes large errors more than MAE
- Same scale as original data
- Commonly used

```python
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

### 2. **Percentage Error Metrics**

#### Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) × Σ|y_i - ŷ_i| / |y_i|
```

**Advantages:**

- Scale-free metric
- Easy to interpret as percentage

```python
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

**Limitations:**

- Undefined when y_i = 0
- Can be inflated for small values

#### Symmetric MAPE (sMAPE)

```
sMAPE = (200/n) × Σ|y_i - ŷ_i| / (|y_i| + |ŷ_i|)
```

```python
def smape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 200
```

### 3. **Scaled Metrics**

#### Mean Absolute Scaled Error (MASE)

```
MASE = MAE / MAE_naive
```

Where MAE_naive is the MAE of a naive forecast (using previous value).

**Advantages:**

- Scale-free
- Comparable across different series
- Always finite

```python
def mase(y_true, y_pred, y_train):
    """
    Calculate Mean Absolute Scaled Error
    """
    # Naive forecast (previous value)
    naive_forecast = y_train[:-1]
    naive_actual = y_train[1:]

    mae_naive = mean_absolute_error(naive_actual, naive_forecast)
    mae_model = mean_absolute_error(y_true, y_pred)

    return mae_model / mae_naive
```

### 4. **Directional Accuracy**

#### Directional Change Accuracy

```
DCA = (1/n) × Σ I((y_i - y_{i-1}) × (ŷ_i - y_{i-1}) > 0)
```

**Use Case:** When direction is more important than magnitude.

```python
def directional_accuracy(y_true, y_pred):
    """
    Calculate directional change accuracy
    """
    actual_changes = np.diff(y_true)
    predicted_changes = np.diff(y_pred)

    directional_matches = np.sum(np.sign(actual_changes) == np.sign(predicted_changes))

    return directional_matches / len(actual_changes)
```

### 5. **Statistical Tests for Forecast Evaluation**

#### Diebold-Mariano Test

Tests if one forecast is significantly better than another.

```python
def diebold_mariano_test(y1, y2, y_true, h=1):
    """
    Diebold-Mariano test for comparing forecasts
    y1, y2: forecasts from two different models
    y_true: actual values
    h: forecast horizon
    """
    e1 = y_true - y1  # forecast errors from model 1
    e2 = y_true - y2  # forecast errors from model 2

    # Loss differential
    d = e1**2 - e2**2

    # Test statistic (simplified version)
    dm_stat = np.mean(d) / np.sqrt(np.var(d) / len(d))

    return dm_stat  # Compare with normal distribution
```

### 6. **Confidence Intervals and Uncertainty**

#### Prediction Intervals for ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(ts_data, order=(1,1,1))
fitted_model = model.fit()

# Get forecast with confidence intervals
forecast = fitted_model.forecast(steps=12)
confidence_intervals = fitted_model.get_forecast(steps=12).conf_int()

# Plot with confidence intervals
plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data.values, label='Historical')
plt.plot(future_dates, forecast.values, label='Forecast', color='red')

plt.fill_between(future_dates,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1],
                 alpha=0.3, color='red', label='95% Confidence Interval')
plt.legend()
plt.show()
```

#### Bootstrap Prediction Intervals for ML Models

```python
def bootstrap_prediction_intervals(model, X_train, y_train, X_test,
                                 n_bootstrap=1000, alpha=0.05):
    """
    Generate bootstrap prediction intervals for ML models
    """
    predictions = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        # Fit model on bootstrap sample
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_boot, y_boot)

        # Make predictions
        pred = model_copy.predict(X_test)
        predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate confidence intervals
    lower = np.percentile(predictions, 100 * alpha/2, axis=0)
    upper = np.percentile(predictions, 100 * (1 - alpha/2), axis=0)
    median = np.median(predictions, axis=0)

    return median, lower, upper
```

---

## Advanced Topics

### 1. **Multivariate Time Series Forecasting**

#### Vector Autoregression (VAR)

```python
from statsmodels.tsa.vector_ar.var_model import VAR

def var_forecasting(data, lags, forecast_steps):
    """
    Vector Autoregression for multivariate forecasting
    """
    # data: DataFrame with multiple time series columns
    model = VAR(data)

    # Select optimal lag order
    lag_order = model.select_order(maxlags=lags)
    optimal_lags = lag_order.aic

    # Fit model
    fitted_model = model.fit(optimal_lags)

    # Generate forecasts
    forecast = fitted_model.forecast(data.values[-optimal_lags:],
                                   steps=forecast_steps)

    return forecast, fitted_model
```

#### Vector Error Correction Model (VECM)

For cointegrated series.

```python
from statsmodels.tsa.vector_ar.vecm import VECM

def vecm_forecasting(data, k_ar_diff, coint_rank, forecast_steps):
    """
    Vector Error Correction Model for cointegrated series
    """
    model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank)
    fitted_model = model.fit()

    forecast = fitted_model.predict(steps=forecast_steps)

    return forecast, fitted_model
```

### 2. **State Space Models**

#### Kalman Filter Implementation

```python
from statsmodels.tsa.statespace import representation

class CustomKalmanFilter:
    def __init__(self, y, params):
        self.y = y
        self.params = params

    def fit(self):
        # Define state space representation
        self.model = representation representation()

        # Transition equation: x_t = F x_{t-1} + w_t
        self.model.transition = np.array([[1, 1], [0, 1]])

        # Observation equation: y_t = H x_t + v_t
        self.model.observation = np.array([[1, 0]])

        # Process and observation noise covariances
        self.model.state_transition_cov = np.array([[self.params[0], 0], [0, self.params[1]]])
        self.model.observation_cov = np.array([[self.params[2]]])

        # Fit the model
        self.filtered_state = self.model.filter(self.y)

        return self

    def forecast(self, steps=1):
        return self.filtered_state.forecast(steps)
```

### 3. **Gaussian Process Regression for Time Series**

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def gp_time_series_forecasting(X_train, y_train, X_test, length_scale=1.0):
    """
    Gaussian Process Regression for time series
    """
    # Define kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))

    # Create GP model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit model
    gp.fit(X_train, y_train)

    # Make predictions
    y_pred, sigma = gp.predict(X_test, return_std=True)

    return y_pred, sigma, gp
```

### 4. **Anomaly Detection in Time Series**

#### Statistical Methods

```python
def detect_statistical_anomalies(data, method='zscore', threshold=3):
    """
    Detect anomalies using statistical methods
    """
    if method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        anomalies = z_scores > threshold

    elif method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = (data < lower_bound) | (data > upper_bound)

    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1)
        anomalies = model.fit_predict(data.values.reshape(-1, 1)) == -1

    return anomalies

def detect_seasonal_anomalies(data, window=30, threshold=2):
    """
    Detect seasonal anomalies
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    # Normalize by seasonal pattern
    seasonal_component = data / rolling_mean
    seasonal_anomalies = np.abs(seasonal_component - 1) > threshold / 100

    return seasonal_anomalies
```

#### LSTM Autoencoder for Anomaly Detection

```python
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

def create_lstm_autoencoder(sequence_length, n_features):
    """
    LSTM Autoencoder for anomaly detection
    """
    model = Sequential([
        # Encoder
        LSTM(50, activation='relu', input_shape=(sequence_length, n_features),
             return_sequences=True),
        LSTM(25, activation='relu', return_sequences=False),

        # Decoder
        RepeatVector(sequence_length),
        LSTM(25, activation='relu', return_sequences=True),
        LSTM(50, activation='relu', return_sequences=True),

        # Output
        TimeDistributed(Dense(n_features))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def detect_anomalies_lstm(data, sequence_length=50, threshold_percentile=95):
    """
    Detect anomalies using LSTM autoencoder
    """
    # Prepare sequences
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])

    sequences = np.array(sequences)

    # Train autoencoder
    autoencoder = create_lstm_autoencoder(sequence_length, 1)
    autoencoder.fit(sequences, sequences, epochs=50, batch_size=32, verbose=0)

    # Get reconstruction errors
    predictions = autoencoder.predict(sequences)
    reconstruction_errors = np.mean(np.square(sequences - predictions), axis=(1,2))

    # Set threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)

    # Detect anomalies
    anomalies = reconstruction_errors > threshold

    return anomalies, reconstruction_errors
```

### 5. **Regime Change Detection**

#### Change Point Detection with PELT Algorithm

```python
def pelt_change_point_detection(data, model='normal', min_size=2, jump=1):
    """
    PELT (Pruned Exact Linear Time) algorithm for change point detection
    """
    # This is a simplified version - use ruptures library for full implementation
    def calculate_cost(segment):
        if model == 'normal':
            return len(segment) * np.log(np.var(segment)) if len(segment) > 1 else 0

    n = len(data)
    change_points = []

    # Simplified PELT implementation
    candidates = [0]
    optimal_cost = [0]

    for t in range(1, n):
        costs = []
        for cp in candidates:
            if t - cp >= min_size:
                segment_cost = calculate_cost(data[cp:t])
                total_cost = optimal_cost[candidates.index(cp)] + segment_cost
                costs.append(total_cost)
            else:
                costs.append(float('inf'))

        optimal_cost.append(min(costs))

        # Add candidate if cost improves
        if min(costs) < float('inf'):
            candidates.append(t)

    # Backtrack to find change points (simplified)
    # In practice, use proper PELT backtracking

    return change_points
```

---

## Industry Applications

### 1. **Financial Markets**

#### Stock Price Prediction

```python
def stock_price_prediction_pipeline():
    """
    Complete pipeline for stock price prediction
    """
    # 1. Data collection
    # - Historical prices, volumes
    # - Economic indicators
    # - News sentiment
    # - Technical indicators

    # 2. Feature engineering
    features = create_technical_indicators(prices, window=14)
    features = create_lag_features(features, [1, 2, 3, 5, 10, 20])

    # 3. Model ensemble
    models = {
        'LSTM': create_lstm_model(input_shape),
        'ARIMA': ARIMA(prices, order=(1,1,1)).fit(),
        'Prophet': Prophet(daily_seasonality=True),
        'RandomForest': RandomForestRegressor(n_estimators=100)
    }

    # 4. Walk-forward validation
    predictions = walk_forward_validation(prices, n_train=1000, n_test=1)

    # 5. Model selection based on MASE
    best_model = select_best_model(predictions, actuals)

    return best_model

# Risk metrics for financial forecasting
def calculate_risk_metrics(returns, predicted_returns, confidence_level=0.95):
    """
    Calculate risk-adjusted performance metrics
    """
    # Value at Risk (VaR)
    var = np.percentile(predicted_returns, (1 - confidence_level) * 100)

    # Expected Shortfall (ES)
    es = np.mean(predicted_returns[predicted_returns <= var])

    # Sharpe Ratio
    excess_returns = predicted_returns - 0.02 / 252  # Assuming 2% risk-free rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)

    # Maximum Drawdown
    cumulative_returns = (1 + predicted_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'VaR': var,
        'Expected_Shortfall': es,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    }
```

#### Cryptocurrency Forecasting

```python
def crypto_forecasting_model():
    """
    Specialized model for cryptocurrency forecasting
    """
    # Cryptocurrencies have unique characteristics:
    # - High volatility
    # - 24/7 trading
    # - Influenced by social sentiment
    # - Regulatory news impact

    # Additional features for crypto
    features = {
        'social_sentiment': social_media_sentiment,  # Twitter, Reddit sentiment
        'on_chain_metrics': blockchain_metrics,     # Network hash rate, active addresses
        'news_sentiment': news_sentiment,           # Regulatory news impact
        'volatility_regimes': detect_volatility_regime(returns)
    }

    return features
```

### 2. **Demand Forecasting**

#### Retail Demand Forecasting

```python
def retail_demand_forecasting():
    """
    Demand forecasting for retail chains
    """
    # Key challenges:
    # - Multiple stores and products
    # - Seasonal patterns
    # - Promotional effects
    # - External factors (weather, events)

    # Hierarchical forecasting approach
    def hierarchical_forecast(hierarchy_data):
        # Top-down approach
        total_forecast = forecast_total_sales()
        store_forecasts = allocate_to_stores(total_forecast)

        # Bottom-up approach
        individual_forecasts = []
        for store_id in stores:
            store_forecast = forecast_store_sales(store_id)
            individual_forecasts.append(store_forecast)

        # Middle-out approach
        final_forecast = middle_out_reconciliation(
            individual_forecasts, store_forecasts
        )

        return final_forecast

    return hierarchical_forecast

# Inventory optimization
def inventory_optimization_with_forecast(demand_forecast, lead_time,
                                       service_level=0.95):
    """
    Calculate optimal inventory levels using demand forecast
    """
    # Safety stock calculation
    z_score = norm.ppf(service_level)
    safety_stock = z_score * np.std(demand_forecast) * np.sqrt(lead_time)

    # Reorder point
    reorder_point = np.mean(demand_forecast) * lead_time + safety_stock

    # Economic order quantity
    demand_rate = np.mean(demand_forecast)
    ordering_cost = 100  # Fixed ordering cost
    holding_cost = 0.25  # Annual holding cost rate

    eoq = np.sqrt(2 * demand_rate * ordering_cost / holding_cost)

    return {
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'economic_order_quantity': eoq
    }
```

### 3. **Energy and Utilities**

#### Load Forecasting

```python
def electricity_load_forecasting():
    """
    Electricity load forecasting for utilities
    """
    # Key factors:
    # - Temperature and weather
    # - Day of week and holidays
    # - Economic activity
    # - Time of day patterns

    def weather_impact_features(temperature, humidity, wind_speed):
        # Cooling degree days (CDD)
        cdd = np.maximum(temperature - 65, 0)

        # Heating degree days (HDD)
        hdd = np.maximum(65 - temperature, 0)

        # Temperature lag effects
        temp_lag_1 = temperature.shift(1)
        temp_lag_24 = temperature.shift(24)

        # Interaction terms
        temp_humidity = temperature * humidity
        wind_cooling = wind_speed * (65 - temperature)

        return {
            'cdd': cdd,
            'hdd': hdd,
            'temp_lag_1': temp_lag_1,
            'temp_lag_24': temp_lag_24,
            'temp_humidity': temp_humidity,
            'wind_cooling': wind_cooling
        }

    # Multi-horizon forecasting
    def multi_horizon_load_forecast(data, horizons=[1, 24, 168]):  # 1h, 1d, 1w
        forecasts = {}

        for horizon in horizons:
            model = create_load_forecast_model(horizon)
            forecasts[f'h_{horizon}'] = model.predict(horizon)

        return forecasts

    return weather_impact_features, multi_horizon_load_forecast

# Renewable energy forecasting
def renewable_energy_forecasting():
    """
    Forecasting solar and wind energy production
    """
    # Solar forecasting factors
    solar_features = {
        'clear_sky_irradiance': calculate_clear_sky_irradiance(),
        'cloud_cover': cloud_cover_data,
        'solar_elevation_angle': calculate_solar_elevation(),
        'temperature': temperature_forecast,
        'humidity': humidity_forecast
    }

    # Wind forecasting factors
    wind_features = {
        'wind_speed': wind_speed_forecast,
        'wind_direction': wind_direction_forecast,
        'atmospheric_pressure': pressure_forecast,
        'temperature_gradient': temperature_gradient,
        'turbulence_intensity': turbulence_intensity
    }

    # Ensemble approach
    def ensemble_renewable_forecast(solar_data, wind_data):
        models = {
            'MLR': MultipleLinearRegression(),
            'RF': RandomForestRegressor(),
            'SVR': SVR(kernel='rbf'),
            'LSTM': create_lstm_model()
        }

        solar_predictions = {}
        wind_predictions = {}

        for name, model in models.items():
            model.fit(solar_data.features, solar_data.target)
            solar_predictions[name] = model.predict(solar_data.test_features)

            model.fit(wind_data.features, wind_data.target)
            wind_predictions[name] = model.predict(wind_data.test_features)

        # Ensemble using weighted average
        weights = {'MLR': 0.2, 'RF': 0.3, 'SVR': 0.2, 'LSTM': 0.3}

        solar_ensemble = np.average(list(solar_predictions.values()),
                                  weights=list(weights.values()), axis=0)
        wind_ensemble = np.average(list(wind_predictions.values()),
                                 weights=list(weights.values()), axis=0)

        return solar_ensemble, wind_ensemble

    return solar_features, wind_features, ensemble_renewable_forecast
```

### 4. **Supply Chain and Logistics**

#### Transportation Forecasting

```python
def transportation_demand_forecasting():
    """
    Forecasting transportation demand and capacity planning
    """
    # Factors affecting transportation demand:
    # - Economic indicators
    # - Seasonality
    # - Fuel prices
    # - Regulatory changes

    def route_demand_forecast(route_data, external_factors):
        """
        Forecast demand for specific transportation routes
        """
        features = {
            'historical_demand': route_data.demand,
            'fuel_prices': external_factors.fuel_prices,
            'economic_index': external_factors.economic_indicators,
            'seasonality_flags': create_seasonal_flags(route_data.dates),
            'capacity_utilization': route_data.capacity_utilization
        }

        # Multi-output model for different demand types
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100))

        return model

    # Fleet optimization
    def fleet_optimization_forecast(demand_forecast, vehicle_specs):
        """
        Optimize fleet size and composition based on demand forecast
        """
        total_capacity_needed = np.sum(demand_forecast)

        # Vehicle type optimization
        vehicle_types = {
            'small_truck': {'capacity': 10, 'cost': 1000, 'fuel_efficiency': 15},
            'medium_truck': {'capacity': 20, 'cost': 1500, 'fuel_efficiency': 12},
            'large_truck': {'capacity': 40, 'cost': 2500, 'fuel_efficiency': 8}
        }

        optimal_fleet = optimize_fleet_composition(
            total_capacity_needed, vehicle_types
        )

        return optimal_fleet

    return route_demand_forecast, fleet_optimization_forecast
```

### 5. **Healthcare**

#### Epidemiological Forecasting

```python
def epidemiological_forecasting():
    """
    Forecasting disease spread and healthcare demand
    """
    # SIR/SEIR models for epidemic forecasting
    class SEIRModel:
        def __init__(self, beta, sigma, gamma, N):
            self.beta = beta    # Infection rate
            self.sigma = sigma  # Incubation rate
            self.gamma = gamma  # Recovery rate
            self.N = N          # Population size

        def seir_equations(self, t, y):
            S, E, I, R = y
            N = self.N

            dSdt = -self.beta * S * I / N
            dEdt = self.beta * S * I / N - self.sigma * E
            dIdt = self.sigma * E - self.gamma * I
            dRdt = self.gamma * I

            return [dSdt, dEdt, dIdt, dRdt]

        def fit(self, data):
            # Calibrate model parameters using observed data
            from scipy.optimize import minimize

            def objective(params):
                self.beta, self.sigma, self.gamma = params
                simulated = self.simulate(len(data))
                return np.sum((simulated - data)**2)

            # Initial parameter guesses
            initial_guess = [0.5, 0.2, 0.1]
            bounds = [(0.01, 2), (0.01, 1), (0.01, 1)]

            result = minimize(objective, initial_guess, bounds=bounds)
            self.beta, self.sigma, self.gamma = result.x

            return self

        def forecast(self, days=30):
            from scipy.integrate import odeint

            # Initial conditions
            initial_conditions = [9990, 5, 5, 0]  # S, E, I, R
            t = np.linspace(0, days, days)

            solution = odeint(self.seir_equations, initial_conditions, t)

            return solution

    # Healthcare resource planning
    def healthcare_resource_planning(epidemic_forecast, hospital_capacity):
        """
        Plan hospital resources based on epidemic forecast
        """
        infected = epidemic_forecast[:, 2]  # I compartment

        # Hospitalization rates by age group
        hospitalization_rates = {
            '0-9': 0.005, '10-19': 0.003, '20-49': 0.01,
            '50-69': 0.03, '70+': 0.05
        }

        # ICU requirements
        icu_rates = {
            '0-9': 0.0003, '10-19': 0.0002, '20-49': 0.001,
            '50-69': 0.005, '70+': 0.02
        }

        # Resource requirements
        hospital_beds_needed = infected * 0.02  # 2% hospitalization rate
        icu_beds_needed = infected * 0.001      # 0.1% ICU rate

        # Staffing requirements
        nurse_to_patient_ratio = 1/5  # 1 nurse per 5 patients
        doctor_to_patient_ratio = 1/20  # 1 doctor per 20 patients

        nurses_needed = hospital_beds_needed * nurse_to_patient_ratio
        doctors_needed = hospital_beds_needed * doctor_to_patient_ratio

        return {
            'hospital_beds_needed': hospital_beds_needed,
            'icu_beds_needed': icu_beds_needed,
            'nurses_needed': nurses_needed,
            'doctors_needed': doctors_needed,
            'capacity_utilization': hospital_beds_needed / hospital_capacity
        }

    return SEIRModel, healthcare_resource_planning
```

### 6. **Manufacturing and Industry 4.0**

#### Predictive Maintenance

```python
def predictive_maintenance_forecasting():
    """
    Time series forecasting for predictive maintenance
    """
    def remaining_useful_life_prediction(sensor_data, failure_history):
        """
        Predict remaining useful life (RUL) of equipment
        """
        # Feature engineering for RUL prediction
        features = {
            'vibration_trend': calculate_vibration_trend(sensor_data),
            'temperature_trend': calculate_temperature_trend(sensor_data),
            'pressure_trend': calculate_pressure_trend(sensor_data),
            'operational_hours': sensor_data.operational_hours,
            'maintenance_intervals': sensor_data.maintenance_history,
            'failure_patterns': failure_history.patterns
        }

        # RUL models
        models = {
            'survival_analysis': CoxPHWrapper(),  # Survival analysis
            'lstm_direct': create_lstm_model(),   # Direct RUL prediction
            'lstm_classification': create_classification_model(),  # Classification approach
            'hybrid': create_hybrid_rul_model()   # Hybrid approach
        }

        return models

    def failure_probability_forecast(sensor_readings, equipment_id, time_horizon):
        """
        Forecast probability of equipment failure
        """
        # Online learning for changing failure patterns
        from sklearn.linear_model import SGDRegressor

        def update_failure_model(new_data):
            model = SGDRegressor(learning_rate='adaptive')
            model.partial_fit(new_data.features, new_data.labels)
            return model

        # Failure modes analysis
        failure_modes = {
            'bearing_wear': {'indicators': ['vibration', 'temperature'],
                           'threshold': 0.8},
            'seal_degradation': {'indicators': ['pressure', 'leakage'],
                               'threshold': 0.7},
            'electrical_fault': {'indicators': ['current', 'voltage'],
                               'threshold': 0.9}
        }

        probabilities = {}
        for mode, specs in failure_modes.items():
            # Calculate failure probability for each mode
            prob = calculate_failure_probability(
                sensor_readings, specs['indicators'], specs['threshold']
            )
            probabilities[mode] = prob

        return probabilities

    return remaining_useful_life_prediction, failure_probability_forecast
```

This comprehensive theory module covers all the essential aspects of time series forecasting, from traditional statistical methods to modern deep learning approaches. The content is based on the scattered time series information found throughout the curriculum and organized into a structured learning path.

---

_This module provides the theoretical foundation for time series forecasting. Move on to the practice exercises to apply these concepts hands-on._
