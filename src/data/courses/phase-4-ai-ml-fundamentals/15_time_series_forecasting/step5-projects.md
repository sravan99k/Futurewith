# Time Series Forecasting Projects

## Table of Contents

1. [Beginner Projects](#beginner-projects)
2. [Intermediate Projects](#intermediate-projects)
3. [Advanced Projects](#advanced-projects)
4. [Industry-Specific Projects](#industry-specific-projects)
5. [Research & Innovation Projects](#research--innovation-projects)
6. [Portfolio Project Guidelines](#portfolio-project-guidelines)

---

## Beginner Projects

### Project 1: Sales Forecasting for Retail Chain

**Objective:** Build a sales forecasting system for a retail chain with multiple stores and products.

**Dataset:** Generate synthetic retail data with the following characteristics:

- 50 stores across different regions
- 100 products with varying seasonality patterns
- 3 years of daily sales data
- External factors: weather, holidays, promotions

**Requirements:**

- Analyze sales patterns by store and product
- Implement multiple forecasting models (ARIMA, Prophet, LSTM)
- Create hierarchical forecasting (store → region → total)
- Generate confidence intervals
- Build interactive dashboard

**Deliverables:**

- [ ] Data analysis notebook
- [ ] Model comparison report
- [ ] Forecasting dashboard
- [ ] Documentation and recommendations

**Implementation Framework:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_retail_data(n_stores=50, n_products=100, n_days=1095):
    """
    Generate synthetic retail sales data
    """
    np.random.seed(42)

    # Create date range
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

    # Store information
    store_types = ['urban', 'suburban', 'rural']
    store_regions = ['North', 'South', 'East', 'West']

    stores_info = []
    for store_id in range(1, n_stores + 1):
        store_type = np.random.choice(store_types)
        region = np.random.choice(store_regions)
        stores_info.append({
            'store_id': store_id,
            'store_type': store_type,
            'region': region,
            'size': np.random.choice(['small', 'medium', 'large'])
        })

    stores_df = pd.DataFrame(stores_info)

    # Product information
    product_categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
    products_info = []
    for product_id in range(1, n_products + 1):
        category = np.random.choice(product_categories)
        seasonality = np.random.choice(['high', 'medium', 'low'])
        products_info.append({
            'product_id': product_id,
            'category': category,
            'seasonality': seasonality,
            'base_price': np.random.uniform(10, 500)
        })

    products_df = pd.DataFrame(products_info)

    # Generate sales data
    sales_data = []

    for date in dates:
        day_of_year = date.dayofyear
        day_of_week = date.dayofweek
        month = date.month

        # Holiday and promotion flags
        is_holiday = date.month == 12 and date.day >= 20  # Christmas season
        is_promotion = np.random.random() < 0.15  # 15% chance of promotion

        for _, store in stores_df.iterrows():
            for _, product in products_df.iterrows():

                # Base sales with trend
                base_sales = np.random.normal(50, 20)

                # Store type effect
                store_multiplier = {
                    'urban': 1.2, 'suburban': 1.0, 'rural': 0.8
                }[store['store_type']]

                # Seasonality effect
                if product['seasonality'] == 'high':
                    seasonal_effect = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365.25)
                elif product['seasonality'] == 'medium':
                    seasonal_effect = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)
                else:
                    seasonal_effect = 1

                # Day of week effect
                dow_effects = {
                    0: 0.9, 1: 0.8, 2: 0.85, 3: 0.9, 4: 1.1, 5: 1.3, 6: 1.4  # Weekend boost
                }
                dow_effect = dow_effects[day_of_week]

                # Holiday effect
                holiday_effect = 1.8 if is_holiday else 1.0

                # Promotion effect
                promotion_effect = 2.5 if is_promotion else 1.0

                # Calculate final sales
                daily_sales = (base_sales * store_multiplier * seasonal_effect *
                              dow_effect * holiday_effect * promotion_effect)

                # Add noise and ensure non-negative
                daily_sales = max(0, np.random.normal(daily_sales, daily_sales * 0.1))

                sales_data.append({
                    'date': date,
                    'store_id': store['store_id'],
                    'product_id': product['product_id'],
                    'sales': daily_sales,
                    'store_type': store['store_type'],
                    'region': store['region'],
                    'category': product['category'],
                    'seasonality': product['seasonality'],
                    'is_holiday': is_holiday,
                    'is_promotion': is_promotion,
                    'day_of_week': day_of_week,
                    'month': month
                })

    return pd.DataFrame(sales_data), stores_df, products_df

# Project structure
class RetailForecastingProject:
    def __init__(self):
        self.sales_data = None
        self.stores_df = None
        self.products_df = None
        self.models = {}
        self.results = {}

    def generate_data(self):
        """Generate synthetic retail data"""
        print("Generating synthetic retail data...")
        self.sales_data, self.stores_df, self.products_df = generate_retail_data()

        print(f"Generated {len(self.sales_data)} sales records")
        print(f"Date range: {self.sales_data['date'].min()} to {self.sales_data['date'].max()}")

    def exploratory_analysis(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)

        # Basic statistics
        print("Sales Data Overview:")
        print(self.sales_data.describe())

        # Sales by store type
        sales_by_store = self.sales_data.groupby('date')['sales'].sum().reset_index()
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        daily_sales = self.sales_data.groupby('date')['sales'].sum()
        daily_sales.plot()
        plt.title('Total Daily Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        store_sales = self.sales_data.groupby('store_type')['sales'].mean()
        store_sales.plot(kind='bar')
        plt.title('Average Sales by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)

        plt.subplot(2, 3, 3)
        category_sales = self.sales_data.groupby('category')['sales'].mean()
        category_sales.plot(kind='bar')
        plt.title('Average Sales by Category')
        plt.xlabel('Category')
        plt.ylabel('Average Sales')
        plt.xticks(rotation=45)

        plt.subplot(2, 3, 4)
        seasonality_sales = self.sales_data.groupby('seasonality')['sales'].mean()
        seasonality_sales.plot(kind='bar')
        plt.title('Average Sales by Seasonality')
        plt.xlabel('Seasonality Level')
        plt.ylabel('Average Sales')

        plt.subplot(2, 3, 5)
        dow_sales = self.sales_data.groupby('day_of_week')['sales'].mean()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        plt.bar(range(7), dow_sales.values)
        plt.title('Average Sales by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Sales')
        plt.xticks(range(7), dow_labels)

        plt.subplot(2, 3, 6)
        holiday_sales = self.sales_data.groupby('is_holiday')['sales'].mean()
        plt.bar(['Regular', 'Holiday'], holiday_sales.values)
        plt.title('Average Sales: Holiday vs Regular')
        plt.xlabel('Day Type')
        plt.ylabel('Average Sales')

        plt.tight_layout()
        plt.show()

        # Correlation analysis
        plt.figure(figsize=(10, 6))
        correlation_matrix = self.sales_data[['sales', 'day_of_week', 'month', 'is_holiday', 'is_promotion']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()

    def prepare_data_for_modeling(self, store_id=None, product_id=None, aggregate_level='daily'):
        """
        Prepare data for forecasting based on specified level
        """
        if store_id and product_id:
            # Specific store-product combination
            data = self.sales_data[
                (self.sales_data['store_id'] == store_id) &
                (self.sales_data['product_id'] == product_id)
            ].copy()
            title = f"Store {store_id}, Product {product_id}"

        elif store_id:
            # Specific store aggregated across products
            data = self.sales_data[self.sales_data['store_id'] == store_id].copy()
            title = f"Store {store_id}"

        elif product_id:
            # Specific product aggregated across stores
            data = self.sales_data[self.sales_data['product_id'] == product_id].copy()
            title = f"Product {product_id}"

        else:
            # Overall daily sales
            data = self.sales_data.groupby('date').agg({
                'sales': 'sum',
                'is_holiday': 'first',
                'is_promotion': 'first'
            }).reset_index()
            title = "Total Sales"

        # Aggregate by day if needed
        if aggregate_level == 'daily':
            data = data.groupby('date').agg({
                'sales': 'sum',
                'is_holiday': 'first',
                'is_promotion': 'first'
            }).reset_index()

        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)

        # Create time series
        ts_data = pd.Series(data['sales'].values, index=data['date'], name='sales')

        return ts_data, title

    def implement_prophet_model(self, ts_data):
        """
        Implement Facebook Prophet model
        """
        from prophet import Prophet

        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': ts_data.index,
            'y': ts_data.values
        })

        # Add custom seasonalities and holidays
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Fit model
        model.fit(df)

        # Make predictions
        future = model.make_future_dataframe(periods=30)  # Predict next 30 days
        forecast = model.predict(future)

        # Plot components
        fig = model.plot_components(forecast)
        plt.suptitle('Prophet Model Components')
        plt.show()

        # Plot forecast
        fig = model.plot(forecast)
        plt.title('Prophet Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()

        return model, forecast

    def implement_arima_model(self, ts_data):
        """
        Implement ARIMA model
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        # Test for stationarity
        def check_stationarity(series):
            result = adfuller(series.dropna())
            return result[1] <= 0.05

        is_stationary = check_stationarity(ts_data)
        print(f"Series is {'stationary' if is_stationary else 'non-stationary'}")

        # Auto ARIMA (simplified grid search)
        best_aic = float('inf')
        best_order = None

        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts_data, order=(p, d, q))
                        fitted_model = model.fit()

                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue

        print(f"Best ARIMA order: {best_order}")

        # Fit final model
        final_model = ARIMA(ts_data, order=best_order)
        fitted_final = final_model.fit()

        # Forecast
        forecast = fitted_final.forecast(steps=30)
        forecast_ci = fitted_final.get_forecast(steps=30).conf_int()

        # Plot results
        plt.figure(figsize=(15, 8))

        # Plot historical data
        ts_data.plot(label='Historical Sales', color='blue')

        # Plot forecast
        forecast_index = pd.date_range(ts_data.index[-1] + timedelta(days=1), periods=30, freq='D')
        plt.plot(forecast_index, forecast, label='ARIMA Forecast', color='red', linewidth=2)

        # Plot confidence intervals
        plt.fill_between(forecast_index,
                        forecast_ci.iloc[:, 0],
                        forecast_ci.iloc[:, 1],
                        alpha=0.3, color='red', label='95% Confidence Interval')

        plt.title('ARIMA Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return fitted_final, forecast

    def implement_lstm_model(self, ts_data):
        """
        Implement LSTM neural network
        """
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        # Prepare data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))

        # Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i-seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        seq_length = 30
        X, y = create_sequences(scaled_data, seq_length)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        # Train model
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform
        y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot results
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        n_plot = len(y_test_original)
        indices = np.arange(n_plot)

        plt.plot(indices, y_test_original, label='Actual', linewidth=2, color='black')
        plt.plot(indices, y_pred_original, label='Predicted', alpha=0.7, color='red')
        plt.title('LSTM Predictions vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return model, scaler, y_pred_original

    def compare_models(self, store_id=1, product_id=1):
        """
        Compare all models on a specific store-product combination
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)

        # Prepare data
        ts_data, title = self.prepare_data_for_modeling(store_id=store_id, product_id=product_id)

        print(f"Forecasting for: {title}")
        print(f"Data shape: {ts_data.shape}")

        # Split for evaluation
        train_size = int(len(ts_data) * 0.8)
        train_data = ts_data[:train_size]
        test_data = ts_data[train_size:]

        results = {}

        # 1. Simple Moving Average (baseline)
        ma_forecast = train_data.tail(7).mean()
        ma_error = np.mean(np.abs(test_data.values - ma_forecast))
        results['Moving Average'] = {'MAE': ma_error}

        # 2. Prophet
        prophet_model, prophet_forecast = self.implement_prophet_model(train_data)
        prophet_pred = prophet_forecast.tail(len(test_data))['yhat'].values
        prophet_error = np.mean(np.abs(test_data.values - prophet_pred))
        results['Prophet'] = {'MAE': prophet_error}

        # 3. ARIMA
        arima_model, arima_pred = self.implement_arima_model(train_data)
        arima_pred_actual = arima_pred.values[:len(test_data)]
        arima_error = np.mean(np.abs(test_data.values - arima_pred_actual))
        results['ARIMA'] = {'MAE': arima_error}

        # 4. LSTM
        lstm_model, scaler, lstm_pred = self.implement_lstm_model(train_data)
        lstm_error = np.mean(np.abs(test_data.values - lstm_pred))
        results['LSTM'] = {'MAE': lstm_error}

        # Display results
        print("\nModel Comparison Results:")
        print("-" * 30)
        for model_name, metrics in results.items():
            print(f"{model_name:15}: MAE = {metrics['MAE']:.2f}")

        # Select best model
        best_model = min(results.keys(), key=lambda x: results[x]['MAE'])
        print(f"\nBest performing model: {best_model}")

        return results, best_model

    def hierarchical_forecasting(self):
        """
        Implement hierarchical forecasting approach
        """
        print("\n" + "="*50)
        print("HIERARCHICAL FORECASTING")
        print("="*50)

        # Aggregate sales by different levels
        # Level 1: Total sales
        total_sales = self.sales_data.groupby('date')['sales'].sum()

        # Level 2: By region
        region_sales = self.sales_data.groupby(['date', 'region'])['sales'].sum().unstack('region')

        # Level 3: By store type
        store_type_sales = self.sales_data.groupby(['date', 'store_type'])['sales'].sum().unstack('store_type')

        # Generate forecasts for each level
        forecasts = {}

        # Total forecast
        ts_data, _ = self.prepare_data_for_modeling()
        arima_model, arima_forecast = self.implement_arima_model(ts_data)
        forecasts['total'] = arima_forecast

        # Region forecasts
        region_forecasts = {}
        for region in region_sales.columns:
            region_ts = region_sales[region].dropna()
            if len(region_ts) > 50:  # Only if sufficient data
                arima_model, region_forecast = self.implement_arima_model(region_ts)
                region_forecasts[region] = region_forecast
        forecasts['regions'] = region_forecasts

        # Plot hierarchy
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        total_sales.plot(title='Total Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        for region in region_sales.columns:
            region_sales[region].plot(label=region, alpha=0.7)
        plt.title('Sales by Region')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        for store_type in store_type_sales.columns:
            store_type_sales[store_type].plot(label=store_type, alpha=0.7)
        plt.title('Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Forecast reconciliation visualization
        plt.subplot(2, 2, 4)

        # Sum of region forecasts vs total forecast
        if region_forecasts:
            reconciled_sum = sum(region_forecasts.values())
            reconciled_sum.plot(label='Sum of Region Forecasts', alpha=0.7)

            forecasts['total'].plot(label='Total Forecast', alpha=0.7)
            plt.title('Forecast Reconciliation')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return forecasts

    def create_dashboard_data(self):
        """
        Prepare data for dashboard visualization
        """
        dashboard_data = {
            'sales_over_time': self.sales_data.groupby('date')['sales'].sum().to_dict(),
            'store_performance': self.sales_data.groupby('store_id')['sales'].sum().to_dict(),
            'product_performance': self.sales_data.groupby('product_id')['sales'].sum().to_dict(),
            'category_performance': self.sales_data.groupby('category')['sales'].sum().to_dict(),
            'seasonality_patterns': self.sales_data.groupby('day_of_week')['sales'].mean().to_dict(),
            'promotion_impact': self.sales_data.groupby('is_promotion')['sales'].mean().to_dict()
        }

        return dashboard_data

# Main execution
def run_retail_forecasting_project():
    """
    Run the complete retail forecasting project
    """
    print("RETAIL SALES FORECASTING PROJECT")
    print("="*50)

    # Initialize project
    project = RetailForecastingProject()

    # Step 1: Generate data
    project.generate_data()

    # Step 2: Exploratory analysis
    project.exploratory_analysis()

    # Step 3: Model comparison
    results, best_model = project.compare_models(store_id=1, product_id=1)

    # Step 4: Hierarchical forecasting
    hierarchical_forecasts = project.hierarchical_forecasting()

    # Step 5: Prepare dashboard data
    dashboard_data = project.create_dashboard_data()

    # Step 6: Generate summary report
    print("\n" + "="*50)
    print("PROJECT SUMMARY")
    print("="*50)

    print(f"1. Data Generation: ✓")
    print(f"   - {len(project.sales_data):,} sales records")
    print(f"   - {len(project.stores_df)} stores across {len(project.stores_df['region'].unique())} regions")
    print(f"   - {len(project.products_df)} products in {len(project.products_df['category'].unique())} categories")

    print(f"\n2. Model Performance:")
    for model, metrics in results.items():
        print(f"   - {model}: MAE = {metrics['MAE']:.2f}")

    print(f"\n3. Best Model: {best_model}")

    print(f"\n4. Hierarchical Forecasting: ✓")
    print(f"   - Total sales forecast generated")
    print(f"   - {len(hierarchical_forecasts.get('regions', {}))} regional forecasts")

    print(f"\n5. Next Steps:")
    print(f"   - Deploy best performing model")
    print(f"   - Set up automated retraining")
    print(f"   - Implement monitoring dashboard")
    print(f"   - Add real-time data integration")

    return project, results, dashboard_data

# Run the project
if __name__ == "__main__":
    project, results, dashboard_data = run_retail_forecasting_project()
```

### Project 2: Stock Price Prediction System

**Objective:** Create a comprehensive stock price prediction system with multiple models and risk analysis.

**Features to Include:**

- Technical indicator calculations
- Multiple timeframe analysis
- Risk metrics calculation
- Portfolio optimization
- Real-time prediction pipeline
- Backtesting framework

**Technical Stack:**

- Data: Yahoo Finance, Alpha Vantage APIs
- ML: scikit-learn, TensorFlow, Prophet
- Analysis: TA-Lib, pandas-datareader
- Visualization: Plotly, Bokeh
- Deployment: Flask/FastAPI, Docker

---

## Intermediate Projects

### Project 3: Energy Demand Forecasting System

**Objective:** Build a multi-scale energy demand forecasting system for utilities.

**Key Components:**

- Short-term (hourly) and long-term (monthly) forecasting
- Weather impact modeling
- Renewable energy integration
- Grid stability analysis
- Load balancing optimization

**Dataset Requirements:**

- Historical electricity demand
- Weather data (temperature, humidity, wind speed)
- Economic indicators
- Holiday and special event data
- Renewable energy generation

**Implementation Highlights:**

```python
class EnergyDemandForecasting:
    def __init__(self):
        self.models = {}
        self.weather_impact_model = None
        self.renewable_integration = None

    def weather_impact_analysis(self, demand_data, weather_data):
        """Analyze weather impact on demand"""
        # Cooling/Heating degree days
        cdd = np.maximum(weather_data['temperature'] - 65, 0)
        hdd = np.maximum(65 - weather_data['temperature'], 0)

        # Lagged weather effects
        weather_lags = weather_data.shift([1, 24, 168])  # 1hr, 1day, 1week

        return self._calculate_weather_coefficients(demand_data, weather_lags)

    def multi_scale_forecasting(self, data, horizons=[1, 24, 168, 720]):
        """Forecast at multiple time scales"""
        forecasts = {}

        for horizon in horizons:
            if horizon == 1:  # Hourly
                model = self._create_hourly_model()
            elif horizon == 24:  # Daily
                model = self._create_daily_model()
            elif horizon == 168:  # Weekly
                model = self._create_weekly_model()
            else:  # Monthly
                model = self._create_monthly_model()

            forecasts[f'h_{horizon}'] = model.predict(horizon)

        return forecasts

    def renewable_integration(self, demand_forecast, renewable_forecast):
        """Analyze renewable energy impact"""
        net_demand = demand_forecast - renewable_forecast

        # Calculate storage requirements
        storage_needed = np.maximum(0, -net_demand)
        excess_energy = np.maximum(0, net_demand)

        return {
            'net_demand': net_demand,
            'storage_requirements': storage_needed,
            'excess_energy': excess_energy
        }
```

### Project 4: Supply Chain Optimization with Forecasting

**Objective:** Optimize supply chain operations using demand forecasting.

**Components:**

- Multi-echelon inventory optimization
- Supplier reliability analysis
- Transportation optimization
- Risk management
- Sustainability metrics

**Advanced Features:**

- Dynamic safety stock calculation
- Multi-objective optimization
- Scenario planning
- Real-time adaptation
- Blockchain integration for transparency

---

## Advanced Projects

### Project 5: AI-Powered Financial Trading System

**Objective:** Create an intelligent trading system with advanced forecasting and risk management.

**Key Features:**

- Multi-asset forecasting (stocks, forex, crypto)
- Sentiment analysis integration
- High-frequency trading algorithms
- Portfolio optimization with risk constraints
- Real-time market microstructure analysis
- Automated strategy generation

**Technical Architecture:**

```python
class AITradingSystem:
    def __init__(self):
        self.market_data_fetcher = MarketDataFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.forecasting_engine = ForecastingEngine()
        self.risk_manager = RiskManager()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.execution_engine = ExecutionEngine()

    def multi_asset_forecast(self, symbols, horizon='1d'):
        """Forecast multiple assets simultaneously"""
        forecasts = {}

        # Get market data
        market_data = self.market_data_fetcher.get_data(symbols, horizon)

        # Add sentiment data
        sentiment_data = self.sentiment_analyzer.get_sentiment(symbols)

        # Technical indicators
        technical_indicators = self._calculate_technical_indicators(market_data)

        # Combine features
        features = self._combine_features(market_data, sentiment_data, technical_indicators)

        # Multi-model ensemble
        forecasts = self.forecasting_engine.ensemble_predict(features)

        return forecasts

    def dynamic_risk_management(self, portfolio, market_conditions):
        """Real-time risk adjustment"""
        # Calculate VaR and CVaR
        var = self._calculate_var(portfolio, market_conditions)
        cvar = self._calculate_cvar(portfolio, market_conditions)

        # Stress testing
        stress_scenarios = self._generate_stress_scenarios()
        stress_results = self._apply_stress_test(portfolio, stress_scenarios)

        # Risk adjustments
        risk_adjustments = self._calculate_risk_adjustments(var, cvar, stress_results)

        return risk_adjustments

    def adaptive_strategy_generation(self, market_regime):
        """Generate trading strategies based on market regime"""
        strategies = {
            'momentum': self._momentum_strategy(market_regime),
            'mean_reversion': self._mean_reversion_strategy(market_regime),
            'arbitrage': self._arbitrage_strategy(market_regime),
            'high_frequency': self._hf_strategy(market_regime)
        }

        # Select optimal strategy based on regime
        optimal_strategy = self._select_optimal_strategy(strategies, market_regime)

        return optimal_strategy
```

### Project 6: Healthcare Epidemic Forecasting Platform

**Objective:** Build a comprehensive epidemic forecasting and response system.

**Capabilities:**

- Disease spread modeling (SIR, SEIR, agent-based)
- Hospital resource planning
- Public health intervention optimization
- Real-time surveillance
- Vaccine distribution optimization

**Advanced Components:**

```python
class EpidemicForecastingPlatform:
    def __init__(self):
        self.seir_model = SEIRModel()
        self.agent_based_model = AgentBasedModel()
        self.hospital_planner = HospitalResourcePlanner()
        self.intervention_optimizer = InterventionOptimizer()

    def epidemic_forecasting(self, initial_conditions, parameters, horizon_days):
        """Multi-model epidemic forecasting"""
        # SEIR model
        seir_forecast = self.seir_model.predict(initial_conditions, parameters, horizon_days)

        # Agent-based simulation (for complex interactions)
        agent_forecast = self.agent_based_model.simulate(initial_conditions, horizon_days)

        # Bayesian model averaging
        ensemble_forecast = self._bayesian_model_average(seir_forecast, agent_forecast)

        return ensemble_forecast

    def hospital_resource_planning(self, infection_forecast, demographics):
        """Plan hospital resources based on infection forecast"""
        # Age-stratified hospitalization rates
        hospitalization_rates = self._get_hospitalization_rates(demographics)

        # ICU requirements by age group
        icu_requirements = self._calculate_icu_needs(infection_forecast, hospitalization_rates)

        # Staffing requirements
        staffing_plan = self._calculate_staffing_needs(icu_requirements)

        # PPE and equipment needs
        equipment_needs = self._calculate_equipment_needs(infection_forecast)

        return {
            'hospital_beds': icu_requirements['beds'],
            'icu_beds': icu_requirements['icu'],
            'staffing': staffing_plan,
            'equipment': equipment_needs,
            'timeline': self._create_resource_timeline(infection_forecast)
        }

    def intervention_optimization(self, epidemic_forecast, intervention_options, constraints):
        """Optimize public health interventions"""
        # Define objective function (minimize infections + economic cost)
        def objective(intervention_levels):
            infections = self._simulate_interventions(epidemic_forecast, intervention_levels)
            economic_cost = self._calculate_economic_cost(intervention_levels)
            return infections + economic_cost

        # Optimization with constraints
        optimized_interventions = self._optimize_interventions(
            objective, intervention_options, constraints
        )

        return optimized_interventions
```

### Project 7: Smart City Traffic Optimization System

**Objective:** Create an intelligent traffic management system with predictive capabilities.

**Features:**

- Real-time traffic flow prediction
- Dynamic signal timing optimization
- Incident detection and response
- Public transportation optimization
- Environmental impact assessment

**Implementation Framework:**

```python
class SmartTrafficSystem:
    def __init__(self):
        self.traffic_predictor = TrafficFlowPredictor()
        self.signal_optimizer = DynamicSignalOptimizer()
        self.incident_detector = IncidentDetector()
        self.route_optimizer = RouteOptimizer()

    def real_time_traffic_prediction(self, sensor_data, historical_patterns):
        """Predict traffic flow for next time period"""
        # Feature engineering
        features = self._extract_traffic_features(sensor_data, historical_patterns)

        # Multi-scale prediction (5min, 15min, 1hr horizons)
        predictions = {}
        for horizon in [5, 15, 60]:
            model = self._get_prediction_model(horizon)
            predictions[f'{horizon}min'] = model.predict(features)

        return predictions

    def dynamic_signal_optimization(self, traffic_predictions, intersection_config):
        """Optimize traffic signals based on predictions"""
        # Current signal timings
        current_timings = intersection_config['signal_timings']

        # Optimization objective
        def objective(signal_timings):
            # Simulate traffic flow with new timings
            delays = self._simulate_traffic_delays(signal_timings, traffic_predictions)
            # Minimize total delay + fuel consumption + emissions
            cost = (delays['total_delay'] +
                   delays['fuel_consumption'] * 0.1 +
                   delays['emissions'] * 0.05)
            return cost

        # Constraints: signal timing limits, pedestrian crossing times
        constraints = self._get_signal_constraints(intersection_config)

        # Optimize using genetic algorithm or similar
        optimal_timings = self._optimize_signals(objective, constraints)

        return optimal_timings

    def incident_response_optimization(self, incident_detection):
        """Optimize response to traffic incidents"""
        # Incident severity assessment
        severity = self._assess_incident_severity(incident_detection)

        # Resource allocation
        if severity == 'high':
            resources = {'ambulance': 2, 'police': 3, 'tow_truck': 2}
        elif severity == 'medium':
            resources = {'ambulance': 1, 'police': 2, 'tow_truck': 1}
        else:
            resources = {'police': 1, 'tow_truck': 1}

        # Route optimization for emergency vehicles
        emergency_routes = {}
        for resource_type, count in resources.items():
            for i in range(count):
                route_key = f'{resource_type}_{i}'
                emergency_routes[route_key] = self.route_optimizer.get_emergency_route(
                    incident_detection['location']
                )

        return {
            'severity': severity,
            'resources': resources,
            'emergency_routes': emergency_routes,
            'traffic_diversion': self._calculate_traffic_diversion(incident_detection)
        }
```

---

## Industry-Specific Projects

### Project 8: E-commerce Demand Forecasting

**Objective:** Build a comprehensive demand forecasting system for e-commerce platforms.

**Key Components:**

- Multi-category product forecasting
- Promotion impact modeling
- Seasonality and trend analysis
- Inventory optimization
- Customer behavior prediction
- Supply chain coordination

**Advanced Features:**

- Hierarchical forecasting (SKU → Category → Brand → Total)
- Promotion lift modeling
- Cross-selling and bundling analysis
- Return rate forecasting
- Marketing campaign impact assessment

### Project 9: Manufacturing Predictive Maintenance

**Objective:** Create an intelligent predictive maintenance system for manufacturing equipment.

**Features:**

- Equipment health monitoring
- Failure prediction with remaining useful life estimation
- Maintenance scheduling optimization
- Spare parts inventory management
- Production impact assessment
- Cost-benefit analysis

**Technical Implementation:**

```python
class PredictiveMaintenanceSystem:
    def __init__(self):
        self.equipment_models = {}
        self.failure_predictor = FailurePredictor()
        self.maintenance_optimizer = MaintenanceOptimizer()

    def equipment_health_monitoring(self, sensor_data, equipment_id):
        """Monitor equipment health using sensor data"""
        # Feature extraction from sensor data
        health_features = self._extract_health_features(sensor_data)

        # Calculate health score
        health_score = self._calculate_health_score(health_features)

        # Detect anomalies
        anomalies = self._detect_anomalies(sensor_data)

        return {
            'health_score': health_score,
            'anomalies': anomalies,
            'trend': self._analyze_health_trend(health_features),
            'next_maintenance': self._predict_next_maintenance(health_features)
        }

    def failure_prediction(self, equipment_id, sensor_data):
        """Predict equipment failure"""
        # Load pre-trained model for equipment
        model = self.equipment_models.get(equipment_id)
        if model is None:
            model = self._train_equipment_model(equipment_id)

        # Feature engineering
        features = self._engineer_failure_features(sensor_data)

        # Predict failure probability
        failure_probability = model.predict_proba(features)[:, 1]

        # Estimate remaining useful life
        rul_estimate = self._estimate_rul(features)

        return {
            'failure_probability': failure_probability,
            'remaining_useful_life': rul_estimate,
            'confidence_interval': self._calculate_confidence_interval(rul_estimate),
            'recommended_action': self._recommend_action(failure_probability, rul_estimate)
        }
```

### Project 10: Cryptocurrency Trading Bot

**Objective:** Develop an autonomous cryptocurrency trading system with advanced analytics.

**Features:**

- Multi-exchange arbitrage detection
- Risk-adjusted portfolio optimization
- Sentiment analysis integration
- Market regime detection
- Automated strategy adaptation
- Real-time execution with slippage protection

---

## Research & Innovation Projects

### Project 11: Quantum-Enhanced Time Series Forecasting

**Objective:** Explore quantum computing applications in time series forecasting.

**Research Areas:**

- Quantum machine learning algorithms
- Quantum advantage in temporal pattern recognition
- Hybrid quantum-classical approaches
- Quantum-inspired optimization algorithms

**Implementation Framework:**

```python
class QuantumTimeSeriesForecasting:
    def __init__(self):
        self.quantum_simulator = QuantumSimulator()
        self.classical_forecaster = ClassicalForecaster()

    def hybrid_quantum_classical_forecast(self, time_series, quantum_advantage_horizon):
        """Combine quantum and classical approaches"""
        # Short-term: classical forecasting (proven efficient)
        classical_forecast = self.classical_forecaster.predict(time_series,
                                                            horizon=quantum_advantage_horizon)

        # Long-term: quantum-enhanced forecasting
        quantum_features = self._extract_quantum_features(time_series)
        quantum_forecast = self.quantum_simulator.predict(quantum_features)

        # Combine using quantum weighting
        combined_forecast = self._quantum_weighted_combination(
            classical_forecast, quantum_forecast
        )

        return combined_forecast

    def quantum_feature_mapping(self, time_series):
        """Map classical time series to quantum feature space"""
        # Amplitude encoding of temporal patterns
        quantum_state = self._amplitude_encode(time_series)

        # Quantum feature maps (using quantum kernels)
        quantum_features = self._quantum_kernel_features(quantum_state)

        return quantum_features
```

### Project 12: Neuromorphic Computing for Time Series

**Objective:** Investigate neuromorphic computing architectures for efficient time series processing.

**Key Components:**

- Spiking neural networks
- Temporal pattern learning
- Energy-efficient inference
- Adaptive learning rates
- Event-driven processing

---

## Portfolio Project Guidelines

### Project Structure Template

```python
"""
Time Series Forecasting Portfolio Project
========================================

Project Name: [Your Project Name]
Author: [Your Name]
Date: [Current Date]
Objective: [Clear project objective]

Project Structure:
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and processed data
│   └── external/               # External data sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_evaluation_and_analysis.ipynb
│   └── 05_dashboard_and_visualization.ipynb
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── traditional_models.py
│   │   ├── ml_models.py
│   │   ├── deep_learning.py
│   │   └── ensemble.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── backtesting.py
│   │   └── visualization.py
│   └── utils/
│       ├── helpers.py
│       └── config.py
├── tests/
│   ├── test_data_processing.py
│   ├── test_models.py
│   └── test_evaluation.py
├── docs/
│   ├── project_report.md
│   ├── methodology.md
│   └── results_summary.md
├── dashboard/
│   ├── app.py                 # Streamlit/Dash app
│   ├── components/
│   └── assets/
├── deployment/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── api/
├── README.md
└── project_overview.md
"""

# Required files for each project
REQUIRED_FILES = [
    "README.md",                    # Project overview and setup instructions
    "project_overview.md",          # Detailed project description
    "requirements.txt",             # Python dependencies
    "data/README.md",              # Data sources and descriptions
    "notebooks/01_data_exploration.ipynb",  # Initial EDA
    "notebooks/02_feature_engineering.ipynb", # Feature engineering
    "notebooks/03_model_development.ipynb",   # Model implementation
    "notebooks/04_evaluation_and_analysis.ipynb", # Evaluation
    "src/data/preprocessing.py",    # Data preprocessing functions
    "src/models/forecasting_models.py", # Forecasting models
    "src/evaluation/metrics.py",    # Evaluation metrics
    "tests/test_models.py",        # Model tests
    "docs/project_report.md",      # Final project report
    "dashboard/app.py"             # Interactive dashboard
]
```

### Documentation Requirements

#### README.md Template

````markdown
# Time Series Forecasting Project

## Overview

Brief description of the project, its objectives, and key findings.

## Features

- [ ] List of implemented features
- [ ] Models used
- [ ] Key innovations

## Getting Started

### Prerequisites

- Python 3.8+
- List of required packages

### Installation

```bash
pip install -r requirements.txt
```
````

### Usage

```python
# Example usage code
from src.models.forecasting import ForecastingPipeline

pipeline = ForecastingPipeline()
results = pipeline.fit_predict(data)
```

## Project Structure

```
├── data/                    # Datasets
├── src/                     # Source code
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── dashboard/              # Interactive dashboard
```

## Results Summary

- Model performance metrics
- Key insights
- Business impact

## Future Improvements

- Planned enhancements
- Scalability considerations
- Additional features

## Author

[Your Name]

````

#### Project Report Template (docs/project_report.md)
```markdown
# Time Series Forecasting Project Report

## Executive Summary
- Project objective and scope
- Methodology overview
- Key findings and results
- Business impact

## Introduction
### Problem Statement
- Clear definition of the forecasting problem
- Business context and importance
- Success criteria

### Data Description
- Dataset characteristics
- Data sources and quality
- Preprocessing steps

## Methodology
### Exploratory Data Analysis
- Key patterns discovered
- Seasonality and trends identified
- Correlation analysis

### Feature Engineering
- Features created and rationale
- Selection criteria
- Impact on model performance

### Model Development
- Models tested and rationale for selection
- Hyperparameter optimization
- Cross-validation strategy

### Evaluation Framework
- Metrics used and justification
- Backtesting approach
- Model selection criteria

## Results
### Model Performance
- Performance metrics table
- Model comparison
- Statistical significance testing

### Insights and Patterns
- Key temporal patterns discovered
- Feature importance analysis
- Business insights

### Model Interpretation
- Explanation of model decisions
- Uncertainty quantification
- Sensitivity analysis

## Implementation
### Production Deployment
- Deployment architecture
- Monitoring strategy
- Performance monitoring

### Scalability
- Computational requirements
- Optimization strategies
- Future scalability plans

## Conclusions
### Key Findings
- Main results and insights
- Model reliability
- Business implications

### Recommendations
- Implementation recommendations
- Model selection advice
- Risk considerations

### Future Work
- Planned improvements
- Research directions
- Long-term enhancements

## Appendices
- Technical specifications
- Code documentation
- Additional analysis
````

### Code Quality Standards

#### Model Implementation Standards

```python
"""
Model Implementation Template
============================
Follow this template for all forecasting models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error

class BaseForecastingModel(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract base class for forecasting models

    Parameters
    ----------
    horizon : int
        Forecast horizon in periods
    frequency : str
        Data frequency (e.g., 'D', 'H', 'M')

    Attributes
    ----------
    fitted_model : object
        Fitted model object
    feature_importance_ : dict or array
        Feature importance scores (if applicable)
    training_metrics_ : dict
        Training performance metrics
    """

    def __init__(self, horizon: int = 1, frequency: str = 'D'):
        self.horizon = horizon
        self.frequency = frequency
        self.fitted_model = None
        self.feature_importance_ = None
        self.training_metrics_ = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseForecastingModel':
        """
        Fit the forecasting model

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with time index
        y : pd.Series
            Target variable with time index

        Returns
        -------
        self : BaseForecastingModel
            Fitted model
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with time index

        Returns
        -------
        predictions : np.ndarray
            Forecasted values
        """
        pass

    def forecast(self, steps: int = None) -> pd.Series:
        """
        Generate multi-step forecast

        Parameters
        ----------
        steps : int, optional
            Number of steps to forecast

        Returns
        -------
        forecast : pd.Series
            Forecasted values with time index
        """
        if steps is None:
            steps = self.horizon

        # Implementation depends on model type
        # This is a template - implement in derived classes
        raise NotImplementedError("Forecast method not implemented")

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Parameters
        ----------
        y_true : pd.Series
            Actual values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        metrics : dict
            Performance metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'bias': np.mean(y_pred - y_true)
        }

        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores

        Returns
        -------
        importance : dict or None
            Feature importance scores
        """
        return self.feature_importance_

    def plot_forecast(self, historical_data: pd.Series,
                     forecast: pd.Series, ax=None) -> None:
        """
        Plot historical data and forecast

        Parameters
        ----------
        historical_data : pd.Series
            Historical time series
        forecast : pd.Series
            Forecasted values
        ax : matplotlib axes, optional
            Axes to plot on
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        # Plot historical data
        historical_data.plot(ax=ax, label='Historical', color='blue', alpha=0.7)

        # Plot forecast
        forecast.plot(ax=ax, label='Forecast', color='red', linewidth=2)

        # Add confidence intervals if available
        # This can be extended for different models

        ax.set_title('Time Series Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

class ARIMAForecaster(BaseForecastingModel):
    """
    ARIMA forecasting model implementation
    """

    def __init__(self, order=(1,1,1), horizon=1, frequency='D'):
        super().__init__(horizon, frequency)
        self.order = order

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ARIMAForecaster':
        """Fit ARIMA model"""
        from statsmodels.tsa.arima.model import ARIMA

        # Train model
        self.fitted_model = ARIMA(y, order=self.order)
        self.fitted_model = self.fitted_model.fit()

        # Store training metrics
        self.training_metrics_ = self.evaluate(y, self.fitted_model.fittedvalues)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        # ARIMA prediction
        forecast = self.fitted_model.forecast(steps=len(X))
        return forecast.values

    def forecast(self, steps: int = None) -> pd.Series:
        """Generate multi-step forecast"""
        if steps is None:
            steps = self.horizon

        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        forecast = self.fitted_model.forecast(steps=steps)
        return pd.Series(forecast.values)

# Usage example
if __name__ == "__main__":
    # Example usage
    model = ARIMAForecaster(order=(1,1,1), horizon=30)

    # Fit and predict (this would need real data)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
```

#### Testing Standards

```python
"""
Unit Tests Template for Time Series Models
==========================================
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.traditional_models import ARIMAForecaster
from models.ml_models import RandomForestForecaster
from evaluation.metrics import calculate_forecast_metrics

class TestARIMAForecaster(unittest.TestCase):
    """Test cases for ARIMA model"""

    def setUp(self):
        """Set up test data"""
        # Create synthetic time series
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        trend = np.linspace(100, 110, 100)
        noise = np.random.normal(0, 2, 100)
        self.y_train = pd.Series(trend + noise, index=dates[:80])
        self.y_test = pd.Series(trend[80:] + noise[80:], index=dates[80:])

        # Create dummy features
        self.X_train = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 80),
            'feature2': np.random.normal(0, 1, 80)
        }, index=dates[:80])

        self.X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20)
        }, index=dates[80:])

        self.model = ARIMAForecaster(order=(1,1,1))

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.order, (1,1,1))
        self.assertEqual(self.model.horizon, 1)
        self.assertIsNone(self.model.fitted_model)

    def test_fit_method(self):
        """Test model fitting"""
        # Fit model
        fitted_model = self.model.fit(self.X_train, self.y_train)

        # Check that model is fitted
        self.assertIsNotNone(self.model.fitted_model)
        self.assertEqual(fitted_model, self.model)

        # Check training metrics are calculated
        self.assertIn('mae', self.model.training_metrics_)
        self.assertIn('rmse', self.model.training_metrics_)

    def test_predict_method(self):
        """Test prediction"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        predictions = self.model.predict(self.X_test)

        # Check prediction shape
        self.assertEqual(len(predictions), len(self.X_test))

        # Check predictions are numeric
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_forecast_method(self):
        """Test multi-step forecasting"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)

        # Generate forecast
        forecast = self.model.forecast(steps=10)

        # Check forecast properties
        self.assertEqual(len(forecast), 10)
        self.assertIsInstance(forecast, pd.Series)
        self.assertTrue(forecast.index.is_monotonic_increasing)

    def test_evaluate_method(self):
        """Test evaluation metrics"""
        # Create dummy predictions
        y_pred = np.random.normal(105, 2, len(self.y_test))

        # Evaluate
        metrics = self.model.evaluate(self.y_test, y_pred)

        # Check all required metrics are present
        required_metrics = ['mae', 'rmse', 'mape', 'bias']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

    def test_feature_importance(self):
        """Test feature importance method"""
        # ARIMA doesn't have traditional feature importance
        importance = self.model.get_feature_importance()
        self.assertIsNone(importance)

    def test_plot_forecast(self):
        """Test forecast plotting"""
        # Fit model first
        self.model.fit(self.X_train, self.y_train)

        # Generate forecast
        forecast = self.model.forecast(steps=10)

        # Test plotting (should not raise error)
        try:
            ax = self.model.plot_forecast(self.y_train, forecast)
            self.assertIsNotNone(ax)
        except Exception as e:
            self.fail(f"Plotting raised an exception: {e}")

    def test_error_handling(self):
        """Test error handling"""
        # Test predict without fitting
        with self.assertRaises(ValueError):
            self.model.predict(self.X_test)

        # Test forecast without fitting
        with self.assertRaises(ValueError):
            self.model.forecast()

class TestForecastMetrics(unittest.TestCase):
    """Test forecast metrics calculation"""

    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.y_true = pd.Series([100, 102, 98, 105, 103])
        self.y_pred = np.array([101, 100, 99, 104, 102])

    def test_calculate_forecast_metrics(self):
        """Test metrics calculation"""
        metrics = calculate_forecast_metrics(self.y_true, self.y_pred)

        # Check all metrics are calculated
        expected_metrics = ['mae', 'rmse', 'mape', 'r2', 'bias']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)

    def test_metrics_values(self):
        """Test metric values are reasonable"""
        metrics = calculate_forecast_metrics(self.y_true, self.y_pred)

        # MAE should be positive
        self.assertGreater(metrics['mae'], 0)

        # RMSE should be greater than or equal to MAE
        self.assertGreaterEqual(metrics['rmse'], metrics['mae'])

        # MAPE should be reasonable
        self.assertLess(metrics['mape'], 1000)

if __name__ == '__main__':
    unittest.main()
```

### Performance Benchmarking

#### Model Comparison Framework

```python
def benchmark_forecasting_models(models, data_splits, metrics=['mae', 'rmse', 'mape']):
    """
    Benchmark multiple forecasting models

    Parameters
    ----------
    models : dict
        Dictionary of model_name: model_instance
    data_splits : dict
        Dictionary of split_name: (X_train, X_test, y_train, y_test)
    metrics : list
        List of metrics to calculate

    Returns
    -------
    results : pd.DataFrame
        Benchmark results
    """
    results = []

    for split_name, (X_train, X_test, y_train, y_test) in data_splits.items():
        for model_name, model in models.items():
            try:
                # Fit model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                split_metrics = {'split': split_name, 'model': model_name}

                if 'mae' in metrics:
                    split_metrics['mae'] = mean_absolute_error(y_test, y_pred)
                if 'rmse' in metrics:
                    split_metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                if 'mape' in metrics:
                    split_metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                if 'r2' in metrics:
                    split_metrics['r2'] = r2_score(y_test, y_pred)

                # Add model-specific metrics
                if hasattr(model, 'training_time'):
                    split_metrics['training_time'] = model.training_time
                if hasattr(model, 'inference_time'):
                    split_metrics['inference_time'] = model.inference_time

                results.append(split_metrics)

            except Exception as e:
                print(f"Error with {model_name} on {split_name}: {str(e)}")
                results.append({
                    'split': split_name,
                    'model': model_name,
                    'error': str(e)
                })

    return pd.DataFrame(results)
```

### Deployment Guide

#### API Implementation

```python
"""
Forecasting API Implementation
=============================
Production-ready API for time series forecasting
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global model registry
models = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_models': list(models.keys())
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            'type': type(model).__name__,
            'horizon': getattr(model, 'horizon', 'unknown'),
            'frequency': getattr(model, 'frequency', 'unknown')
        }

    return jsonify(model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions using specified model

    Expected JSON payload:
    {
        "model_name": "arima_model",
        "data": {
            "features": [...],  # Feature values
            "timestamp": "2023-01-01"  # Timestamp
        },
        "horizon": 7  # Optional: forecast horizon
    }
    """
    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        model_name = data.get('model_name')
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        model = models[model_name]

        # Extract features
        features = data.get('features', [])
        timestamp = data.get('timestamp')
        horizon = data.get('horizon', model.horizon)

        # Convert to DataFrame
        if isinstance(features, list):
            if len(features) == 1:
                # Single prediction
                X = pd.DataFrame([features])
            else:
                # Multiple predictions
                X = pd.DataFrame(features)
        else:
            return jsonify({'error': 'Invalid features format'}), 400

        # Make prediction
        predictions = model.predict(X)

        # Format response
        if len(predictions) == 1:
            result = {
                'prediction': float(predictions[0]),
                'model': model_name,
                'horizon': horizon,
                'timestamp': timestamp
            }
        else:
            result = {
                'predictions': [float(p) for p in predictions],
                'model': model_name,
                'horizon': horizon,
                'timestamp': timestamp
            }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Generate multi-step forecast

    Expected JSON payload:
    {
        "model_name": "arima_model",
        "horizon": 30,
        "frequency": "D"
    }
    """
    try:
        data = request.get_json()

        model_name = data.get('model_name')
        horizon = data.get('horizon', 30)
        frequency = data.get('frequency', 'D')

        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        model = models[model_name]

        # Generate forecast
        forecast = model.forecast(steps=horizon)

        # Convert to list for JSON serialization
        forecast_data = []
        for idx, value in forecast.items():
            forecast_data.append({
                'timestamp': idx.isoformat(),
                'value': float(value)
            })

        return jsonify({
            'forecast': forecast_data,
            'model': model_name,
            'horizon': horizon,
            'frequency': frequency
        })

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/<model_name>/retrain', methods=['POST'])
def retrain_model(model_name):
    """
    Retrain a model with new data

    Expected JSON payload:
    {
        "data": {
            "X": [[...], [...]],  # Features
            "y": [...]            # Target values
        },
        "parameters": {...}  # Optional model parameters
    }
    """
    try:
        data = request.get_json()

        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not found'}), 404

        # Extract training data
        train_data = data.get('data', {})
        X_train = pd.DataFrame(train_data.get('X'))
        y_train = pd.Series(train_data.get('y'))

        # Get model instance
        model = models[model_name]

        # Optional parameters
        parameters = data.get('parameters', {})
        for param, value in parameters.items():
            if hasattr(model, param):
                setattr(model, param, value)

        # Retrain
        model.fit(X_train, y_train)

        # Update global model
        models[model_name] = model

        return jsonify({
            'message': f'Model {model_name} retrained successfully',
            'training_samples': len(X_train)
        })

    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def load_models(model_paths):
    """Load models from files"""
    for name, path in model_paths.items():
        try:
            model = joblib.load(path)
            models[name] = model
            logger.info(f"Loaded model: {name}")
        except Exception as e:
            logger.error(f"Failed to load model {name}: {str(e)}")

if __name__ == '__main__':
    # Load models (replace with your model paths)
    model_paths = {
        'arima_model': 'models/arima_model.pkl',
        'lstm_model': 'models/lstm_model.pkl',
        'prophet_model': 'models/prophet_model.pkl'
    }

    load_models(model_paths)

    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False)
```

This comprehensive project structure provides a complete framework for developing professional-grade time series forecasting projects. Each project includes hands-on implementation, documentation, testing, and deployment components that are essential for a strong portfolio.

---

_Choose projects that align with your interests and career goals. Focus on quality over quantity - it's better to complete a few projects thoroughly than to rush through many incomplete ones._
