# Model Selection & Data Science Pipeline: Universal Guide

## Clear Steps for Everyone

**Date:** November 1, 2025  
**Version:** Universal Edition  
**Total Lines:** 4,500+

---

## Table of Contents

1. [Introduction to Data Science Pipeline](#introduction)
2. [Data Collection Strategies](#data-collection)
3. [Data Cleaning & Preprocessing](#data-cleaning)
4. [Exploratory Data Analysis (EDA)](#eda)
5. [Feature Engineering](#feature-engineering)
6. [Model Selection Framework](#model-selection)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Cross-Validation Strategies](#cross-validation)
9. [Evaluation Metrics Deep Dive](#evaluation-metrics)
10. [Model Interpretation & Explainability](#model-interpretation)
11. [Algorithm Selection Flowcharts](#algorithm-selection)
12. [Complete End-to-End Pipeline](#end-to-end)
13. [Hardware & Infrastructure Requirements](#hardware)
14. [Career Guidance](#career-guidance)
15. [Real-World Applications](#real-world-applications)

---

## Introduction to Data Science Pipeline {#introduction}

### What is a Data Science Pipeline? 🤔

Imagine you're building a **castle** (your AI model). You can't just grab random stones and build a perfect castle immediately! You need to:

1. **Collect the right stones** (data)
2. **Clean and shape them** (preprocessing)
3. **Arrange them in the best pattern** (model selection)
4. **Test if it stands strong** (evaluation)
5. **Understand why it works** (interpretation)

**Data Science Pipeline** is like following a recipe step-by-step to create the perfect AI dish!

### Why Do We Need a Pipeline?

Think of it like building a **car factory**:

- **Quality Control:** Check each car before it leaves (data validation)
- **Assembly Line:** Organized steps make everything work together smoothly
- **Testing:** Make sure every car drives safely (model validation)
- **Improvements:** Learn from each car to make the next one better

### The 6 Stages of Every AI Project 🚀

```
1. Collect Data → 2. Clean Data → 3. Explore Data → 4. Build Model → 5. Test Model → 6. Deploy Model
```

**Stage 1: Data Collection** - Gather raw materials (data) from the right sources
**Stage 2: Data Cleaning** - Remove dirt and fix broken pieces (missing values, outliers)
**Stage 3: Data Exploration** - Look at your materials to understand what you've got
**Stage 4: Model Building** - Choose the right algorithm and train it
**Stage 5: Model Testing** - Check if it works well on new, unseen data
**Stage 6: Model Deployment** - Put your trained model to work in the real world

### Simple Analogy: Baking a Cake 🍰

Let's compare our data science pipeline to baking a cake:

1. **Get Ingredients (Data Collection):** Collect flour, sugar, eggs, etc.
2. **Clean & Prepare (Preprocessing):** Sift flour, crack eggs carefully
3. **Mix & Test (Exploration):** Taste the batter, check consistency
4. **Choose Recipe (Model Selection):** Decide between chocolate cake, vanilla, etc.
5. **Bake & Time (Training):** Set the right temperature and timing
6. **Taste Test (Evaluation):** Check if it's sweet enough, not burnt
7. **Serve & Enjoy (Deployment):** Share your delicious cake with others

### What Makes a Good Pipeline?

Think of it like **organizing your bedroom**:

1. **Everything in the right place** - Data is well-structured and organized
2. **Clean and tidy** - No missing values or dirty data
3. **Easy to find what you need** - Clear documentation and workflow
4. **Can adapt to changes** - Flexible enough for new requirements
5. **Works consistently** - Produces reliable results every time

### Types of Data Science Problems

**Classification** - "Is this email spam or not spam?" (like sorting toys into red box or blue box)

**Regression** - "What will the house price be?" (like predicting how tall you'll be when you grow up)

**Clustering** - "Group similar customers together" (like sorting your Lego pieces by color without being told)

**Dimensionality Reduction** - "Simplify data while keeping important information" (like creating a small poster from a big painting)

---

## Data Collection Strategies {#data-collection}

### What is Data Collection? 📊

Data collection is like being a **treasure hunter** looking for the perfect pieces of information. You need to know:

- **WHERE** to look for treasure (data sources)
- **WHAT** kind of treasure you need (data types)
- **HOW** to gather it safely (collection methods)
- **WHEN** to collect it (timing and frequency)

### Types of Data Sources 🏭

#### 1. **Internal Data Sources** (From Inside Your Company)

Think of this like collecting your **allowance money** from different jars at home:

- **Customer Databases:** Store records of every customer interaction
- **Sales Records:** Information about what you sold and when
- **Website Analytics:** How people use your website (page views, clicks, time spent)
- **Mobile App Data:** User behavior in your mobile app
- **Email Records:** Marketing campaign results
- **Financial Reports:** Revenue, costs, profits over time

**Example Code for Collecting Internal Data:**

```python
import pandas as pd
import numpy as np

def collect_internal_data():
    """
    Collect data from different internal sources
    Think of this as checking different pockets in your backpack for coins
    """

    # Simulate collecting customer data (like checking your wallet)
    customers_data = {
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 34, 28, 45, 31],
        'income': [50000, 75000, 45000, 120000, 80000],
        'purchases_last_month': [5, 12, 3, 20, 8]
    }

    # Simulate sales data (like checking your piggy bank)
    sales_data = {
        'sale_id': [101, 102, 103, 104, 105],
        'customer_id': [1, 2, 1, 3, 4],
        'amount': [25.99, 150.50, 89.99, 299.99, 75.00],
        'date': ['2025-01-15', '2025-01-16', '2025-01-17', '2025-01-18', '2025-01-19']
    }

    return pd.DataFrame(customers_data), pd.DataFrame(sales_data)

# Let's collect our internal treasure!
customers_df, sales_df = collect_internal_data()
print("Customer Data Shape:", customers_df.shape)
print("Sales Data Shape:", sales_df.shape)
```

#### 2. **External Data Sources** (From Outside Your Company)

Think of this like **baking cookies** - you need ingredients from different stores:

- **Government Databases:** Census data, economic indicators, weather data
- **Social Media APIs:** Twitter, Facebook, Instagram data
- **Web Scraping:** Information from websites (reviews, prices, news)
- **Open Datasets:** Kaggle, UCI Machine Learning Repository, Google Dataset Search
- **APIs (Application Programming Interfaces):** Financial data, weather APIs, news APIs
- **Sensor Data:** IoT devices, GPS data, satellite imagery

**Example Code for External Data Collection:**

```python
import requests
import json
from datetime import datetime

def collect_external_data():
    """
    Collect data from external sources - like getting ingredients from different stores
    """

    # Simulate collecting weather data (like checking the weather app)
    weather_data = {
        'city': ['New York', 'London', 'Tokyo', 'Sydney'],
        'temperature': [22, 18, 25, 28],  # Celsius
        'humidity': [65, 70, 60, 55],     # Percentage
        'date': ['2025-10-30'] * 4
    }

    # Simulate collecting financial data (like checking stock prices)
    financial_data = {
        'symbol': ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
        'price': [150.25, 2800.50, 330.75, 3150.80],
        'volume': [50000000, 1200000, 25000000, 3000000],
        'market_cap': [2400000000000, 1800000000000, 2500000000000, 1600000000000]
    }

    return pd.DataFrame(weather_data), pd.DataFrame(financial_data)

# Collect external data sources
weather_df, financial_df = collect_external_data()
print("Weather Data Shape:", weather_df.shape)
print("Financial Data Shape:", financial_df.shape)
```

### Data Collection Methods 📱

#### 1. **Automated Collection** (Like Having a Robot Assistant)

**Web Scraping:** Automatically collect data from websites
**API Integration:** Pull data from third-party services automatically  
**Database Queries:** Extract data from existing databases
**Sensor Integration:** Collect data from IoT devices automatically

**Web Scraping Example:**

```python
from bs4 import BeautifulSoup
import requests

def web_scraping_example():
    """
    Collect news headlines - like having a robot that reads newspapers for you
    """

    # Simulate scraping news headlines
    news_headlines = [
        "AI Breakthrough: New Model Achieves 99% Accuracy",
        "Climate Change: New Data Shows Rising Temperatures",
        "Space Exploration: Mars Mission Launches Successfully",
        "Healthcare: AI Helps Detect Diseases Earlier",
        "Transportation: Self-Driving Cars Reach New Milestone"
    ]

    scraped_data = {
        'headline': news_headlines,
        'category': ['Technology', 'Environment', 'Science', 'Healthcare', 'Transportation'],
        'source': ['TechNews'] * 5,
        'timestamp': ['2025-10-30 12:00:00'] * 5
    }

    return pd.DataFrame(scraped_data)

news_df = web_scraping_example()
print("Scraped News Headlines:")
print(news_df.head())
```

#### 2. **Manual Collection** (Like Taking Notes Yourself)

**Surveys and Questionnaires:** Ask people directly for information
**Data Entry:** Manually input data from physical documents
**User Generated Content:** Let users submit data through forms
**Observational Studies:** Record behavior by watching people

**Survey Data Collection Example:**

```python
def collect_survey_data():
    """
    Collect survey responses - like interviewing people directly
    """

    survey_responses = {
        'respondent_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'age_group': ['18-25', '26-35', '36-45', '46-55', '18-25', '26-35', '36-45', '46-55'],
        'satisfaction_score': [8, 9, 7, 6, 9, 8, 7, 8],  # Scale 1-10
        'would_recommend': [True, True, True, False, True, True, True, True],
        'feedback': [
            'Great service!',
            'Very satisfied with the product',
            'Could be better',
            'Not impressed',
            'Excellent experience',
            'Good value for money',
            'Satisfactory',
            'Would definitely use again'
        ]
    }

    return pd.DataFrame(survey_responses)

survey_df = collect_survey_data()
print("Survey Data Sample:")
print(survey_df.head())
```

### Data Quality Assessment 🔍

Before collecting data, think about the **quality of ingredients** for your recipe:

#### 1. **Completeness** - Is all the data there?

Like checking if you have all ingredients before baking:

- How much data is missing?
- Are there empty cells in important columns?
- Do you have enough historical data?

#### 2. **Accuracy** - Is the data correct?

Like checking if your measuring cup has the right markings:

- Are the values reasonable and realistic?
- Do different data sources agree with each other?
- Are there obvious errors or inconsistencies?

#### 3. **Consistency** - Is the data consistent over time?

Like making sure your oven temperature stays the same:

- Is data formatted the same way?
- Do definitions stay the same over time?
- Are there any conflicting records?

#### 4. **Timeliness** - Is the data up to date?

Like using fresh ingredients instead of expired ones:

- How recent is the data?
- How often is it updated?
- Does it reflect current conditions?

### Data Collection Planning 🗺️

#### Step 1: Define Your Data Needs

Think of this like making a **grocery list** before going to the store:

**Questions to Ask:**

- What specific information do I need?
- What time period should I cover?
- What level of detail do I need?
- How frequently do I need updates?

```python
def plan_data_collection():
    """
    Plan your data collection strategy like making a grocery list
    """

    project_requirements = {
        'project_name': 'Customer Churn Prediction',
        'business_question': 'Which customers are likely to stop using our service?',
        'required_data': {
            'customer_data': {
                'customer_id': 'Unique identifier',
                'age': 'Customer age',
                'tenure': 'How long they\'ve been a customer',
                'monthly_charges': 'Average monthly spending',
                'contract_type': 'Month-to-month, yearly, etc.',
                'churn_status': 'Did they leave? (TARGET VARIABLE)'
            },
            'service_data': {
                'support_tickets': 'Number of customer support calls',
                'service_usage': 'How often they use the service',
                'billing_issues': 'Any payment problems',
                'feature_usage': 'Which features they use most'
            }
        },
        'data_sources': [
            'Internal CRM system',
            'Billing database',
            'Customer support system',
            'Usage analytics platform'
        ],
        'timeline': 'Last 2 years of customer data',
        'quality_requirements': 'Minimum 95% completeness on key variables'
    }

    return project_requirements

requirements = plan_data_collection()
print("Data Collection Plan:")
for key, value in requirements.items():
    print(f"{key}: {value}")
```

#### Step 2: Data Source Evaluation

Like evaluating different stores before shopping:

```python
def evaluate_data_sources():
    """
    Compare different data sources - like comparing different grocery stores
    """

    source_comparison = {
        'internal_crm': {
            'cost': 'Free (already have)',
            'accessibility': 'High',
            'data_quality': 'Good',
            'update_frequency': 'Real-time',
            'completeness': '95%'
        },
        'external_api': {
            'cost': '$500/month',
            'accessibility': 'Medium',
            'data_quality': 'Excellent',
            'update_frequency': 'Daily',
            'completeness': '100%'
        },
        'web_scraping': {
            'cost': '$1000 setup',
            'accessibility': 'Medium',
            'data_quality': 'Variable',
            'update_frequency': 'Weekly',
            'completeness': '80%'
        }
    }

    return source_comparison

sources = evaluate_data_sources()
print("\nData Source Evaluation:")
for source, metrics in sources.items():
    print(f"\n{source.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
```

### Data Collection Best Practices ✅

#### 1. **Documentation** 📝

Keep track of everything like writing down your homework assignment:

```python
def create_data_documentation():
    """
    Create documentation for your data collection process
    """

    documentation = {
        'collection_date': '2025-10-30',
        'data_source': 'Customer Database',
        'collection_method': 'SQL query',
        'records_collected': 10000,
        'fields_included': ['customer_id', 'age', 'income', 'region', 'churn_flag'],
        'quality_checks': [
            'Removed duplicates',
            'Handled missing values',
            'Validated age ranges (18-100)',
            'Checked for outliers in income'
        ],
        'collection_notes': 'Data represents active customers as of October 2025'
    }

    return documentation

docs = create_data_documentation()
print("Data Collection Documentation:")
for key, value in docs.items():
    print(f"{key}: {value}")
```

#### 2. **Data Versioning** 📚

Keep different versions like saving different drafts of your essay:

```python
def manage_data_versions():
    """
    Track different versions of your collected data
    """

    version_history = {
        'v1.0': {
            'date': '2025-10-15',
            'description': 'Initial customer data collection',
            'records': 9500,
            'issues': 'Missing region data for 5% of customers'
        },
        'v1.1': {
            'date': '2025-10-20',
            'description': 'Added missing region data from billing system',
            'records': 10000,
            'issues': 'None'
        },
        'v1.2': {
            'date': '2025-10-25',
            'description': 'Added updated churn labels',
            'records': 10000,
            'issues': 'None'
        }
    }

    return version_history

versions = manage_data_versions()
print("\nData Version History:")
for version, details in versions.items():
    print(f"\n{version}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

#### 3. **Data Privacy & Ethics** 🔒

Protect sensitive information like keeping your diary private:

```python
def ensure_data_privacy():
    """
    Implement privacy and ethical considerations
    """

    privacy_checklist = {
        'personal_identifiers': 'Remove or encrypt customer names, emails, addresses',
        'sensitive_data': 'Handle financial information securely with encryption',
        'consent': 'Ensure all data collection has proper user consent',
        'anonymization': 'Replace identifiable information with anonymous IDs',
        'access_controls': 'Limit data access to authorized personnel only',
        'data_retention': 'Delete data after retention period expires',
        'audit_trail': 'Log who accessed what data and when'
    }

    return privacy_checklist

privacy = ensure_data_privacy()
print("\nData Privacy Checklist:")
for item, description in privacy.items():
    print(f"✓ {item}: {description}")
```

### Data Collection Tools & Technologies 🛠️

#### Beginner Level Tools

Think of these as **basic kitchen utensils**:

1. **Excel/Google Sheets** - Simple data entry and basic collection
2. **Survey Tools** (Google Forms, Typeform) - Collect responses easily
3. **Web Scrapers** (Octoparse, ParseHub) - Visual web scraping
4. **API Testing Tools** (Postman) - Test API connections

#### Intermediate Level Tools

Think of these as **professional kitchen equipment**:

1. **Python Libraries:**
   - `requests` - Make API calls
   - `beautifulsoup4` - Parse HTML and web scraping
   - `pandas` - Data manipulation and collection
   - `selenium` - Automate web browser interactions
   - `pymongo` - Collect data from MongoDB
   - `sqlalchemy` - Connect to various databases

```python
# Collection of Python tools for data collection
import collections

data_collection_tools = {
    'web_scraping': {
        'requests': 'Make HTTP requests to websites',
        'beautifulsoup4': 'Parse HTML and extract data',
        'selenium': 'Automate browser interactions',
        'scrapy': 'Professional web scraping framework'
    },
    'api_collection': {
        'requests': 'Call REST APIs',
        'httpx': 'Modern async HTTP client',
        'tweepy': 'Twitter API client',
        'google-api-python-client': 'Google services APIs'
    },
    'database_collection': {
        'pymongo': 'MongoDB database',
        'psycopg2': 'PostgreSQL database',
        'sqlite3': 'SQLite database',
        'sqlalchemy': 'ORM for multiple databases'
    },
    'file_collection': {
        'pandas': 'Read CSV, Excel, JSON files',
        'openpyxl': 'Read/write Excel files',
        'pdfplumber': 'Extract text from PDF files',
        'pytesseract': 'Extract text from images'
    }
}

print("Data Collection Tools by Category:")
for category, tools in data_collection_tools.items():
    print(f"\n{category.upper()}:")
    for tool, description in tools.items():
        print(f"  • {tool}: {description}")
```

#### Advanced Level Tools

Think of these as **industrial kitchen equipment**:

1. **Big Data Platforms:**
   - Apache Kafka - Real-time data streaming
   - Apache Spark - Large-scale data processing
   - Hadoop - Distributed storage and processing

2. **Cloud Data Collection:**
   - AWS Glue - Data integration service
   - Google Cloud Dataflow - Stream processing
   - Azure Data Factory - Data movement service

### Data Collection Challenges & Solutions 🚧

#### Challenge 1: **Missing Data**

Like trying to bake a cake without flour:

```python
def handle_missing_data():
    """
    Strategies for handling missing data like finding missing ingredients
    """

    strategies = {
        'complete_case_analysis': {
            'description': 'Remove rows with any missing values',
            'when_to_use': 'When you have enough data and missing is random',
            'pros': 'Simple and effective',
            'cons': 'May lose important information'
        },
        'mean_median_mode_imputation': {
            'description': 'Fill missing values with average or most common value',
            'when_to_use': 'When missing data is small and random',
            'pros': 'Preserves data size',
            'cons': 'May reduce variance'
        },
        'forward_backward_fill': {
            'description': 'Use previous/next value to fill missing',
            'when_to_use': 'Time series data where values are related',
            'pros': 'Makes sense for sequential data',
            'cons': 'May introduce bias'
        },
        'multiple_imputation': {
            'description': 'Create multiple versions with different imputations',
            'when_to_use': 'When you want to account for imputation uncertainty',
            'pros': 'More statistically sound',
            'cons': 'Complex to implement'
        }
    }

    return strategies

missing_data_strategies = handle_missing_data()
print("Missing Data Handling Strategies:")
for strategy, details in missing_data_strategies.items():
    print(f"\n{strategy.upper()}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

#### Challenge 2: **Data Quality Issues**

Like dealing with moldy or expired ingredients:

```python
def assess_data_quality():
    """
    Check data quality like inspecting food for freshness
    """

    quality_checks = {
        'outlier_detection': {
            'method': 'Statistical methods (IQR, Z-score)',
            'example': 'Age = 150 years (impossible)',
            'solution': 'Remove or investigate unusual values'
        },
        'inconsistency_check': {
            'method': 'Compare across related fields',
            'example': 'Age = 25, Birth Year = 1990 (inconsistent)',
            'solution': 'Standardize data entry and validation'
        },
        'duplicate_detection': {
            'method': 'Find identical or near-identical records',
            'example': 'Same customer appears twice with same details',
            'solution': 'Remove duplicates based on business rules'
        },
        'validity_check': {
            'method': 'Check against business rules',
            'example': 'Negative price, Email without @ symbol',
            'solution': 'Implement input validation rules'
        }
    }

    return quality_checks

quality_issues = assess_data_quality()
print("Data Quality Assessment:")
for issue, details in quality_issues.items():
    print(f"\n{issue.upper()}:")
    for key, value in details.items():
        print(f"  {key}: {value}")
```

### Sample Data Collection Project 🍕

Let's collect data for a **pizza delivery optimization** project:

```python
def pizza_delivery_data_collection():
    """
    Complete data collection for pizza delivery optimization
    Like gathering ingredients for the perfect pizza recipe
    """

    # Step 1: Define what data we need
    data_requirements = {
        'order_data': [
            'order_id', 'customer_id', 'pizza_type', 'size', 'toppings',
            'order_time', 'delivery_time', 'delivery_address', 'distance_km',
            'weather_condition', 'traffic_level', 'delivery_fee'
        ],
        'customer_data': [
            'customer_id', 'age', 'location_zone', 'past_orders',
            'preferred_payment', 'phone_type', 'app_usage_frequency'
        ],
        'operational_data': [
            'driver_id', 'vehicle_type', 'restaurant_id', 'staff_count',
            'peak_hours', 'kitchen_efficiency', 'driver_performance'
        ]
    }

    # Step 2: Collect data from different sources
    # Simulate order data collection
    np.random.seed(42)  # For reproducible results

    orders = []
    for i in range(1000):  # Collect 1000 orders
        order = {
            'order_id': f'ORD_{i+1:04d}',
            'customer_id': f'CUST_{np.random.randint(1, 500):03d}',
            'pizza_type': np.random.choice(['Margherita', 'Pepperoni', 'Supreme', 'Veggie']),
            'size': np.random.choice(['Small', 'Medium', 'Large', 'Extra Large']),
            'order_time': np.random.randint(11, 23),  # 11 AM to 10 PM
            'distance_km': np.random.exponential(2.5),  # Exponential distribution
            'weather_condition': np.random.choice(['Sunny', 'Rainy', 'Cloudy', 'Snowy']),
            'traffic_level': np.random.choice(['Low', 'Medium', 'High']),
            'delivery_fee': np.random.uniform(2.50, 8.50)
        }

        # Calculate delivery time based on conditions
        base_time = 20  # 20 minutes base
        weather_factor = {'Sunny': 1.0, 'Cloudy': 1.1, 'Rainy': 1.3, 'Snowy': 1.5}[order['weather_condition']]
        traffic_factor = {'Low': 1.0, 'Medium': 1.2, 'High': 1.5}[order['traffic_level']]
        distance_factor = 1 + (order['distance_km'] * 0.3)

        order['delivery_time'] = base_time * weather_factor * traffic_factor * distance_factor
        orders.append(order)

    # Step 3: Organize collected data
    order_df = pd.DataFrame(orders)

    print("Pizza Delivery Data Collection Summary:")
    print(f"Total Orders Collected: {len(order_df)}")
    print(f"Average Delivery Time: {order_df['delivery_time'].mean():.1f} minutes")
    print(f"Weather Conditions: {order_df['weather_condition'].value_counts().to_dict()}")
    print(f"Traffic Levels: {order_df['traffic_level'].value_counts().to_dict()}")

    return order_df

# Collect our pizza delivery data!
pizza_data = pizza_delivery_data_collection()

# Show sample of collected data
print("\nSample of Collected Data:")
print(pizza_data.head())

print(f"\nData Collection Complete!")
print(f"Shape: {pizza_data.shape}")
print(f"Columns: {list(pizza_data.columns)}")
```

### Summary: Data Collection Checklist ✅

Before you start collecting data, make sure you have:

- [ ] **Clear objective** - What specific question are you trying to answer?
- [ ] **Data source list** - Where will you get your data from?
- [ ] **Quality requirements** - How clean and complete should your data be?
- [ ] **Privacy considerations** - How will you protect sensitive information?
- [ ] **Technical setup** - Do you have the right tools and access?
- [ ] **Timeline plan** - When will you collect and process the data?
- [ ] **Storage plan** - Where will you save your collected data?
- [ ] **Documentation** - Will you record your collection process?

---

## Data Cleaning & Preprocessing {#data-cleaning}

### What is Data Cleaning? 🧹

Think of data cleaning like **washing vegetables** before cooking:

1. **Remove dirt** (inaccurate or wrong data)
2. **Check for bruises** (outliers and anomalies)
3. **Cut off bad parts** (missing values)
4. **Organize properly** (standardize formats)
5. **Ready to use** (clean, consistent, reliable data)

### Why Clean Data Matters 🎯

Clean data is like having a **well-organized toolbox**:

- **Faster work:** No time wasted searching for things
- **Better results:** Tools work as expected
- **Fewer mistakes:** Everything is in the right place
- **Easy collaboration:** Others can find what they need

### Common Data Problems & Solutions 🏥

#### Problem 1: **Missing Data** (Like Missing Puzzle Pieces)

**What it looks like:**

```
Name    Age    Income    City
John    25     50000     New York
Mary    30     NULL      Boston    ← Missing value
Bob     NULL   75000     Chicago   ← Missing value
```

**Solutions:**

1. **Remove missing data** (if only a few pieces missing)
2. **Fill with average** (like guessing the missing puzzle piece)
3. **Use intelligent guess** (like knowing it should be a corner piece)

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

def handle_missing_data_examples():
    """
    Different ways to handle missing data - like solving a puzzle
    """

    # Create sample data with missing values
    data = {
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 30, np.nan, 45, 50],           # Missing age for customer 3
        'income': [50000, 75000, 60000, np.nan, 80000],  # Missing income for customer 4
        'city': ['New York', 'Boston', 'Chicago', np.nan, 'Miami'],  # Missing city for customer 4
        'purchases': [5, 12, 8, 15, np.nan]        # Missing purchases for customer 5
    }

    df = pd.DataFrame(data)
    print("Original Data with Missing Values:")
    print(df)
    print(f"\nMissing values per column:")
    print(df.isnull().sum())

    # Solution 1: Remove rows with missing values (if you can afford to lose data)
    df_complete = df.dropna()
    print(f"\n1. After removing rows with any missing values:")
    print(df_complete)
    print(f"Records remaining: {len(df_complete)} out of {len(df)}")

    # Solution 2: Fill with mean/median for numerical data
    df_mean_filled = df.copy()
    df_mean_filled['age'].fillna(df['age'].mean(), inplace=True)
    df_mean_filled['income'].fillna(df['income'].mean(), inplace=True)
    df_mean_filled['purchases'].fillna(df['purchases'].mean(), inplace=True)

    print(f"\n2. After filling numerical missing values with mean:")
    print(df_mean_filled)

    # Solution 3: Fill with mode for categorical data
    df_mode_filled = df_mean_filled.copy()
    df_mode_filled['city'].fillna(df['city'].mode()[0], inplace=True)  # mode()[0] gets the most common value

    print(f"\n3. After filling categorical missing values with mode:")
    print(df_mode_filled)

    # Solution 4: Use KNN imputation (more sophisticated)
    imputer = KNNImputer(n_neighbors=2)
    numerical_cols = ['age', 'income', 'purchases']
    df_imputed = df.copy()
    df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    print(f"\n4. After KNN imputation:")
    print(df_imputed)

    return df, df_mode_filled, df_imputed

# Demonstrate missing data handling
original, filled, imputed = handle_missing_data_examples()
```

#### Problem 2: **Duplicate Data** (Like Buying the Same Item Twice)

**What it looks like:**

```
Name    Email           Purchase
John    john@email.com   100
Mary    mary@email.com   200
John    john@email.com   100    ← Duplicate!
Sarah   sarah@email.com  150
```

**Solutions:**

1. **Remove exact duplicates** (keep only one copy)
2. **Remove near duplicates** (based on similarity)
3. **Merge duplicate records** (combine information)

```python
def handle_duplicate_data():
    """
    Handle duplicate data - like removing duplicate items from your backpack
    """

    # Create sample data with duplicates
    data = {
        'customer_id': [1, 2, 3, 1, 4, 2],        # Customer 1 and 2 appear twice
        'name': ['John', 'Mary', 'Bob', 'John', 'Sarah', 'Mary'],
        'email': ['john@email', 'mary@email', 'bob@email', 'john@email', 'sarah@email', 'mary@email'],
        'purchase_amount': [100, 200, 150, 100, 300, 200]  # Exact duplicates
    }

    df = pd.DataFrame(data)
    print("Original Data with Duplicates:")
    print(df)
    print(f"Total duplicates: {df.duplicated().sum()}")

    # Solution 1: Remove exact duplicates
    df_no_dup = df.drop_duplicates()
    print(f"\n1. After removing exact duplicates:")
    print(df_no_dup)

    # Solution 2: Keep first occurrence of each customer
    df_first_occurrence = df.drop_duplicates(subset=['customer_id'], keep='first')
    print(f"\n2. Keeping first occurrence of each customer:")
    print(df_first_occurrence)

    # Solution 3: Aggregate duplicate records
    df_aggregated = df.groupby('customer_id').agg({
        'name': 'first',
        'email': 'first',
        'purchase_amount': 'sum'  # Add up purchase amounts
    }).reset_index()

    print(f"\n3. After aggregating duplicates (sum purchase amounts):")
    print(df_aggregated)

    return df, df_no_dup, df_aggregated

# Demonstrate duplicate handling
original_data, no_duplicates, aggregated = handle_duplicate_data()
```

#### Problem 3: **Outliers** (Like Finding a Giant Apple in a Basket of Normal Apples)

**What it looks like:**

```
Age values: [25, 30, 28, 32, 27, 100]  ← 100 is an outlier!
Income values: [50000, 75000, 60000, 1200000]  ← 1.2M is an outlier!
```

**Solutions:**

1. **Remove outliers** (throw away the giant apple)
2. **Cap outliers** (cut the giant apple down to normal size)
3. **Transform data** (use a different way to measure)

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def handle_outliers():
    """
    Handle outliers - like finding unusual items in your backpack
    """

    # Create sample data with outliers
    np.random.seed(42)
    data = {
        'age': [25, 30, 28, 32, 27, 50, 29, 31, 100],  # 100 is outlier
        'income': [50000, 75000, 60000, 55000, 80000, 70000, 1200000, 65000, 58000],  # 1.2M is outlier
        'experience_years': [2, 5, 3, 7, 1, 4, 8, 6, 95]  # 95 is outlier
    }

    df = pd.DataFrame(data)
    print("Original Data with Outliers:")
    print(df)
    print(f"\nStatistical Summary:")
    print(df.describe())

    # Method 1: Using IQR (Interquartile Range)
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"\n{column} outlier detection:")
        print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        print(f"Outliers found: {len(outliers)}")
        if len(outliers) > 0:
            print(f"Outlier values: {outliers[column].tolist()}")

        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Remove outliers from age column
    df_no_age_outliers = remove_outliers_iqr(df, 'age')
    print(f"\nAfter removing age outliers: {len(df_no_age_outliers)} records")

    # Method 2: Using Z-Score
    def remove_outliers_zscore(df, column, threshold=2):
        z_scores = np.abs(stats.zscore(df[column]))
        print(f"\n{column} Z-scores: {z_scores}")
        outliers_z = df[z_scores > threshold]
        print(f"Outliers (Z-score > {threshold}): {len(outliers_z)}")
        if len(outliers_z) > 0:
            print(f"Outlier values: {outliers_z[column].tolist()}")

        return df[z_scores <= threshold]

    df_no_income_outliers = remove_outliers_zscore(df_no_age_outliers, 'income')
    print(f"After removing income outliers: {len(df_no_income_outliers)} records")

    # Method 3: Capping (Winsorization)
    df_capped = df.copy()
    for column in ['age', 'income', 'experience_years']:
        Q1 = df_capped[column].quantile(0.05)  # 5th percentile
        Q99 = df_capped[column].quantile(0.95)  # 95th percentile

        df_capped[column] = df_capped[column].clip(lower=Q1, upper=Q99)
        print(f"\n{column} capped at [{Q1}, {Q99}]")

    print(f"\nAfter capping outliers:")
    print(df_capped)

    return df, df_no_income_outliers, df_capped

# Demonstrate outlier handling
original_df, no_outliers, capped_df = handle_outliers()
```

### Data Type Issues 🔧

**Problem:** Data stored in wrong format (like storing numbers as text)

```python
def fix_data_types():
    """
    Fix data type issues - like organizing your tools by type
    """

    # Create data with wrong types
    data = {
        'customer_id': ['1', '2', '3', '4'],          # Should be integer
        'age': ['25', '30', '28', '35'],              # Should be integer
        'income': ['50000', '75000', '60000', '80000'], # Should be integer
        'signup_date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'], # Should be date
        'is_premium': ['True', 'False', 'True', 'False'], # Should be boolean
        'rating': ['4.5', '3.2', '4.8', '2.9']       # Should be float
    }

    df = pd.DataFrame(data)
    print("Original Data (wrong types):")
    print(df.dtypes)
    print(f"\nData sample:")
    print(df.head())

    # Fix data types
    df_fixed = df.copy()

    # Convert to appropriate types
    df_fixed['customer_id'] = df_fixed['customer_id'].astype(int)
    df_fixed['age'] = df_fixed['age'].astype(int)
    df_fixed['income'] = df_fixed['income'].astype(int)
    df_fixed['signup_date'] = pd.to_datetime(df_fixed['signup_date'])
    df_fixed['is_premium'] = df_fixed['is_premium'].astype(bool)
    df_fixed['rating'] = df_fixed['rating'].astype(float)

    print(f"\nAfter fixing data types:")
    print(df_fixed.dtypes)
    print(f"\nData sample:")
    print(df_fixed.head())

    return df, df_fixed

# Demonstrate data type fixing
original_types, fixed_types = fix_data_types()
```

### Text Data Cleaning 📝

**Problem:** Text data with inconsistencies (like having messy handwriting)

```python
import re

def clean_text_data():
    """
    Clean text data - like correcting messy handwriting
    """

    # Create messy text data
    messy_texts = [
        "Hello World!!!",
        "  HELLO world  ",
        "hello-world",
        "Hello-World!!",
        "HELLO WORLD",
        "hello123world",
        "hello.world",
        "Hello World?"
    ]

    df = pd.DataFrame({'text': messy_texts})

    print("Original Messy Text:")
    print(df['text'].tolist())

    # Text cleaning steps
    def clean_text(text):
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    df['cleaned_text'] = df['text'].apply(clean_text)

    print(f"\nCleaned Text:")
    print(df['cleaned_text'].tolist())

    # Advanced text cleaning
    def advanced_clean_text(text):
        # Remove stopwords
        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = text.split()
        words = [word for word in words if word not in stopwords]

        # Stemming (remove word endings)
        words = [word[:3] + word[3:].replace('ing', '').replace('ed', '').replace('ly', '') for word in words]

        return ' '.join(words)

    df['advanced_cleaned'] = df['cleaned_text'].apply(advanced_clean_text)

    print(f"\nAdvanced Cleaned Text (stopwords removed, stemmed):")
    print(df['advanced_cleaned'].tolist())

    return df

# Demonstrate text cleaning
text_cleaning_results = clean_text_data()
```

### Date and Time Data Cleaning 📅

```python
def clean_date_data():
    """
    Clean date and time data - like organizing your calendar properly
    """

    # Create messy date data
    messy_dates = [
        '01/15/2025',
        '2025-02-20',
        'March 15, 2025',
        '15-Mar-2025',
        '2025.03.20',
        '2025/03/25 14:30:00',
        'Mar 30 2025 3:45 PM',
        '2025-04-01T16:00:00Z'
    ]

    df = pd.DataFrame({'date_string': messy_dates})

    print("Original Messy Dates:")
    print(df['date_string'].tolist())

    # Convert to proper datetime
    df['parsed_date'] = pd.to_datetime(df['date_string'])

    print(f"\nParsed Dates:")
    print(df['parsed_date'].tolist())

    # Extract useful features
    df['year'] = df['parsed_date'].dt.year
    df['month'] = df['parsed_date'].dt.month
    df['day'] = df['parsed_date'].dt.day
    df['weekday'] = df['parsed_date'].dt.day_name()
    df['is_weekend'] = df['parsed_date'].dt.weekday >= 5  # Saturday=5, Sunday=6

    print(f"\nDate Features:")
    print(df[['date_string', 'parsed_date', 'year', 'month', 'weekday', 'is_weekend']])

    return df

# Demonstrate date cleaning
date_cleaning_results = clean_date_data()
```

### Categorical Data Cleaning 🏷️

```python
def clean_categorical_data():
    """
    Clean categorical data - like organizing your closet by categories
    """

    # Create messy categorical data
    messy_categories = {
        'product_category': ['Electronics', 'electronics', 'ELECTRONICS', 'Electronics ', ' gadgets', 'GADGETS'],
        'color': ['Red', 'RED', 'red', ' Red ', 'blue', 'Blue', 'BLUE'],
        'size': ['Small', 'SMALL', 'small', ' S ', 'Medium', 'MEDIUM', 'medium', 'M'],
        'brand': ['Apple', 'apple', 'APPLE', 'Apple Inc.', 'Samsung', 'samsung', 'SAMSUNG']
    }

    df = pd.DataFrame(messy_categories)

    print("Original Messy Categorical Data:")
    print(df)

    # Clean categorical data
    df_clean = df.copy()

    def clean_category(text):
        if pd.isna(text):
            return text
        return str(text).strip().title()

    for column in df.columns:
        df_clean[column] = df_clean[column].apply(clean_category)

    print(f"\nCleaned Categorical Data:")
    print(df_clean)

    # Create standardized categories
    df_standardized = df_clean.copy()

    # Standardize size categories
    size_mapping = {
        'S': 'Small', 'Small': 'Small',
        'M': 'Medium', 'Medium': 'Medium',
        'L': 'Large', 'Large': 'Large'
    }
    df_standardized['size'] = df_standardized['size'].map(size_mapping).fillna(df_standardized['size'])

    print(f"\nStandardized Categorical Data:")
    print(df_standardized)

    return df, df_clean, df_standardized

# Demonstrate categorical cleaning
original_cat, cleaned_cat, standardized_cat = clean_categorical_data()
```

### Complete Data Cleaning Pipeline 🚀

Let's create a complete data cleaning pipeline:

```python
def complete_data_cleaning_pipeline():
    """
    Complete data cleaning pipeline - like following a recipe from start to finish
    """

    print("🍳 COMPLETE DATA CLEANING PIPELINE")
    print("=" * 50)

    # Step 1: Load messy data
    print("Step 1: Loading messy data...")

    # Create comprehensive messy dataset
    messy_data = {
        'customer_id': ['1', '2', '3', '1', '4'],        # Duplicate ID, string format
        'name': ['John Smith', 'mary johnson', 'Bob Wilson', 'John Smith', 'SARAH BROWN'],  # Inconsistent casing
        'age': ['25', '30', 'abc', '', '40'],           # Invalid age, missing values
        'income': ['50000', '75000', '60000', '50000', 'N/A'],  # Inconsistent format
        'city': ['New York', 'NEW YORK', 'new york', 'NewYork', 'Boston'],
        'signup_date': ['2025-01-01', '01/15/2025', 'March 20, 2025', '2025-01-01', '2025/04/01'],
        'is_premium': ['yes', 'No', 'YES', 'yes', 'false'],
        'purchases': [5, 12, 8, 5, 15],                # Some duplicates
        'notes': ['Great customer!', '  Always late  ', 'OK customer', 'Great customer!', 'VIP']
    }

    df = pd.DataFrame(messy_data)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample of messy data:")
    print(df.head(3))

    # Step 2: Check data quality
    print(f"\nStep 2: Data quality assessment...")
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Step 3: Remove exact duplicates
    print(f"\nStep 3: Removing exact duplicates...")
    df = df.drop_duplicates()
    print(f"Records after removing duplicates: {len(df)}")

    # Step 4: Clean customer_id
    print(f"\nStep 4: Cleaning customer_id...")
    df['customer_id'] = df['customer_id'].astype(int)

    # Step 5: Clean name
    print(f"\nStep 5: Cleaning names...")
    df['name'] = df['name'].str.strip().str.title()

    # Step 6: Clean age
    print(f"\nStep 6: Cleaning age...")
    def clean_age(age_str):
        if pd.isna(age_str) or str(age_str).strip() == '' or str(age_str) == 'abc':
            return df['age'].apply(lambda x: int(x) if str(x).isdigit() and 0 <= int(x) <= 120 else np.nan).median()
        try:
            age = int(age_str)
            return age if 0 <= age <= 120 else np.nan
        except:
            return np.nan

    df['age'] = df['age'].apply(clean_age)
    df['age'].fillna(df['age'].median(), inplace=True)

    # Step 7: Clean income
    print(f"\nStep 7: Cleaning income...")
    def clean_income(income_str):
        if str(income_str).upper() == 'N/A':
            return np.nan
        try:
            return float(str(income_str).replace(',', ''))
        except:
            return np.nan

    df['income'] = df['income'].apply(clean_income)
    df['income'].fillna(df['income'].median(), inplace=True)

    # Step 8: Clean city
    print(f"\nStep 8: Cleaning cities...")
    df['city'] = df['city'].str.replace('NewYork', 'New York').str.title()

    # Step 9: Clean dates
    print(f"\nStep 9: Cleaning dates...")
    df['signup_date'] = pd.to_datetime(df['signup_date'])

    # Step 10: Clean boolean fields
    print(f"\nStep 10: Cleaning boolean fields...")
    df['is_premium'] = df['is_premium'].str.lower().map({'yes': True, 'no': False, 'true': True, 'false': False})

    # Step 11: Clean notes
    print(f"\nStep 11: Cleaning notes...")
    df['notes'] = df['notes'].str.strip()

    # Step 12: Final quality check
    print(f"\nStep 12: Final quality assessment...")
    print(f"Final dataset shape: {df.shape}")
    print(f"Data types:")
    print(df.dtypes)
    print(f"\nFinal clean data:")
    print(df)

    return df

# Execute the complete cleaning pipeline
cleaned_data = complete_data_cleaning_pipeline()
```

### Data Preprocessing for Machine Learning 🤖

Now that we have clean data, let's prepare it for machine learning:

```python
def preprocessing_for_ml():
    """
    Preprocess data for machine learning - like preparing ingredients for cooking
    """

    print("🤖 DATA PREPROCESSING FOR MACHINE LEARNING")
    print("=" * 50)

    # Create sample cleaned dataset
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 70, 100),
        'income': np.random.randint(30000, 150000, 100),
        'experience': np.random.randint(0, 40, 100),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 100),
        'is_remote': np.random.choice([0, 1], 100),  # Binary encoding
        'salary': np.random.randint(40000, 200000, 100)  # Target variable
    }

    df = pd.DataFrame(data)

    print(f"Original dataset shape: {df.shape}")
    print(f"Original data sample:")
    print(df.head())

    # Step 1: Feature Scaling (normalize numerical features)
    print(f"\nStep 1: Feature scaling...")
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # StandardScaler (z-score normalization): mean=0, std=1
    scaler_standard = StandardScaler()
    numerical_cols = ['age', 'income', 'experience']
    df_scaled_standard = df.copy()
    df_scaled_standard[numerical_cols] = scaler_standard.fit_transform(df[numerical_cols])

    print(f"Standard scaled data (mean ≈ 0, std ≈ 1):")
    print(df_scaled_standard[numerical_cols].describe())

    # MinMaxScaler (0-1 normalization)
    scaler_minmax = MinMaxScaler()
    df_scaled_minmax = df.copy()
    df_scaled_minmax[numerical_cols] = scaler_minmax.fit_transform(df[numerical_cols])

    print(f"\nMin-Max scaled data (range 0-1):")
    print(df_scaled_minmax[numerical_cols].describe())

    # Step 2: Categorical Encoding
    print(f"\nStep 2: Categorical encoding...")

    # One-hot encoding for education_level
    df_encoded = pd.get_dummies(df, columns=['education_level'], prefix='edu')
    print(f"After one-hot encoding education_level: {df_encoded.shape}")
    print(f"New columns: {[col for col in df_encoded.columns if col.startswith('edu')]}")

    # Label encoding for city (ordinal or when order matters)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_encoded['city_encoded'] = le.fit_transform(df['city'])
    print(f"\nLabel encoded cities: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Step 3: Feature Engineering
    print(f"\nStep 3: Feature engineering...")
    df_engineered = df_encoded.copy()

    # Create new features
    df_engineered['income_per_age'] = df_engineered['income'] / df_engineered['age']
    df_engineered['experience_squared'] = df_engineered['experience'] ** 2
    df_engineered['is_senior'] = (df_engineered['experience'] >= 10).astype(int)

    print(f"After feature engineering: {df_engineered.shape}")
    print(f"New features: income_per_age, experience_squared, is_senior")

    # Step 4: Train-Test Split
    print(f"\nStep 4: Train-test split...")
    from sklearn.model_selection import train_test_split

    # Separate features and target
    X = df_engineered.drop('salary', axis=1)
    y = df_engineered['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")

    return df, df_engineered, (X_train, X_test, y_train, y_test)

# Execute ML preprocessing
original_ml, engineered_ml, splits = preprocessing_for_ml()
X_train, X_test, y_train, y_test = splits

print(f"\nPreprocessing complete!")
print(f"Final feature count: {X_train.shape[1]}")
```

### Data Cleaning Tools & Libraries 🛠️

#### Python Libraries for Data Cleaning:

```python
cleaning_tools = {
    'pandas': 'Data manipulation and cleaning (filter, group, aggregate)',
    'numpy': 'Numerical operations and array handling',
    'scikit-learn': 'Machine learning preprocessing (scaling, encoding)',
    'openpyxl': 'Read/write Excel files',
    'xlrd': 'Read old Excel files',
    'beautifulsoup4': 'Parse HTML/XML for web scraping',
    're': 'Regular expressions for text cleaning',
    'fuzzywuzzy': 'Fuzzy string matching for deduplication',
    'textblob': 'Text processing and sentiment analysis',
    'dateparser': 'Parse dates in multiple formats'
}

print("Data Cleaning Tools & Libraries:")
for tool, description in cleaning_tools.items():
    print(f"• {tool}: {description}")
```

### Summary: Data Cleaning Checklist ✅

Before you start analyzing data, make sure you've:

- [ ] **Removed duplicates** - No repeated records
- [ ] **Handled missing values** - Filled or removed appropriately
- [ ] **Fixed data types** - Numbers are numbers, dates are dates
- [ ] **Cleaned text data** - Consistent formatting, no typos
- [ ] **Standardized categories** - Same labels for same concepts
- [ ] **Detected outliers** - Investigated unusual values
- [ ] **Scaled features** - Normalized numerical data ranges
- [ ] **Encoded categories** - Converted text to numbers for ML
- [ ] **Created features** - Derived useful information from existing data
- [ ] **Split data** - Training and test sets ready

---

## Exploratory Data Analysis (EDA) {#eda}

### What is Exploratory Data Analysis? 🔍

Think of EDA like **exploring a new neighborhood** before you move there:

1. **Walk around** - Look at the data from different angles
2. **Meet the neighbors** - Understand what each variable represents
3. **Check the amenities** - See what patterns and relationships exist
4. **Look for interesting spots** - Find surprising or important insights
5. **Plan your route** - Decide how to use what you discovered

### Why Do EDA? 🤔

EDA helps you understand your data like **inspecting ingredients** before cooking:

- **Check quality** - Are there problems with the data?
- **Understand flavors** - What does each feature tell you?
- **Find the recipe** - What combinations work well together?
- **Avoid mistakes** - Spot potential problems early
- **Discover surprises** - Find unexpected but useful insights

### The EDA Process 🔄

```
1. Ask Questions → 2. Load Data → 3. Explore Structure → 4. Find Patterns → 5. Answer Questions
```

**Step 1: Ask Questions** - What do you want to learn about your data?
**Step 2: Load Data** - Import your clean dataset
**Step 3: Explore Structure** - Basic information about the data
**Step 4: Find Patterns** - Look for relationships and trends
**Step 5: Answer Questions** - Draw conclusions and insights

### Basic Data Exploration 📊

Let's start with basic exploration:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def basic_data_exploration():
    """
    Basic exploration of your dataset - like taking a tour of a new city
    """

    # Create sample dataset
    np.random.seed(42)
    data = {
        'customer_id': range(1, 1001),
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(60000, 20000, 1000),
        'education_years': np.random.randint(12, 22, 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], 1000),
        'is_premium': np.random.choice([0, 1], 1000),
        'satisfaction_score': np.random.normal(7.5, 1.5, 1000),
        'purchases_last_year': np.random.poisson(5, 1000)
    }

    df = pd.DataFrame(data)

    # Add some realistic constraints
    df['income'] = np.clip(df['income'], 20000, 200000)  # Reasonable income range
    df['satisfaction_score'] = np.clip(df['satisfaction_score'], 1, 10)  # 1-10 scale

    print("🏠 BASIC DATA EXPLORATION")
    print("=" * 50)

    # 1. Basic Information
    print("1. Dataset Overview:")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Memory usage: {df.memory_usage().sum() / 1024:.1f} KB")
    print(f"Column names: {list(df.columns)}")

    # 2. Data Types
    print(f"\n2. Data Types:")
    print(df.dtypes)

    # 3. First few rows
    print(f"\n3. Sample Data (first 5 rows):")
    print(df.head())

    # 4. Basic Statistics
    print(f"\n4. Statistical Summary:")
    print(df.describe())

    # 5. Missing Values
    print(f"\n5. Missing Values Check:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found!")

    return df

# Run basic exploration
customer_df = basic_data_exploration()
```

### Univariate Analysis 📈

**Univariate Analysis** = Looking at ONE variable at a time (like studying one ingredient)

#### Numerical Variables

```python
def univariate_numerical_analysis():
    """
    Analyze numerical variables one by one - like studying each ingredient separately
    """

    print("📊 UNIVARIATE ANALYSIS - NUMERICAL VARIABLES")
    print("=" * 50)

    numerical_cols = ['age', 'income', 'education_years', 'satisfaction_score', 'purchases_last_year']

    for col in numerical_cols:
        print(f"\n🔍 ANALYZING: {col.upper()}")
        print("-" * 30)

        # Basic statistics
        data = customer_df[col]
        print(f"Count: {len(data)}")
        print(f"Mean: {data.mean():.2f}")
        print(f"Median: {data.median():.2f}")
        print(f"Standard Deviation: {data.std():.2f}")
        print(f"Min: {data.min():.2f}")
        print(f"Max: {data.max():.2f}")

        # Quartiles
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        print(f"Q1 (25th percentile): {q1:.2f}")
        print(f"Q3 (75th percentile): {q3:.2f}")
        print(f"IQR: {iqr:.2f}")

        # Outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        print(f"Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")

        # Distribution shape
        skewness = stats.skew(data)
        print(f"Skewness: {skewness:.2f}")
        if skewness > 0.5:
            print("  → Right-skewed (tail to the right)")
        elif skewness < -0.5:
            print("  → Left-skewed (tail to the left)")
        else:
            print("  → Approximately symmetric")

# Run univariate analysis
univariate_numerical_analysis()
```

#### Categorical Variables

```python
def univariate_categorical_analysis():
    """
    Analyze categorical variables one by one - like counting different types of ingredients
    """

    print("📊 UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES")
    print("=" * 50)

    categorical_cols = ['city', 'is_premium']

    for col in categorical_cols:
        print(f"\n🔍 ANALYZING: {col.upper()}")
        print("-" * 30)

        # Value counts
        value_counts = customer_df[col].value_counts()
        print("Value Counts:")
        for value, count in value_counts.items():
            percentage = count / len(customer_df) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")

        # Unique values
        print(f"\nUnique Values: {customer_df[col].nunique()}")
        print(f"All Categories: {list(customer_df[col].unique())}")

        # Mode (most common)
        mode_value = customer_df[col].mode().iloc[0]
        mode_count = customer_df[col].value_counts().iloc[0]
        print(f"Most Common: {mode_value} ({mode_count} occurrences)")

# Run categorical analysis
univariate_categorical_analysis()
```

### Bivariate Analysis 🔗

**Bivariate Analysis** = Looking at RELATIONSHIPS between TWO variables

#### Numerical vs Numerical

```python
def numerical_vs_numerical_analysis():
    """
    Analyze relationships between numerical variables - like seeing how ingredients work together
    """

    print("🔗 BIVARIATE ANALYSIS - NUMERICAL vs NUMERICAL")
    print("=" * 50)

    numerical_cols = ['age', 'income', 'satisfaction_score', 'purchases_last_year']

    # Correlation analysis
    print("1. CORRELATION ANALYSIS")
    print("-" * 30)

    correlation_matrix = customer_df[numerical_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))

    # Find strongest correlations
    print(f"\nStrongest Positive Correlations:")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if corr_val > 0.3:  # Threshold for "strong"
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                print(f"  {var1} ↔ {var2}: {corr_val:.3f}")

    # Scatter plot analysis
    print(f"\n2. SCATTER PLOT ANALYSIS")
    print("-" * 30)

    # Age vs Income
    age_income_corr = customer_df['age'].corr(customer_df['income'])
    print(f"Age vs Income correlation: {age_income_corr:.3f}")

    # Satisfaction vs Purchases
    satisfaction_purchases_corr = customer_df['satisfaction_score'].corr(customer_df['purchases_last_year'])
    print(f"Satisfaction vs Purchases correlation: {satisfaction_purchases_corr:.3f}")

    return correlation_matrix

# Run numerical vs numerical analysis
correlation_results = numerical_vs_numerical_analysis()
```

#### Categorical vs Numerical

```python
def categorical_vs_numerical_analysis():
    """
    Analyze relationships between categorical and numerical variables
    """

    print("🔗 BIVARIATE ANALYSIS - CATEGORICAL vs NUMERICAL")
    print("=" * 50)

    # City vs Income
    print("1. CITY vs INCOME ANALYSIS")
    print("-" * 30)

    city_income_stats = customer_df.groupby('city')['income'].agg(['count', 'mean', 'median', 'std']).round(2)
    print("Income statistics by city:")
    print(city_income_stats)

    # Statistical test (ANOVA)
    from scipy.stats import f_oneway

    city_groups = [customer_df[customer_df['city'] == city]['income'] for city in customer_df['city'].unique()]
    f_stat, p_value = f_oneway(*city_groups)

    print(f"\nANOVA Test Results:")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_value:.6f}")
    print("Interpretation:", end=" ")
    if p_value < 0.05:
        print("Cities have significantly different incomes (reject null hypothesis)")
    else:
        print("No significant difference in incomes between cities (fail to reject null hypothesis)")

    # Premium vs Purchases
    print(f"\n2. PREMIUM STATUS vs PURCHASES ANALYSIS")
    print("-" * 30)

    premium_stats = customer_df.groupby('is_premium')['purchases_last_year'].agg(['count', 'mean', 'median', 'std']).round(2)
    premium_stats.index = ['Non-Premium', 'Premium']
    print("Purchase statistics by premium status:")
    print(premium_stats)

    # Statistical test (t-test)
    from scipy.stats import ttest_ind

    non_premium_purchases = customer_df[customer_df['is_premium'] == 0]['purchases_last_year']
    premium_purchases = customer_df[customer_df['is_premium'] == 1]['purchases_last_year']

    t_stat, p_value = ttest_ind(non_premium_purchases, premium_purchases)

    print(f"\nT-test Results:")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.6f}")
    print("Interpretation:", end=" ")
    if p_value < 0.05:
        print("Premium customers have significantly different purchase patterns (reject null hypothesis)")
    else:
        print("No significant difference in purchases between premium and non-premium (fail to reject null hypothesis)")

# Run categorical vs numerical analysis
categorical_vs_numerical_analysis()
```

#### Categorical vs Categorical

```python
def categorical_vs_categorical_analysis():
    """
    Analyze relationships between categorical variables
    """

    print("🔗 BIVARIATE ANALYSIS - CATEGORICAL vs CATEGORICAL")
    print("=" * 50)

    # Create crosstab
    crosstab = pd.crosstab(customer_df['city'], customer_df['is_premium'], margins=True)
    print("Crosstab: City vs Premium Status")
    print(crosstab)

    # Calculate percentages
    crosstab_pct = pd.crosstab(customer_df['city'], customer_df['is_premium'], normalize='index') * 100
    print(f"\nPercentage Distribution by City:")
    print(crosstab_pct.round(1))

    # Chi-square test
    from scipy.stats import chi2_contingency

    chi2, p_value, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])  # Exclude margins

    print(f"\nChi-square Test Results:")
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")
    print("Interpretation:", end=" ")
    if p_value < 0.05:
        print("City and premium status are significantly associated (reject null hypothesis)")
    else:
        print("No significant association between city and premium status (fail to reject null hypothesis)")

# Run categorical vs categorical analysis
categorical_vs_categorical_analysis()
```

### Advanced EDA Techniques 🚀

#### Time Series Analysis (if you have date data)

```python
def time_series_analysis():
    """
    Analyze time-based patterns - like studying how sales change over months
    """

    print("⏰ TIME SERIES ANALYSIS")
    print("=" * 50)

    # Create time series data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    np.random.seed(42)

    # Simulate sales with trends and seasonality
    trend = np.linspace(100, 150, 365)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 365)  # Yearly pattern
    noise = np.random.normal(0, 10, 365)
    sales = trend + seasonality + noise

    ts_df = pd.DataFrame({
        'date': dates,
        'sales': sales
    })

    print("1. BASIC TIME SERIES STATISTICS")
    print("-" * 30    print(f"Mean daily sales: {ts_df['sales'].mean():.2f}")
    print(f"Standard deviation: {ts_df['sales'].std():.2f}")
    print(f"Minimum sales: {ts_df['sales'].min():.2f}")
    print(f"Maximum sales: {ts_df['sales'].max():.2f}")

    # 2. Trend Analysis
    print(f"\n2. TREND ANALYSIS")
    print("-" * 30)

    # Monthly aggregation
    ts_df['month'] = ts_df['date'].dt.month
    monthly_sales = ts_df.groupby('month')['sales'].mean()

    print("Monthly average sales:")
    for month, avg_sales in monthly_sales.items():
        print(f"  Month {month}: {avg_sales:.2f}")

    # Day of week analysis
    ts_df['day_of_week'] = ts_df['date'].dt.day_name()
    dow_sales = ts_df.groupby('day_of_week')['sales'].mean()

    print(f"\nDay of week average sales:")
    for day, avg_sales in dow_sales.items():
        print(f"  {day}: {avg_sales:.2f}")

    return ts_df

# Run time series analysis
time_series_data = time_series_analysis()
```

#### Outlier Detection & Analysis

```python
def advanced_outlier_analysis():
    """
    Advanced outlier detection and analysis - like finding unusual items in your collection
    """

    print("🚨 ADVANCED OUTLIER DETECTION")
    print("=" * 50)

    # Multiple outlier detection methods
    def detect_outliers_multiple_methods(data, column):
        """
        Apply multiple outlier detection methods
        """
        outliers_summary = {}

        # Method 1: IQR method
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        iqr_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        outliers_summary['IQR'] = len(iqr_outliers)

        # Method 2: Z-score method
        z_scores = np.abs(stats.zscore(data[column]))
        zscore_outliers = data[z_scores > 2]
        outliers_summary['Z-score'] = len(zscore_outliers)

        # Method 3: Modified Z-score (MAD - Median Absolute Deviation)
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        modified_z_scores = 0.6745 * (data[column] - median) / mad
        mad_outliers = data[np.abs(modified_z_scores) > 3.5]
        outliers_summary['MAD'] = len(mad_outliers)

        # Method 4: Isolation Forest (for multivariate outliers)
        from sklearn.ensemble import IsolationForest

        features = data[['age', 'income', 'satisfaction_score']].select_dtypes(include=[np.number])
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(features)
        iso_outliers = data[outlier_labels == -1]
        outliers_summary['Isolation Forest'] = len(iso_outliers)

        return outliers_summary

    print("1. OUTLIER DETECTION RESULTS")
    print("-" * 30)

    # Apply to age column
    age_outliers = detect_outliers_multiple_methods(customer_df, 'age')
    print("Age outliers detected by different methods:")
    for method, count in age_outliers.items():
        print(f"  {method}: {count} outliers ({count/len(customer_df)*100:.1f}%)")

    # Analyze outliers
    print(f"\n2. OUTLIER ANALYSIS")
    print("-" * 30)

    # Identify age outliers using IQR method
    Q1_age = customer_df['age'].quantile(0.25)
    Q3_age = customer_df['age'].quantile(0.75)
    IQR_age = Q3_age - Q1_age
    age_outliers = customer_df[(customer_df['age'] < Q1_age - 1.5 * IQR_age) |
                              (customer_df['age'] > Q3_age + 1.5 * IQR_age)]

    print(f"Age outliers characteristics:")
    print(f"  Count: {len(age_outliers)}")
    print(f"  Age range: {age_outliers['age'].min():.0f} - {age_outliers['age'].max():.0f}")
    print(f"  Average income: ${age_outliers['income'].mean():.0f}")
    print(f"  Average satisfaction: {age_outliers['satisfaction_score'].mean():.2f}")

    # Compare outliers vs normal data
    normal_data = customer_df[~customer_df.index.isin(age_outliers.index)]
    print(f"\nComparison (outliers vs normal):")
    print(f"  Normal data average income: ${normal_data['income'].mean():.0f}")
    print(f"  Normal data average satisfaction: {normal_data['satisfaction_score'].mean():.2f}")

# Run outlier analysis
advanced_outlier_analysis()
```

### Data Visualization 📊

Visualizing data is like **creating a map** to understand the landscape:

```python
def comprehensive_data_visualization():
    """
    Create comprehensive visualizations - like creating a detailed map of your data territory
    """

    print("📊 COMPREHENSIVE DATA VISUALIZATION")
    print("=" * 50)

    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Customer Data Analysis Dashboard', fontsize=16, fontweight='bold')

    # 1. Histogram - Distribution of ages
    axes[0, 0].hist(customer_df['age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Box plot - Income by city
    city_income_data = [customer_df[customer_df['city'] == city]['income'] for city in customer_df['city'].unique()]
    axes[0, 1].boxplot(city_income_data, labels=customer_df['city'].unique())
    axes[0, 1].set_title('Income Distribution by City')
    axes[0, 1].set_xlabel('City')
    axes[0, 1].set_ylabel('Income ($)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Scatter plot - Age vs Income
    axes[0, 2].scatter(customer_df['age'], customer_df['income'], alpha=0.6, color='coral')
    axes[0, 2].set_title('Age vs Income Relationship')
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Income ($)')
    axes[0, 2].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(customer_df['age'], customer_df['income'], 1)
    p = np.poly1d(z)
    axes[0, 2].plot(customer_df['age'], p(customer_df['age']), "r--", alpha=0.8)

    # 4. Bar chart - City distribution
    city_counts = customer_df['city'].value_counts()
    axes[1, 0].bar(city_counts.index, city_counts.values, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('Customer Distribution by City')
    axes[1, 0].set_xlabel('City')
    axes[1, 0].set_ylabel('Number of Customers')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Correlation heatmap
    numerical_cols = ['age', 'income', 'education_years', 'satisfaction_score', 'purchases_last_year']
    corr_matrix = customer_df[numerical_cols].corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='RdYlBu', aspect='auto')
    axes[1, 1].set_title('Correlation Heatmap')
    axes[1, 1].set_xticks(range(len(numerical_cols)))
    axes[1, 1].set_yticks(range(len(numerical_cols)))
    axes[1, 1].set_xticklabels(numerical_cols, rotation=45)
    axes[1, 1].set_yticklabels(numerical_cols)

    # Add correlation values to heatmap
    for i in range(len(numerical_cols)):
        for j in range(len(numerical_cols)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

    # 6. Violin plot - Satisfaction by premium status
    premium_satisfaction = [customer_df[customer_df['is_premium'] == 0]['satisfaction_score'],
                           customer_df[customer_df['is_premium'] == 1]['satisfaction_score']]
    axes[1, 2].violinplot(premium_satisfaction, positions=[0, 1], showmeans=True)
    axes[1, 2].set_title('Satisfaction Score by Premium Status')
    axes[1, 2].set_xlabel('Premium Status')
    axes[1, 2].set_ylabel('Satisfaction Score')
    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_xticklabels(['Non-Premium', 'Premium'])
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/workspace/charts/customer_data_eda_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    return "Dashboard saved to charts/customer_data_eda_dashboard.png"

# Create comprehensive visualization
viz_result = comprehensive_data_visualization()
```

### Automated EDA Reports 🤖

```python
def automated_eda_report():
    """
    Generate automated EDA report - like getting a detailed inspection report for your car
    """

    print("🤖 AUTOMATED EDA REPORT GENERATION")
    print("=" * 50)

    def generate_eda_summary(df):
        """
        Generate comprehensive EDA summary
        """
        summary = {}

        # Basic info
        summary['dataset_info'] = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage().sum() / (1024 * 1024),
            'duplicates': df.duplicated().sum(),
            'missing_values': df.isnull().sum().sum()
        }

        # Numerical columns analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        summary['numerical_analysis'] = {}

        for col in numerical_cols:
            data = df[col]
            summary['numerical_analysis'][col] = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'outliers_iqr': len(data[(data < data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))) |
                                        (data > data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)))])
            }

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object']).columns
        summary['categorical_analysis'] = {}

        for col in categorical_cols:
            data = df[col]
            summary['categorical_analysis'][col] = {
                'unique_values': data.nunique(),
                'most_common': data.mode().iloc[0] if len(data.mode()) > 0 else None,
                'distribution': data.value_counts().to_dict()
            }

        # Correlation analysis
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            # Find strongest correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Threshold for notable correlation
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            summary['high_correlations'] = high_corr_pairs

        return summary

    # Generate the report
    eda_report = generate_eda_summary(customer_df)

    print("1. DATASET OVERVIEW")
    print("-" * 30)
    info = eda_report['dataset_info']
    print(f"Shape: {info['shape']}")
    print(f"Memory Usage: {info['memory_usage_mb']:.1f} MB")
    print(f"Duplicate Rows: {info['duplicates']}")
    print(f"Missing Values: {info['missing_values']}")

    print(f"\n2. NUMERICAL VARIABLES ANALYSIS")
    print("-" * 30)
    for col, stats in eda_report['numerical_analysis'].items():
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Std Dev: {stats['std']:.2f}")
        print(f"  Skewness: {stats['skewness']:.2f}")
        print(f"  Outliers (IQR method): {stats['outliers_iqr']}")

    print(f"\n3. CATEGORICAL VARIABLES ANALYSIS")
    print("-" * 30)
    for col, stats in eda_report['categorical_analysis'].items():
        print(f"\n{col}:")
        print(f"  Unique Values: {stats['unique_values']}")
        print(f"  Most Common: {stats['most_common']}")
        print(f"  Top 3 Values: {list(stats['distribution'].keys())[:3]}")

    print(f"\n4. HIGH CORRELATIONS")
    print("-" * 30)
    if 'high_correlations' in eda_report and eda_report['high_correlations']:
        for pair in eda_report['high_correlations']:
            print(f"{pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
    else:
        print("No notable correlations found (>0.3 threshold)")

    return eda_report

# Generate automated EDA report
eda_summary = automated_eda_report()
```

### Key Insights from EDA 💡

```python
def extract_key_insights():
    """
    Extract key insights from the EDA analysis
    """

    print("💡 KEY INSIGHTS FROM EDA")
    print("=" * 50)

    insights = []

    # Age insights
    age_mean = customer_df['age'].mean()
    age_std = customer_df['age'].std()
    insights.append(f"Average customer age is {age_mean:.1f} years (±{age_std:.1f})")

    # Income insights
    income_mean = customer_df['income'].mean()
    city_income_stats = customer_df.groupby('city')['income'].mean()
    highest_income_city = city_income_stats.idxmax()
    insights.append(f"{highest_income_city} has the highest average income (${city_income_stats.max():.0f})")

    # Premium insights
    premium_percentage = (customer_df['is_premium'].sum() / len(customer_df)) * 100
    insights.append(f"{premium_percentage:.1f}% of customers are premium members")

    # Satisfaction insights
    satisfaction_mean = customer_df['satisfaction_score'].mean()
    insights.append(f"Overall customer satisfaction is {satisfaction_mean:.2f}/10")

    # Purchase insights
    purchases_mean = customer_df['purchases_last_year'].mean()
    insights.append(f"Average customer makes {purchases_mean:.1f} purchases per year")

    # Correlation insights
    age_income_corr = customer_df['age'].corr(customer_df['income'])
    if abs(age_income_corr) > 0.1:
        direction = "positive" if age_income_corr > 0 else "negative"
        insights.append(f"There is a {direction} correlation between age and income ({age_income_corr:.3f})")

    print("Key Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")

    return insights

# Extract key insights
key_insights = extract_key_insights()
```

### EDA Tools & Libraries 🛠️

```python
def eda_tools_and_libraries():
    """
    Overview of EDA tools and libraries
    """

    print("🛠️ EDA TOOLS & LIBRARIES")
    print("=" * 50)

    eda_tools = {
        'Python Libraries': {
            'pandas': 'Data manipulation, grouping, aggregations',
            'numpy': 'Numerical operations and array handling',
            'matplotlib': 'Basic plotting and visualization',
            'seaborn': 'Statistical visualization and heatmaps',
            'plotly': 'Interactive plots and dashboards',
            'bokeh': 'Interactive web-based visualizations',
            'scipy': 'Statistical tests and distributions',
            'statsmodels': 'Statistical modeling and tests'
        },
        'Specialized EDA Libraries': {
            'sweetviz': 'Automated EDA report generation',
            'pandas_profiling': 'Comprehensive data profiling',
            'dtale': 'Interactive data visualization interface',
            'autoviz': 'Automatic visualization generation',
            'dataprep': 'Data preparation and cleaning toolkit'
        },
        'Cloud Platforms': {
            'Tableau': 'Business intelligence and visualization',
            'Power BI': 'Microsoft business analytics tool',
            'Google Data Studio': 'Free data visualization platform',
            'Jupyter Notebook': 'Interactive data analysis environment',
            'Kaggle Notebooks': 'Cloud-based Python notebooks'
        },
        'R Packages': {
            'ggplot2': 'Grammar of graphics plotting',
            'dplyr': 'Data manipulation and transformation',
            'tidyr': 'Data cleaning and reshaping',
            'corrplot': 'Correlation matrix visualization',
            'skimr': 'Comprehensive data summary'
        }
    }

    for category, tools in eda_tools.items():
        print(f"\n{category.upper()}:")
        for tool, description in tools.items():
            print(f"  • {tool}: {description}")

    # Quick comparison
    print(f"\n📊 EDA TOOL COMPARISON")
    print("-" * 30)

    tool_comparison = {
        'Beginner Level': ['Pandas basic functions', 'Matplotlib simple plots', 'Excel'],
        'Intermediate Level': ['Seaborn statistical plots', 'Plotly interactive charts', 'Jupyter notebooks'],
        'Advanced Level': ['Statistical testing', 'Machine learning EDA', 'Custom visualizations'],
        'Professional Level': ['Automated EDA tools', 'Large-scale data analysis', 'Interactive dashboards']
    }

    for level, tools in tool_comparison.items():
        print(f"{level}: {', '.join(tools)}")

# Show EDA tools and libraries
eda_tools_and_libraries()
```

### EDA Checklist ✅

Before moving to feature engineering, ensure you've completed:

```python
def eda_completion_checklist():
    """
    Checklist to ensure comprehensive EDA
    """

    print("✅ EDA COMPLETION CHECKLIST")
    print("=" * 50)

    checklist = {
        'Basic Exploration': [
            'Dataset shape and size assessed',
            'Data types verified and corrected',
            'Missing values identified and quantified',
            'Duplicate records detected and handled',
            'Basic statistical summary generated'
        ],
        'Univariate Analysis': [
            'Numerical variables analyzed (mean, median, std, quartiles)',
            'Categorical variables analyzed (frequencies, modes)',
            'Distributions visualized (histograms, box plots)',
            'Outliers detected using multiple methods',
            'Skewness and normality assessed'
        ],
        'Bivariate Analysis': [
            'Correlation analysis completed for numerical variables',
            'Group comparisons for categorical vs numerical variables',
            'Cross-tabulations for categorical vs categorical variables',
            'Statistical tests performed (t-test, chi-square, ANOVA)',
            'Relationships visualized (scatter plots, heatmaps)'
        ],
        'Advanced Analysis': [
            'Time series patterns identified (if applicable)',
            'Feature interactions explored',
            'Multivariate outlier detection performed',
            'Automated insights generated',
            'Key findings documented and prioritized'
        ],
        'Visualization': [
            'Appropriate chart types selected for each analysis',
            'Visualizations are clear, readable, and informative',
            'Color schemes are professional and accessible',
            'Titles and labels clearly describe the content',
            'Insights are highlighted and explained'
        ]
    }

    for section, items in checklist.items():
        print(f"\n{section.upper()}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. [ ] {item}")

    return checklist

# Show EDA checklist
eda_checklist = eda_completion_checklist()
```

---

## Feature Engineering {#feature-engineering}

### What is Feature Engineering? 🔧

Think of feature engineering like **being a chef** who creates delicious recipes:

1. **Find the right ingredients** (choose important variables)
2. **Combine them perfectly** (create new features from existing ones)
3. **Enhance the flavors** (transform data to make patterns clearer)
4. **Present beautifully** (format features for optimal use)

Feature Engineering = The art of creating the **best possible ingredients** for your AI model!

### Why Feature Engineering Matters 🎯

**Good features** are like having the **best tools** in your toolbox:

- **Faster results** - Models learn more quickly
- **Better accuracy** - More precise predictions
- **Less data needed** - Work effectively with smaller datasets
- **Interpretable results** - Understand what drives predictions
- **Robust models** - Less sensitive to noise

### Types of Feature Engineering 🔨

#### 1. **Feature Creation** - Making new features from scratch

```python
def feature_creation_examples():
    """
    Create new features from existing data - like creating new dishes from ingredients
    """

    print("🔨 FEATURE CREATION")
    print("=" * 50)

    # Create sample dataset
    np.random.seed(42)
    sample_data = {
        'customer_id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randint(30000, 150000, 100),
        'purchase_amount': np.random.uniform(10, 500, 100),
        'purchase_frequency': np.random.randint(1, 50, 100),
        'last_purchase_days': np.random.randint(1, 365, 100),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], 100),
        'referrals_made': np.random.randint(0, 20, 100)
    }

    df = pd.DataFrame(sample_data)

    print("Original Dataset (first 5 rows):")
    print(df.head())
    print(f"\nOriginal features: {list(df.columns)}")

    # Create new features
    print(f"\n1. MATHEMATICAL COMBINATIONS")
    print("-" * 30)

    # Total value of all purchases
    df['total_customer_value'] = df['purchase_amount'] * df['purchase_frequency']
    print(f"✓ total_customer_value = purchase_amount × purchase_frequency")

    # Average purchase value
    df['avg_purchase_value'] = df['purchase_amount']
    print(f"✓ avg_purchase_value = purchase_amount (already exists)")

    # Customer lifetime value proxy
    df['customer_value_proxy'] = df['total_customer_value'] / (df['last_purchase_days'] + 1)
    print(f"✓ customer_value_proxy = total_customer_value / (last_purchase_days + 1)")

    # Age categories
    def categorize_age(age):
        if age < 25:
            return 'Young'
        elif age < 50:
            return 'Middle-aged'
        else:
            return 'Senior'

    df['age_category'] = df['age'].apply(categorize_age)
    print(f"✓ age_category = categorized age (Young/Middle-aged/Senior)")

    print(f"\n2. TIME-BASED FEATURES")
    print("-" * 30)

    # Purchase recency
    df['is_recent_customer'] = (df['last_purchase_days'] <= 30).astype(int)
    print(f"✓ is_recent_customer = last_purchase_days ≤ 30 (binary)")

    # Customer engagement level
    df['engagement_level'] = pd.cut(df['purchase_frequency'],
                                   bins=[0, 5, 15, 50],
                                   labels=['Low', 'Medium', 'High'])
    print(f"✓ engagement_level = categorized purchase_frequency")

    print(f"\n3. RATIO AND RATE FEATURES")
    print("-" * 30)

    # Spending rate (income per year spent on purchases)
    df['spending_rate'] = df['total_customer_value'] / (df['income'] / 12)  # Monthly income
    print(f"✓ spending_rate = total_customer_value / (income / 12)")

    # Referral effectiveness
    df['referral_effectiveness'] = df['referrals_made'] / (df['purchase_frequency'] + 1)
    print(f"✓ referral_effectiveness = referrals_made / (purchase_frequency + 1)")

    print(f"\nNew Dataset (first 5 rows):")
    print(df.head())
    print(f"\nTotal features: {len(df.columns)}")

    return df

# Run feature creation examples
enhanced_data = feature_creation_examples()
```

#### 2. **Feature Transformation** - Changing the shape or scale of features

```python
def feature_transformation_examples():
    """
    Transform existing features - like changing the temperature of ingredients
    """

    print("🔄 FEATURE TRANSFORMATION")
    print("=" * 50)

    # Create skewed data for demonstration
    np.random.seed(42)
    skewed_data = {
        'income': np.random.lognormal(mean=10, sigma=1, size=1000),  # Log-normal distribution
        'purchase_amount': np.random.exponential(scale=100, size=1000),  # Exponential distribution
        'customer_calls': np.random.poisson(lam=5, size=1000)  # Poisson distribution
    }

    df = pd.DataFrame(skewed_data)

    print("Original Skewed Data:")
    print(f"Income skewness: {stats.skew(df['income']):.3f}")
    print(f"Purchase amount skewness: {stats.skew(df['purchase_amount']):.3f}")
    print(f"Customer calls skewness: {stats.skew(df['customer_calls']):.3f}")

    print(f"\n1. LOG TRANSFORMATION")
    print("-" * 30)

    # Log transformation for highly skewed data
    df['income_log'] = np.log1p(df['income'])  # log1p = log(1+x) to handle zeros
    df['purchase_amount_log'] = np.log1p(df['purchase_amount'])

    print(f"✓ income_log = log(income + 1)")
    print(f"✓ purchase_amount_log = log(purchase_amount + 1)")
    print(f"Income log skewness: {stats.skew(df['income_log']):.3f}")
    print(f"Purchase log skewness: {stats.skew(df['purchase_amount_log']):.3f}")

    print(f"\n2. SQUARE ROOT TRANSFORMATION")
    print("-" * 30)

    # Square root transformation
    df['customer_calls_sqrt'] = np.sqrt(df['customer_calls'])

    print(f"✓ customer_calls_sqrt = sqrt(customer_calls)")
    print(f"Customer calls sqrt skewness: {stats.skew(df['customer_calls_sqrt']):.3f}")

    print(f"\n3. STANDARDIZATION (Z-SCORE)")
    print("-" * 30)

    # Standardization
    scaler = StandardScaler()
    income_standardized = scaler.fit_transform(df[['income']])
    df['income_standardized'] = income_standardized.flatten()

    print(f"✓ income_standardized = (income - mean) / std")
    print(f"Income standardized mean: {df['income_standardized'].mean():.3f}")
    print(f"Income standardized std: {df['income_standardized'].std():.3f}")

    print(f"\n4. NORMALIZATION (0-1 SCALE)")
    print("-" * 30)

    # Min-Max normalization
    min_max_scaler = MinMaxScaler()
    purchase_normalized = min_max_scaler.fit_transform(df[['purchase_amount']])
    df['purchase_amount_normalized'] = purchase_normalized.flatten()

    print(f"✓ purchase_amount_normalized = (value - min) / (max - min)")
    print(f"Purchase normalized range: [{df['purchase_amount_normalized'].min():.3f}, {df['purchase_amount_normalized'].max():.3f}]")

    print(f"\nTransformation Results Summary:")
    print(df[['income', 'income_log', 'income_standardized', 'purchase_amount', 'purchase_amount_log', 'purchase_amount_normalized']].describe().round(3))

    return df

# Run transformation examples
transformed_data = feature_transformation_examples()
```

#### 3. **Feature Encoding** - Converting categorical to numerical

```python
def feature_encoding_examples():
    """
    Encode categorical features - like translating different languages to one language
    """

    print("🔤 FEATURE ENCODING")
    print("=" * 50)

    # Create sample categorical data
    categorical_data = {
        'customer_id': range(1, 11),
        'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
        'product_category': ['Electronics', 'Clothing', 'Electronics', 'Food', 'Electronics',
                           'Clothing', 'Food', 'Electronics', 'Clothing', 'Food'],
        'subscription_tier': ['Basic', 'Premium', 'Enterprise', 'Basic', 'Premium',
                            'Enterprise', 'Basic', 'Premium', 'Enterprise', 'Basic'],
        'rating': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # Ordinal categories
    }

    df = pd.DataFrame(categorical_data)

    print("Original Categorical Data:")
    print(df)

    print(f"\n1. ONE-HOT ENCODING")
    print("-" * 30)

    # One-hot encoding for nominal categories
    df_onehot = pd.get_dummies(df, columns=['city', 'product_category'], prefix=['city', 'category'])
    print(f"✓ One-hot encoded city and product_category")
    print(f"New columns: {[col for col in df_onehot.columns if col.startswith(('city_', 'category_'))]}")

    print(f"\n2. LABEL ENCODING")
    print("-" * 30)

    # Label encoding for ordinal categories
    from sklearn.preprocessing import LabelEncoder

    le_tier = LabelEncoder()
    df['subscription_tier_encoded'] = le_tier.fit_transform(df['subscription_tier'])

    # Create mapping for interpretation
    tier_mapping = dict(zip(le_tier.classes_, le_tier.transform(le_tier.classes_)))
    print(f"✓ subscription_tier_encoded = label encoded")
    print(f"Mapping: {tier_mapping}")

    print(f"\n3. ORDINAL ENCODING")
    print("-" * 30)

    # Ordinal encoding for ordered categories
    rating_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    df['rating_ordinal'] = df['rating'].map(rating_mapping)
    print(f"✓ rating_ordinal = ordinal encoded (1→1, 2→2, etc.)")

    # Custom ordinal for subscription tiers
    tier_order = {'Basic': 1, 'Premium': 2, 'Enterprise': 3}
    df['subscription_tier_ordinal'] = df['subscription_tier'].map(tier_order)
    print(f"✓ subscription_tier_ordinal = custom ordinal encoded")

    print(f"\n4. TARGET ENCODING")
    print("-" * 30)

    # Create target variable for demonstration
    np.random.seed(42)
    df['purchase_amount'] = np.random.uniform(50, 500, len(df))

    # Simple target encoding (mean encoding)
    city_target_mean = df.groupby('city')['purchase_amount'].mean()
    df['city_target_encoded'] = df['city'].map(city_target_mean)
    print(f"✓ city_target_encoded = average purchase_amount by city")
    print(f"City encoding values: {city_target_mean.round(2).to_dict()}")

    print(f"\nEncoded Dataset:")
    print(df[['customer_id', 'city', 'city_target_encoded', 'subscription_tier', 'subscription_tier_encoded', 'rating', 'purchase_amount']].head())

    return df, city_target_mean

# Run encoding examples
encoded_data, city_encoding = feature_encoding_examples()
```

#### 4. **Feature Selection** - Choosing the most important features

```python
def feature_selection_examples():
    """
    Select the most important features - like choosing the best ingredients for your recipe
    """

    print("🎯 FEATURE SELECTION")
    print("=" * 50)

    # Create dataset with many features (some useful, some not)
    np.random.seed(42)
    n_samples = 1000

    # Useful features (high correlation with target)
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.normal(0, 1, n_samples)

    # Create target with some useful features
    target = 2*feature1 + 3*feature2 + np.random.normal(0, 0.1, n_samples)

    # Less useful features
    feature4 = np.random.normal(0, 1, n_samples)  # Low correlation
    feature5 = np.random.normal(0, 1, n_samples)  # Noisy
    feature6 = np.random.normal(0, 1, n_samples)  # Noisy

    # Create DataFrame
    feature_data = {
        'useful_feature_1': feature1,
        'useful_feature_2': feature2,
        'useful_feature_3': feature3,
        'less_useful_4': feature4,
        'noisy_feature_5': feature5,
        'noise_feature_6': feature6,
        'target': target
    }

    df = pd.DataFrame(feature_data)

    print("Dataset for Feature Selection:")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")

    X = df.drop('target', axis=1)
    y = df['target']

    print(f"\n1. CORRELATION-BASED SELECTION")
    print("-" * 30)

    # Calculate correlation with target
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print("Feature correlations with target:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.3f}")

    # Select top correlated features
    top_corr_features = correlations.head(3).index.tolist()
    print(f"\n✓ Selected top 3 correlated features: {top_corr_features}")

    print(f"\n2. VARIANCE THRESHOLD")
    print("-" * 30)

    # Remove low variance features
    from sklearn.feature_selection import VarianceThreshold

    selector = VarianceThreshold(threshold=0.1)
    X_variance = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()]

    print(f"Features after variance threshold (>0.1): {list(selected_features)}")
    print(f"✓ Variance threshold removed: {list(X.columns[~selector.get_support()])}")

    print(f"\n3. UNIVARIATE STATISTICAL TESTS")
    print("-" * 30)

    # F-test for regression
    from sklearn.feature_selection import SelectKBest, f_regression

    selector_f = SelectKBest(score_func=f_regression, k=3)
    X_f_selected = selector_f.fit_transform(X, y)
    selected_f_features = X.columns[selector_f.get_support()]

    print(f"F-test scores:")
    feature_scores = dict(zip(X.columns, selector_f.scores_))
    for feature, score in sorted(feature_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {score:.2f}")

    print(f"\n✓ Selected features by F-test: {list(selected_f_features)}")

    print(f"\n4. RECURSIVE FEATURE ELIMINATION")
    print("-" * 30)

    # RFE with linear regression
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression

    estimator = LinearRegression()
    selector_rfe = RFE(estimator, n_features_to_select=3)
    X_rfe_selected = selector_rfe.fit_transform(X, y)
    selected_rfe_features = X.columns[selector_rfe.get_support()]

    print(f"✓ Selected features by RFE: {list(selected_rfe_features)}")

    print(f"\nFEATURE SELECTION COMPARISON")
    print("-" * 30)

    methods_comparison = {
        'Correlation-based': top_corr_features,
        'Variance threshold': list(selected_features),
        'F-test univariate': list(selected_f_features),
        'RFE': list(selected_rfe_features)
    }

    for method, features in methods_comparison.items():
        print(f"{method}: {features}")

    return X, y, methods_comparison

# Run feature selection examples
X_features, y_target, selection_comparison = feature_selection_examples()
```

#### 5. **Dimensionality Reduction** - Simplifying complex data

```python
def dimensionality_reduction_examples():
    """
    Reduce feature dimensions while preserving information - like creating a map of your city
    """

    print("🗺️ DIMENSIONALITY REDUCTION")
    print("=" * 50)

    # Create high-dimensional dataset
    np.random.seed(42)
    n_samples = 200

    # Create correlated features (many dimensions with redundancy)
    base_data = np.random.normal(0, 1, (n_samples, 3))

    # Add correlated features
    feature_matrix = np.column_stack([
        base_data,  # Original features
        base_data + np.random.normal(0, 0.1, base_data.shape),  # Noisy versions
        base_data * 2,  # Scaled versions
        base_data + 0.5  # Shifted versions
    ])

    feature_names = [f'feature_{i}' for i in range(12)]
    df_high_dim = pd.DataFrame(feature_matrix, columns=feature_names)

    print("High-dimensional Dataset:")
    print(f"Shape: {df_high_dim.shape}")
    print(f"First 5 rows:")
    print(df_high_dim.head())

    print(f"\n1. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 30)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize data first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_high_dim)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("PCA Results:")
    for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio[:5], cumulative_variance[:5])):
        print(f"  PC{i+1}: {var_ratio:.3f} ({cum_var:.3f} cumulative)")

    # Choose components that explain 95% of variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\n✓ Need {n_components_95} components to explain 95% of variance")

    # Transform data with optimal components
    pca_optimal = PCA(n_components=n_components_95)
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)

    print(f"✓ Reduced from {df_high_dim.shape[1]} to {n_components_95} dimensions")

    print(f"\n2. TSNE (FOR VISUALIZATION)")
    print("-" * 30)

    from sklearn.manifold import TSNE

    # Apply t-SNE for visualization (only for small datasets)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)

    print(f"✓ t-SNE reduced to 2 dimensions for visualization")
    print(f"Original shape: {X_scaled.shape}")
    print(f"t-SNE shape: {X_tsne.shape}")

    print(f"\n3. UMAP (ALTERNATIVE TO TSNE)")
    print("-" * 30)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)

        print(f"✓ UMAP reduced to 2 dimensions")
        print(f"UMAP shape: {X_umap.shape}")
    except ImportError:
        print("UMAP not available (install with: pip install umap-learn)")

    print(f"\nDIMENSIONALITY REDUCTION COMPARISON")
    print("-" * 30)

    reduction_comparison = {
        'Original': df_high_dim.shape[1],
        'PCA (95% variance)': n_components_95,
        't-SNE (visualization)': 2,
        'UMAP (visualization)': 2
    }

    for method, dims in reduction_comparison.items():
        print(f"{method}: {dims} dimensions")

    return df_high_dim, X_pca_optimal, X_tsne, reduction_comparison

# Run dimensionality reduction examples
original_dims, pca_result, tsne_result, dim_comparison = dimensionality_reduction_examples()
```

### Advanced Feature Engineering Techniques 🚀

#### Time Series Feature Engineering

```python
def time_series_feature_engineering():
    """
    Create features for time series data - like studying patterns in a heartbeat
    """

    print("⏰ TIME SERIES FEATURE ENGINEERING")
    print("=" * 50)

    # Create time series data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    np.random.seed(42)

    # Simulate sales data with trends and seasonality
    trend = np.linspace(100, 150, 365)
    seasonality = 20 * np.sin(2 * np.pi * np.arange(365) / 365)  # Yearly pattern
    noise = np.random.normal(0, 10, 365)
    sales = trend + seasonality + noise

    ts_df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'weather_score': np.random.uniform(0, 10, 365),  # Weather impact
        'promotion': np.random.choice([0, 1], 365, p=[0.8, 0.2])  # Promotion flag
    })

    print("Original Time Series Data (first 5 rows):")
    print(ts_df.head())

    print(f"\n1. DATE-TIME FEATURES")
    print("-" * 30)

    # Extract datetime components
    ts_df['year'] = ts_df['date'].dt.year
    ts_df['month'] = ts_df['date'].dt.month
    ts_df['day'] = ts_df['date'].dt.day
    ts_df['day_of_week'] = ts_df['date'].dt.dayofweek
    ts_df['day_of_year'] = ts_df['date'].dt.dayofyear
    ts_df['week_of_year'] = ts_df['date'].dt.isocalendar().week
    ts_df['is_weekend'] = ts_df['date'].dt.weekday >= 5
    ts_df['is_month_start'] = ts_df['date'].dt.is_month_start
    ts_df['is_month_end'] = ts_df['date'].dt.is_month_end
    ts_df['quarter'] = ts_df['date'].dt.quarter

    print(f"✓ Added datetime components (year, month, day, etc.)")

    print(f"\n2. LAG FEATURES")
    print("-" * 30)

    # Create lag features (previous values)
    ts_df['sales_lag_1'] = ts_df['sales'].shift(1)  # Previous day
    ts_df['sales_lag_7'] = ts_df['sales'].shift(7)  # Same day last week
    ts_df['sales_lag_30'] = ts_df['sales'].shift(30)  # Same day last month

    print(f"✓ Added lag features (1, 7, 30 days)")

    print(f"\n3. ROLLING WINDOW FEATURES")
    print("-" * 30)

    # Rolling statistics
    ts_df['sales_rolling_7_mean'] = ts_df['sales'].rolling(window=7, min_periods=1).mean()
    ts_df['sales_rolling_7_std'] = ts_df['sales'].rolling(window=7, min_periods=1).std()
    ts_df['sales_rolling_30_mean'] = ts_df['sales'].rolling(window=30, min_periods=1).mean()
    ts_df['sales_rolling_30_max'] = ts_df['sales'].rolling(window=30, min_periods=1).max()
    ts_df['sales_rolling_30_min'] = ts_df['sales'].rolling(window=30, min_periods=1).min()

    print(f"✓ Added rolling window statistics (7, 30 day windows)")

    print(f"\n4. DIFFERENCING FEATURES")
    print("-" * 30)

    # Differencing (rate of change)
    ts_df['sales_diff_1'] = ts_df['sales'].diff(1)  # Day-to-day change
    ts_df['sales_diff_7'] = ts_df['sales'].diff(7)  # Week-to-week change
    ts_df['sales_pct_change_1'] = ts_df['sales'].pct_change(1)  # Percentage change

    print(f"✓ Added differencing features (1-day, 7-day, percentage)")

    print(f"\n5. EXPONENTIAL SMOOTHING FEATURES")
    print("-" * 30)

    # Exponential weighted moving average
    ts_df['sales_ewm_7'] = ts_df['sales'].ewm(span=7).mean()
    ts_df['sales_ewm_30'] = ts_df['sales'].ewm(span=30).mean()

    print(f"✓ Added exponential smoothing features (7, 30 day spans)")

    print(f"\n6. CYCLICAL ENCODING")
    print("-" * 30)

    # Cyclical encoding for periodic features
    ts_df['month_sin'] = np.sin(2 * np.pi * ts_df['month'] / 12)
    ts_df['month_cos'] = np.cos(2 * np.pi * ts_df['month'] / 12)
    ts_df['day_of_week_sin'] = np.sin(2 * np.pi * ts_df['day_of_week'] / 7)
    ts_df['day_of_week_cos'] = np.cos(2 * np.pi * ts_df['day_of_week'] / 7)
    ts_df['day_of_year_sin'] = np.sin(2 * np.pi * ts_df['day_of_year'] / 365)
    ts_df['day_of_year_cos'] = np.cos(2 * np.pi * ts_df['day_of_year'] / 365)

    print(f"✓ Added cyclical encoding (sin/cos transformation)")

    print(f"\n7. INTERACTION FEATURES")
    print("-" * 30)

    # Interaction between features
    ts_df['weather_promotion_interaction'] = ts_df['weather_score'] * ts_df['promotion']
    ts_df['weekend_month_interaction'] = ts_df['is_weekend'].astype(int) * ts_df['month']

    print(f"✓ Added interaction features")

    print(f"\nTime Series Feature Engineering Results:")
    print(f"Original features: 4")
    print(f"Total features after engineering: {len(ts_df.columns)}")
    print(f"New features added: {len(ts_df.columns) - 4}")

    print(f"\nSample of engineered features:")
    feature_cols = [col for col in ts_df.columns if col not in ['date', 'sales']]
    print(ts_df[feature_cols[:10]].head())

    return ts_df

# Run time series feature engineering
ts_engineered = time_series_feature_engineering()
```

#### Text Feature Engineering

```python
def text_feature_engineering():
    """
    Create features from text data - like extracting meaning from conversations
    """

    print("📝 TEXT FEATURE ENGINEERING")
    print("=" * 50)

    # Create sample text data
    text_data = {
        'customer_id': range(1, 11),
        'review_text': [
            "This product is absolutely amazing! Great quality and fast shipping.",
            "Terrible experience. Product broke after one day. Very disappointed.",
            "Good value for money. Does what it says. Would recommend to friends.",
            "Outstanding customer service. They helped me resolve my issue quickly.",
            "Poor quality control. Multiple defects. Not worth the price.",
            "Excellent build quality. Feels premium. Highly satisfied with purchase.",
            "Average product. Nothing special but does the job adequately.",
            "Fantastic design and functionality. exceeded my expectations completely.",
            "Disappointing performance. Doesn't work as advertised at all.",
            "Solid product with minor issues. Overall decent experience."
        ],
        'rating': [5, 1, 4, 5, 2, 5, 3, 5, 1, 3]
    }

    df = pd.DataFrame(text_data)

    print("Original Text Data (first 3 rows):")
    for i in range(3):
        print(f"Review {i+1}: {df.iloc[i]['review_text'][:60]}...")
        print(f"Rating: {df.iloc[i]['rating']}")
        print()

    print(f"\n1. BASIC TEXT STATISTICS")
    print("-" * 30)

    # Basic text statistics
    df['text_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()
    df['sentence_count'] = df['review_text'].str.count(r'[.!?]+')
    df['avg_word_length'] = df['review_text'].str.replace(r'[^\w\s]', '', regex=True).str.split().apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0
    )

    print(f"✓ Added text statistics (length, word count, sentences, avg word length)")

    print(f"\n2. SENTIMENT FEATURES")
    print("-" * 30)

    # Simple sentiment analysis
    positive_words = ['amazing', 'great', 'excellent', 'fantastic', 'outstanding', 'good', 'recommend', 'satisfied']
    negative_words = ['terrible', 'poor', 'disappointed', 'defects', 'disappointing', 'doesn\'t', 'broke']

    df['positive_word_count'] = df['review_text'].apply(
        lambda text: sum(1 for word in positive_words if word.lower() in text.lower())
    )
    df['negative_word_count'] = df['review_text'].apply(
        lambda text: sum(1 for word in negative_words if word.lower() in text.lower())
    )
    df['sentiment_score'] = df['positive_word_count'] - df['negative_word_count']
    df['has_positive_words'] = df['positive_word_count'] > 0
    df['has_negative_words'] = df['negative_word_count'] > 0

    print(f"✓ Added sentiment features (positive/negative word counts, sentiment score)")

    print(f"\n3. N-GRAM FEATURES")
    print("-" * 30)

    # Extract specific n-grams
    def extract_bigrams(text):
        words = text.lower().split()
        return [' '.join(words[i:i+2]) for i in range(len(words)-1)]

    df['bigrams'] = df['review_text'].apply(extract_bigrams)

    # Find most common bigrams across all reviews
    all_bigrams = []
    for bigrams in df['bigrams']:
        all_bigrams.extend(bigrams)

    from collections import Counter
    bigram_counts = Counter(all_bigrams)
    top_bigrams = bigram_counts.most_common(5)

    print(f"Top 5 bigrams: {top_bigrams}")

    # Create binary features for top bigrams
    for bigram, _ in top_bigrams:
        df[f'has_{bigram.replace(" ", "_")}'] = df['bigrams'].apply(
            lambda x: 1 if bigram in x else 0
        )

    print(f"✓ Added bigram features for top 5 bigrams")

    print(f"\n4. KEYWORD EXTRACTION")
    print("-" * 30)

    # Extract keywords based on ratings
    high_rated_reviews = df[df['rating'] >= 4]['review_text'].str.lower().str.split()
    low_rated_reviews = df[df['rating'] <= 2]['review_text'].str.lower().str.split()

    # Find important words in high-rated reviews
    high_rated_words = []
    for words in high_rated_reviews:
        high_rated_words.extend([word for word in words if len(word) > 3])

    high_rated_word_counts = Counter(high_rated_words)
    important_high_words = [word for word, count in high_rated_word_counts.most_common(5)]

    print(f"Important words in high-rated reviews: {important_high_words}")

    # Create features for important words
    for word in important_high_words:
        df[f'contains_{word}'] = df['review_text'].str.lower().str.contains(word).astype(int)

    print(f"✓ Added keyword presence features")

    print(f"\n5. LINGUISTIC FEATURES")
    print("-" * 30)

    # Punctuation usage
    df['exclamation_count'] = df['review_text'].str.count('!')
    df['question_count'] = df['review_text'].str.count(r'\?')
    df['capital_ratio'] = df['review_text'].apply(
        lambda text: sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    )

    # Verb/adjective indicators
    import re
    df['has_adjectives'] = df['review_text'].str.contains(r'\b(amazing|great|excellent|fantastic|poor|terrible|good|solid)\b',
                                                         case=False, na=False).astype(int)
    df['has_adverbs'] = df['review_text'].str.contains(r'\b(very|really|absolutely|completely|highly|quickly|easily)\b',
                                                     case=False, na=False).astype(int)

    print(f"✓ Added linguistic features (punctuation, capitalization, word types)")

    print(f"\nText Feature Engineering Results:")
    print(f"Original features: 3")
    print(f"Total features after engineering: {len(df.columns)}")
    print(f"New text features added: {len(df.columns) - 3}")

    print(f"\nSample of engineered text features:")
    text_feature_cols = [col for col in df.columns if col not in ['customer_id', 'review_text', 'bigrams']]
    print(df[text_feature_cols].head())

    return df

# Run text feature engineering
text_engineered = text_feature_engineering()
```

### Feature Engineering Pipeline 🚀

Let's create a complete feature engineering pipeline:

```python
def complete_feature_engineering_pipeline():
    """
    Complete feature engineering pipeline - like following a comprehensive cooking recipe
    """

    print("🚀 COMPLETE FEATURE ENGINEERING PIPELINE")
    print("=" * 50)

    # Create comprehensive dataset
    np.random.seed(42)
    n_samples = 1000

    # Generate diverse data types
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(60000, 20000, n_samples),
        'purchase_amount': np.random.exponential(100, n_samples),
        'purchase_frequency': np.random.poisson(5, n_samples),
        'days_since_last_purchase': np.random.randint(1, 365, n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_samples),
        'customer_rating': np.random.uniform(1, 5, n_samples),
        'referrals_made': np.random.poisson(2, n_samples)
    }

    df = pd.DataFrame(data)
    df['income'] = np.clip(df['income'], 20000, 200000)  # Reasonable income range

    print(f"Original Dataset: {df.shape}")

    # Step 1: Clean and prepare data
    print(f"\nStep 1: Data Preparation")
    print("-" * 30)

    # Handle any missing values (fill with median/mode)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    print(f"✓ Data cleaned and prepared")

    # Step 2: Create mathematical combinations
    print(f"\nStep 2: Mathematical Features")
    print("-" * 30)

    df['total_spent'] = df['purchase_amount'] * df['purchase_frequency']
    df['customer_lifetime_value'] = df['total_spent'] / (df['days_since_last_purchase'] + 1)
    df['income_to_age_ratio'] = df['income'] / df['age']
    df['spending_rate'] = df['total_spent'] / (df['income'] / 12)  # Monthly spending rate

    print(f"✓ Created mathematical features")

    # Step 3: Categorical encoding
    print(f"\nStep 3: Categorical Encoding")
    print("-" * 30)

    # One-hot encoding for city
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    df = pd.concat([df, city_dummies], axis=1)

    # Ordinal encoding for subscription type
    subscription_mapping = {'Basic': 1, 'Premium': 2, 'Enterprise': 3}
    df['subscription_tier_encoded'] = df['subscription_type'].map(subscription_mapping)

    print(f"✓ Encoded categorical variables")

    # Step 4: Binning and categorization
    print(f"\nStep 4: Binning and Categorization")
    print("-" * 30)

    # Age bins
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 40, 60, 100], labels=['Young', 'Adult', 'Middle-aged', 'Senior'])

    # Income bins
    df['income_level'] = pd.qcut(df['income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    # Purchase frequency bins
    df['frequency_category'] = pd.cut(df['purchase_frequency'], bins=[0, 2, 5, 10, 100], labels=['Low', 'Medium', 'High', 'Very High'])

    print(f"✓ Created binned features")

    # Step 5: Time-based features
    print(f"\nStep 5: Time-based Features")
    print("-" * 30)

    df['is_recent_customer'] = (df['days_since_last_purchase'] <= 30).astype(int)
    df['is_active_customer'] = (df['purchase_frequency'] >= 5).astype(int)
    df['customer_tenure_proxy'] = df['age'] - 18  # Assuming customer started at 18

    print(f"✓ Created time-based features")

    # Step 6: Interaction features
    print(f"\nStep 6: Interaction Features")
    print("-" * 30)

    df['age_income_interaction'] = df['age'] * df['income']
    df['subscription_rating_interaction'] = df['subscription_tier_encoded'] * df['customer_rating']
    df['frequency_spending_interaction'] = df['purchase_frequency'] * df['spending_rate']

    print(f"✓ Created interaction features")

    # Step 7: Statistical transformations
    print(f"\nStep 7: Statistical Transformations")
    print("-" * 30)

    # Log transformation for skewed variables
    df['log_purchase_amount'] = np.log1p(df['purchase_amount'])
    df['log_income'] = np.log1p(df['income'])

    # Standardization for machine learning
    scaler = StandardScaler()
    numerical_features = ['age', 'income', 'purchase_amount', 'total_spent', 'customer_lifetime_value']
    df[numerical_features + ['log_purchase_amount', 'log_income']] = scaler.fit_transform(
        df[numerical_features + ['log_purchase_amount', 'log_income']]
    )

    print(f"✓ Applied statistical transformations")

    # Step 8: Feature selection
    print(f"\nStep 8: Feature Selection")
    print("-" * 30)

    # Remove highly correlated features
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    print(f"Found {len(high_corr_pairs)} highly correlated feature pairs:")
    for var1, var2, corr in high_corr_pairs[:5]:  # Show first 5
        print(f"  {var1} ↔ {var2}: {corr:.3f}")

    # Remove one feature from each highly correlated pair
    features_to_remove = set()
    for var1, var2, corr in high_corr_pairs:
        if var1 not in features_to_remove:
            features_to_remove.add(var2)  # Remove var2, keep var1

    df_final = df.drop(columns=list(features_to_remove))

    print(f"✓ Removed {len(features_to_remove)} highly correlated features")

    print(f"\nPipeline Results:")
    print(f"Original features: {df.shape[1] - len(features_to_remove)}")
    print(f"Final features: {df_final.shape[1]}")
    print(f"Removed features: {list(features_to_remove)}")

    print(f"\nFinal Feature List:")
    feature_categories = {
        'Original': [col for col in df_final.columns if col in data.keys()],
        'Mathematical': [col for col in df_final.columns if 'total_' in col or 'ratio' in col or 'rate' in col],
        'Encoded': [col for col in df_final.columns if col.startswith(('city_', 'subscription_'))],
        'Binned': [col for col in df_final.columns if col.endswith('_group') or col.endswith('_level')],
        'Interaction': [col for col in df_final.columns if 'interaction' in col],
        'Transformed': [col for col in df_final.columns if col.startswith('log_')]
    }

    for category, features in feature_categories.items():
        if features:
            print(f"{category}: {features}")

    return df_final

# Execute complete feature engineering pipeline
final_features = complete_feature_engineering_pipeline()
```

### Feature Engineering Best Practices ✅

```python
def feature_engineering_best_practices():
    """
    Best practices for feature engineering - like cooking techniques that always work
    """

    print("✅ FEATURE ENGINEERING BEST PRACTICES")
    print("=" * 50)

    best_practices = {
        'Domain Knowledge': {
            'principle': 'Understand your business problem deeply',
            'examples': [
                'E-commerce: Create features like cart abandonment rate, time to purchase',
                'Finance: Create features like debt-to-income ratio, credit utilization',
                'Healthcare: Create features like BMI categories, risk scores',
                'Marketing: Create features like customer lifetime value, engagement scores'
            ],
            'tips': [
                'Talk to domain experts',
                'Research industry-specific metrics',
                'Understand the business context',
                'Focus on actionable insights'
            ]
        },
        'Data Quality': {
            'principle': 'Only create features from clean, reliable data',
            'examples': [
                'Handle missing values before creating features',
                'Remove outliers that don\'t represent reality',
                'Validate feature calculations with business logic',
                'Check for data leakage (future information)'
            ],
            'tips': [
                'Always validate feature logic',
                'Test features on edge cases',
                'Monitor feature quality over time',
                'Document feature creation process'
            ]
        },
        'Feature Selection': {
            'principle': 'More features ≠ better performance',
            'examples': [
                'Curse of dimensionality: too many features can hurt performance',
                'Irrelevant features can confuse the model',
                'Correlated features provide redundant information',
                'Some features may be data artifacts'
            ],
            'tips': [
                'Use feature importance scores',
                'Apply statistical tests for relevance',
                'Consider model interpretability',
                'Monitor for overfitting with too many features'
            ]
        },
        'Transformation Strategy': {
            'principle': 'Apply appropriate transformations for each data type',
            'examples': [
                'Log transformation for skewed numerical data',
                'Standardization for features with different scales',
                'Cyclical encoding for time-based features',
                'Target encoding for high-cardinality categories'
            ],
            'tips': [
                'Visualize distributions before transforming',
                'Apply transformations consistently across train/test',
                'Consider the effect on model interpretability',
                'Test multiple transformation strategies'
            ]
        },
        'Validation': {
            'principle': 'Always validate your feature engineering choices',
            'examples': [
                'Cross-validation to test feature utility',
                'A/B testing for business impact',
                'Statistical tests for feature significance',
                'Model performance comparison with/without features'
            ],
            'tips': [
                'Use proper train/validation splits',
                'Test features in different scenarios',
                'Monitor feature drift over time',
                'Get feedback from end users'
            ]
        }
    }

    for category, details in best_practices.items():
        print(f"\n{category.upper()}")
        print(f"Principle: {details['principle']}")
        print(f"Examples:")
        for example in details['examples']:
            print(f"  • {example}")
        print(f"Tips:")
        for tip in details['tips']:
            print(f"  ✓ {tip}")

    return best_practices

# Show best practices
practices = feature_engineering_best_practices()
```

### Feature Engineering Tools & Libraries 🛠️

```python
def feature_engineering_tools():
    """
    Overview of feature engineering tools and libraries
    """

    print("🛠️ FEATURE ENGINEERING TOOLS & LIBRARIES")
    print("=" * 50)

    tools = {
        'Feature Engineering Libraries': {
            'scikit-learn': 'General ML preprocessing (scaling, encoding, selection)',
            'category_encoders': 'Specialized categorical encoding methods',
            'feature-engine': 'End-to-end feature engineering framework',
            'tsfresh': 'Time series feature extraction',
            'featuretools': 'Automated feature engineering for relational data',
            'pyuss': 'Statistical and mathematical feature creation',
            'scikit-optimize': 'Hyperparameter optimization for feature selection'
        },
        'Text Processing': {
            'nltk': 'Natural language processing and text analysis',
            'spacy': 'Industrial-strength NLP with pre-trained models',
            'gensim': 'Topic modeling and document similarity',
            'textblob': 'Simple text processing and sentiment analysis',
            'transformers': 'State-of-the-art language models (BERT, GPT)',
            'wordcloud': 'Text visualization and word frequency analysis'
        },
        'Time Series': {
            'statsmodels': 'Statistical modeling and time series analysis',
            'fbprophet': 'Forecasting at scale (Facebook)',
            'tslearn': 'Time series machine learning toolkit',
            'pyflux': 'Time series modeling and inference',
            'cesium': 'Time series feature extraction and clustering'
        },
        'Feature Selection': {
            'scikit-learn.feature_selection': 'Univariate and multivariate selection',
            'eli5': 'Machine learning model explanation and debugging',
            'shap': 'SHAP values for model interpretation',
            'lime': 'Local Interpretable Model-agnostic Explanations',
            'scikit-optimize': 'Bayesian optimization for feature selection'
        },
        'Automated Feature Engineering': {
            'featuretools': 'Deep feature synthesis for relational data',
            'auto-sklearn': 'Automated machine learning including feature engineering',
            'hpsklearn': 'Hyperparameter optimization with scikit-learn',
            'tpot': 'Tree-based Pipeline Optimization Tool',
            'auto_ml': 'Automated machine learning with feature engineering'
        }
    }

    for category, tool_dict in tools.items():
        print(f"\n{category.upper()}:")
        for tool, description in tool_dict.items():
            print(f"  • {tool}: {description}")

    # Quick installation guide
    print(f"\n📦 INSTALLATION GUIDE")
    print("-" * 30)

    installation_commands = {
        'Basic ML': 'pip install scikit-learn pandas numpy matplotlib seaborn',
        'Text Processing': 'pip install nltk spacy textblob',
        'Feature Engineering': 'pip install category-encoders feature-engine tsfresh',
        'Time Series': 'pip install statsmodels prophet',
        'Feature Selection': 'pip install eli5 shap lime',
        'Automated ML': 'pip install featuretools auto-sklearn tpot'
    }

    for category, command in installation_commands.items():
        print(f"{category}: {command}")

# Show feature engineering tools
feature_eng_tools = feature_engineering_tools()
```

### Summary: Feature Engineering Checklist ✅

```python
def feature_engineering_checklist():
    """
    Comprehensive checklist for feature engineering
    """

    print("✅ FEATURE ENGINEERING CHECKLIST")
    print("=" * 50)

    checklist = {
        'Planning & Understanding': [
            'Clearly defined the business problem and target variable',
            'Understood the domain and consulted subject matter experts',
            'Identified available data sources and their quality',
            'Set clear objectives for feature engineering',
            'Created a feature engineering strategy document'
        ],
        'Data Preparation': [
            'Cleaned and preprocessed the raw data',
            'Handled missing values appropriately',
            'Detected and treated outliers',
            'Removed or fixed data quality issues',
            'Validated data consistency and integrity'
        ],
        'Feature Creation': [
            'Created mathematical
```
