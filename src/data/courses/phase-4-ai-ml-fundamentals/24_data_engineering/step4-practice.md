# Data Engineering - Practice

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Apache Spark Exercises](#apache-spark-exercises)
3. [Data Pipeline Implementation](#data-pipeline-implementation)
4. [Data Quality Validation](#data-quality-validation)
5. [Stream Processing](#stream-processing)
6. [ETL Pipeline Creation](#etl-pipeline-creation)
7. [Database Operations](#database-operations)
8. [Cloud Data Engineering](#cloud-data-engineering)
9. [Performance Optimization](#performance-optimization)
10. [End-to-End Project](#end-to-end-project)

## Setup and Installation

### Local Spark Setup

```bash
# Install Spark locally
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
export SPARK_HOME=/path/to/spark-3.5.0-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH

# Start Spark shell
spark-shell  # Scala
pyspark      # Python
```

### Docker Setup

```yaml
# docker-compose.yml for Spark cluster
version: "3.8"
services:
  spark-master:
    image: bitnami/spark:latest
    environment:
      - SPARK_MODE=master
    ports:
      - "8080:8080"

  spark-worker:
    image: bitnami/spark:latest
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master
```

### Python Environment Setup

```python
# requirements.txt
pyspark==3.5.0
kafka-python==2.0.2
great-expectations==0.18.8
pandas==2.1.0
sqlalchemy==2.0.0
psycopg2-binary==2.9.7
```

## Apache Spark Exercises

### Exercise 1: Basic Spark Operations

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize Spark
spark = SparkSession.builder \
    .appName("DataEngineeringExercises") \
    .getOrCreate()

# Create sample data
data = [
    (1, "John", 25, "NYC", 5000),
    (2, "Jane", 30, "LA", 6000),
    (3, "Bob", 35, "NYC", 7000),
    (4, "Alice", 28, "LA", 5500),
    (5, "Charlie", 32, "Chicago", 6500)
]

# Define schema
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("city", StringType(), True),
    StructField("salary", IntegerType(), True)
])

# Create DataFrame
df = spark.createDataFrame(data, schema)
df.show()

# Basic transformations
# 1. Filter data
filtered_df = df.filter(df.age > 25)
filtered_df.show()

# 2. Select specific columns
selected_df = df.select("name", "city", "salary")
selected_df.show()

# 3. Group by and aggregate
grouped_df = df.groupBy("city").agg(
    avg("salary").alias("avg_salary"),
    count("*").alias("count")
)
grouped_df.show()

# 4. Sort data
sorted_df = df.orderBy(desc("salary"))
sorted_df.show()

# 5. Add new columns
df_with_experience = df.withColumn(
    "experience_level",
    when(df.age < 30, "Junior")
    .when(df.age < 35, "Mid")
    .otherwise("Senior")
)
df_with_experience.show()

# Save to different formats
df.write.mode("overwrite").parquet("output/employees.parquet")
df.write.mode("overwrite").csv("output/employees.csv", header=True)
df.write.mode("overwrite").json("output/employees.json")

spark.stop()
```

### Exercise 2: Complex Data Transformations

```python
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Create orders data
orders_data = [
    (1, "2023-01-01", "product_a", 5, 100.0),
    (2, "2023-01-01", "product_b", 3, 75.0),
    (3, "2023-01-02", "product_a", 2, 40.0),
    (4, "2023-01-02", "product_c", 1, 50.0),
    (5, "2023-01-03", "product_b", 4, 100.0)
]

orders_schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("order_date", StringType(), True),
    StructField("product", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("revenue", DoubleType(), True)
])

orders_df = spark.createDataFrame(orders_data, orders_schema)

# Convert string date to date type
orders_df = orders_df.withColumn("order_date", to_date("order_date", "yyyy-MM-dd"))

# 1. Window functions - Running totals
window_spec = Window.orderBy("order_date")
orders_with_running_total = orders_df.withColumn(
    "running_revenue",
    sum("revenue").over(window_spec)
)

# 2. Partition by product
product_window = Window.orderBy("order_date").partitionBy("product")
orders_with_product_rank = orders_df.withColumn(
    "product_rank",
    rank().over(product_window)
)

# 3. Pivot data
pivot_df = orders_df.groupBy("order_date").pivot("product").sum("revenue")
pivot_df.show()

# 4. Join operations
products_data = [
    ("product_a", "Electronics", "Brand X"),
    ("product_b", "Electronics", "Brand Y"),
    ("product_c", "Books", "Brand Z")
]

products_schema = StructType([
    StructField("product", StringType(), True),
    StructField("category", StringType(), True),
    StructField("brand", StringType(), True)
])

products_df = spark.createDataFrame(products_data, products_schema)

# Join orders with products
joined_df = orders_df.join(products_df, "product")
joined_df.show()

# 5. Aggregations with cube
cube_df = orders_df.cube("product", "order_date").sum("revenue")
cube_df.show()
```

## Data Pipeline Implementation

### Exercise 3: ETL Pipeline with Spark

```python
import os
from datetime import datetime

class DataPipeline:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.source_path = "/data/source/"
        self.target_path = "/data/target/"

    def extract(self, file_format="csv"):
        """Extract data from source files"""
        if file_format == "csv":
            df = self.spark.read.csv(
                self.source_path,
                header=True,
                inferSchema=True
            )
        elif file_format == "json":
            df = self.spark.read.json(self.source_path)
        elif file_format == "parquet":
            df = self.spark.read.parquet(self.source_path)

        return df

    def transform(self, df):
        """Transform data"""
        # Clean data
        df = df.dropna()

        # Add calculated fields
        df = df.withColumn("timestamp", current_timestamp())
        df = df.withColumn("year", year("timestamp"))

        # Standardize formats
        df = df.withColumn(
            "email",
            lower(trim(col("email")))
        )

        return df

    def load(self, df, table_name):
        """Load data to target"""
        output_path = f"{self.target_path}{table_name}/"
        df.write.mode("overwrite").parquet(output_path)

        # Also create a summary
        summary = df.agg(
            count("*").alias("record_count"),
            countDistinct("id").alias("unique_records")
        )
        summary_path = f"{self.target_path}{table_name}_summary/"
        summary.coalesce(1).write.mode("overwrite").parquet(summary_path)

        return df

    def run_pipeline(self, table_name, file_format="csv"):
        """Run complete ETL pipeline"""
        print(f"Starting ETL pipeline for {table_name}")

        # Extract
        df = self.extract(file_format)
        print(f"Extracted {df.count()} records")

        # Transform
        df_transformed = self.transform(df)
        print(f"Transformed data")

        # Load
        df_loaded = self.load(df_transformed, table_name)
        print(f"Loaded data to {self.target_path}{table_name}")

        return df_loaded

# Usage
pipeline = DataPipeline(spark)
customers_df = pipeline.run_pipeline("customers", "csv")
```

### Exercise 4: Data Quality Validation

```python
import great_expectations as ge

class DataQualityChecker:
    def __init__(self, df, table_name):
        self.df = df
        self.table_name = table_name
        self.context = ge.get_context()

    def create_expectation_suite(self):
        """Create expectation suite based on table type"""
        suite_name = f"{self.table_name}_suite"
        suite = self.context.create_expectation_suite(suite_name)

        if self.table_name == "customers":
            # Customer-specific expectations
            suite.expect_column_to_exist("customer_id")
            suite.expect_column_to_exist("email")
            suite.expect_column_to_exist("registration_date")

            suite.expect_column_values_to_not_be_null("customer_id")
            suite.expect_column_values_to_be_unique("customer_id")
            suite.expect_column_values_to_match_regex("email", r".+@.+\..+")
            suite.expect_column_values_to_be_between(
                "age", min_value=0, max_value=120
            )

        elif self.table_name == "orders":
            # Order-specific expectations
            suite.expect_column_to_exist("order_id")
            suite.expect_column_values_to_not_be_null("order_id")
            suite.expect_column_values_to_be_unique("order_id")
            suite.expect_column_values_to_be_greater_than("amount", 0)
            suite.expect_column_values_to_be_in_set(
                "status", ["pending", "processing", "shipped", "delivered"]
            )

        return suite

    def validate_data(self):
        """Run data quality validation"""
        suite = self.create_expectation_suite()

        # Convert Spark DataFrame to Great Expectations format
        ge_df = ge.dataset.SparkDFDataset(self.df)

        # Validate
        results = ge_df.validate(expectation_suite=suite)

        # Print results
        if results["success"]:
            print(f"✅ Data validation passed for {self.table_name}")
        else:
            print(f"❌ Data validation failed for {self.table_name}")
            for result in results["results"]:
                if not result["success"]:
                    print(f"  - {result['expectation_config']['expectation_type']}")
                    print(f"    {result['exception_info']['exception_message']}")

        return results

    def generate_data_profile(self):
        """Generate data profiling report"""
        ge_df = ge.dataset.SparkDFDataset(self.df)
        profile = ge_df.profile()
        return profile

# Usage
quality_checker = DataQualityChecker(customers_df, "customers")
validation_results = quality_checker.validate_data()
profile = quality_checker.generate_data_profile()
```

## Stream Processing

### Exercise 5: Kafka Stream Processing

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class StreamProcessor:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("StreamProcessing") \
            .getOrCreate()

        self.consumer = KafkaConsumer(
            'sensor_data',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def process_stream_spark(self):
        """Process streaming data with Spark"""
        # Define schema for sensor data
        sensor_schema = StructType([
            StructField("sensor_id", StringType(), True),
            StructField("temperature", DoubleType(), True),
            StructField("humidity", DoubleType(), True),
            StructField("timestamp", LongType(), True)
        ])

        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "sensor_data") \
            .load()

        # Parse JSON data
        parsed_df = df.select(
            from_json(col("value").cast("string"), sensor_schema).alias("data")
        ).select("data.*")

        # Add processing timestamp
        processed_df = parsed_df.withColumn(
            "processed_at", current_timestamp()
        )

        # Aggregation over 1-minute windows
        windowed_df = processed_df \
            .withWatermark("timestamp", "1 minute") \
            .groupBy(
                window(col("timestamp"), "1 minute"),
                col("sensor_id")
            ) \
            .agg(
                avg("temperature").alias("avg_temperature"),
                avg("humidity").alias("avg_humidity"),
                max("temperature").alias("max_temperature"),
                min("temperature").alias("min_temperature"),
                count("*").alias("readings_count")
            )

        # Write to console (for debugging)
        query = windowed_df \
            .writeStream \
            .outputMode("update") \
            .format("console") \
            .option("truncate", False) \
            .trigger(processingTime="10 seconds") \
            .start()

        # Write aggregated data back to Kafka
        kafka_query = windowed_df \
            .selectExpr("to_json(struct(*)) AS value") \
            .writeStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", "aggregated_sensor_data") \
            .outputMode("update") \
            .trigger(processingTime="10 seconds") \
            .start()

        return query, kafka_query

    def simulate_sensor_data(self):
        """Generate simulated sensor data for testing"""
        import random

        sensors = ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5"]

        for _ in range(100):
            sensor_id = random.choice(sensors)
            temperature = round(random.uniform(20.0, 30.0), 2)
            humidity = round(random.uniform(40.0, 80.0), 2)
            timestamp = int(time.time() * 1000)

            data = {
                "sensor_id": sensor_id,
                "temperature": temperature,
                "humidity": humidity,
                "timestamp": timestamp
            }

            self.producer.send('sensor_data', data)
            time.sleep(0.5)  # Send data every 500ms

# Usage
processor = StreamProcessor()
query, kafka_query = processor.process_stream_spark()

# Start generating data
processor.simulate_sensor_data()

# Wait for processing
query.awaitTermination(timeout=60)
kafka_query.awaitTermination(timeout=60)
```

## ETL Pipeline Creation

### Exercise 6: Complete ETL Pipeline with Scheduling

```python
import schedule
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ETLPipelineScheduler:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.logger = logging.getLogger(__name__)

    def extract_customers(self):
        """Extract customer data from multiple sources"""
        # Read from CSV
        customers_csv = self.spark.read.csv(
            "/data/source/customers.csv",
            header=True,
            inferSchema=True
        )

        # Read from database
        customers_db = self.spark.read.format("jdbc") \
            .option("url", "jdbc:postgresql://localhost:5432/db") \
            .option("dbtable", "legacy_customers") \
            .option("user", "user") \
            .option("password", "password") \
            .load()

        # Union both sources
        all_customers = customers_csv.union(customers_db)

        self.logger.info(f"Extracted {all_customers.count()} customer records")
        return all_customers

    def extract_orders(self):
        """Extract order data"""
        orders = self.spark.read.parquet("/data/source/orders/")

        # Filter recent orders (last 30 days)
        orders = orders.filter(
            col("order_date") >= date_add(current_date(), -30)
        )

        self.logger.info(f"Extracted {orders.count()} recent orders")
        return orders

    def transform_customers(self, df):
        """Transform customer data"""
        transformed = df \
            .dropDuplicates(["email"]) \
            .withColumn("email", lower(trim(col("email")))) \
            .withColumn("customer_segment",
                when(col("total_orders") >= 10, "VIP")
                .when(col("total_orders") >= 5, "Loyal")
                .otherwise("New")
            ) \
            .withColumn("processed_at", current_timestamp())

        return transformed

    def transform_orders(self, df):
        """Transform order data"""
        transformed = df \
            .withColumn("order_month", trunc("order_date", "Month")) \
            .withColumn("revenue_category",
                when(col("total_amount") >= 1000, "High")
                .when(col("total_amount") >= 500, "Medium")
                .otherwise("Low")
            ) \
            .withColumn("processed_at", current_timestamp())

        return transformed

    def load_to_warehouse(self, customers_df, orders_df):
        """Load to data warehouse"""
        # Create dimension and fact tables
        customer_dim = customers_df.select(
            "customer_id", "email", "name", "customer_segment", "processed_at"
        )

        order_fact = orders_df.join(
            customers_df.select("customer_id", "customer_segment"),
            "customer_id"
        ).select(
            "order_id", "customer_id", "order_date", "order_month",
            "total_amount", "revenue_category", "customer_segment",
            "processed_at"
        )

        # Write to warehouse
        customer_dim.coalesce(1) \
            .write \
            .mode("overwrite") \
            .partitionBy("customer_segment") \
            .parquet("/data/warehouse/customer_dimension/")

        order_fact.coalesce(1) \
            .write \
            .mode("overwrite") \
            .partitionBy("order_month") \
            .parquet("/data/warehouse/order_fact/")

        self.logger.info("Loaded data to warehouse")

    def run_daily_etl(self):
        """Run daily ETL job"""
        try:
            self.logger.info("Starting daily ETL pipeline")

            # Extract
            customers_df = self.extract_customers()
            orders_df = self.extract_orders()

            # Transform
            customers_transformed = self.transform_customers(customers_df)
            orders_transformed = self.transform_orders(orders_df)

            # Load
            self.load_to_warehouse(customers_transformed, orders_transformed)

            self.logger.info("Daily ETL pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            raise

    def setup_schedule(self):
        """Setup ETL job scheduling"""
        # Schedule daily at 2 AM
        schedule.every().day.at("02:00").do(self.run_daily_etl)

        # Schedule hourly data validation
        schedule.every().hour.do(self.validate_data_quality)

        self.logger.info("Scheduled ETL jobs")

    def validate_data_quality(self):
        """Validate data quality checks"""
        try:
            # Check customer data quality
            customers = self.spark.read.parquet("/data/warehouse/customer_dimension/")

            # Basic quality checks
            customer_count = customers.count()
            null_emails = customers.filter(col("email").isNull()).count()
            duplicate_emails = customers.groupBy("email").count().filter(col("count") > 1).count()

            if null_emails > 0:
                self.logger.warning(f"Found {null_emails} customers with null emails")

            if duplicate_emails > 0:
                self.logger.warning(f"Found {duplicate_emails} duplicate emails")

            self.logger.info(f"Data quality check passed: {customer_count} customers, {null_emails} null emails, {duplicate_emails} duplicates")

        except Exception as e:
            self.logger.error(f"Data quality validation failed: {str(e)}")

# Usage
scheduler = ETLPipelineScheduler(spark)
scheduler.setup_schedule()

# Run continuously
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Database Operations

### Exercise 7: Database Integration

```python
from sqlalchemy import create_engine, text
import pandas as pd

class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.connection = None

    def connect(self):
        """Establish database connection"""
        self.connection = self.engine.connect()
        return self.connection

    def read_from_database(self, query):
        """Read data from database"""
        if not self.connection:
            self.connect()

        result = pd.read_sql(query, self.connection)
        return result

    def write_to_database(self, df, table_name, if_exists='replace'):
        """Write DataFrame to database"""
        df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
        print(f"Written {len(df)} rows to {table_name}")

    def execute_query(self, query):
        """Execute SQL query"""
        if not self.connection:
            self.connect()

        result = self.connection.execute(text(query))
        return result

class DataLakeManager:
    def __init__(self, spark_session):
        self.spark = spark_session

    def read_from_s3(self, bucket_path, file_format="parquet"):
        """Read data from S3"""
        if file_format == "parquet":
            df = self.spark.read.parquet(f"s3a://{bucket_path}")
        elif file_format == "csv":
            df = self.spark.read.csv(f"s3a://{bucket_path}", header=True, inferSchema=True)
        elif file_format == "json":
            df = self.spark.read.json(f"s3a://{bucket_path}")

        return df

    def write_to_s3(self, df, bucket_path, file_format="parquet"):
        """Write data to S3"""
        if file_format == "parquet":
            df.write.mode("overwrite").parquet(f"s3a://{bucket_path}")
        elif file_format == "csv":
            df.write.mode("overwrite").csv(f"s3a://{bucket_path}", header=True)
        elif file_format == "json":
            df.write.mode("overwrite").json(f"s3a://{bucket_path}")

    def manage_partitions(self, df, partition_column, output_path):
        """Manage data partitions"""
        df.write.mode("overwrite").partitionBy(partition_column).parquet(output_path)

        # List partitions
        partitions = self.spark.sql(f"SHOW PARTITIONS {output_path}")
        return partitions

# Example usage
# PostgreSQL connection
db_manager = DatabaseManager("postgresql://user:pass@localhost:5432/mydb")

# Read from database
customers = db_manager.read_from_database("SELECT * FROM customers WHERE status = 'active'")

# Process with Spark
processed_customers = spark.createDataFrame(customers)
processed_customers = processed_customers.withColumn("processed_date", current_date())

# Write back to different database table
db_manager.write_to_database(processed_customers.toPandas(), "processed_customers")

# S3 integration
s3_manager = DataLakeManager(spark)

# Read from S3
raw_data = s3_manager.read_from_s3("my-bucket/raw-data/", "csv")

# Process and write back
processed_data = raw_data.filter(col("amount") > 0)
s3_manager.write_to_s3(processed_data, "my-bucket/processed-data/", "parquet")
```

## Cloud Data Engineering

### Exercise 8: AWS Data Engineering

```python
import boto3
import pandas as pd
from botocore.exceptions import ClientError

class AWSDataPipeline:
    def __init__(self, aws_access_key, aws_secret_key, region='us-east-1'):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.glue_client = boto3.client('glue', region_name=region)
        self.athena_client = boto3.client('athena', region_name=region)

    def upload_to_s3(self, local_file_path, bucket_name, s3_key):
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(local_file_path, bucket_name, s3_key)
            print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            print(f"Upload failed: {e}")

    def download_from_s3(self, bucket_name, s3_key, local_file_path):
        """Download file from S3"""
        try:
            self.s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f"Downloaded s3://{bucket_name}/{s3_key} to {local_file_path}")
        except ClientError as e:
            print(f"Download failed: {e}")

    def create_glue_crawler(self, crawler_name, s3_path, database_name):
        """Create AWS Glue crawler"""
        try:
            response = self.glue_client.create_crawler(
                Name=crawler_name,
                Role='GlueServiceRole',
                DatabaseName=database_name,
                Targets={
                    'S3Targets': [
                        {
                            'Path': s3_path
                        }
                    ]
                }
            )
            print(f"Created crawler: {crawler_name}")
            return response
        except ClientError as e:
            print(f"Crawler creation failed: {e}")

    def run_glue_crawler(self, crawler_name):
        """Run Glue crawler"""
        try:
            response = self.glue_client.start_crawler(Name=crawler_name)
            print(f"Started crawler: {crawler_name}")
            return response
        except ClientError as e:
            print(f"Crawler start failed: {e}")

    def query_athena(self, query, database, output_location):
        """Query data using Athena"""
        try:
            # Start query execution
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': database
                },
                ResultConfiguration={
                    'OutputLocation': output_location
                }
            )

            query_execution_id = response['QueryExecutionId']

            # Wait for completion
            while True:
                result = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )
                status = result['QueryExecution']['Status']['State']

                if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break

                time.sleep(5)

            # Get results
            if status == 'SUCCEEDED':
                results = self.athena_client.get_query_results(
                    QueryExecutionId=query_execution_id
                )
                return results
            else:
                print(f"Query failed with status: {status}")
                return None

        except ClientError as e:
            print(f"Athena query failed: {e}")

# Example usage
aws_pipeline = AWSDataPipeline(
    aws_access_key="your_access_key",
    aws_secret_key="your_secret_key"
)

# Upload processed data to S3
aws_pipeline.upload_to_s3(
    "output/processed_customers.parquet",
    "my-data-lake",
    "processed/customers/2023-01-01/customers.parquet"
)

# Create and run Glue crawler
aws_pipeline.create_glue_crawler(
    "customer_crawler",
    "s3://my-data-lake/processed/",
    "analytics_db"
)

aws_pipeline.run_glue_crawler("customer_crawler")

# Query with Athena
query = """
SELECT customer_segment,
       COUNT(*) as count,
       AVG(total_orders) as avg_orders
FROM customers
WHERE processed_date >= DATE('2023-01-01')
GROUP BY customer_segment
"""

results = aws_pipeline.query_athena(
    query,
    "analytics_db",
    "s3://my-athena-results/"
)
```

## Performance Optimization

### Exercise 9: Spark Performance Optimization

```python
class SparkOptimizer:
    def __init__(self, spark_session):
        self.spark = spark_session

    def optimize_data_format(self):
        """Demonstrate optimal data formats"""
        # Create sample data
        data = [(i, f"customer_{i}", random.random() * 1000) for i in range(100000)]
        df = self.spark.createDataFrame(data, ["id", "name", "amount"])

        # Write in different formats and compare
        formats = ["csv", "json", "parquet", "orc"]

        for fmt in formats:
            start_time = time.time()

            # Write data
            df.write.mode("overwrite").format(fmt).save(f"temp_data_{fmt}/")

            # Read data back
            read_df = self.spark.read.format(fmt).load(f"temp_data_{fmt}/")
            count = read_df.count()

            end_time = time.time()

            print(f"{fmt.upper()}: Write/Read time = {end_time - start_time:.2f}s, Records = {count}")

        # Cleanup
        for fmt in formats:
            self.spark.sparkContext.parallelize([]).map(lambda x: None).collect()

    def optimize_joins(self):
        """Optimize join operations"""
        # Create large datasets
        users_data = [(i, f"user_{i}", i % 100) for i in range(100000)]
        orders_data = [(i, i % 1000, random.random() * 100) for i in range(500000)]

        users_df = self.spark.createDataFrame(users_data, ["user_id", "name", "segment"])
        orders_df = self.spark.createDataFrame(orders_data, ["order_id", "user_id", "amount"])

        # Repartition for better join performance
        users_repartitioned = users_df.repartition(200, "segment")
        orders_repartitioned = orders_df.repartition(200, "user_id")

        # Broadcast small table (if users_df is small)
        from pyspark.sql.functions import broadcast

        # Perform join with broadcast optimization
        result = orders_df.join(broadcast(users_df), "user_id")

        # Show execution plan
        result.explain(True)

        return result

    def optimize_memory_usage(self):
        """Optimize memory usage"""
        # Configure Spark for memory optimization
        optimized_spark = SparkSession.builder \
            .appName("OptimizedSpark") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "4") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

        # Use efficient serialization
        self.spark.sparkContext.getConf().set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        return optimized_spark

    def monitor_performance(self):
        """Monitor query performance"""
        # Enable SQL metrics
        self.spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        self.spark.conf.set("spark.sql.adaptive.enabled", "true")

        # Create test DataFrame
        df = self.spark.range(1000000).select(
            (col("id") % 1000).alias("category"),
            (col("id") * 2).alias("value")
        )

        # Profile query performance
        import time

        start_time = time.time()
        result = df.groupBy("category").agg(sum("value"))
        count = result.count()
        end_time = time.time()

        print(f"Query executed in {end_time - start_time:.2f}s, returned {count} rows")

        # Show Spark UI metrics (available in web UI)
        return result

# Usage
optimizer = SparkOptimizer(spark)

# Run optimizations
optimizer.optimize_data_format()
join_result = optimizer.optimize_joins()
optimized_spark = optimizer.optimize_memory_usage()
performance_result = optimizer.monitor_performance()
```

## End-to-End Project

### Exercise 10: Complete Data Engineering Project

```python
class CompleteDataEngineeringProject:
    def __init__(self, spark_session):
        self.spark = spark_session
        self.config = {
            'source_path': '/data/source/',
            'processed_path': '/data/processed/',
            'warehouse_path': '/data/warehouse/',
            'reports_path': '/data/reports/'
        }

    def setup_project_structure(self):
        """Setup project directory structure"""
        import os

        paths = [
            self.config['source_path'],
            self.config['processed_path'],
            self.config['warehouse_path'],
            self.config['reports_path']
        ]

        for path in paths:
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

    def generate_sample_data(self):
        """Generate sample datasets for the project"""
        import random
        from datetime import datetime, timedelta

        # Generate customers
        customers = []
        for i in range(10000):
            customers.append((
                i + 1,
                f"customer_{i+1}@email.com",
                random.choice(['John', 'Jane', 'Bob', 'Alice', 'Charlie']),
                random.randint(18, 80),
                random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix']),
                random.randint(0, 50)
            ))

        # Generate orders
        orders = []
        for i in range(50000):
            customer_id = random.randint(1, 10000)
            order_date = datetime.now() - timedelta(days=random.randint(0, 365))
            amount = round(random.uniform(10, 1000), 2)

            orders.append((
                i + 1,
                customer_id,
                order_date.strftime('%Y-%m-%d'),
                random.choice(['Electronics', 'Books', 'Clothing', 'Home', 'Sports']),
                amount,
                random.choice(['pending', 'processing', 'shipped', 'delivered'])
            ))

        # Save to CSV
        customers_df = self.spark.createDataFrame(customers, [
            "customer_id", "email", "name", "age", "city", "total_orders"
        ])

        orders_df = self.spark.createDataFrame(orders, [
            "order_id", "customer_id", "order_date", "category", "amount", "status"
        ])

        customers_df.coalesce(1).write.mode("overwrite").csv(
            f"{self.config['source_path']}customers/", header=True
        )

        orders_df.coalesce(1).write.mode("overwrite").csv(
            f"{self.config['source_path']}orders/", header=True
        )

        print(f"Generated {len(customers)} customers and {len(orders)} orders")

    def extract_data(self):
        """Extract data from source files"""
        # Read customers
        customers = self.spark.read.csv(
            f"{self.config['source_path']}customers/",
            header=True,
            inferSchema=True
        )

        # Read orders
        orders = self.spark.read.csv(
            f"{self.config['source_path']}orders/",
            header=True,
            inferSchema=True
        )

        print(f"Extracted {customers.count()} customers and {orders.count()} orders")
        return customers, orders

    def transform_data(self, customers, orders):
        """Transform and clean data"""
        # Transform customers
        customers_clean = customers \
            .dropDuplicates(["email"]) \
            .withColumn("email", lower(trim(col("email")))) \
            .withColumn("age_group",
                when(col("age") < 30, "18-29")
                .when(col("age") < 50, "30-49")
                .otherwise("50+")
            ) \
            .withColumn("customer_lifetime_value",
                col("total_orders") * 50  # Simple CLV calculation
            ) \
            .withColumn("processed_date", current_date())

        # Transform orders
        orders_clean = orders \
            .withColumn("order_date", to_date("order_date", "yyyy-MM-dd")) \
            .withColumn("order_year", year("order_date")) \
            .withColumn("order_month", month("order_date")) \
            .withColumn("revenue_tier",
                when(col("amount") >= 500, "High")
                .when(col("amount") >= 100, "Medium")
                .otherwise("Low")
            ) \
            .withColumn("is_weekend", dayofweek("order_date").isin([1, 7])) \
            .withColumn("processed_date", current_date())

        print("Data transformation completed")
        return customers_clean, orders_clean

    def load_to_warehouse(self, customers, orders):
        """Load to data warehouse"""
        # Create dimension tables
        customer_dim = customers.select(
            "customer_id", "email", "name", "age", "age_group",
            "city", "customer_lifetime_value", "processed_date"
        )

        date_dim = orders.select(
            "order_date", "order_year", "order_month"
        ).dropDuplicates().withColumn("date_key",
            concat(col("order_year"),
                   lpad(col("order_month"), 2, '0')))

        product_dim = orders.select("category").dropDuplicates()

        # Create fact table
        order_fact = orders.join(
            customers.select("customer_id", "age_group", "customer_lifetime_value"),
            "customer_id"
        ).join(
            date_dim.select("order_date", "date_key"),
            "order_date"
        ).select(
            "order_id", "customer_id", "date_key", "category",
            "amount", "revenue_tier", "status", "is_weekend",
            "age_group", "customer_lifetime_value", "processed_date"
        )

        # Write to warehouse
        customer_dim.write.mode("overwrite").partitionBy("age_group") \
            .parquet(f"{self.config['warehouse_path']}customer_dimension/")

        date_dim.write.mode("overwrite").parquet(
            f"{self.config['warehouse_path']}date_dimension/"
        )

        product_dim.write.mode("overwrite").parquet(
            f"{self.config['warehouse_path']}product_dimension/"
        )

        order_fact.write.mode("overwrite").partitionBy("order_year", "order_month") \
            .parquet(f"{self.config['warehouse_path']}order_fact/")

        print("Data loaded to warehouse")

    def create_analytics_reports(self):
        """Create analytics reports"""
        # Read warehouse tables
        customer_dim = self.spark.read.parquet(
            f"{self.config['warehouse_path']}customer_dimension/"
        )
        order_fact = self.spark.read.parquet(
            f"{self.config['warehouse_path']}order_fact/"
        )

        # Report 1: Customer segmentation analysis
        customer_report = customer_dim.groupBy("age_group", "city") \
            .agg(
                count("*").alias("customer_count"),
                avg("customer_lifetime_value").alias("avg_clv"),
                sum("customer_lifetime_value").alias("total_clv")
            ) \
            .orderBy(desc("total_clv"))

        customer_report.coalesce(1).write.mode("overwrite") \
            .parquet(f"{self.config['reports_path']}customer_segmentation/")

        # Report 2: Monthly sales analysis
        monthly_sales = order_fact.groupBy("order_year", "order_month") \
            .agg(
                sum("amount").alias("total_revenue"),
                count("*").alias("total_orders"),
                countDistinct("customer_id").alias("unique_customers"),
                avg("amount").alias("avg_order_value")
            ) \
            .withColumn("revenue_per_customer",
                       col("total_revenue") / col("unique_customers")) \
            .orderBy("order_year", "order_month")

        monthly_sales.coalesce(1).write.mode("overwrite") \
            .parquet(f"{self.config['reports_path']}monthly_sales/")

        # Report 3: Category performance
        category_performance = order_fact.groupBy("category", "revenue_tier") \
            .agg(
                sum("amount").alias("total_revenue"),
                count("*").alias("order_count"),
                countDistinct("customer_id").alias("unique_customers")
            ) \
            .orderBy(desc("total_revenue"))

        category_performance.coalesce(1).write.mode("overwrite") \
            .parquet(f"{self.config['reports_path']}category_performance/")

        print("Analytics reports created")

        return customer_report, monthly_sales, category_performance

    def validate_data_quality(self):
        """Validate data quality"""
        print("Running data quality validation...")

        # Check warehouse tables
        tables = ['customer_dimension', 'order_fact']

        for table in tables:
            df = self.spark.read.parquet(f"{self.config['warehouse_path']}{table}/")
            count = df.count()
            print(f"{table}: {count} records")

            # Check for nulls in key columns
            if table == 'customer_dimension':
                null_ids = df.filter(col("customer_id").isNull()).count()
                print(f"  - Null customer_ids: {null_ids}")

            elif table == 'order_fact':
                null_order_ids = df.filter(col("order_id").isNull()).count()
                print(f"  - Null order_ids: {null_order_ids}")

        print("Data quality validation completed")

    def run_complete_pipeline(self):
        """Run the complete data engineering pipeline"""
        print("Starting Complete Data Engineering Project Pipeline")
        print("=" * 60)

        # Setup
        self.setup_project_structure()

        # Generate sample data
        self.generate_sample_data()

        # Extract
        customers, orders = self.extract_data()

        # Transform
        customers_clean, orders_clean = self.transform_data(customers, orders)

        # Load to warehouse
        self.load_to_warehouse(customers_clean, orders_clean)

        # Create reports
        customer_report, monthly_sales, category_performance = \
            self.create_analytics_reports()

        # Validate
        self.validate_data_quality()

        print("=" * 60)
        print("Pipeline completed successfully!")

        return {
            'customers': customers_clean,
            'orders': orders_clean,
            'customer_report': customer_report,
            'monthly_sales': monthly_sales,
            'category_performance': category_performance
        }

# Usage
project = CompleteDataEngineeringProject(spark)
results = project.run_complete_pipeline()

# Show sample results
print("\nSample Results:")
print("\nCustomer Segmentation:")
results['customer_report'].show(5)

print("\nMonthly Sales:")
results['monthly_sales'].show(5)

print("\nCategory Performance:")
results['category_performance'].show(5)

spark.stop()
```

This comprehensive practice module covers all essential aspects of data engineering with hands-on exercises ranging from basic Spark operations to complete end-to-end projects. Each exercise builds upon the previous ones and provides practical experience with real-world data engineering scenarios.
