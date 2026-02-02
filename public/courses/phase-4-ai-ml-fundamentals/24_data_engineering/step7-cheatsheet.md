# Data Engineering - Cheatsheet

## Quick Reference Guide

## Spark Operations Cheatsheet

### DataFrame Creation

```python
# From list of tuples
df = spark.createDataFrame(data, ["id", "name", "age"])

# From pandas DataFrame
df = spark.createDataFrame(pandas_df)

# From RDD
rdd = spark.sparkContext.parallelize(data)
df = rdd.toDF(["id", "name", "age"])

# Read from files
df = spark.read.csv("path", header=True, inferSchema=True)
df = spark.read.json("path")
df = spark.read.parquet("path")
```

### Essential DataFrame Operations

```python
# Select and filter
df.select("column1", "column2")
df.filter(df.age > 25)
df.where(col("city") == "NYC")

# Aggregations
df.groupBy("city").agg(avg("salary"), count("*"))

# Joins
df1.join(df2, "id")
df1.join(df2, df1.id == df2.id, "left")

# Window functions
from pyspark.sql.window import Window
window = Window.orderBy("date")
df.withColumn("running_sum", sum("amount").over(window))

# Sorting and limiting
df.orderBy(desc("salary")).limit(10)
```

### Common Transformations

```python
# Add columns
df.withColumn("new_col", col("col1") + col("col2"))
df.withColumn("category", when(col("age") < 30, "young").otherwise("old"))

# String operations
df.withColumn("email", lower(col("email")))
df.withColumn("name", trim(col("name")))

# Date operations
df.withColumn("year", year(col("date")))
df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

# Drop duplicates
df.dropDuplicates(["email"])
df.distinct()
```

## Streaming Operations

### Kafka Integration

```python
# Read from Kafka
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic_name") \
    .load()

# Parse JSON from Kafka
from pyspark.sql.functions import from_json
schema = StructType([StructField("id", IntegerType())])
df = df.select(from_json(col("value").cast("string"), schema).alias("data"))

# Write to Kafka
query = df.selectExpr("to_json(struct(*)) AS value") \
    .writeStream.format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "output_topic") \
    .start()
```

### Window Operations

```python
# Tumbling window
df.groupBy(window(col("timestamp"), "1 hour")).agg(avg("value"))

# Sliding window
df.groupBy(window(col("timestamp"), "1 hour", "15 minutes")).agg(sum("value"))

# Session window
df.groupBy(session_window(col("timestamp"), "10 minutes")).agg(count("*"))
```

## Data Quality Checks

### Great Expectations

```python
import great_expectations as ge

# Create expectation suite
context = ge.get_context()
suite = context.create_expectation_suite("my_suite")

# Common expectations
suite.expect_column_to_exist("customer_id")
suite.expect_column_values_to_not_be_null("email")
suite.expect_column_values_to_be_unique("id")
suite.expect_column_values_to_match_regex("email", r".+@.+\..+")
suite.expect_column_values_to_be_between("age", 0, 120)
suite.expect_column_values_to_be_in_set("status", ["active", "inactive"])

# Validate
ge_df = ge.dataset.SparkDFDataset(df)
results = ge_df.validate(expectation_suite=suite)
```

### Manual Data Quality

```python
# Check for nulls
null_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])

# Check duplicates
duplicate_count = df.count() - df.distinct().count()

# Check data types
df.dtypes

# Basic statistics
df.describe().show()

# Value counts for categorical columns
df.groupBy("category").count().show()
```

## ETL Pipeline Patterns

### Basic ETL Structure

```python
class ETLPipeline:
    def extract(self, source_path):
        return spark.read.csv(source_path, header=True, inferSchema=True)

    def transform(self, df):
        return df.dropna().withColumn("processed_at", current_timestamp())

    def load(self, df, target_path):
        df.write.mode("overwrite").parquet(target_path)

    def run(self, source, target):
        raw_data = self.extract(source)
        clean_data = self.transform(raw_data)
        self.load(clean_data, target)
        return clean_data
```

### Incremental Load Pattern

```python
# Read current watermark
current_max = spark.sql("SELECT MAX(updated_at) FROM target_table").collect()[0][0]

# Read only new data
new_data = spark.read.parquet("source/") \
    .filter(col("updated_at") > current_max)

# Union with existing and deduplicate
if current_max:
    existing_data = spark.read.parquet("target_table/")
    full_data = new_data.union(existing_data).distinct()
else:
    full_data = new_data

full_data.write.mode("overwrite").parquet("target_table/")
```

## Performance Optimization

### Partitioning

```python
# Partition by column
df.write.partitionBy("year", "month").parquet("output/")

# Repartition for parallelism
df_repartitioned = df.repartition(200, "customer_id")

# Coalesce to reduce files
df.coalesce(1).write.mode("overwrite").csv("output/", header=True)
```

### Caching and Persistence

```python
# Cache frequently accessed DataFrames
df.cache()
df.persist(StorageLevel.MEMORY_AND_DISK)

# Unpersist when done
df.unpersist()
```

### Broadcast Variables

```python
# For small lookup tables
lookup_df = spark.read.csv("small_table.csv")
broadcast_lookup = broadcast(lookup_df)

# Use in join
result = large_df.join(broadcast_lookup, "key")
```

## Data Formats Comparison

| Format  | Use Case             | Pros                 | Cons                     |
| ------- | -------------------- | -------------------- | ------------------------ |
| CSV     | Simple data exchange | Universal support    | No compression, slow     |
| JSON    | Semi-structured data | Flexible schema      | Large file size          |
| Parquet | Analytics            | Columnar, compressed | Requires specific reader |
| Avro    | Streaming            | Schema evolution     | Complex setup            |
| ORC     | Hive analytics       | Highly optimized     | Hadoop ecosystem only    |

## Common SQL Patterns

### Slowly Changing Dimensions

```sql
-- Type 2 SCD
INSERT INTO customer_dim (customer_id, name, effective_date, end_date, is_current)
SELECT
    customer_id,
    name,
    CURRENT_DATE as effective_date,
    NULL as end_date,
    TRUE as is_current
FROM source_customers
WHERE NOT EXISTS (
    SELECT 1 FROM customer_dim cd
    WHERE cd.customer_id = source_customers.customer_id
    AND cd.is_current = TRUE
);

-- Close previous records
UPDATE customer_dim
SET end_date = CURRENT_DATE, is_current = FALSE
WHERE customer_id IN (
    SELECT customer_id FROM source_customers
    WHERE name != customer_dim.name
) AND is_current = TRUE;
```

### Data Quality Queries

```sql
-- Check for duplicates
SELECT id, COUNT(*)
FROM table_name
GROUP BY id
HAVING COUNT(*) > 1;

-- Check for nulls in key columns
SELECT COUNT(*) - COUNT(customer_id) as null_customer_ids
FROM orders;

-- Data freshness check
SELECT MAX(updated_at) as last_update,
       DATEDIFF(CURRENT_DATE, MAX(updated_at)) as days_since_update
FROM table_name;
```

## Cloud Services Quick Reference

### AWS

```python
# S3 Operations
import boto3
s3 = boto3.client('s3')
s3.upload_file('local_file.csv', 'bucket-name', 'key/path/file.csv')
s3.download_file('bucket-name', 'key/path/file.csv', 'local_file.csv')

# Glue Crawler
glue = boto3.client('glue')
glue.create_crawler(
    Name='my_crawler',
    Role='GlueServiceRole',
    DatabaseName='my_database',
    Targets={'S3Targets': [{'Path': 's3://bucket/path/'}]}
)
```

### GCP

```python
from google.cloud import storage, bigquery

# GCS
storage_client = storage.Client()
bucket = storage_client.bucket('bucket-name')
blob = bucket.blob('path/file.csv')
blob.upload_from_filename('local_file.csv')

# BigQuery
client = bigquery.Client()
query = "SELECT * FROM `project.dataset.table` LIMIT 1000"
query_job = client.query(query)
```

### Azure

```python
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient

# Blob Storage
blob_service_client = BlobServiceClient(account_url="https://account.blob.core.windows.net")
container_client = blob_service_client.get_container_client("container-name")
blob_client = container_client.get_blob_client("file.csv")
with open("local_file.csv", "rb") as data:
    blob_client.upload_blob(data)

# Cosmos DB
cosmos_client = CosmosClient(url, credential)
database = cosmos_client.get_database_client("database-name")
container = database.get_container_client("container-name")
```

## Monitoring and Alerting

### Spark Metrics

```python
# Enable SQL metrics
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.adaptive.enabled", "true")

# Check executor metrics
spark.sparkContext.statusTracker().getExecutorInfos()

# Resource usage monitoring
df.explain()  # Query plan
df.count()  # Action trigger
```

### Pipeline Monitoring

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineMonitor:
    def __init__(self):
        self.start_time = None

    def log_start(self, pipeline_name):
        self.start_time = datetime.now()
        logger.info(f"Starting {pipeline_name}")

    def log_end(self, pipeline_name, record_count):
        duration = datetime.now() - self.start_time
        logger.info(f"Completed {pipeline_name}: {record_count} records in {duration}")

        # Send alert if duration exceeds threshold
        if duration.total_seconds() > 3600:  # 1 hour
            logger.warning(f"Pipeline {pipeline_name} took longer than expected")
```

## Error Handling Patterns

### Retry Logic

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        return wrapper
    return decorator

@retry(max_attempts=3)
def process_data(df):
    return df.groupBy("category").sum("amount")
```

### Data Validation with Fallbacks

```python
def safe_transform(df):
    try:
        # Try advanced transformation
        result = df.withColumn("complex_calc", complex_udf(col("data")))
        return result
    except Exception as e:
        logger.warning(f"Advanced transformation failed, using fallback: {e}")
        # Fallback to simple transformation
        return df.withColumn("simple_calc", col("data") * 2)
```

## Security Best Practices

### Data Masking

```python
from pyspark.sql.functions import udf
import hashlib

def mask_email(email):
    if "@" in email:
        username, domain = email.split("@")
        masked_username = username[:2] + "*" * (len(username) - 2)
        return f"{masked_username}@{domain}"
    return "***@***.***"

mask_email_udf = udf(mask_email)
masked_df = df.withColumn("masked_email", mask_email_udf(col("email")))
```

### Encryption

```python
from cryptography.fernet import Fernet

# Generate key
key = Fernet.generate_key()
cipher = Fernet(key)

def encrypt_value(value):
    return cipher.encrypt(str(value).encode()).decode()

encrypt_udf = udf(encrypt_value)
encrypted_df = df.withColumn("encrypted_ssn", encrypt_udf(col("ssn")))
```

## Common Data Engineering Patterns

### Medallion Architecture (Bronze, Silver, Gold)

```python
# Bronze Layer - Raw Data
bronze_df = spark.read.json("s3://raw-data/")
bronze_df.write.mode("append").partitionBy("ingestion_date").parquet("s3://bronze/")

# Silver Layer - Cleaned Data
silver_df = bronze_df.dropDuplicates().dropna()
silver_df.write.mode("append").partitionBy("process_date").parquet("s3://silver/")

# Gold Layer - Business Logic
gold_df = silver_df.groupBy("customer_id").agg(sum("amount").alias("total_revenue"))
gold_df.write.mode("overwrite").parquet("s3://gold/customer_revenue/")
```

### Data Lake Pattern

```python
# Lambda architecture implementation
# Batch layer
batch_results = spark.read.parquet("s3://raw/") \
    .groupBy("category") \
    .count() \
    .write.parquet("s3://batch-results/")

# Speed layer (real-time)
streaming_results = spark.readStream \
    .format("kafka") \
    .option("subscribe", "events") \
    .load() \
    .groupBy("category") \
    .count() \
    .writeStream \
    .outputMode("update") \
    .parquet("s3://speed-results/")
```

## Testing Patterns

### Unit Testing Spark Code

```python
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("Test") \
        .master("local[*]") \
        .getOrCreate()

def test_data_transformation(spark):
    # Create test data
    test_data = [(1, "John", 25), (2, "Jane", 30)]
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])
    df = spark.createDataFrame(test_data, schema)

    # Apply transformation
    result = df.withColumn("age_group",
        when(col("age") < 30, "young").otherwise("old"))

    # Assertions
    assert result.count() == 2
    assert result.filter(col("age_group") == "young").count() == 1
```

## Quick Commands Reference

### Spark Shell

```bash
# Start Spark shell
spark-shell  # Scala
pyspark      # Python

# Spark-submit
spark-submit --master yarn --deploy-mode cluster app.py
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1 app.py
```

### File Operations

```bash
# List files in HDFS
hdfs dfs -ls /data/
hdfs dfs -mkdir /data/output/
hdfs dfs -put local_file.csv /data/input/

# S3 operations
aws s3 ls s3://bucket-name/
aws s3 cp local_file.csv s3://bucket-name/path/
aws s3 sync local_dir/ s3://bucket-name/path/
```

### Database Commands

```sql
-- PostgreSQL
\dt                    -- List tables
\d table_name          -- Describe table
\copy table FROM file  -- Copy from file
\copy table TO file    -- Copy to file

-- MySQL
SHOW TABLES;
DESCRIBE table_name;
LOAD DATA INFILE 'file.csv' INTO TABLE table_name;
```

This cheatsheet provides quick reference for the most commonly used data engineering patterns, operations, and best practices. Keep it handy for daily development work!
