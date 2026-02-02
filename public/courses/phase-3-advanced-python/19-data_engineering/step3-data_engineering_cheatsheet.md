# Data Engineering Cheatsheet: Quick Reference Guide

## Table of Contents

1. [Essential Commands](#essential-commands)
2. [SQL Quick Reference](#sql-quick-reference)
3. [Python Data Tools](#python-data-tools)
4. [Database Operations](#database-operations)
5. [Data Pipeline Commands](#data-pipeline-commands)
6. [Cloud Platform Commands](#cloud-platform-commands)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Configuration Templates](#configuration-templates)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Best Practices Checklist](#best-practices-checklist)

## Essential Commands

### File Operations

```bash
# Check file sizes
du -sh /path/to/data/* | sort -hr

# Count lines in CSV files
wc -l *.csv

# Check file types
file data/*

# Split large files
split -l 1000000 large_file.csv small_file_

# Merge files
cat file_*.csv > combined_file.csv

# Remove duplicates
awk '!seen[$0]++' input.csv > output.csv

# Extract specific columns
cut -d',' -f1,3,5 data.csv

# Filter rows
grep "pattern" data.csv

# Sort and remove duplicates
sort data.csv | uniq > unique_data.csv
```

### Data Format Conversion

```bash
# CSV to JSON
python3 -c "import csv, json; print(json.dumps([dict(r) for r in csv.DictReader(open('data.csv'))]))"

# JSON to CSV
python3 -c "import csv, json; w=csv.writer(open('data.csv','w')); w.writerow(json.load(open('data.json'))[0].keys()); [w.writerow([v for v in item.values()]) for item in json.load(open('data.json'))]"

# Convert encoding
iconv -f ISO-8859-1 -t UTF-8 input.csv > output.csv

# Compress files
gzip large_file.csv
gunzip large_file.csv.gz

# Create archives
tar -czf data_archive.tar.gz data_directory/
tar -xzf data_archive.tar.gz
```

### Data Validation

```bash
# Check CSV format
head -n 5 data.csv | column -t -s ','

# Count records by field
cut -d',' -f1 data.csv | sort | uniq -c

# Check for null values
awk -F',' 'NR>1 && ($3=="" || $3=="NULL") {print NR}' data.csv

# Validate JSON format
python3 -m json.tool data.json

# Check for duplicate records
awk -F',' 'NR>1 {key=$1","$2} seen[key]++ {print "Duplicate:", key}' data.csv
```

## SQL Quick Reference

### Basic Queries

```sql
-- Select with filtering
SELECT column1, column2
FROM table_name
WHERE condition
ORDER BY column1 DESC
LIMIT 100;

-- Aggregate functions
SELECT
    COUNT(*) as total_rows,
    SUM(amount) as total_amount,
    AVG(price) as avg_price,
    MIN(created_at) as earliest_date,
    MAX(updated_at) as latest_date
FROM orders;

-- Group by with having
SELECT
    category,
    COUNT(*) as item_count,
    AVG(price) as avg_price
FROM products
GROUP BY category
HAVING COUNT(*) > 10;

-- Window functions
SELECT
    customer_id,
    order_date,
    amount,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_sequence,
    LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_amount
FROM orders;
```

### Data Transformation

```sql
-- Pivot data
SELECT
    customer_id,
    SUM(CASE WHEN product_category = 'Electronics' THEN amount ELSE 0 END) as electronics_total,
    SUM(CASE WHEN product_category = 'Clothing' THEN amount ELSE 0 END) as clothing_total,
    SUM(CASE WHEN product_category = 'Books' THEN amount ELSE 0 END) as books_total
FROM orders
GROUP BY customer_id;

-- Unpivot data
SELECT
    customer_id,
    'Electronics' as category,
    electronics_total as amount
FROM (SELECT customer_id, electronics_total FROM customer_totals) t
UNION ALL
SELECT
    customer_id,
    'Clothing' as category,
    clothing_total as amount
FROM (SELECT customer_id, clothing_total FROM customer_totals) t;

-- Recursive CTE
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level employees
    SELECT
        employee_id,
        manager_id,
        name,
        0 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: subordinates
    SELECT
        e.employee_id,
        e.manager_id,
        e.name,
        eh.level + 1
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy ORDER BY level, name;
```

### Performance Optimization

```sql
-- Use EXPLAIN to analyze query performance
EXPLAIN ANALYZE
SELECT * FROM large_table
WHERE date_column >= '2023-01-01'
AND date_column < '2023-02-01';

-- Create indexes for better performance
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_orders_amount ON orders(amount) WHERE amount > 1000;

-- Partition large tables
CREATE TABLE orders_2023 PARTITION OF orders
FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- Use materialized views for complex aggregations
CREATE MATERIALIZED VIEW daily_sales_summary AS
SELECT
    DATE(order_date) as sale_date,
    COUNT(*) as order_count,
    SUM(amount) as total_sales
FROM orders
GROUP BY DATE(order_date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW daily_sales_summary;
```

### Data Quality Checks

```sql
-- Check for null values
SELECT
    'column_name' as column_name,
    COUNT(*) as total_rows,
    COUNT(column_name) as non_null_count,
    COUNT(*) - COUNT(column_name) as null_count
FROM table_name;

-- Check for duplicates
SELECT column1, column2, COUNT(*)
FROM table_name
GROUP BY column1, column2
HAVING COUNT(*) > 1;

-- Validate data ranges
SELECT *
FROM table_name
WHERE date_column < '1900-01-01'
OR date_column > CURRENT_DATE + INTERVAL '1 day';

-- Check referential integrity
SELECT *
FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```

## Python Data Tools

### Pandas Operations

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')
df = pd.read_json('data.json')
df = pd.read_parquet('data.parquet')

# Basic operations
df.head()                    # First 5 rows
df.info()                    # Data types and memory usage
df.describe()                # Statistical summary
df.shape                     # Shape (rows, columns)
df.columns                   # Column names

# Data filtering
df[df['column'] > 100]                    # Filter rows
df[(df['col1'] > 10) & (df['col2'] == 'value')]  # Multiple conditions
df.query('column > 100 and column2 == "value"')  # Query syntax

# Data transformation
df['new_column'] = df['column1'] + df['column2']  # Create new column
df['date'] = pd.to_datetime(df['date'])           # Convert to datetime
df['category'] = df['category'].astype('category') # Convert to category

# Groupby operations
df.groupby('category').agg({
    'amount': ['sum', 'mean', 'count'],
    'date': 'min'
}).round(2)

# Pivot tables
pd.pivot_table(df,
               values='amount',
               index='customer_id',
               columns='product_category',
               aggfunc='sum',
               fill_value=0)

# Data cleaning
df.dropna()                  # Remove rows with NaN
df.fillna(0)                 # Fill NaN with 0
df.drop_duplicates()         # Remove duplicates
df.drop(['column1', 'column2'], axis=1)  # Drop columns

# Data merging
pd.merge(df1, df2, on='key', how='inner')  # Inner join
pd.concat([df1, df2], ignore_index=True)   # Concatenate
```

### Data Validation with Great Expectations

```python
import great_expectations as ge

# Create context
context = ge.get_context()

# Load data
df = ge.read_csv('data.csv')

# Define expectations
df.expect_column_to_exist('customer_id')
df.expect_column_values_to_not_be_null('customer_id')
df.expect_column_values_to_be_of_type('amount', 'float')
df.expect_column_values_to_be_between('amount', min_value=0)
df.expect_column_values_to_be_in_set('status', ['active', 'inactive'])

# Validate data
results = df.validate()

if results['success']:
    print("Data validation passed!")
else:
    print("Data validation failed!")
    for result in results['results']:
        if not result['success']:
            print(f"- {result['expectation_config']['expectation_type']}")
```

### Data Processing with Dask

```python
import dask.dataframe as dd

# Load large datasets
df = dd.read_csv('large_file.csv')

# Perform operations
result = df.groupby('category').amount.sum().compute()
filtered_df = df[df.amount > 100].compute()

# Parallel processing
df = df.repartition(npartitions=100)  # Optimize partitions
```

## Database Operations

### PostgreSQL Commands

```sql
-- Connection
psql -h hostname -U username -d database_name

-- Database management
CREATE DATABASE mydb;
DROP DATABASE mydb;
\l                             -- List databases
\c mydb                        -- Connect to database

-- Table operations
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE users;
\d users                       -- Describe table

-- Index operations
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created ON users(created_at DESC);
DROP INDEX idx_users_email;

-- Query optimization
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com';
VACUUM ANALYZE users;          -- Update statistics
REINDEX TABLE users;           -- Rebuild indexes

-- Backup and restore
pg_dump mydb > backup.sql
psql mydb < backup.sql
```

### MySQL Commands

```sql
-- Connection
mysql -h hostname -u username -p database_name

-- Database management
CREATE DATABASE mydb;
DROP DATABASE mydb;
SHOW DATABASES;
USE mydb;

-- Table operations
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE users;
DESCRIBE users;

-- Index operations
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created ON users(created_at);
DROP INDEX idx_users_email ON users;

-- Query optimization
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
ANALYZE TABLE users;           -- Update table statistics
```

### MongoDB Operations

```javascript
// Connection
mongo mongodb://username:password@hostname:port/database

// Database operations
use mydb                         // Switch to database
db.createCollection('users')     // Create collection
db.users.drop()                  // Drop collection

// Document operations
db.users.insertOne({
    name: "John Doe",
    email: "john@example.com",
    created_at: new Date()
});

db.users.find({email: "john@example.com"})
db.users.updateOne(
    {email: "john@example.com"},
    {$set: {name: "John Smith"}}
);
db.users.deleteOne({email: "john@example.com"});

// Aggregation
db.users.aggregate([
    {$group: {_id: "$status", count: {$sum: 1}}},
    {$sort: {count: -1}}
]);
```

## Data Pipeline Commands

### Apache Airflow

```bash
# Start Airflow
airflow webserver -p 8080
airflow scheduler

# DAG operations
airflow dags list
airflow dags list-runs -d my_dag
airflow tasks list my_dag
airflow dags trigger my_dag

# Manual task execution
airflow tasks test my_dag my_task 2023-01-01

# Reset database (development only)
airflow db reset
```

### DBT (Data Build Tool)

```bash
# Initialize dbt project
dbt init my_project

# Development commands
dbt compile                    # Compile SQL files
dbt run                        # Run models
dbt test                       # Run tests
dbt docs generate             # Generate documentation
dbt docs serve                # Serve documentation

# Operations
dbt snapshot                  # Take data snapshots
dbt seed                      # Load seed files
dbt source freshness          # Check source data freshness

# Production deployment
dbt run --models +my_model    # Run model and dependencies
dbt run --models @latest      # Run latest models only
```

### Apache Kafka

```bash
# Start Kafka
kafka-server-start.sh config/server.properties

# Topic operations
kafka-topics.sh --create --topic my-topic --partitions 3 --replication-factor 1
kafka-topics.sh --list --bootstrap-server localhost:9092
kafka-topics.sh --delete --topic my-topic --bootstrap-server localhost:9092

# Producer and consumer
kafka-console-producer.sh --broker-list localhost:9092 --topic my-topic
kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic my-topic --from-beginning

# Consumer groups
kafka-consumer-groups.sh --list --bootstrap-server localhost:9092
kafka-consumer-groups.sh --describe --group my-group --bootstrap-server localhost:9092
```

### Apache Spark

```bash
# Submit Spark job
spark-submit --master yarn --deploy-mode cluster my_spark_job.py

# Start Spark shell
spark-shell --master yarn
pyspark --master yarn

# Key configurations
--executor-memory 4g
--executor-cores 4
--num-executors 10
--driver-memory 2g
```

## Cloud Platform Commands

### AWS CLI

```bash
# AWS S3 operations
aws s3 ls s3://my-bucket/
aws s3 cp local-file.csv s3://my-bucket/data/
aws s3 sync local-dir/ s3://my-bucket/data/
aws s3 rm s3://my-bucket/data/file.csv

# AWS Redshift
aws redshift describe-clusters
aws redshift create-cluster-snapshot --cluster-identifier my-cluster --snapshot-identifier my-snapshot

# AWS Glue
aws glue get-tables --database-name mydb
aws glue start-crawler --name my-crawler
aws glue get-crawler --name my-crawler
```

### Google Cloud Platform

```bash
# Cloud Storage
gsutil ls gs://my-bucket/
gsutil cp local-file.csv gs://my-bucket/data/
gsutil -m rsync -r local-dir/ gs://my-bucket/data/
gsutil rm gs://my-bucket/data/file.csv

# BigQuery
bq ls                              # List datasets
bq ls my_dataset                   # List tables
bq query "SELECT * FROM my_dataset.my_table LIMIT 10"
bq load --source_format=CSV my_dataset.my_table gs://my-bucket/data/file.csv

# Dataflow
gcloud dataflow jobs run my-job --gcs-location=gs://my-templates/template
```

### Azure CLI

```bash
# Azure Storage
az storage blob upload --file local-file.csv --container-name mycontainer --name data/file.csv
az storage blob list --container-name mycontainer
az storage blob download --container-name mycontainer --name data/file.csv --file local-file.csv

# Azure Data Factory
az datafactory pipeline create --factory-name myfactory --name mypipeline --resource-group myrg
az datafactory pipeline run --factory-name myfactory --name mypipeline --parameters '{}'
```

## Monitoring and Logging

### Prometheus Queries

```promql
# Data pipeline metrics
rate(data_pipeline_runs_total[5m])
rate(data_pipeline_errors_total[5m])
data_pipeline_run_duration_seconds{job="etl_job"}

# Database metrics
database_connections_active
database_queries_per_second
database_slow_queries_total

# System metrics
cpu_usage_percent
memory_usage_percent
disk_usage_percent
network_io_bytes_total

# Custom application metrics
data_quality_score
data_freshness_minutes
processing_lag_seconds
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Data Pipeline Monitoring",
    "panels": [
      {
        "title": "Pipeline Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(data_pipeline_runs_total[5m]) / rate(data_pipeline_runs_total[5m]) * 100",
            "legendFormat": "Success Rate %"
          }
        ]
      },
      {
        "title": "Data Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(data_records_processed_total[5m])",
            "legendFormat": "Records/sec"
          }
        ]
      },
      {
        "title": "Pipeline Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "data_pipeline_duration_seconds",
            "legendFormat": "{{pipeline_name}}"
          }
        ]
      }
    ]
  }
}
```

### Log Analysis Commands

```bash
# Search logs
grep "ERROR" application.log
grep -i "timeout" *.log
grep -E "2023-01-01.*ERROR" logs/

# Count log entries
grep "ERROR" application.log | wc -l
grep -c "Exception" *.log

# Time-based filtering
grep "2023-01-01 10:" application.log
sed -n '/2023-01-01 10:00:00/,/2023-01-01 11:00:00/p' application.log

# Extract specific fields
awk -F',' '{print $1, $3, $5}' data.csv
```

## Configuration Templates

### Docker Compose for Data Stack

```yaml
version: "3.8"

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: analytics
      POSTGRES_USER: etl_user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  airflow:
    image: apache/airflow:2.5.0
    environment:
      AIRFLOW__CORE__LOAD_EXAMPLES: "false"
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://etl_user:password@postgres:5432/airflow
    depends_on:
      - postgres
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
    ports:
      - "8080:8080"

  jupyter:
    image: jupyter/datascience-notebook
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      JUPYTER_ENABLE_LAB: "yes"

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-pipeline
  template:
    metadata:
      labels:
        app: data-pipeline
    spec:
      containers:
        - name: data-processor
          image: data-processor:latest
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            exec:
              command:
                - python
                - health_check.py
            initialDelaySeconds: 30
            periodSeconds: 10
```

### Environment Configuration

```yaml
# config/development.yml
database:
  host: localhost
  port: 5432
  name: analytics_dev
  username: etl_user
  password: dev_password

pipeline:
  batch_size: 1000
  retry_attempts: 3
  retry_delay: 300
  timeout: 3600

monitoring:
  metrics_enabled: true
  log_level: DEBUG
  alerts:
    email:
      enabled: false
    slack:
      enabled: false

---
# config/production.yml
database:
  host: prod-db.company.com
  port: 5432
  name: analytics_prod
  username: etl_user
  password: ${DB_PASSWORD}

pipeline:
  batch_size: 10000
  retry_attempts: 5
  retry_delay: 600
  timeout: 7200

monitoring:
  metrics_enabled: true
  log_level: INFO
  alerts:
    email:
      enabled: true
      recipients:
        - data-team@company.com
    slack:
      enabled: true
      channel: "#data-alerts"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Database Connection Issues

```bash
# Test database connectivity
telnet hostname 5432
nc -zv hostname 5432

# Check database status
psql -h hostname -U username -d database_name -c "SELECT 1;"

# Check connection pool
SELECT count(*) FROM pg_stat_activity;
SELECT * FROM pg_stat_activity WHERE state = 'active';
```

#### Memory Issues

```bash
# Check memory usage
free -h
top -o %MEM

# Check process memory
ps aux --sort=-%mem | head -10

# Monitor memory over time
watch -n 1 'free -h'

# Check for memory leaks
valgrind --tool=memcheck ./data_processor
```

#### Performance Issues

```bash
# Check CPU usage
top -o %CPU
htop

# Check I/O usage
iostat -x 1
iotop

# Check network usage
iftop
nethogs

# Analyze slow queries
EXPLAIN ANALYZE SELECT * FROM large_table WHERE date > '2023-01-01';
```

#### Data Pipeline Failures

```bash
# Check Airflow logs
airflow tasks states-for-dag-run my_dag 2023-01-01

# Check DBT logs
dbt run --models +my_model --log-level debug

# Check Spark logs
yarn logs -applicationId application_1234567890123_0001

# Check Kafka lag
kafka-consumer-groups.sh --describe --group my-group --bootstrap-server localhost:9092
```

### Debug Commands

```bash
# Network debugging
ping hostname
nslookup hostname
dig hostname
traceroute hostname

# Process debugging
strace -p PID
lsof -p PID
pstree -p PID

# File debugging
lsof filename
fuser filename
stat filename

# Database debugging
SHOW PROCESSLIST;                    # MySQL
SELECT * FROM pg_stat_activity;     # PostgreSQL
db.currentOp()                      # MongoDB
```

## Best Practices Checklist

### Data Quality

- [ ] **Data Validation**
  - [ ] Schema validation on ingestion
  - [ ] Data type validation
  - [ ] Range validation
  - [ ] Format validation
  - [ ] Completeness checks

- [ ] **Data Cleaning**
  - [ ] Handle null values appropriately
  - [ ] Remove duplicates
  - [ ] Standardize formats
  - [ ] Handle outliers
  - [ ] Validate referential integrity

### Performance

- [ ] **Query Optimization**
  - [ ] Use appropriate indexes
  - [ ] Avoid SELECT \*
  - [ ] Use WHERE clauses effectively
  - [ ] Limit result sets
  - [ ] Use appropriate data types

- [ ] **Data Processing**
  - [ ] Use appropriate batch sizes
  - [ ] Implement parallel processing
  - [ ] Monitor resource usage
  - [ ] Use efficient file formats
  - [ ] Partition large tables

### Security

- [ ] **Access Control**
  - [ ] Use least privilege principle
  - [ ] Implement role-based access
  - [ ] Encrypt sensitive data
  - [ ] Use secure connections
  - [ ] Regular access reviews

- [ ] **Data Protection**
  - [ ] Backup strategies
  - [ ] Data retention policies
  - [ ] PII handling
  - [ ] Audit logging
  - [ ] Compliance checks

### Monitoring

- [ ] **System Monitoring**
  - [ ] CPU/Memory usage
  - [ ] Disk space
  - [ ] Network I/O
  - [ ] Database connections
  - [ ] Query performance

- [ ] **Data Pipeline Monitoring**
  - [ ] Job success/failure rates
  - [ ] Processing times
  - [ ] Data quality metrics
  - [ ] Data freshness
  - [ ] Error rates

### Documentation

- [ ] **Code Documentation**
  - [ ] Clear function descriptions
  - [ ] Parameter documentation
  - [ ] Usage examples
  - [ ] Error handling
  - [ ] Performance notes

- [ ] **Architecture Documentation**
  - [ ] System diagrams
  - [ ] Data flow documentation
  - [ ] API documentation
  - [ ] Configuration guide
  - [ ] Deployment procedures

### Testing

- [ ] **Unit Testing**
  - [ ] Data transformation functions
  - [ ] Validation functions
  - [ ] Business logic
  - [ ] Error handling
  - [ ] Edge cases

- [ ] **Integration Testing**
  - [ ] End-to-end pipelines
  - [ ] Database connections
  - [ ] External API integrations
  - [ ] Data quality checks
  - [ ] Performance testing

This cheatsheet provides quick access to essential data engineering commands, configurations, and best practices. Keep it handy for daily operations and troubleshooting!
