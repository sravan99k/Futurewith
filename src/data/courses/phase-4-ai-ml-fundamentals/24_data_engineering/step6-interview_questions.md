# Data Engineering - Interview Preparation

## Table of Contents

1. [Technical Interview Overview](#technical-interview-overview)
2. [Core Concepts Questions](#core-concepts-questions)
3. [Spark and Big Data Questions](#spark-and-big-data-questions)
4. [System Design Questions](#system-design-questions)
5. [Coding Challenges](#coding-chunkenges)
6. [Behavioral Questions](#behavioral-questions)
7. [Case Studies](#case-studies)
8. [Preparation Strategy](#preparation-strategy)

## Technical Interview Overview

### Interview Structure

- **Round 1: Technical Screening** (45-60 minutes)
  - Basic concepts and experience discussion
  - Simple coding question
  - Data modeling scenario

- **Round 2: Technical Deep Dive** (60-90 minutes)
  - Complex coding challenge
  - System design discussion
  - Architecture trade-offs

- **Round 3: System Design** (60-90 minutes)
  - Large-scale system design
  - Scalability discussions
  - Technology choices

- **Round 4: Cultural Fit** (30-45 minutes)
  - Behavioral questions
  - Team collaboration scenarios
  - Leadership and communication

### Key Skills Assessed

- **Programming**: Python, SQL, Scala
- **Big Data Technologies**: Spark, Hadoop, Kafka
- **Database Systems**: SQL, NoSQL, Data Warehouses
- **Cloud Platforms**: AWS, GCP, Azure
- **System Design**: Scalability, reliability, performance
- **Data Modeling**: ETL pipelines, data warehousing

## Core Concepts Questions

### 1. What is data engineering and how does it differ from data science?

**Answer Framework:**

- **Data Engineering**: Focuses on data collection, storage, transformation, and making data available for analysis
- **Key Responsibilities**: Building data pipelines, infrastructure, ETL processes, data quality
- **Technical Skills**: Programming, databases, cloud platforms, distributed systems
- **Time Horizon**: Real-time to daily processing

**Differences from Data Science:**

- **Focus**: Infrastructure vs analysis
- **Output**: Working data pipelines vs insights/models
- **Skills**: Systems and programming vs statistics and domain knowledge
- **Collaboration**: Works with data scientists to provide clean data

### 2. Explain the difference between ETL and ELT.

**ETL (Extract, Transform, Load):**

- Transform data before loading to warehouse
- Traditional approach
- Pros: Clean data in warehouse, faster queries
- Cons: Limited by warehouse compute power

**ELT (Extract, Load, Transform):**

- Load raw data first, then transform
- Modern cloud-based approach
- Pros: Leverages warehouse compute power, flexible
- Cons: Requires powerful warehouse, complex transformations

**Modern Approach:**

- Hybrid: Light transformation during load, heavy transformation when needed
- Example: Snowflake's transformation capabilities

### 3. What are the different types of data architectures?

**Lambda Architecture:**

- **Components**: Batch layer, speed layer, serving layer
- **Pros**: Fault-tolerant, handles batch and real-time
- **Cons**: Complex, duplicate code paths
- **Use Case**: High availability requirements

**Kappa Architecture:**

- **Approach**: Everything as streaming
- **Pros**: Simpler, single code path
- **Cons**: Requires sophisticated streaming
- **Use Case**: Real-time focused applications

**Data Mesh:**

- **Principles**: Domain ownership, data as product, federated governance
- **Pros**: Scalable, aligns with business
- **Cons**: Complex governance, cultural change
- **Use Case**: Large organizations with multiple domains

### 4. Explain data partitioning strategies.

**Horizontal Partitioning:**

- Split rows across multiple tables/files
- **Range**: Based on value ranges (e.g., date ranges)
- **Hash**: Based on hash function for even distribution
- **List**: Based on specific values

**Vertical Partitioning:**

- Split columns across tables
- Separates frequently accessed columns
- Reduces I/O for large records

**Best Practices:**

- Partition on high-cardinality, frequently filtered columns
- Keep partition sizes between 100MB-1GB
- Avoid over-partitioning
- Consider partition pruning for query optimization

### 5. What is data lineage and why is it important?

**Definition:**

- Tracking data flow from source to destination
- Including transformations applied
- Visual representation of data dependencies

**Importance:**

- **Compliance**: Regulatory requirements (GDPR, SOX)
- **Debugging**: Quickly identify data issues
- **Impact Analysis**: Understand effects of changes
- **Trust**: Build confidence in data quality

**Implementation:**

- Metadata management tools (Apache Atlas, DataHub)
- Automated tracking in ETL tools
- Manual documentation for complex transformations
- Version control for transformation logic

## Spark and Big Data Questions

### 1. Explain Spark's execution model.

**Architecture Components:**

- **Driver**: Coordinates Spark applications
- **Cluster Manager**: Allocates resources (YARN, Mesos, Kubernetes)
- **Executors**: Execute tasks on worker nodes
- **RDDs**: Resilient Distributed Datasets

**Execution Flow:**

1. Driver creates logical plan
2. Catalyst optimizer creates physical plan
3. DAG scheduler creates stages and tasks
4. Task scheduler assigns tasks to executors
5. Executors run tasks and return results

**Key Concepts:**

- **Lazy Evaluation**: Transformations are not immediately executed
- **Lineage**: RDDs track transformation history
- **Fault Tolerance**: Can recompute lost partitions using lineage

### 2. What are the different types of joins in Spark and their trade-offs?

**Broadcast Hash Join:**

- **Use Case**: One table small enough to broadcast
- **Pros**: No shuffle, very fast
- **Cons**: Memory pressure if table too large
- **Threshold**: Default 10MB (configurable)

**Sort-Merge Join:**

- **Use Case**: Both tables large, sorted by join key
- **Pros**: Efficient for large datasets
- **Cons**: Requires shuffle and sort phases
- **Performance**: O(n log n) complexity

**Shuffle Hash Join:**

- **Use Case**: Medium-sized tables
- **Pros**: Better than sort-merge for some cases
- **Cons**: Requires full shuffle
- **Default**: Spark 3.0+ uses AQE to choose optimal

**Join Strategies Selection:**

- Spark automatically chooses based on table sizes
- Can hint with broadcast() function
- Consider partitioning for frequent joins

### 3. How do you optimize Spark applications?

**Data Format Optimization:**

- Use columnar formats (Parquet, ORC)
- Enable compression (Snappy, ZSTD)
- Avoid CSV for analytical workloads

**Partitioning Strategy:**

- Repartition data appropriately (200-400 partitions per core)
- Partition on high-cardinality, frequently filtered columns
- Avoid small files problem

**Caching and Persistence:**

- Cache frequently accessed DataFrames
- Choose appropriate storage level (MEMORY_AND_DISK)
- Unpersist when no longer needed

**Join Optimization:**

- Broadcast small tables
- Use bucketing for frequent joins
- Consider join ordering

**Configuration Tuning:**

- Increase executor memory and cores appropriately
- Enable Adaptive Query Execution (AQE)
- Configure shuffle partitions
- Set appropriate garbage collection

### 4. Explain streaming vs batch processing in Spark.

**Batch Processing:**

- **Spark Core**: Process bounded datasets
- **Transformation**: filter, groupBy, join
- **Action**: collect, write, count
- **Use Case**: Historical data analysis

**Streaming Processing:**

- **Spark Streaming**: Micro-batch processing
- **Structured Streaming**: Continuous processing
- **Transformations**: Same as batch + window operations
- **Output**: Append, update, complete modes

**Window Operations:**

- **Tumbling**: Non-overlapping, fixed intervals
- **Sliding**: Overlapping, fixed intervals
- **Session**: Grouped by activity gaps

**State Management:**

- **Checkpointing**: Persist state for fault tolerance
- **State stores**: Built-in or custom implementations
- **Watermarking**: Handle late data

## System Design Questions

### 1. Design a real-time analytics system for an e-commerce platform.

**Requirements:**

- Process 100K events per second
- Dashboard with real-time metrics
- Historical data for trend analysis
- Sub-second latency for critical metrics

**Architecture Components:**

**Data Ingestion Layer:**

- **Apache Kafka**: Message queue for event streaming
- **Producers**: Website, mobile app, backend services
- **Partitioning**: By user_id for parallelism
- **Retention**: 7 days for real-time, archive to S3 for historical

**Processing Layer:**

- **Apache Spark Streaming**: Real-time aggregation
- **Window Operations**: 1-minute tumbling windows
- **State Management**: Checkpointing for aggregations
- **SLA**: < 500ms processing latency

**Storage Layer:**

- **Time-Series Database**: InfluxDB for metrics
- **Data Warehouse**: Snowflake for historical analysis
- **Cache**: Redis for real-time dashboard data

**Serving Layer:**

- **API Gateway**: REST endpoints for dashboards
- **GraphQL**: Flexible data fetching for UIs
- **WebSockets**: Real-time updates to clients

**Key Design Decisions:**

- Use Kafka for durability and replayability
- Spark for complex processing logic
- Separate real-time and batch pipelines
- Multi-layer caching strategy

### 2. Design a data lake for a multi-national company.

**Requirements:**

- Store data from 50+ sources
- Support multiple analytics tools
- Cost-effective storage
- Data governance and security

**Architecture Design:**

**Ingestion Layer:**

- **Landing Zone**: Raw data storage (S3)
- **Structured Ingestion**: Database CDC, API connectors
- **Unstructured Ingestion**: File uploads, log collection
- **Schema Registry**: Track evolving data schemas

**Storage Layer:**

- **Bronze Layer**: Raw, immutable data
- **Silver Layer**: Cleaned, deduplicated data
- **Gold Layer**: Business-ready, aggregated data
- **Formats**: Parquet for analytics, Delta Lake for ACID

**Processing Layer:**

- **Apache Spark**: Large-scale batch processing
- **Apache Airflow**: Workflow orchestration
- **DBT**: SQL-based transformations
- **Schema Evolution**: Handle changing data structures

**Governance Layer:**

- **Data Catalog**: Apache Atlas, AWS Glue
- **Lineage Tracking**: Automatic for all transformations
- **Access Control**: Fine-grained permissions
- **Data Quality**: Great Expectations for validation

**Access Layer:**

- **Presto/Trino**: Interactive analytics
- **Sagemaker**: ML workloads
- **Tableau/PowerBI**: Business intelligence
- **APIs**: Programmatic access

**Cost Optimization:**

- Lifecycle policies for storage tiers
- Compression and partitioning
- Spot instances for batch processing
- Query result caching

### 3. Design a pipeline for machine learning data preparation.

**Requirements:**

- Ingest training data from multiple sources
- Feature engineering and selection
- Version control for datasets and features
- Support for both batch and real-time inference

**Pipeline Architecture:**

**Data Ingestion:**

- **Source Connectors**: Database, APIs, file systems
- **CDC**: Change data capture for databases
- **Streaming**: Kafka for real-time features
- **Schema Evolution**: Handle changing data structures

**Data Processing:**

- **Feature Store**: Feast for feature management
- **Data Validation**: Great Expectations for quality
- **Feature Engineering**: Spark for transformations
- **Data Splitting**: Stratified splits for training

**Storage:**

- **Training Data**: Parquet files in S3
- **Feature Store**: Online (Redis) and offline (BigQuery)
- **Metadata**: MLflow for experiment tracking
- **Artifacts**: Model versions and datasets

**Orchestration:**

- **Apache Airflow**: Batch pipeline scheduling
- **Kubeflow**: ML workflow orchestration
- **CI/CD**: Automated pipeline deployment
- **Monitoring**: Data drift and model performance

**Key Components:**

- Feature store for consistency between training and inference
- Version control for reproducibility
- Automated data validation
- A/B testing framework

## Coding Challenges

### Challenge 1: Find top N customers by revenue

```python
def top_customers_by_revenue(df, n=10):
    """
    Find top N customers by total revenue
    Input: DataFrame with columns: customer_id, revenue
    Output: DataFrame with top N customers
    """
    return (
        df.groupBy("customer_id")
          .agg(sum("revenue").alias("total_revenue"))
          .orderBy(desc("total_revenue"))
          .limit(n)
    )

# Test with sample data
data = [
    (1, 100.0), (2, 150.0), (1, 50.0),
    (3, 200.0), (2, 100.0), (1, 75.0)
]
df = spark.createDataFrame(data, ["customer_id", "revenue"])
result = top_customers_by_revenue(df, 2)
result.show()
```

### Challenge 2: Deduplicate data with business logic

```python
def deduplicate_customers(df):
    """
    Remove duplicate customers keeping the most recent
    Input: DataFrame with customer_id, name, email, updated_at
    """
    window = Window.partitionBy("customer_id").orderBy(desc("updated_at"))

    return (
        df.withColumn("row_num", row_number().over(window))
          .filter(col("row_num") == 1)
          .drop("row_num")
    )
```

### Challenge 3: Calculate moving average

```python
def moving_average(df, window_size=7):
    """
    Calculate moving average for stock prices
    Input: DataFrame with date and price columns
    """
    window_spec = Window.orderBy("date").rowsBetween(-window_size+1, 0)

    return (
        df.withColumn("moving_avg", avg("price").over(window_spec))
          .orderBy("date")
    )
```

### Challenge 4: Join optimization problem

```python
def optimize_large_join(orders_df, customers_df, products_df):
    """
    Optimize join of large tables
    """
    # Broadcast small dimension tables
    from pyspark.sql.functions import broadcast

    # Optimize join order: fact table with broadcast dimensions
    result = (
        orders_df
        .join(broadcast(customers_df), "customer_id")
        .join(broadcast(products_df), "product_id")
        .select("order_id", "customer_name", "product_name", "amount")
    )

    # Repartition for downstream processing
    return result.repartition(200, "customer_id")
```

### Challenge 5: Streaming window aggregation

```python
def streaming_window_aggregation():
    """
    Implement streaming aggregation with window
    """
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "events") \
        .load()

    # Parse JSON and define schema
    schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("action", StringType(), True),
        StructField("timestamp", LongType(), True)
    ])

    parsed_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    # Window aggregation
    result = (
        parsed_df
        .withWatermark("timestamp", "10 minutes")
        .groupBy(
            window(col("timestamp"), "5 minutes"),
            col("action")
        )
        .agg(count("*").alias("action_count"))
    )

    # Write to console and Kafka
    console_query = result.writeStream \
        .outputMode("update") \
        .format("console") \
        .start()

    kafka_query = result.selectExpr("to_json(struct(*)) AS value") \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("topic", "aggregated_events") \
        .start()

    return console_query, kafka_query
```

## Behavioral Questions

### 1. Tell me about a time when you had to debug a complex data pipeline issue.

**Answer Framework (STAR Method):**

**Situation:**

- Describe the context and scale of the problem
- "We had a daily ETL pipeline that was timing out..."

**Task:**

- What was your responsibility?
- "I was responsible for investigating and fixing the timeout..."

**Action:**

- What steps did you take?
- "1. Added comprehensive logging to track each step 2. Analyzed Spark UI to identify slow tasks 3. Found that one transformation was causing data skew 4. Repartitioned the data and added broadcast joins..."

**Result:**

- What was the outcome?
- "Reduced pipeline runtime from 2 hours to 20 minutes, and it has been stable since."

### 2. How do you ensure data quality in your pipelines?

**Approach:**

- **Validation Framework**: Great Expectations for automated checks
- **Data Profiling**: Statistical analysis of new data sources
- **Monitoring**: Alert on quality degradation
- **Documentation**: Clear expectations and error handling

**Example:**

- "For a customer data pipeline, we validate:
  - Email format with regex
  - Required fields are not null
  - Age is within reasonable bounds
  - No duplicate customer IDs
- We alert on any validation failures and have rollback procedures."

### 3. Describe a time when you had to work with a difficult stakeholder.

**Situation:** "A product manager needed real-time analytics but our infrastructure could only provide batch processing."

**Action:**

- "Scheduled a technical meeting to understand their requirements
- Explained current limitations and proposed a phased approach
- Implemented a simplified real-time metric using Redis
- Provided batch processing for detailed analytics"

**Result:** "Satisfied immediate needs while building toward long-term solution."

### 4. How do you stay updated with technology trends?

**Approach:**

- **Online Learning**: Coursera, Udemy for new technologies
- **Open Source**: Contributing to projects, reading GitHub
- **Communities**: Stack Overflow, Reddit, Discord
- **Conferences**: Strata Data Conference, Spark Summit
- **Podcasts**: Data Engineering Podcast, Software Engineering Daily

### 5. Describe your approach to code reviews.

**Best Practices:**

- **Review for Logic**: Correctness, efficiency, edge cases
- **Review for Style**: Consistent formatting, naming conventions
- **Review for Security**: Data handling, credentials, permissions
- **Review for Testing**: Unit tests, integration tests
- **Provide Constructive Feedback**: Specific suggestions, positive reinforcement

## Case Studies

### Case Study 1: E-commerce Recommendation Engine

**Challenge:** Build a real-time recommendation system for an e-commerce platform

**Requirements:**

- Process 50K product views per minute
- Generate recommendations in < 100ms
- Support 1M daily active users
- Handle product catalog changes

**Solution Discussion Points:**

- **Data Ingestion**: Kafka for event streaming
- **Real-time Processing**: Spark Streaming with session windows
- **Feature Store**: Redis for user profiles, product features
- **Model Serving**: A/B testing framework
- **Monitoring**: Latency, accuracy, business metrics

**Technical Decisions:**

- Window size selection for user behavior
- Fallback mechanisms for cold start
- Scalability planning for traffic spikes

### Case Study 2: Financial Data Compliance

**Challenge:** Build a compliance reporting system for a bank

**Requirements:**

- Daily reports for regulatory submission
- 100% data accuracy requirement
- Complete audit trail
- Multi-year historical analysis

**Solution Discussion Points:**

- **Data Lineage**: Track every transformation
- **Data Quality**: Multi-layer validation
- **Audit Logging**: Immutable logs of all operations
- **Disaster Recovery**: Multiple backup locations
- **Compliance**: SOX, GDPR, regional regulations

**Technical Challenges:**

- Managing large historical datasets
- Ensuring data consistency across systems
- Performance optimization for complex queries

### Case Study 3: IoT Sensor Data Platform

**Challenge:** Process sensor data from 10,000 devices

**Requirements:**

- Collect data from sensors every 5 seconds
- Store data for 5 years
- Generate alerts for anomalies
- Provide dashboard for real-time monitoring

**Solution Discussion Points:**

- **Ingestion**: MQTT for device communication
- **Stream Processing**: Real-time anomaly detection
- **Storage**: Time-series database + data lake
- **Analytics**: Machine learning for predictive maintenance

**Scalability Considerations:**

- Handling device provisioning at scale
- Data compression and archiving
- Geographic distribution of processing

## Preparation Strategy

### 1. Technical Preparation (4-6 weeks)

**Week 1-2: Fundamentals**

- Review core data engineering concepts
- Practice SQL queries (window functions, CTEs)
- Study database design principles
- Review cloud platforms (AWS/GCP/Azure)

**Week 3-4: Big Data Technologies**

- Hands-on Spark practice (Spark SQL, Streaming)
- Study Kafka for event streaming
- Learn data warehousing concepts
- Practice system design problems

**Week 5-6: Advanced Topics**

- Study distributed systems concepts
- Review monitoring and alerting
- Practice coding challenges
- Prepare system design presentations

### 2. Practical Experience

**Hands-on Projects:**

1. **Build an ETL Pipeline**: Extract, transform, and load data using Spark
2. **Real-time Analytics**: Stream processing with Kafka and Spark
3. **Data Warehouse Design**: Star schema implementation
4. **Data Quality Framework**: Automated validation system

**Portfolio Presentation:**

- Document architectural decisions
- Explain scalability considerations
- Show performance metrics
- Discuss challenges and solutions

### 3. Mock Interview Practice

**Technical Screening:**

- Practice explaining complex concepts simply
- Be ready to write SQL queries on whiteboard
- Prepare to discuss specific technologies you've used

**System Design:**

- Practice drawing architecture diagrams
- Prepare for scalability questions
- Be ready to discuss trade-offs

**Behavioral:**

- Prepare STAR stories for common scenarios
- Practice explaining technical concepts to non-technical stakeholders
- Be ready to discuss leadership and mentoring experiences

### 4. Resources and Study Materials

**Books:**

- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Fundamentals of Data Engineering" by Joe Reis
- "Spark: The Definitive Guide" by Bill Chambers

**Online Resources:**

- Spark documentation and tutorials
- AWS/GCP/Azure documentation
- Data engineering blogs and newsletters
- YouTube channels (Data Eng Podcast, etc.)

**Practice Platforms:**

- LeetCode for coding practice
- HackerRank for SQL challenges
- System design resources (Grokking the System Design Interview)

### 5. Interview Day Preparation

**Before the Interview:**

- Review job requirements and company tech stack
- Prepare specific examples from your experience
- Test technical setup (camera, microphone, screen sharing)
- Have water and notes ready

**During the Interview:**

- Think out loud to show your reasoning
- Ask clarifying questions
- Break down complex problems into smaller parts
- Be honest about what you don't know

**After the Interview:**

- Send a thank-you email
- Reflect on questions that were challenging
- Follow up if you don't hear back within expected timeframe

Remember: The key to success in data engineering interviews is demonstrating both technical depth and practical problem-solving skills. Focus on understanding fundamental concepts and being able to apply them to real-world scenarios.
