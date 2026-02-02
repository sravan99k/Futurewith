# Data Engineering Pipeline Practice: Implementation Guide

## Table of Contents

1. [Pipeline Development Framework](#pipeline-development-framework)
2. [ETL/ELT Pipeline Implementation](#etlelt-pipeline-implementation)
3. [Stream Processing Pipelines](#stream-processing-pipelines)
4. [Data Quality and Validation](#data-quality-and-validation)
5. [Orchestration and Scheduling](#orchestration-and-scheduling)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Testing Strategies](#testing-strategies)
8. [Performance Optimization](#performance-optimization)
9. [Security Implementation](#security-implementation)
10. [Production Deployment](#production-deployment)

## Pipeline Development Framework

### 1. Pipeline Architecture Patterns

#### Modular Pipeline Design

```python
# Modular pipeline framework
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class PipelineContext:
    """Context shared across pipeline stages"""
    pipeline_id: str
    execution_id: str
    start_time: datetime
    configuration: Dict[str, Any]
    metadata: Dict[str, Any]
    results: Dict[str, Any] = None

    def __post_init__(self):
        if self.results is None:
            self.results = {}

class PipelineStage(ABC):
    """Abstract base class for pipeline stages"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"pipeline.{name}")

    @abstractmethod
    async def execute(self, context: PipelineContext, data: Any) -> Any:
        """Execute the pipeline stage"""
        pass

    async def validate_input(self, data: Any) -> bool:
        """Validate input data for this stage"""
        return True  # Default implementation - override as needed

    async def validate_output(self, data: Any) -> bool:
        """Validate output data from this stage"""
        return True  # Default implementation - override as needed

class DataExtractionStage(PipelineStage):
    """Extract data from various sources"""

    async def execute(self, context: PipelineContext, data: Any) -> Any:
        """Extract data based on configuration"""

        source_config = context.configuration.get('source', {})
        source_type = source_config.get('type')

        if source_type == 'database':
            return await self._extract_from_database(source_config)
        elif source_type == 'api':
            return await self._extract_from_api(source_config)
        elif source_type == 'file':
            return await self._extract_from_file(source_config)
        elif source_type == 'stream':
            return await self._extract_from_stream(source_config)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    async def _extract_from_database(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from database"""
        # Implementation for database extraction
        # This would use appropriate database drivers
        pass

    async def _extract_from_api(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from API"""
        # Implementation for API extraction
        pass

    async def _extract_from_file(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from file"""
        # Implementation for file extraction
        pass

    async def _extract_from_stream(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from streaming source"""
        # Implementation for stream extraction
        pass

class DataTransformationStage(PipelineStage):
    """Transform data according to business rules"""

    async def execute(self, context: PipelineContext, data: Any) -> Any:
        """Apply transformations to data"""

        transformations = context.configuration.get('transformations', [])

        for transform_config in transformations:
            transform_type = transform_config.get('type')

            if transform_type == 'filter':
                data = await self._apply_filter(data, transform_config)
            elif transform_type == 'aggregate':
                data = await self._apply_aggregation(data, transform_config)
            elif transform_type == 'join':
                data = await self._apply_join(data, transform_config)
            elif transform_type == 'calculate':
                data = await self._apply_calculation(data, transform_config)
            elif transform_type == 'normalize':
                data = await self._apply_normalization(data, transform_config)

        return data

    async def _apply_filter(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply filter transformation"""
        condition = config.get('condition')

        # Simple filter implementation
        if isinstance(condition, dict):
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')

            filtered_data = []
            for record in data:
                if self._evaluate_condition(record.get(field), operator, value):
                    filtered_data.append(record)
            return filtered_data

        return data

    async def _apply_aggregation(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply aggregation transformation"""
        group_by = config.get('group_by', [])
        aggregations = config.get('aggregations', [])

        # Group data by specified fields
        groups = {}
        for record in data:
            group_key = tuple(record.get(field) for field in group_by)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record)

        # Apply aggregations to each group
        aggregated_data = []
        for group_key, group_records in groups.items():
            aggregated_record = {}

            # Add group by fields
            for i, field in enumerate(group_by):
                aggregated_record[field] = group_key[i]

            # Apply aggregations
            for agg_config in aggregations:
                field = agg_config.get('field')
                function = agg_config.get('function')

                values = [record.get(field) for record in group_records if field in record]

                if function == 'sum':
                    aggregated_record[f"{field}_sum"] = sum(values)
                elif function == 'avg':
                    aggregated_record[f"{field}_avg"] = sum(values) / len(values)
                elif function == 'count':
                    aggregated_record[f"{field}_count"] = len(values)
                elif function == 'max':
                    aggregated_record[f"{field}_max"] = max(values)
                elif function == 'min':
                    aggregated_record[f"{field}_min"] = min(values)

            aggregated_data.append(aggregated_record)

        return aggregated_data

class DataLoadingStage(PipelineStage):
    """Load data into target systems"""

    async def execute(self, context: PipelineContext, data: Any) -> Any:
        """Load data to target system"""

        target_config = context.configuration.get('target', {})
        target_type = target_config.get('type')

        if target_type == 'database':
            return await self._load_to_database(data, target_config)
        elif target_type == 'data_lake':
            return await self._load_to_data_lake(data, target_config)
        elif target_type == 'data_warehouse':
            return await self._load_to_data_warehouse(data, target_config)
        elif target_type == 'api':
            return await self._load_to_api(data, target_config)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

class DataPipeline:
    """Orchestrates pipeline execution"""

    def __init__(self, pipeline_id: str, stages: List[PipelineStage]):
        self.pipeline_id = pipeline_id
        self.stages = stages
        self.logger = logging.getLogger(f"pipeline.{pipeline_id}")

    async def execute(self, configuration: Dict[str, Any], initial_data: Any = None) -> Dict[str, Any]:
        """Execute the complete pipeline"""

        execution_id = str(datetime.now().timestamp())
        context = PipelineContext(
            pipeline_id=self.pipeline_id,
            execution_id=execution_id,
            start_time=datetime.now(),
            configuration=configuration,
            metadata={}
        )

        self.logger.info(f"Starting pipeline execution: {execution_id}")

        try:
            data = initial_data

            # Execute each stage
            for stage in self.stages:
                self.logger.info(f"Executing stage: {stage.name}")

                # Validate input
                if not await stage.validate_input(data):
                    raise ValueError(f"Input validation failed for stage: {stage.name}")

                # Execute stage
                data = await stage.execute(context, data)

                # Validate output
                if not await stage.validate_output(data):
                    raise ValueError(f"Output validation failed for stage: {stage.name}")

                # Store results
                context.results[stage.name] = {
                    'status': 'success',
                    'output_type': type(data).__name__,
                    'record_count': len(data) if isinstance(data, (list, dict)) else 1
                }

                self.logger.info(f"Completed stage: {stage.name}")

            # Pipeline completed successfully
            result = {
                'execution_id': execution_id,
                'status': 'success',
                'start_time': context.start_time,
                'end_time': datetime.now(),
                'duration_seconds': (datetime.now() - context.start_time).total_seconds(),
                'stages_completed': len(self.stages),
                'results': context.results
            }

            self.logger.info(f"Pipeline completed successfully: {execution_id}")
            return result

        except Exception as e:
            # Pipeline failed
            error_result = {
                'execution_id': execution_id,
                'status': 'failed',
                'start_time': context.start_time,
                'end_time': datetime.now(),
                'duration_seconds': (datetime.now() - context.start_time).total_seconds(),
                'error': str(e),
                'stages_completed': len([s for s in context.results if context.results[s]['status'] == 'success'])
            }

            self.logger.error(f"Pipeline failed: {execution_id}, Error: {str(e)}")
            return error_result
```

### 2. Configuration Management

#### Dynamic Pipeline Configuration

```python
# Configuration management for pipelines
import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class PipelineType(Enum):
    BATCH = "batch"
    STREAM = "stream"
    HYBRID = "hybrid"

@dataclass
class DataSource:
    type: str
    connection_string: str
    credentials: Dict[str, str]
    schema: Optional[Dict[str, Any]] = None
    partitioning: Optional[Dict[str, Any]] = None

@dataclass
class DataTarget:
    type: str
    connection_string: str
    table_name: str
    write_mode: str = "append"  # append, overwrite, upsert
    partitioning: Optional[Dict[str, Any]] = None

@dataclass
class Transformation:
    type: str
    parameters: Dict[str, Any]

@dataclass
class QualityRule:
    type: str
    field: str
    parameters: Dict[str, Any]

@dataclass
class PipelineConfig:
    name: str
    type: PipelineType
    schedule: Optional[str] = None
    source: Optional[DataSource] = None
    target: Optional[DataTarget] = None
    transformations: List[Transformation] = None
    quality_rules: List[QualityRule] = None
    monitoring: Dict[str, Any] = None
    error_handling: Dict[str, Any] = None

class PipelineConfigManager:
    """Manages pipeline configurations"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.configs: Dict[str, PipelineConfig] = {}
        self._load_configs()

    def _load_configs(self):
        """Load all pipeline configurations"""
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)

            for pipeline_name, pipeline_config in config_data.get('pipelines', {}).items():
                config = self._parse_pipeline_config(pipeline_name, pipeline_config)
                self.configs[pipeline_name] = config

        except Exception as e:
            raise ConfigurationError(f"Failed to load configurations: {str(e)}")

    def _parse_pipeline_config(self, name: str, config_data: Dict[str, Any]) -> PipelineConfig:
        """Parse individual pipeline configuration"""

        # Parse source
        source = None
        if 'source' in config_data:
            source_data = config_data['source']
            source = DataSource(
                type=source_data['type'],
                connection_string=source_data['connection_string'],
                credentials=source_data.get('credentials', {}),
                schema=source_data.get('schema'),
                partitioning=source_data.get('partitioning')
            )

        # Parse target
        target = None
        if 'target' in config_data:
            target_data = config_data['target']
            target = DataTarget(
                type=target_data['type'],
                connection_string=target_data['connection_string'],
                table_name=target_data['table_name'],
                write_mode=target_data.get('write_mode', 'append'),
                partitioning=target_data.get('partitioning')
            )

        # Parse transformations
        transformations = []
        for transform_data in config_data.get('transformations', []):
            transformation = Transformation(
                type=transform_data['type'],
                parameters=transform_data.get('parameters', {})
            )
            transformations.append(transformation)

        # Parse quality rules
        quality_rules = []
        for rule_data in config_data.get('quality_rules', []):
            quality_rule = QualityRule(
                type=rule_data['type'],
                field=rule_data['field'],
                parameters=rule_data.get('parameters', {})
            )
            quality_rules.append(quality_rule)

        return PipelineConfig(
            name=name,
            type=PipelineType(config_data['type']),
            schedule=config_data.get('schedule'),
            source=source,
            target=target,
            transformations=transformations,
            quality_rules=quality_rules,
            monitoring=config_data.get('monitoring', {}),
            error_handling=config_data.get('error_handling', {})
        )

    def get_pipeline_config(self, pipeline_name: str) -> Optional[PipelineConfig]:
        """Get configuration for specific pipeline"""
        return self.configs.get(pipeline_name)

    def get_all_pipelines(self) -> Dict[str, PipelineConfig]:
        """Get all pipeline configurations"""
        return self.configs.copy()

    def update_pipeline_config(self, pipeline_name: str, updates: Dict[str, Any]):
        """Update pipeline configuration"""
        if pipeline_name not in self.configs:
            raise ValueError(f"Pipeline {pipeline_name} not found")

        # Update configuration
        config = self.configs[pipeline_name]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Save updated configuration
        self._save_configs()

    def create_pipeline_config(self, config: PipelineConfig):
        """Create new pipeline configuration"""
        if config.name in self.configs:
            raise ValueError(f"Pipeline {config.name} already exists")

        self.configs[config.name] = config
        self._save_configs()

    def delete_pipeline_config(self, pipeline_name: str):
        """Delete pipeline configuration"""
        if pipeline_name not in self.configs:
            raise ValueError(f"Pipeline {pipeline_name} not found")

        del self.configs[pipeline_name]
        self._save_configs()

    def _save_configs(self):
        """Save all configurations to file"""
        # Convert configs to dictionary format
        configs_data = {'pipelines': {}}

        for name, config in self.configs.items():
            config_dict = asdict(config)

            # Convert enum to string
            config_dict['type'] = config.type.value

            configs_data['pipelines'][name] = config_dict

        # Save to file
        with open(self.config_path, 'w') as file:
            yaml.dump(configs_data, file, default_flow_style=False, indent=2)

# Example configuration file (YAML)
"""
pipelines:
  customer_data_etl:
    type: batch
    schedule: "0 2 * * *"  # Daily at 2 AM
    source:
      type: database
      connection_string: "postgresql://user:pass@source-db:5432/source_db"
      schema:
        table: "customers"
        columns:
          - customer_id
          - name
          - email
          - created_at
    target:
      type: data_warehouse
      connection_string: "postgresql://user:pass@dw:5432/analytics"
      table_name: "dim_customers"
      write_mode: "overwrite"
    transformations:
      - type: filter
        parameters:
          condition:
            field: "status"
            operator: "eq"
            value: "active"
      - type: calculate
        parameters:
          expressions:
            - field: "customer_lifetime_days"
              formula: "current_date - created_at"
    quality_rules:
      - type: not_null
        field: "customer_id"
      - type: unique
        field: "customer_id"
      - type: date_format
        field: "created_at"
        parameters:
          format: "YYYY-MM-DD"
    monitoring:
      alerts:
        - type: "email"
          recipients: ["data-team@company.com"]
        - type: "slack"
          channel: "#data-alerts"
    error_handling:
      strategy: "retry"
      max_retries: 3
      retry_delay: 300  # 5 minutes

  real_time_fraud_detection:
    type: stream
    source:
      type: stream
      connection_string: "kafka://kafka-broker:9092/transactions"
      topic: "financial_transactions"
    target:
      type: stream
      connection_string: "kafka://kafka-broker:9092/fraud_alerts"
      topic: "fraud_alerts"
    transformations:
      - type: calculate
        parameters:
          expressions:
            - field: "amount_usd"
              formula: "amount * exchange_rate"
      - type: filter
        parameters:
          condition:
            field: "amount_usd"
            operator: "gt"
            value: 10000
    monitoring:
      alerts:
        - type: "pagerduty"
          service_key: "your-pagerduty-key"
"""
```

## ETL/ELT Pipeline Implementation

### 1. Traditional ETL Pipeline

```python
# ETL pipeline implementation with error handling and recovery
import asyncio
import aiofiles
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

@dataclass
class ETLMetrics:
    """Metrics for ETL pipeline execution"""
    start_time: datetime
    end_time: Optional[datetime] = None
    records_read: int = 0
    records_written: int = 0
    records_failed: int = 0
    error_count: int = 0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        total = self.records_read
        if total == 0:
            return 1.0
        return (total - self.records_failed) / total

class ETLPipeline:
    """Traditional ETL pipeline with comprehensive error handling"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("etl_pipeline")
        self.metrics = ETLMetrics(start_time=datetime.now())
        self.error_handler = ETLErrorHandler(config.get('error_handling', {}))

    async def execute(self) -> ETLMetrics:
        """Execute the complete ETL pipeline"""

        try:
            self.logger.info("Starting ETL pipeline execution")

            # Extract phase
            extracted_data = await self._extract()
            self.metrics.records_read = len(extracted_data)

            # Transform phase
            transformed_data = await self._transform(extracted_data)

            # Load phase
            await self._load(transformed_data)
            self.metrics.records_written = len(transformed_data)

            # Complete metrics
            self.metrics.end_time = datetime.now()

            self.logger.info(
                f"ETL pipeline completed. "
                f"Read: {self.metrics.records_read}, "
                f"Written: {self.metrics.records_written}, "
                f"Failed: {self.metrics.records_failed}, "
                f"Duration: {self.metrics.duration_seconds:.2f}s"
            )

            return self.metrics

        except Exception as e:
            self.metrics.end_time = datetime.now()
            self.metrics.error_count += 1

            self.logger.error(f"ETL pipeline failed: {str(e)}")

            # Handle error based on configuration
            await self.error_handler.handle_error(e, self.metrics)

            raise

    async def _extract(self) -> List[Dict[str, Any]]:
        """Extract data from source"""

        self.logger.info("Starting data extraction")

        source_config = self.config['source']
        source_type = source_config['type']

        try:
            if source_type == 'database':
                return await self._extract_from_database(source_config)
            elif source_type == 'file':
                return await self._extract_from_file(source_config)
            elif source_type == 'api':
                return await self._extract_from_api(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

        except Exception as e:
            self.logger.error(f"Data extraction failed: {str(e)}")
            raise ETL ExtractionError(f"Failed to extract data: {str(e)}")

    async def _extract_from_database(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from database"""

        # This would use appropriate database driver
        # Example with asyncpg for PostgreSQL
        import asyncpg

        connection = await asyncpg.connect(config['connection_string'])

        try:
            # Build query with parameters
            query = config['query']
            params = config.get('parameters', {})

            # Execute query
            rows = await connection.fetch(query, *params.values())

            # Convert to list of dictionaries
            data = [dict(row) for row in rows]

            self.logger.info(f"Extracted {len(data)} records from database")
            return data

        finally:
            await connection.close()

    async def _extract_from_file(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from file"""

        file_path = config['file_path']
        file_format = config.get('format', 'csv')

        try:
            if file_format == 'csv':
                data = pd.read_csv(file_path).to_dict('records')
            elif file_format == 'json':
                async with aiofiles.open(file_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
            elif file_format == 'parquet':
                data = pd.read_parquet(file_path).to_dict('records')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

            self.logger.info(f"Extracted {len(data)} records from file: {file_path}")
            return data

        except Exception as e:
            raise ETL ExtractionError(f"Failed to extract from file {file_path}: {str(e)}")

    async def _extract_from_api(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data from API"""

        import aiohttp

        url = config['url']
        headers = config.get('headers', {})
        params = config.get('parameters', {})

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Handle pagination if needed
                        if config.get('pagination'):
                            data = await self._handle_api_pagination(session, config, data)

                        self.logger.info(f"Extracted {len(data)} records from API: {url}")
                        return data
                    else:
                        raise ETL ExtractionError(f"API request failed with status {response.status}")

            except Exception as e:
                raise ETL ExtractionError(f"Failed to extract from API {url}: {str(e)}")

    async def _transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform extracted data"""

        self.logger.info("Starting data transformation")

        transformations = self.config.get('transformations', [])
        transformed_data = data

        for transform_config in transformations:
            transform_type = transform_config['type']

            try:
                if transform_type == 'filter':
                    transformed_data = await self._apply_filter_transform(transformed_data, transform_config)
                elif transform_type == 'map':
                    transformed_data = await self._apply_map_transform(transformed_data, transform_config)
                elif transform_type == 'aggregate':
                    transformed_data = await self._apply_aggregate_transform(transformed_data, transform_config)
                elif transform_type == 'join':
                    transformed_data = await self._apply_join_transform(transformed_data, transform_config)
                elif transform_type == 'calculate':
                    transformed_data = await self._apply_calculation_transform(transformed_data, transform_config)
                elif transform_type == 'normalize':
                    transformed_data = await self._apply_normalization_transform(transformed_data, transform_config)
                else:
                    self.logger.warning(f"Unknown transformation type: {transform_type}")

                self.logger.debug(f"Applied {transform_type} transformation. Records: {len(transformed_data)}")

            except Exception as e:
                self.logger.error(f"Transformation {transform_type} failed: {str(e)}")

                # Handle transformation error based on configuration
                error_config = self.config.get('error_handling', {})
                on_transform_error = error_config.get('on_transform_error', 'skip')

                if on_transform_error == 'skip':
                    self.metrics.warnings.append(f"Skipped transformation {transform_type}: {str(e)}")
                elif on_transform_error == 'fail':
                    raise ETL TransformationError(f"Transformation {transform_type} failed: {str(e)}")

        self.logger.info(f"Transformation completed. Records: {len(transformed_data)}")
        return transformed_data

    async def _apply_filter_transform(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply filter transformation"""

        condition = config['condition']
        field = condition['field']
        operator = condition['operator']
        value = condition['value']

        filtered_data = []
        for record in data:
            try:
                record_value = record.get(field)
                if self._evaluate_condition(record_value, operator, value):
                    filtered_data.append(record)
            except Exception as e:
                self.logger.warning(f"Failed to evaluate condition for record: {str(e)}")
                self.metrics.records_failed += 1

        return filtered_data

    async def _apply_calculation_transform(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply calculation transformation"""

        expressions = config.get('expressions', [])

        for record in data:
            for expression in expressions:
                field = expression['field']
                formula = expression['formula']

                try:
                    # Simple formula evaluation (in practice, use a safe expression evaluator)
                    result = self._evaluate_formula(formula, record)
                    record[field] = result
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {field} for record: {str(e)}")
                    self.metrics.records_failed += 1

        return data

    async def _load(self, data: List[Dict[str, Any]]) -> None:
        """Load transformed data to target"""

        self.logger.info("Starting data loading")

        target_config = self.config['target']
        target_type = target_config['type']

        try:
            if target_type == 'database':
                await self._load_to_database(data, target_config)
            elif target_type == 'file':
                await self._load_to_file(data, target_config)
            elif target_type == 'data_lake':
                await self._load_to_data_lake(data, target_config)
            else:
                raise ValueError(f"Unsupported target type: {target_type}")

            self.logger.info(f"Successfully loaded {len(data)} records to target")

        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise ETL LoadError(f"Failed to load data: {str(e)}")

    async def _load_to_database(self, data: List[Dict], config: Dict) -> None:
        """Load data to database"""

        import asyncpg

        connection = await asyncpg.connect(config['connection_string'])

        try:
            table_name = config['table_name']
            write_mode = config.get('write_mode', 'append')

            if write_mode == 'append':
                await self._append_to_table(connection, table_name, data)
            elif write_mode == 'overwrite':
                await self._overwrite_table(connection, table_name, data)
            elif write_mode == 'upsert':
                await self._upsert_to_table(connection, table_name, data)
            else:
                raise ValueError(f"Unsupported write mode: {write_mode}")

        finally:
            await connection.close()

    async def _append_to_table(self, connection, table_name: str, data: List[Dict]) -> None:
        """Append data to table"""

        if not data:
            return

        # Get column names from first record
        columns = list(data[0].keys())

        # Build insert query
        placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
        column_names = ', '.join(columns)

        query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

        # Insert records in batch
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]

            values = []
            for record in batch:
                values.extend([record[col] for col in columns])

            await connection.executemany(query, values)

class ETLErrorHandler:
    """Handles errors in ETL pipeline"""

    def __init__(self, error_config: Dict[str, Any]):
        self.config = error_config
        self.logger = logging.getLogger("etl_error_handler")

    async def handle_error(self, error: Exception, metrics: ETLMetrics):
        """Handle pipeline error"""

        error_type = type(error).__name__
        error_message = str(error)

        # Log error
        self.logger.error(f"ETL Error [{error_type}]: {error_message}")

        # Update metrics
        metrics.error_count += 1

        # Get error handling strategy
        strategy = self.config.get('strategy', 'fail')

        if strategy == 'retry':
            await self._handle_retry(error, metrics)
        elif strategy == 'continue':
            self.logger.warning("Continuing pipeline despite error")
        elif strategy == 'fail':
            raise error

        # Send alerts if configured
        await self._send_alerts(error, metrics)

    async def _handle_retry(self, error: Exception, metrics: ETLMetrics):
        """Handle error with retry logic"""

        max_retries = self.config.get('max_retries', 3)
        retry_delay = self.config.get('retry_delay', 300)  # 5 minutes

        if metrics.error_count <= max_retries:
            self.logger.info(f"Retrying after {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
        else:
            self.logger.error(f"Max retries ({max_retries}) exceeded")
            raise error

    async def _send_alerts(self, error: Exception, metrics: ETLMetrics):
        """Send error alerts"""

        alerts_config = self.config.get('alerts', [])

        for alert_config in alerts_config:
            alert_type = alert_config['type']

            try:
                if alert_type == 'email':
                    await self._send_email_alert(alert_config, error, metrics)
                elif alert_type == 'slack':
                    await self._send_slack_alert(alert_config, error, metrics)
                elif alert_type == 'pagerduty':
                    await self._send_pagerduty_alert(alert_config, error, metrics)
            except Exception as alert_error:
                self.logger.error(f"Failed to send alert: {str(alert_error)}")

# Custom exceptions
class ETL ExtractionError(Exception):
    pass

class ETL TransformationError(Exception):
    pass

class ETL LoadError(Exception):
    pass
```

### 2. Modern ELT Pipeline

```python
# ELT pipeline with dbt integration
import subprocess
import asyncio
from typing import Dict, Any, List
from pathlib import Path

class ELTPipeline:
    """Modern ELT pipeline using dbt for transformations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("elt_pipeline")
        self.dbt_project_path = config['dbt_project_path']
        self.target_schema = config.get('target_schema', 'analytics')

    async def execute(self) -> Dict[str, Any]:
        """Execute ELT pipeline"""

        self.logger.info("Starting ELT pipeline execution")
        start_time = datetime.now()

        try:
            # Step 1: Extract - Load raw data
            await self._extract_and_load()

            # Step 2: Transform using dbt
            await self._transform_with_dbt()

            # Step 3: Validate results
            validation_results = await self._validate_results()

            end_time = datetime.now()

            result = {
                'status': 'success',
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'validation_results': validation_results
            }

            self.logger.info(f"ELT pipeline completed successfully in {result['duration_seconds']:.2f}s")
            return result

        except Exception as e:
            end_time = datetime.now()

            result = {
                'status': 'failed',
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'error': str(e)
            }

            self.logger.error(f"ELT pipeline failed: {str(e)}")
            return result

    async def _extract_and_load(self) -> None:
        """Extract data from sources and load to staging"""

        self.logger.info("Starting extract and load phase")

        sources = self.config.get('sources', [])

        for source_config in sources:
            source_type = source_config['type']

            if source_type == 'database':
                await self._extract_from_db_to_staging(source_config)
            elif source_type == 'api':
                await self._extract_from_api_to_staging(source_config)
            elif source_type == 'file':
                await self._extract_from_file_to_staging(source_config)

        self.logger.info("Extract and load phase completed")

    async def _extract_from_db_to_staging(self, config: Dict[str, Any]) -> None:
        """Extract from database and load to staging tables"""

        # Use COPY or bulk insert to staging
        staging_table = f"staging_{config['table_name']}"

        # Create staging table if it doesn't exist
        await self._create_staging_table(staging_table, config)

        # Extract data and load to staging
        # This would implement the actual extraction logic
        pass

    async def _transform_with_dbt(self) -> None:
        """Run dbt transformations"""

        self.logger.info("Starting dbt transformations")

        # Set environment variables
        env = {
            'DBT_TARGET_SCHEMA': self.target_schema,
            'DBT_PROFILE_DIR': str(Path(self.dbt_project_path).parent / 'profiles')
        }

        # Run dbt commands
        commands = [
            ['dbt', 'compile', '--project-dir', self.dbt_project_path],
            ['dbt', 'run', '--project-dir', self.dbt_project_path],
            ['dbt', 'test', '--project-dir', self.dbt_project_path]
        ]

        for command in commands:
            try:
                self.logger.info(f"Running command: {' '.join(command)}")
                result = await asyncio.create_subprocess_exec(
                    *command,
                    env={**dict(os.environ), **env},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await result.communicate()

                if result.returncode != 0:
                    raise Exception(f"DBT command failed: {stderr.decode()}")

                self.logger.info(f"DBT command completed: {' '.join(command)}")

            except Exception as e:
                self.logger.error(f"DBT transformation failed: {str(e)}")
                raise

    async def _validate_results(self) -> Dict[str, Any]:
        """Validate transformation results"""

        self.logger.info("Validating transformation results")

        validation_results = {}

        # Run dbt tests
        try:
            result = await asyncio.create_subprocess_exec(
                'dbt', 'test', '--project-dir', self.dbt_project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                validation_results['dbt_tests'] = {'status': 'passed', 'output': stdout.decode()}
            else:
                validation_results['dbt_tests'] = {'status': 'failed', 'error': stderr.decode()}

        except Exception as e:
            validation_results['dbt_tests'] = {'status': 'error', 'error': str(e)}

        # Check row counts
        validation_results['row_counts'] = await self._check_row_counts()

        # Check data freshness
        validation_results['data_freshness'] = await self._check_data_freshness()

        self.logger.info("Validation completed")
        return validation_results

    async def _check_row_counts(self) -> Dict[str, str]:
        """Check row counts in target tables"""

        # This would query the database to get row counts
        # and compare against expected values
        pass

    async def _check_data_freshness(self) -> Dict[str, Any]:
        """Check if data is fresh enough"""

        # This would check timestamps in the data
        # to ensure data is within acceptable freshness bounds
        pass
```

This comprehensive guide covers the practical implementation of data engineering pipelines, from basic ETL patterns to modern ELT approaches. The examples demonstrate:

1. **Modular Architecture**: Reusable pipeline components
2. **Configuration Management**: Dynamic pipeline configuration
3. **Error Handling**: Comprehensive error recovery strategies
4. **Monitoring**: Metrics and alerting integration
5. **Testing**: Data validation and quality checks
6. **Performance**: Optimized data processing patterns

These practical implementations provide a foundation for building robust, scalable data pipelines in production environments.
