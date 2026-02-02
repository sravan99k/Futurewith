# Async Programming & Production Python - Practice Exercises

## Table of Contents

1. [Basic Async Concepts](#basic-concepts)
2. [Event Loop Exercises](#event-loop-exercises)
3. [Task Management](#task-management)
4. [Async HTTP Client Exercises](#async-http-client)
5. [Database Operations](#database-operations)
6. [Production Patterns](#production-patterns)
7. [Error Handling](#error-handling)
8. [Performance Optimization](#performance)

## Basic Async Concepts {#basic-concepts}

### Exercise 1: Create Your First Coroutine

```python
import asyncio
import time

# Exercise: Create a simple coroutine that demonstrates async behavior
async def greet_with_delay(name, delay=1):
    """Async function that greets after a delay"""
    print(f"Starting to greet {name}...")
    await asyncio.sleep(delay)
    print(f"Hello, {name}! (after {delay} seconds)")
    return f"Greeting completed for {name}"

# Practice: Run multiple greetings concurrently
async def practice_basic_async():
    print("=== Basic Async Practice ===")

    # Sequential execution (slow)
    print("\n1. Sequential execution:")
    start_time = time.time()
    result1 = await greet_with_delay("Alice", 1)
    result2 = await greet_with_delay("Bob", 1)
    result3 = await greet_with_delay("Charlie", 1)
    sequential_time = time.time() - start_time
    print(f"Sequential took: {sequential_time:.2f} seconds")

    # Concurrent execution (fast)
    print("\n2. Concurrent execution:")
    start_time = time.time()
    results = await asyncio.gather(
        greet_with_delay("Alice", 1),
        greet_with_delay("Bob", 1),
        greet_with_delay("Charlie", 1)
    )
    concurrent_time = time.time() - start_time
    print(f"Concurrent took: {concurrent_time:.2f} seconds")

    print(f"\nSpeedup: {sequential_time / concurrent_time:.2f}x faster!")
    return results

# Run the practice
# asyncio.run(practice_basic_async())
```

### Exercise 2: Async File Operations

```python
import asyncio
import aiofiles
import os

# Exercise: Create async file operations
async def async_file_reader(filename):
    """Async file reading"""
    try:
        async with aiofiles.open(filename, 'r') as file:
            content = await file.read()
            return content
    except FileNotFoundError:
        return f"File {filename} not found"

async def async_file_writer(filename, content):
    """Async file writing"""
    async with aiofiles.open(filename, 'w') as file:
        await file.write(content)
    return f"Written to {filename}"

# Practice: Concurrent file operations
async def practice_file_operations():
    print("=== Async File Operations Practice ===")

    # Create multiple test files
    files_data = {
        "test1.txt": "Content of file 1",
        "test2.txt": "Content of file 2",
        "test3.txt": "Content of file 3"
    }

    # Write files concurrently
    write_tasks = [
        async_file_writer(filename, content)
        for filename, content in files_data.items()
    ]
    write_results = await asyncio.gather(*write_tasks)

    # Read files concurrently
    read_tasks = [
        async_file_reader(filename)
        for filename in files_data.keys()
    ]
    read_results = await asyncio.gather(*read_tasks)

    print("Write results:", write_results)
    print("Read results:", read_results)

    # Cleanup
    for filename in files_data.keys():
        if os.path.exists(filename):
            os.remove(filename)

# Run the practice
# asyncio.run(practice_file_operations())
```

## Event Loop Exercises {#event-loop-exercises}

### Exercise 3: Custom Event Loop Control

```python
import asyncio
import threading
import time

# Exercise: Manual event loop control
async def long_running_task(task_id, duration=3):
    """Simulates a long-running async task"""
    print(f"Task {task_id} starting...")
    await asyncio.sleep(duration)
    print(f"Task {task_id} completed!")
    return f"Result from task {task_id}"

async def event_loop_practice():
    print("=== Event Loop Practice ===")

    # Method 1: asyncio.run() - simplest
    print("\n1. Using asyncio.run():")
    result1 = await long_running_task("A", 2)

    # Method 2: Manual event loop control
    print("\n2. Manual event loop control:")
    loop = asyncio.get_event_loop()
    result2 = await loop.create_task(long_running_task("B", 2))

    # Method 3: Running in a separate thread
    print("\n3. Event loop in separate thread:")

    def run_async_in_thread():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(long_running_task("C", 1))
        finally:
            new_loop.close()

    thread_result = await asyncio.get_event_loop().run_in_executor(
        None, run_async_in_thread
    )

    print(f"Results: {result1}, {result2}, {thread_result}")

# Run the practice
# asyncio.run(event_loop_practice())
```

### Exercise 4: Event Loop Monitoring

```python
import asyncio
import psutil
import time

# Exercise: Monitor event loop performance
class AsyncMonitor:
    def __init__(self):
        self.start_time = None
        self.completed_tasks = 0

    async def monitored_task(self, task_id, duration=1):
        """Task with monitoring"""
        if not self.start_time:
            self.start_time = time.time()

        print(f"Task {task_id} started at {time.time() - self.start_time:.2f}s")
        await asyncio.sleep(duration)

        self.completed_tasks += 1
        print(f"Task {task_id} completed. Total completed: {self.completed_tasks}")
        return task_id

async def monitor_event_loop():
    print("=== Event Loop Monitoring Practice ===")

    monitor = AsyncMonitor()

    # Create and monitor multiple tasks
    tasks = [
        monitor.monitored_task(i, duration=0.5)
        for i in range(5)
    ]

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    print(f"\nAll tasks completed in {end_time - start_time:.2f} seconds")
    print(f"Results: {results}")

# Run the practice
# asyncio.run(monitor_event_loop())
```

## Task Management {#task-management}

### Exercise 5: Task Creation and Management

```python
import asyncio
import uuid

# Exercise: Advanced task management
class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.results = {}

    async def create_task_with_id(self, func, task_id=None):
        """Create a task with custom ID"""
        if task_id is None:
            task_id = str(uuid.uuid4())

        task = asyncio.create_task(func(), name=task_id)
        self.tasks[task_id] = task
        return task_id

    async def run_task_group(self, functions, max_concurrent=3):
        """Run tasks with concurrency limit"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_task(func, task_id):
            async with semaphore:
                return await self.create_task_with_id(func, task_id)

        tasks = [
            limited_task(func, f"task_{i}")
            for i, func in enumerate(functions)
        ]

        return await asyncio.gather(*tasks)

    async def wait_for_task(self, task_id, timeout=None):
        """Wait for specific task with timeout"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
            self.results[task_id] = result
            return result
        except asyncio.TimeoutError:
            print(f"Task {task_id} timed out!")
            return None

    def cancel_task(self, task_id):
        """Cancel a specific task"""
        if task_id in self.tasks:
            self.tasks[task_id].cancel()
            del self.tasks[task_id]
            return True
        return False

# Practice functions
async def sample_task(duration, result_value):
    await asyncio.sleep(duration)
    return result_value

async def task_management_practice():
    print("=== Task Management Practice ===")

    manager = TaskManager()

    # Create various tasks
    functions = [
        lambda: sample_task(1, "Task 1"),
        lambda: sample_task(2, "Task 2"),
        lambda: sample_task(1.5, "Task 3"),
        lambda: sample_task(0.5, "Task 4"),
        lambda: sample_task(3, "Task 5")
    ]

    print("\n1. Running tasks with concurrency limit:")
    task_ids = await manager.run_task_group(functions, max_concurrent=3)
    print(f"Created tasks: {task_ids}")

    print("\n2. Waiting for specific tasks:")
    for task_id in task_ids[:3]:
        result = await manager.wait_for_task(task_id, timeout=2)
        print(f"Task {task_id}: {result}")

    print("\n3. Canceling remaining tasks:")
    for task_id in task_ids[3:]:
        manager.cancel_task(task_id)
        print(f"Canceled task {task_id}")

# Run the practice
# asyncio.run(task_management_practice())
```

## Async HTTP Client Exercises {#async-http-client}

### Exercise 6: Async HTTP Client (requires aiohttp)

```python
import asyncio
import aiohttp
import json

# Exercise: Async HTTP operations
class AsyncHTTPClient:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_url(self, url, method='GET', data=None, headers=None):
        """Fetch URL with error handling"""
        try:
            if method.upper() == 'GET':
                async with self.session.get(url, headers=headers) as response:
                    return {
                        'url': url,
                        'status': response.status,
                        'content': await response.text(),
                        'headers': dict(response.headers)
                    }
            elif method.upper() == 'POST':
                async with self.session.post(url, json=data, headers=headers) as response:
                    return {
                        'url': url,
                        'status': response.status,
                        'content': await response.text(),
                        'headers': dict(response.headers)
                    }
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'status': None
            }

    async def fetch_multiple_urls(self, urls, max_concurrent=5):
        """Fetch multiple URLs with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(url):
            async with semaphore:
                return await self.fetch_url(url)

        tasks = [fetch_with_limit(url) for url in urls]
        return await asyncio.gather(*tasks)

# Practice exercises
async def http_client_practice():
    print("=== Async HTTP Client Practice ===")

    # Test URLs (replace with actual APIs for real testing)
    test_urls = [
        "https://httpbin.org/json",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/status/200",
        "https://httpbin.org/uuid",
        "https://httpbin.org/user-agent"
    ]

    async with AsyncHTTPClient() as client:
        print("\n1. Fetching multiple URLs concurrently:")
        start_time = time.time()
        results = await client.fetch_multiple_urls(test_urls, max_concurrent=3)
        end_time = time.time()

        print(f"Completed in {end_time - start_time:.2f} seconds")

        for result in results:
            if 'error' in result:
                print(f"Error for {result['url']}: {result['error']}")
            else:
                print(f"✓ {result['url']}: Status {result['status']}")

        print("\n2. Testing POST request:")
        post_data = {"test": "data", "timestamp": time.time()}
        post_result = await client.fetch_url(
            "https://httpbin.org/post",
            method='POST',
            data=post_data
        )
        print(f"POST result: {post_result['status']}")

# Run the practice
# asyncio.run(http_client_practice())
```

## Database Operations {#database-operations}

### Exercise 7: Async Database Simulation

```python
import asyncio
import json
from typing import List, Dict, Any

# Exercise: Simulate async database operations
class AsyncDatabase:
    def __init__(self):
        self.data = {}
        self.connection_pool = []

    async def connect(self):
        """Simulate database connection"""
        print("Connecting to database...")
        await asyncio.sleep(0.1)  # Simulate connection time
        self.connection_pool = [f"connection_{i}" for i in range(5)]
        print("Database connected!")

    async def execute_query(self, query, params=None):
        """Execute async query with simulated delay"""
        # Simulate query execution time
        query_time = 0.1 + len(query) * 0.001
        await asyncio.sleep(query_time)

        if query.upper().startswith("SELECT"):
            # Simulate SELECT query
            return {
                'query': query,
                'results': self.data.get('users', []),
                'row_count': len(self.data.get('users', [])),
                'execution_time': query_time
            }
        elif query.upper().startswith("INSERT"):
            # Simulate INSERT query
            user_data = params or {}
            user_id = len(self.data.get('users', [])) + 1
            user_data['id'] = user_id

            if 'users' not in self.data:
                self.data['users'] = []
            self.data['users'].append(user_data)

            return {
                'query': query,
                'inserted_id': user_id,
                'row_count': 1,
                'execution_time': query_time
            }

    async def batch_insert(self, users_data: List[Dict]):
        """Batch insert multiple users"""
        tasks = [
            self.execute_query(
                "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
                [user.get('name'), user.get('email'), user.get('age')]
            ) for user in users_data
        ]
        return await asyncio.gather(*tasks)

# Practice exercises
async def database_practice():
    print("=== Async Database Practice ===")

    db = AsyncDatabase()
    await db.connect()

    # Exercise 1: Single queries
    print("\n1. Sequential queries:")
    start_time = time.time()

    for i in range(3):
        result = await db.execute_query(
            "INSERT INTO users (name, email, age) VALUES (?, ?, ?)",
            [f"User{i}", f"user{i}@email.com", 25 + i]
        )
        print(f"Inserted user {i}: ID {result['inserted_id']}")

    sequential_time = time.time() - start_time

    # Exercise 2: Batch operations
    print("\n2. Batch insertion:")
    start_time = time.time()

    batch_users = [
        {'name': f'BatchUser{i}', 'email': f'batch{i}@email.com', 'age': 30}
        for i in range(3)
    ]

    batch_results = await db.batch_insert(batch_users)
    print(f"Batch inserted {len(batch_results)} users")

    batch_time = time.time() - start_time

    # Exercise 3: Concurrent queries
    print("\n3. Concurrent queries:")
    start_time = time.time()

    query_tasks = [
        db.execute_query("SELECT * FROM users") for _ in range(5)
    ]

    query_results = await asyncio.gather(*query_tasks)
    concurrent_time = time.time() - start_time

    print(f"\nTiming Results:")
    print(f"Sequential: {sequential_time:.3f}s")
    print(f"Batch: {batch_time:.3f}s")
    print(f"Concurrent: {concurrent_time:.3f}s")
    print(f"\nUsers in database: {query_results[0]['row_count']}")

# Run the practice
# asyncio.run(database_practice())
```

## Production Patterns {#production-patterns}

### Exercise 8: Resource Management with Context Managers

```python
import asyncio
import contextlib

# Exercise: Advanced resource management
class AsyncResourcePool:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.available_resources = asyncio.Queue(maxsize=max_size)
        self.allocated_resources = set()

    async def initialize(self):
        """Initialize the resource pool"""
        for i in range(self.max_size):
            resource = f"resource_{i}"
            await self.available_resources.put(resource)

    @contextlib.asynccontextmanager
    async def acquire(self):
        """Acquire resource from pool"""
        resource = await self.available_resources.get()
        self.allocated_resources.add(resource)
        try:
            yield resource
        finally:
            self.allocated_resources.remove(resource)
            await self.available_resources.put(resource)

    async def use_resource(self, resource_id, task_duration):
        """Simulate using a resource"""
        async with self.acquire() as resource:
            print(f"Using {resource} for task {resource_id}")
            await asyncio.sleep(task_duration)
            print(f"Finished {resource} for task {resource_id}")
            return f"Completed task {resource_id}"

# Exercise: Configuration management
class AsyncConfig:
    def __init__(self):
        self.config_data = {}
        self.reload_event = asyncio.Event()

    async def load_config(self, config_path):
        """Load configuration asynchronously"""
        print(f"Loading config from {config_path}")
        await asyncio.sleep(0.1)  # Simulate file I/O

        # Simulate config data
        self.config_data = {
            'database_url': 'postgresql://localhost/db',
            'max_connections': 100,
            'timeout': 30,
            'features': ['async', 'monitoring', 'logging']
        }

        self.reload_event.set()
        print("Configuration loaded!")

    async def watch_config(self):
        """Watch for config changes"""
        while True:
            await self.reload_event.wait()
            print(f"Config updated: {len(self.config_data)} keys")
            self.reload_event.clear()
            await asyncio.sleep(1)  # Check every second

# Practice exercises
async def production_patterns_practice():
    print("=== Production Patterns Practice ===")

    # Exercise 1: Resource pool
    print("\n1. Resource Pool Management:")
    pool = AsyncResourcePool(max_size=3)
    await pool.initialize()

    # Create tasks that use resources
    tasks = [
        pool.use_resource(f"task_{i}", duration=0.5)
        for i in range(6)
    ]

    results = await asyncio.gather(*tasks)
    print(f"Pool results: {results}")

    # Exercise 2: Configuration management
    print("\n2. Configuration Management:")
    config = AsyncConfig()

    # Load config and watch for changes
    config_task = asyncio.create_task(config.load_config("config.json"))
    watch_task = asyncio.create_task(config.watch_config())

    await config_task
    print(f"Initial config: {config.config_data}")

    # Simulate config reload
    await asyncio.sleep(0.2)
    await config.load_config("config.json")  # This triggers the watch

    watch_task.cancel()

# Run the practice
# asyncio.run(production_patterns_practice())
```

## Error Handling {#error-handling}

### Exercise 9: Comprehensive Error Handling

```python
import asyncio
import logging
from enum import Enum
from typing import Optional

# Exercise: Advanced error handling patterns
class ErrorType(Enum):
    NETWORK = "network_error"
    DATABASE = "database_error"
    VALIDATION = "validation_error"
    TIMEOUT = "timeout_error"

class AsyncErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {error_type: 0 for error_type in ErrorType}

    async def safe_operation(self, operation, error_type: ErrorType, max_retries=3, timeout=5):
        """Execute operation with comprehensive error handling"""
        for attempt in range(max_retries + 1):
            try:
                # Add timeout to operation
                result = await asyncio.wait_for(operation(), timeout=timeout)
                self.logger.info(f"Operation succeeded on attempt {attempt + 1}")
                return result

            except asyncio.TimeoutError:
                self.error_counts[ErrorType.TIMEOUT] += 1
                self.logger.warning(f"Timeout on attempt {attempt + 1} for {error_type.value}")

            except Exception as e:
                self.error_counts[error_type] += 1
                self.logger.error(f"Error on attempt {attempt + 1}: {e}")

                if attempt == max_retries:
                    self.logger.error(f"Max retries reached for {error_type.value}")
                    raise

                # Exponential backoff
                delay = (2 ** attempt)
                await asyncio.sleep(delay)

    async def circuit_breaker(self, operation, failure_threshold=3, timeout=10):
        """Circuit breaker pattern"""
        failure_count = 0
        last_failure_time = None

        while True:
            if failure_count >= failure_threshold:
                # Check if timeout has passed
                if last_failure_time and (time.time() - last_failure_time) < timeout:
                    await asyncio.sleep(1)
                    continue
                else:
                    # Reset circuit breaker
                    failure_count = 0
                    self.logger.info("Circuit breaker reset")

            try:
                result = await operation()
                failure_count = 0  # Reset on success
                return result

            except Exception as e:
                failure_count += 1
                last_failure_time = time.time()
                self.logger.error(f"Circuit breaker failure: {e}")

                if failure_count >= failure_threshold:
                    self.logger.error("Circuit breaker opened")

                await asyncio.sleep(0.1)

# Mock operations for testing
async def mock_network_operation():
    """Simulate network operation that may fail"""
    await asyncio.sleep(0.1)
    if random.random() < 0.3:  # 30% failure rate
        raise ConnectionError("Network connection failed")
    return "Network operation successful"

async def mock_database_operation():
    """Simulate database operation"""
    await asyncio.sleep(0.2)
    if random.random() < 0.2:  # 20% failure rate
        raise DatabaseError("Database query failed")
    return "Database operation successful"

class DatabaseError(Exception):
    pass

async def error_handling_practice():
    print("=== Error Handling Practice ===")

    handler = AsyncErrorHandler()

    # Exercise 1: Safe operations with retry
    print("\n1. Safe operations with retry:")

    try:
        result = await handler.safe_operation(
            lambda: mock_network_operation(),
            ErrorType.NETWORK,
            max_retries=3,
            timeout=2
        )
        print(f"Network operation result: {result}")
    except Exception as e:
        print(f"Network operation failed: {e}")

    try:
        result = await handler.safe_operation(
            lambda: mock_database_operation(),
            ErrorType.DATABASE,
            max_retries=2,
            timeout=1
        )
        print(f"Database operation result: {result}")
    except Exception as e:
        print(f"Database operation failed: {e}")

    # Exercise 2: Circuit breaker
    print("\n2. Circuit breaker pattern:")

    async def unreliable_operation():
        await asyncio.sleep(0.1)
        if random.random() < 0.8:  # 80% failure rate
            raise ConnectionError("Operation failed")
        return "Operation successful"

    try:
        result = await handler.circuit_breaker(unreliable_operation, failure_threshold=3, timeout=5)
        print(f"Circuit breaker result: {result}")
    except Exception as e:
        print(f"Circuit breaker failed: {e}")

    print(f"\nError statistics: {handler.error_counts}")

# Run the practice
# asyncio.run(error_handling_practice())
```

## Performance Optimization {#performance}

### Exercise 10: Performance Monitoring and Optimization

```python
import asyncio
import psutil
import time
from dataclasses import dataclass
from typing import List, Dict, Any

# Exercise: Performance monitoring and optimization
@dataclass
class PerformanceMetrics:
    operation_name: str
    start_time: float
    end_time: float
    memory_usage: float
    cpu_usage: float
    result_size: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def throughput(self) -> float:
        return self.result_size / self.duration if self.duration > 0 else 0

class AsyncPerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.baseline_memory = psutil.Process().memory_info().rss

    async def measure_operation(self, operation_name: str, operation, *args, **kwargs):
        """Measure performance of an async operation"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = await operation(*args, **kwargs)
            result_size = len(str(result)) if result else 0

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            # Estimate CPU usage (simplified)
            cpu_usage = psutil.cpu_percent(interval=0.1)

            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                memory_usage=(end_memory - start_memory) / 1024 / 1024,  # MB
                cpu_usage=cpu_usage,
                result_size=result_size
            )

            self.metrics.append(metrics)
            return result

        except Exception as e:
            end_time = time.time()
            print(f"Operation {operation_name} failed: {e}")
            raise

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {}

        durations = [m.duration for m in self.metrics]
        memory_usage = [m.memory_usage for m in self.metrics]
        throughputs = [m.throughput for m in self.metrics]

        return {
            'total_operations': len(self.metrics),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_memory_mb': sum(memory_usage) / len(memory_usage),
            'avg_throughput': sum(throughputs) / len(throughputs),
            'total_execution_time': sum(durations)
        }

    def print_report(self):
        """Print detailed performance report"""
        summary = self.get_performance_summary()
        if not summary:
            print("No performance data available")
            return

        print("\n=== Performance Report ===")
        print(f"Total Operations: {summary['total_operations']}")
        print(f"Average Duration: {summary['avg_duration']:.3f}s")
        print(f"Duration Range: {summary['min_duration']:.3f}s - {summary['max_duration']:.3f}s")
        print(f"Average Memory Usage: {summary['avg_memory_mb']:.2f} MB")
        print(f"Average Throughput: {summary['avg_throughput']:.0f} chars/sec")
        print(f"Total Execution Time: {summary['total_execution_time']:.3f}s")

# Practice operations
async def data_processing_task(data_size: int) -> str:
    """Simulate data processing"""
    data = list(range(data_size))

    # Simulate processing
    processed = []
    for item in data:
        processed.append(item ** 2)
        if len(processed) % 1000 == 0:
            await asyncio.sleep(0.001)  # Yield control

    return f"Processed {len(processed)} items"

async def concurrent_batch_processing(items: List[int]) -> List[str]:
    """Process items in batches for better performance"""
    batch_size = 10
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [data_processing_task(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

        # Small delay between batches
        await asyncio.sleep(0.01)

    return results

async def performance_practice():
    print("=== Performance Optimization Practice ===")

    monitor = AsyncPerformanceMonitor()

    # Exercise 1: Single operation performance
    print("\n1. Single operation performance:")

    result = await monitor.measure_operation(
        "data_processing_small",
        data_processing_task,
        data_size=1000
    )
    print(f"Result: {result}")

    # Exercise 2: Concurrent processing
    print("\n2. Concurrent batch processing:")

    items = [100, 200, 300, 400, 500]

    result = await monitor.measure_operation(
        "concurrent_batch_processing",
        concurrent_batch_processing,
        items
    )
    print(f"Processed {len(result)} batches")

    # Exercise 3: Memory-efficient processing
    print("\n3. Memory-efficient processing:")

    async def memory_efficient_processing():
        """Process large dataset efficiently"""
        async for chunk in generate_large_dataset(10000):
            await process_chunk(chunk)
            await asyncio.sleep(0)  # Yield control

    async def generate_large_dataset(size):
        """Generate data in chunks"""
        chunk_size = 1000
        for i in range(0, size, chunk_size):
            yield list(range(i, min(i + chunk_size, size)))

    async def process_chunk(chunk):
        """Process a single chunk"""
        # Simulate processing
        await asyncio.sleep(0.001)
        return sum(chunk)

    await monitor.measure_operation(
        "memory_efficient_processing",
        memory_efficient_processing
    )

    # Print performance report
    monitor.print_report()

# Run the practice
# asyncio.run(performance_practice())
```

## Summary

These practice exercises cover all essential aspects of async programming:

1. **Basic Concepts**: Coroutines, event loops, and async/await syntax
2. **File Operations**: Async file I/O with proper resource management
3. **Event Loop Control**: Manual event loop management and monitoring
4. **Task Management**: Creating, managing, and monitoring async tasks
5. **HTTP Clients**: Async HTTP operations with concurrency control
6. **Database Operations**: Async database simulation and batch operations
7. **Production Patterns**: Resource pooling and configuration management
8. **Error Handling**: Comprehensive error handling with retry mechanisms
9. **Performance**: Monitoring and optimizing async operations

Each exercise includes realistic scenarios that you'll encounter in production applications. Practice these patterns to master async Python development!

## Running the Exercises

To run these exercises, uncomment the function calls at the bottom of each exercise and execute:

```python
import asyncio

# Run individual exercises
# asyncio.run(practice_basic_async())
# asyncio.run(http_client_practice())
# asyncio.run(database_practice())
# asyncio.run(production_patterns_practice())
# asyncio.run(error_handling_practice())
# asyncio.run(performance_practice())
```
