# Async Programming & Production Python - Theory Guide

## Table of Contents

1. [Introduction to Asynchronous Programming](#introduction)
2. [Event Loops and Coroutines](#event-loops)
3. [Async/Await Syntax](#async-await)
4. [Concurrency vs Parallelism](#concurrency)
5. [Async Libraries and Tools](#async-libraries)
6. [Production Patterns](#production-patterns)
7. [Error Handling in Async](#error-handling)
8. [Performance Considerations](#performance)

## Introduction to Asynchronous Programming {#introduction}

Asynchronous programming allows programs to perform multiple tasks concurrently without blocking execution. Unlike traditional synchronous programming where tasks execute one after another, async programming enables efficient handling of I/O-bound operations like network requests, file operations, and database queries.

### Key Benefits

- **Improved Performance**: Better resource utilization during I/O operations
- **Scalability**: Handle more concurrent connections and requests
- **Responsiveness**: Applications remain responsive during long-running operations
- **Resource Efficiency**: Reduced memory and CPU overhead

### When to Use Async

- Network programming (web scraping, API calls)
- Web applications (Django, Flask with async support)
- Real-time applications (chat, gaming)
- Data processing pipelines
- Microservices architecture

## Event Loops and Coroutines {#event-loops}

### Event Loop

The event loop is the core of async programming. It continuously checks for events and executes callback functions.

```python
import asyncio

# Get the current event loop
loop = asyncio.get_event_loop()

# Run until all tasks are complete
loop.run_until_complete(main())
```

### Coroutines

Coroutines are special functions that can be paused and resumed. They're defined using `async def`:

```python
import asyncio

async def simple_coroutine():
    print("Starting coroutine")
    await asyncio.sleep(1)  # Pause here
    print("Resumed after 1 second")
    return "Completed"

# Create and run coroutine
result = await simple_coroutine()
```

### Task Management

```python
async def main():
    # Create multiple tasks
    tasks = [
        asyncio.create_task(simple_coroutine()),
        asyncio.create_task(another_coroutine()),
        asyncio.create_task(third_coroutine())
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    return results
```

## Async/Await Syntax {#async-await}

### Async Functions

```python
async def fetch_data(url: str) -> dict:
    """Async function to fetch data from URL"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def process_multiple_urls(urls: list) -> list:
    """Process multiple URLs concurrently"""
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)
```

### Awaitable Objects

Objects that can be awaited:

- Coroutines
- Tasks
- Futures

```python
# Creating tasks
task1 = asyncio.create_task(fetch_data("https://api.example.com/data1"))
task2 = asyncio.create_task(fetch_data("https://api.example.com/data2"))

# Awaiting tasks
result1 = await task1
result2 = await task2
```

## Concurrency vs Parallelism {#concurrency}

### Concurrency

- Multiple tasks make progress without blocking each other
- Single-threaded or multi-threaded execution
- Efficient for I/O-bound operations

### Parallelism

- Multiple tasks execute simultaneously on different cores
- Requires multiple threads or processes
- Best for CPU-bound operations

### Choosing the Right Approach

```python
# Concurrency for I/O-bound operations
async def io_bound_example():
    urls = ["https://api1.com", "https://api2.com", "https://api3.com"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Parallelism for CPU-bound operations
import multiprocessing

def cpu_bound_task(n):
    return sum(i**2 for i in range(n))

def parallel_example():
    with multiprocessing.Pool() as pool:
        results = pool.map(cpu_bound_task, [1000000, 2000000, 3000000])
        return results
```

## Async Libraries and Tools {#async-libraries}

### HTTP Clients

```python
# aiohttp - Async HTTP client/server
import aiohttp

async def fetch_with_aiohttp(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# httpx - Modern async HTTP client
import httpx

async def fetch_with_httpx(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Web Frameworks

```python
# FastAPI - Modern async web framework
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/data/{item_id}")
async def get_item(item_id: int):
    # Simulate async database operation
    await asyncio.sleep(0.1)
    return {"item_id": item_id, "name": f"Item {item_id}"}

# Quart - Flask-like async framework
from quart import Quart
import asyncio

app = Quart(__name__)

@app.route("/api/data")
async def get_data():
    data = await fetch_data_from_database()
    return {"data": data}
```

### Database Drivers

```python
# Async database drivers
import asyncpg  # PostgreSQL
import aiomysql  # MySQL
import motor  # MongoDB

# Example with asyncpg
async def connect_to_db():
    conn = await asyncpg.connect(
        "postgresql://user:password@localhost/dbname"
    )
    return conn

async def query_database(conn, query):
    return await conn.fetch(query)
```

## Production Patterns {#production-patterns}

### Resource Management

```python
class AsyncResourceManager:
    def __init__(self):
        self.connections = []

    async def __aenter__(self):
        # Initialize resources
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup resources
        for connection in self.connections:
            await connection.close()

# Usage
async def example_usage():
    async with AsyncResourceManager() as manager:
        # Use resources
        result = await manager.process_data()
        return result
```

### Connection Pooling

```python
import asyncpg
import asyncio

class DatabasePool:
    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20
        )

    async def execute_query(self, query, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
```

### Graceful Shutdown

```python
import asyncio
import signal

class AsyncApplication:
    def __init__(self):
        self.shutdown_event = asyncio.Event()

    async def setup_signal_handlers(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            self.shutdown_event.set
        )

    async def run(self):
        await self.setup_signal_handlers()

        while not self.shutdown_event.is_set():
            try:
                await self.process_requests()
            except Exception as e:
                await self.handle_error(e)

    async def shutdown(self):
        self.shutdown_event.set()
        # Cleanup operations
        await self.cleanup_resources()
```

## Error Handling in Async {#error-handling}

### Exception Propagation

```python
async def handle_exceptions():
    try:
        result = await risky_operation()
        return result
    except SpecificError as e:
        await handle_specific_error(e)
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### Timeout Handling

```python
import asyncio

async def timeout_example():
    try:
        result = await asyncio.wait_for(
            long_running_task(),
            timeout=10.0
        )
        return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
```

### Retry Mechanisms

```python
async def retry_operation(operation, max_retries=3, delay=1):
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
```

## Performance Considerations {#performance}

### Memory Management

```python
# Use async context managers for cleanup
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        data = await response.read()

# Avoid memory leaks in long-running applications
import gc

async def memory_efficient_processing():
    for chunk in large_dataset:
        # Process chunk
        process(chunk)

        # Periodically force garbage collection
        if gc.collect() > 0:
            pass
```

### Monitoring and Profiling

```python
import time
import asyncio

async def profile_async_function(func):
    start_time = time.time()
    try:
        result = await func()
        return result
    finally:
        end_time = time.time()
        print(f"Function took {end_time - start_time:.2f} seconds")

# Usage
await profile_async_function(heavy_computation)
```

### Connection Limits

```python
import asyncio
import aiohttp

# Limit concurrent connections
async def limited_requests(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(url):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()

    tasks = [fetch_with_limit(url) for url in urls]
    return await asyncio.gather(*tasks)
```

## Best Practices

### 1. Avoid Blocking Operations

```python
# Good
async def good_example():
    await async_database_query()
    await async_file_read()

# Bad - blocks the event loop
import time
async def bad_example():
    time.sleep(5)  # This blocks!
    return "Done"
```

### 2. Use Appropriate Async Libraries

```python
# Good - use async versions
import aiofiles  # for async file operations
import asyncpg    # for async database

# Bad - use blocking libraries
import requests  # blocks for HTTP
import sqlite3   # blocks for database
```

### 3. Proper Error Handling

```python
async def robust_async_function():
    try:
        result = await potentially_failing_operation()
        return result
    except asyncio.TimeoutError:
        logger.warning("Operation timed out")
        return default_value
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise
```

### 4. Context Switching

```python
# Minimize context switching
async def batch_operations():
    # Group related operations
    db_operations = [async_db_query(sql) for sql in queries]
    http_operations = [async_http_request(url) for url in urls]

    # Execute in logical groups
    db_results = await asyncio.gather(*db_operations)
    http_results = await asyncio.gather(*http_operations)
```

## Summary

Async programming is essential for building scalable, high-performance Python applications. Key concepts include:

- **Event Loop**: Core mechanism for async execution
- **Coroutines**: Functions that can be paused and resumed
- **Tasks**: Scheduled coroutines for concurrent execution
- **Resource Management**: Proper cleanup and connection handling
- **Error Handling**: Robust exception management
- **Performance**: Efficient memory and connection management

Master these concepts to build production-ready async Python applications that can handle thousands of concurrent operations efficiently.
