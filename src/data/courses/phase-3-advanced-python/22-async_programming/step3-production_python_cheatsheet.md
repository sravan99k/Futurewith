# Async Programming & Production Python - Quick Reference Cheatsheet

## Table of Contents

1. [Core Async Concepts](#core-concepts)
2. [Essential Syntax](#essential-syntax)
3. [Common Patterns](#common-patterns)
4. [Error Handling](#error-handling)
5. [Performance Tips](#performance-tips)
6. [Production Checklist](#production-checklist)
7. [Quick Code Snippets](#quick-code-snippets)

## Core Async Concepts {#core-concepts}

### Async vs Sync Comparison

```python
# SYNC - Blocks execution
import requests
def sync_api_call():
    response = requests.get("https://api.example.com/data")
    return response.json()

# ASYNC - Non-blocking
import aiohttp
import asyncio
async def async_api_call():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            return await response.json()
```

### Event Loop Basics

```python
import asyncio

# Method 1: Simple (Python 3.7+)
async def main():
    await some_async_function()

asyncio.run(main())

# Method 2: Manual control
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

## Essential Syntax {#essential-syntax}

### Coroutine Definition

```python
# Basic coroutine
async def greet(name: str) -> str:
    await asyncio.sleep(1)
    return f"Hello, {name}!"

# Call coroutine
result = await greet("Alice")
```

### Task Creation and Management

```python
# Create tasks
task1 = asyncio.create_task(greet("Alice"))
task2 = asyncio.create_task(greet("Bob"))

# Wait for all tasks
results = await asyncio.gather(task1, task2)

# Wait for any task to complete
done, pending = await asyncio.wait(
    [task1, task2],
    return_when=asyncio.FIRST_COMPLETED
)

# Cancel tasks
task1.cancel()
task2.cancel()
```

### Concurrency Control

```python
# Semaphore - limit concurrent operations
semaphore = asyncio.Semaphore(5)

async def limited_operation():
    async with semaphore:
        # Only 5 operations run at a time
        await do_work()

# Queue - producer/consumer pattern
async def producer(queue):
    for item in items:
        await queue.put(item)
    await queue.put(None)  # Signal completion

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await process(item)
```

## Common Patterns {#common-patterns}

### HTTP Client (aiohttp)

```python
import aiohttp

# Session management
async with aiohttp.ClientSession() as session:
    # GET request
    async with session.get('https://api.example.com/data') as response:
        data = await response.json()

    # POST request
    async with session.post('https://api.example.com/data',
                           json={'key': 'value'}) as response:
        result = await response.json()

# Concurrent requests
async def fetch_multiple_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]
```

### File Operations (aiofiles)

```python
import aiofiles

# Read file
async def read_file(filename):
    async with aiofiles.open(filename, 'r') as file:
        return await file.read()

# Write file
async def write_file(filename, content):
    async with aiofiles.open(filename, 'w') as file:
        await file.write(content)

# Multiple files
async def process_files(filenames):
    async with aiofiles.open(filenames[0], 'r') as f1, \
                 aiofiles.open(filenames[1], 'r') as f2:
        content1 = await f1.read()
        content2 = await f2.read()
        return content1 + content2
```

### Database Operations

```python
# PostgreSQL with asyncpg
import asyncpg

async def db_operations():
    conn = await asyncpg.connect('postgresql://user:pass@localhost/db')

    # Single query
    row = await conn.fetchrow('SELECT * FROM users WHERE id = $1', 1)

    # Multiple queries
    users = await conn.fetch('SELECT * FROM users')

    # Transaction
    async with conn.transaction():
        await conn.execute('INSERT INTO users (name) VALUES ($1)', 'Alice')
        await conn.execute('INSERT INTO users (name) VALUES ($1)', 'Bob')

    await conn.close()

# MySQL with aiomysql
import aiomysql

async def mysql_operations():
    conn = await aiomysql.connect(
        host='localhost',
        user='user',
        password='pass',
        db='mydb'
    )

    async with conn.cursor() as cur:
        await cur.execute('SELECT * FROM users')
        rows = await cur.fetchall()

    conn.close()
```

### Context Managers

```python
# Custom async context manager
class AsyncResourceManager:
    def __init__(self):
        self.resource = None

    async def __aenter__(self):
        self.resource = await acquire_resource()
        return self.resource

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await release_resource(self.resource)

# Usage
async def use_resource():
    async with AsyncResourceManager() as resource:
        await do_work_with_resource(resource)
```

## Error Handling {#error-handling}

### Basic Exception Handling

```python
async def safe_operation():
    try:
        result = await potentially_failing_operation()
        return result
    except SpecificError as e:
        logger.error(f"Specific error: {e}")
        return default_value
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### Timeout Handling

```python
import asyncio

async def with_timeout():
    try:
        result = await asyncio.wait_for(
            long_running_operation(),
            timeout=10.0
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("Operation timed out")
        return None
```

### Retry Logic

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

### Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    async def call(self, operation):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = 'HALF_OPEN'

        try:
            result = await operation()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

## Performance Tips {#performance-tips}

### Memory Management

```python
# Avoid memory leaks
async def memory_efficient_processing():
    async for chunk in stream_large_data():
        # Process chunk without storing all data
        result = process_chunk(chunk)
        yield result

# Context manager for cleanup
class AsyncCleanup:
    async def __aenter__(self):
        self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
```

### Connection Pooling

```python
# HTTP connection pooling
connector = aiohttp.TCPConnector(
    limit=100,
    limit_per_host=30,
    ttl_dns_cache=300,
    use_dns_cache=True,
)

async with aiohttp.ClientSession(connector=connector) as session:
    # Reuses connections automatically

# Database connection pooling
pool = await asyncpg.create_pool(
    'postgresql://user:pass@localhost/db',
    min_size=5,
    max_size=20,
    command_timeout=60
)

async with pool.acquire() as connection:
    await connection.execute('SELECT 1')
```

### Batching Operations

```python
# Batch HTTP requests
async def batch_requests(urls, batch_size=10):
    semaphore = asyncio.Semaphore(batch_size)

    async def limited_request(url):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.json()

    tasks = [limited_request(url) for url in urls]
    return await asyncio.gather(*tasks)

# Batch database operations
async def batch_db_operations(operations, batch_size=100):
    for i in range(0, len(operations), batch_size):
        batch = operations[i:i + batch_size]
        await asyncio.gather(*[op() for op in batch])
```

## Production Checklist {#production-checklist}

### ✅ Required Libraries

```python
# Install essential async libraries
pip install aiohttp aiofiles asyncpg aiomysql uvloop

# For development
pip install pytest-asyncio aioresponses
```

### ✅ Error Handling

```python
# 1. Always handle exceptions
# 2. Implement timeouts
# 3. Add retry logic for transient failures
# 4. Log errors appropriately
# 5. Use circuit breakers for external services

import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### ✅ Resource Management

```python
# 1. Use context managers for cleanup
# 2. Implement connection pooling
# 3. Set appropriate timeouts
# 4. Monitor resource usage
# 5. Handle graceful shutdown

import signal
import asyncio

class AsyncApplication:
    def __init__(self):
        self.shutdown_event = asyncio.Event()

    async def setup_signal_handlers(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(
            signal.SIGTERM,
            self.shutdown_event.set
        )
        loop.add_signal_handler(
            signal.SIGINT,
            self.shutdown_event.set
        )

    async def run(self):
        await self.setup_signal_handlers()

        try:
            while not self.shutdown_event.is_set():
                await self.process_requests()
        finally:
            await self.cleanup()
```

### ✅ Performance Monitoring

```python
import time
import psutil

class AsyncMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

    async def record_request(self, duration, success=True):
        self.request_count += 1
        if not success:
            self.error_count += 1

        # Log metrics
        logger.info(
            "request_completed",
            duration=duration,
            success=success,
            request_count=self.request_count,
            error_rate=self.error_count / self.request_count,
            memory_usage=psutil.Process().memory_info().rss / 1024 / 1024
        )
```

### ✅ Configuration Management

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    database_url: str
    max_connections: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            database_url=os.getenv('DATABASE_URL', 'sqlite:///app.db'),
            max_connections=int(os.getenv('MAX_CONNECTIONS', '10')),
            request_timeout=int(os.getenv('REQUEST_TIMEOUT', '30')),
            retry_attempts=int(os.getenv('RETRY_ATTEMPTS', '3'))
        )

config = Config.from_env()
```

## Quick Code Snippets {#quick-code-snippets}

### Web Scraping Template

```python
import asyncio
import aiohttp
from bs4 import BeautifulSoup

async def scrape_website(base_url, max_pages=10):
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(5)

        async def fetch_page(url):
            async with semaphore:
                async with session.get(url) as response:
                    return await response.text()

        async def parse_page(html):
            soup = BeautifulSoup(html, 'html.parser')
            # Extract data
            return soup.title.string if soup.title else "No title"

        urls = [f"{base_url}/page/{i}" for i in range(1, max_pages + 1)]
        html_pages = await asyncio.gather(*[fetch_page(url) for url in urls])
        results = await asyncio.gather(*[parse_page(html) for html in html_pages])

        return results
```

### Real-time Data Processing

```python
import asyncio
import json

async def real_time_processor():
    queue = asyncio.Queue(maxsize=1000)

    async def data_producer():
        while True:
            data = await get_external_data()
            await queue.put(data)

    async def data_consumer():
        while True:
            data = await queue.get()
            result = await process_data(data)
            await store_result(result)
            queue.task_done()

    # Start multiple consumers for parallel processing
    consumers = [
        asyncio.create_task(data_consumer())
        for _ in range(5)
    ]

    await asyncio.gather(
        data_producer(),
        *consumers
    )
```

### API Rate Limiting

```python
import asyncio
import time

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    async def acquire(self):
        now = time.time()
        # Remove old requests
        self.requests = [req_time for req_time in self.requests
                        if now - req_time < self.time_window]

        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0])
            await asyncio.sleep(sleep_time)
            return await self.acquire()

        self.requests.append(now)

# Usage
rate_limiter = RateLimiter(max_requests=100, time_window=60)

async def limited_api_call(url):
    await rate_limiter.acquire()
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### Background Task Scheduler

```python
import asyncio
from datetime import datetime, timedelta

class BackgroundScheduler:
    def __init__(self):
        self.tasks = []

    async def schedule_task(self, func, delay, interval=None):
        async def task_wrapper():
            await asyncio.sleep(delay)
            await func()

            if interval:
                while True:
                    await asyncio.sleep(interval)
                    await func()

        task = asyncio.create_task(task_wrapper())
        self.tasks.append(task)
        return task

    async def cancel_all(self):
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()

# Usage
scheduler = BackgroundScheduler()

# Schedule one-time task
await scheduler.schedule_task(
    cleanup_old_data,
    delay=3600  # Run after 1 hour
)

# Schedule recurring task
await scheduler.schedule_task(
    check_system_health,
    delay=0,  # Start immediately
    interval=300  # Every 5 minutes
)
```

### Testing Async Code

```python
import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_function():
    # Mock async function
    mock_function = AsyncMock(return_value="expected_result")

    # Test the function
    result = await mock_function()
    assert result == "expected_result"

    # Verify it was called
    mock_function.assert_called_once()

@pytest.mark.asyncio
async def test_async_context_manager():
    with pytest.raises(ValueError):
        async with some_async_context_manager():
            raise ValueError("Test error")

# Using pytest-asyncio
@pytest.fixture
async def async_client():
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.mark.asyncio
async def test_api_call(async_client):
    response = await async_client.get('https://api.example.com/test')
    assert response.status == 200
```

## Debugging Tips

### Async Debugging

```python
# Enable debug mode
import asyncio
asyncio.get_event_loop().set_debug(True)

# Add debugging to coroutines
import functools

def debug_coroutine(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            print(f"Completed {func.__name__}")
            return result
        except Exception as e:
            print(f"Failed {func.__name__}: {e}")
            raise
    return wrapper

@debug_coroutine
async def my_async_function():
    await asyncio.sleep(1)
    return "done"
```

### Memory Profiling

```python
import tracemalloc
import asyncio

async def memory_intensive_task():
    tracemalloc.start()

    # Your async code here
    data = [i for i in range(100000)]
    await asyncio.sleep(1)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()
```

This cheatsheet provides quick reference for all essential async programming patterns and production considerations. Bookmark it for fast access during development!
