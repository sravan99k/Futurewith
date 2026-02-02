# Async Programming Practice Exercises

This comprehensive practice guide provides hands-on exercises to reinforce your understanding of Python asynchronous programming with asyncio. Complete each exercise to build practical skills in writing concurrent, high-performance Python applications.

## Table of Contents

1. [Introduction to Async Programming](#introduction-to-async-programming)
2. [Coroutines and Tasks](#coroutines-and-tasks)
3. [Async Context Managers and Iterators](#async-context-managers-and-iterators)
4. [Concurrent HTTP Requests](#concurrent-http-requests)
5. [Database Operations with Async](#database-operations-with-async)
6. [Error Handling and Resilience](#error-handling-and-resilience)
7. [Real-World Async Applications](#real-world-async-applications)

---

## Introduction to Async Programming

### Exercise 1: Understanding the Event Loop

**Objective:** Understand how the asyncio event loop works and how coroutines are scheduled.

**Task:** Create a demonstration of event loop behavior:

```python
import asyncio
import time
from typing import List

async def async_task(task_name: str, delay: float) -> str:
    """Simulate an asynchronous task."""
    print(f"  Starting task: {task_name}")
    await asyncio.sleep(delay)
    print(f"  Completed task: {task_name}")
    return f"{task_name} completed after {delay}s"

def synchronous_work():
    """Simulate synchronous blocking work."""
    print("Synchronous work starting")
    time.sleep(0.1)
    print("Synchronous work completed")
    return "sync result"

async def demonstrate_event_loop():
    """Demonstrate event loop behavior."""
    print("=" * 60)
    print("ASYNC PROGRAMMING DEMONSTRATION")
    print("=" * 60)
    
    # Part 1: Synchronous vs Asynchronous
    print("\n1. Synchronous Execution (Blocking)")
    print("-" * 40)
    start = time.time()
    result1 = await async_task("Task 1", 0.1)
    result2 = await async_task("Task 2", 0.1)
    result3 = await async_task("Task 3", 0.1)
    sync_time = time.time() - start
    print(f"  Sequential async time: {sync_time:.3f}s")
    
    # Part 2: Concurrent execution
    print("\n2. Concurrent Execution (Non-blocking)")
    print("-" * 40)
    start = time.time()
    results = await asyncio.gather(
        async_task("Task A", 0.1),
        async_task("Task B", 0.1),
        async_task("Task C", 0.1)
    )
    concurrent_time = time.time() - start
    print(f"  Concurrent async time: {concurrent_time:.3f}s")
    
    # Part 3: Mixed sync and async
    print("\n3. Mixed Synchronous and Async Execution")
    print("-" * 40)
    
    async def mixed_operations():
        # Run sync function in thread pool
        loop = asyncio.get_running_loop()
        sync_result = await loop.run_in_executor(
            None, synchronous_work
        )
        
        # Run async operations concurrently
        async_results = await asyncio.gather(
            async_task("Async 1", 0.05),
            async_task("Async 2", 0.05)
        )
        
        return sync_result, async_results
    
    start = time.time()
    sync, async_res = await mixed_operations()
    mixed_time = time.time() - start
    print(f"  Mixed operations time: {mixed_time:.3f}s")
    
    # Summary
    print("\n4. Performance Summary")
    print("-" * 40)
    print(f"  Sequential:    {sync_time:.3f}s (baseline)")
    print(f"  Concurrent:    {concurrent_time:.3f}s ({sync_time/concurrent_time:.1f}x faster)")
    print(f"  Mixed:         {mixed_time:.3f}s")

# Additional exploration
async def explore_event_loop():
    """Explore event loop mechanisms."""
    print("\n5. Event Loop Exploration")
    print("-" * 40)
    
    loop = asyncio.get_running_loop()
    print(f"  Loop: {loop}")
    print(f"  Is running: {loop.is_running()}")
    
    # Schedule tasks
    async def scheduled_task(task_id: int):
        print(f"    Task {task_id} starting")
        await asyncio.sleep(0.05)
        print(f"    Task {task_id} completed")
        return task_id * 2
    
    # Create tasks without awaiting immediately
    tasks = [asyncio.create_task(scheduled_task(i)) for i in range(3)]
    
    # Tasks are already running
    print(f"  Tasks scheduled: {len(tasks)}")
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    print(f"  All tasks completed: {results}")

if __name__ == "__main__":
    asyncio.run(demonstrate_event_loop())
    asyncio.run(explore_event_loop())
```

**Analysis Questions:**
1. Why is concurrent execution faster than sequential?
2. What happens if you try to run synchronous blocking code directly in an async function?
3. How does run_in_executor() bridge synchronous and asynchronous code?

### Exercise 2: Creating and Managing Tasks

**Objective:** Learn to create, manage, and coordinate multiple async tasks.

**Task:** Build a task management system:

```python
import asyncio
import time
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: int
    name: str
    duration: float
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None

class TaskManager:
    """Async task manager with priority scheduling."""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: List[Task] = []
        self.running_tasks: List[asyncio.Task] = []
    
    def add_task(self, task: Task) -> None:
        """Add a task to the manager."""
        task.created_at = datetime.now()
        self.tasks.append(task)
    
    async def _execute_task(self, task: Task) -> str:
        """Execute a single task with semaphore control."""
        async with self.semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            print(f"  [START] {task.name} (Priority: {task.priority})")
            
            try:
                # Simulate work
                await asyncio.sleep(task.duration)
                
                task.status = TaskStatus.COMPLETED
                task.result = f"Result from {task.name}"
                task.completed_at = datetime.now()
                print(f"  [DONE] {task.name} completed in {task.duration}s")
                
                return task.result
                
            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                print(f"  [CANCELLED] {task.name}")
                raise
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                print(f"  [ERROR] {task.name}: {e}")
                raise
    
    async def run_all(self) -> List[Task]:
        """Run all tasks with priority scheduling."""
        # Sort by priority (higher first)
        sorted_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)
        
        # Create task objects
        async_tasks = [
            asyncio.create_task(self._execute_task(task))
            for task in sorted_tasks
        ]
        
        # Wait for completion
        try:
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            return self.tasks
        except Exception as e:
            print(f"Error during task execution: {e}")
            return self.tasks
    
    async def run_with_timeout(self, timeout: float) -> List[Task]:
        """Run all tasks with a timeout."""
        try:
            async with asyncio.timeout(timeout):
                return await self.run_all()
        except asyncio.TimeoutError:
            print(f"\n  [TIMEOUT] Tasks did not complete within {timeout}s")
            # Cancel all running tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()
            return self.tasks

# Priority queue implementation
class PriorityTaskQueue:
    """Thread-safe priority queue for async tasks."""
    
    def __init__(self):
        self._queue: List[tuple] = []  # (priority, task)
        self._lock = asyncio.Lock()
    
    async def put(self, priority: int, coro) -> None:
        """Add a task with priority."""
        async with self._lock:
            heapq.heappush(self._queue, (priority, coro))
    
    async def get(self) -> tuple:
        """Get the highest priority task."""
        async with self._lock:
            return heapq.heappop(self._queue) if self._queue else None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

if __name__ == "__main__":
    import heapq
    
    print("=" * 60)
    print("TASK MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Create task manager with max 2 concurrent tasks
    manager = TaskManager(max_concurrent=2)
    
    # Add tasks with different priorities
    tasks_data = [
        (1, "Low Priority Task", 0.1, 1),
        (2, "High Priority Task", 0.15, 5),
        (3, "Medium Priority Task", 0.1, 3),
        (4, "Another Low Task", 0.05, 1),
        (5, "Critical Task", 0.2, 10),
    ]
    
    for task_id, name, duration, priority in tasks_data:
        task = Task(
            id=task_id,
            name=name,
            duration=duration,
            priority=priority
        )
        manager.add_task(task)
    
    # Run all tasks
    print("\nRunning tasks with max 2 concurrent...")
    start = time.time()
    
    completed_tasks = asyncio.run(manager.run_all())
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.3f}s")
    
    # Show results
    print("\nTask Summary:")
    print("-" * 60)
    for task in sorted(completed_tasks, key=lambda t: t.priority, reverse=True):
        duration = (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else 0
        print(f"  {task.name}: {task.status.value} ({duration:.3f}s)")
```

---

## Async Context Managers and Iterators

### Exercise 3: Creating Async Context Managers

**Objective:** Learn to create and use async context managers for resource management.

**Task:** Build an async connection pool with proper context management:

```python
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class DatabaseConnection:
    """Represents a database connection."""
    connection_id: int
    in_use: bool = False
    
    async def execute(self, query: str) -> str:
        """Execute a query."""
        await asyncio.sleep(0.01)  # Simulate network latency
        return f"Result for '{query}' from connection {self.connection_id}"
    
    async def close(self):
        """Close the connection."""
        print(f"  Closing connection {self.connection_id}")

class AsyncConnectionPool:
    """Async connection pool with context manager support."""
    
    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.available: List[DatabaseConnection] = []
        self.in_use: Dict[int, DatabaseConnection] = {}
        self.connection_counter = 0
        
        # Pre-create connections
        for _ in range(pool_size):
            self.connection_counter += 1
            self.available.append(DatabaseConnection(self.connection_counter))
    
    async def acquire(self) -> DatabaseConnection:
        """Acquire a connection from the pool."""
        if not self.available:
            # Wait for a connection to become available
            while not self.available:
                await asyncio.sleep(0.01)
        
        connection = self.available.pop()
        connection.in_use = True
        self.in_use[connection.connection_id] = connection
        return connection
    
    async def release(self, connection: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        if connection.connection_id in self.in_use:
            del self.in_use[connection.connection_id]
            connection.in_use = False
            self.available.append(connection)
    
    async def __aenter__(self) -> 'AsyncConnectionPool':
        """Enter context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and cleanup."""
        # Close all connections
        for connection in self.available:
            await connection.close()
        for connection in self.in_use.values():
            await connection.close()
        self.available.clear()
        self.in_use.clear()

# Usage with async context manager
async def use_connection_pool():
    """Demonstrate connection pool usage."""
    print("\nConnection Pool Demo")
    print("-" * 40)
    
    async with AsyncConnectionPool(pool_size=3) as pool:
        # Acquire multiple connections
        conn1 = await pool.acquire()
        conn2 = await pool.acquire()
        
        print(f"  Using {len(pool.in_use)} connections")
        
        # Use connections concurrently
        results = await asyncio.gather(
            conn1.execute("SELECT * FROM users"),
            conn2.execute("SELECT * FROM orders")
        )
        
        print(f"  Results: {len(results)} queries executed")
        
        # Release connections
        await pool.release(conn1)
        await pool.release(conn2)
        print(f"  Available connections: {len(pool.available)}")

# Async context manager for rate limiting
class RateLimiter:
    """Async context manager for rate limiting."""
    
    def __init__(self, max_calls: int, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls: List[float] = []
        self.semaphore = asyncio.Semaphore(max_calls)
    
    async def __aenter__(self):
        """Acquire semaphore slot."""
        await self.semaphore.acquire()
        self.calls.append(time.time())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release semaphore slot."""
        self.semaphore.release()
        # Clean up old calls
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]

# Resource timeout context manager
class TimeoutContext:
    """Context manager that enforces a timeout."""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.task = None
    
    async def __aenter__(self):
        """Start the timeout task."""
        async def timeout_watchdog():
            await asyncio.sleep(self.timeout)
            raise TimeoutError(f"Operation timed out after {self.timeout}s")
        
        self.task = asyncio.create_task(timeout_watchdog())
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cancel the timeout task."""
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(use_connection_pool())
```

### Exercise 4: Creating Async Iterators and Generators

**Objective:** Learn to create async iterators for streaming data.

**Task:** Build async data pipelines with async generators:

```python
import asyncio
from typing import AsyncGenerator, List
from dataclasses import dataclass

@dataclass
class DataPoint:
    """A data point for streaming."""
    timestamp: float
    value: float
    source: str

class AsyncDataStream:
    """Async data stream with multiple sources."""
    
    def __init__(self, sources: List[str]):
        self.sources = sources
    
    async def generate_data(self, source: str, count: int = 10) -> AsyncGenerator[DataPoint, None]:
        """Generate streaming data from a source."""
        import random
        import time
        
        for i in range(count):
            yield DataPoint(
                timestamp=time.time(),
                value=random.random() * 100,
                source=source
            )
            await asyncio.sleep(0.05)  # Simulate data arrival
    
    async def stream_all_sources(self) -> AsyncGenerator[DataPoint, None]:
        """Stream data from all sources concurrently."""
        tasks = [
            self.generate_data(source)
            for source in self.sources
        ]
        
        # Merge streams
        queues: List[asyncio.Queue] = []
        for task in tasks:
            q = asyncio.Queue()
            queues.append(q)
            
            async def feed_queue(gen, queue):
                async for item in gen:
                    await queue.put(item)
                await queue.put(None)  # Signal completion
            
            asyncio.create_task(feed_queue(task, q))
        
        # Yield from queues as they complete
        pending = list(range(len(queues)))
        while pending:
            for i in pending[:]:
                try:
                    item = await asyncio.wait_for(queues[i].get(), timeout=0.1)
                    if item is None:
                        pending.remove(i)
                    else:
                        yield item
                except asyncio.TimeoutError:
                    pass

# Async batch processor
class AsyncBatchProcessor:
    """Process async data in batches."""
    
    def __init__(self, batch_size: int = 100, timeout: float = 1.0):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
    
    async def add_item(self, item) -> None:
        """Add item to processor."""
        await self.queue.put(item)
    
    async def process_batch(self, batch: List) -> List:
        """Process a batch of items."""
        # Simulate processing
        await asyncio.sleep(0.01)
        return [item * 2 for item in batch]
    
    async def run(self) -> AsyncGenerator[List, None]:
        """Run the batch processor."""
        batch = []
        
        while True:
            try:
                # Wait for next item with timeout
                item = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=self.timeout
                )
                batch.append(item)
                
                # Process if batch is full
                if len(batch) >= self.batch_size:
                    result = await self.process_batch(batch)
                    yield result
                    batch = []
                    
            except asyncio.TimeoutError:
                # Process remaining items
                if batch:
                    result = await self.process_batch(batch)
                    yield result
                    batch = []
                # Check for shutdown signal
                if self.queue.empty() and not asyncio.current_task().cancelling():
                    continue
                elif self.queue.empty():
                    break

# Usage demonstration
async def demonstrate_async_iterators():
    """Demonstrate async iterators and generators."""
    print("\n" + "=" * 60)
    print("ASYNC ITERATORS AND GENERATORS DEMO")
    print("=" * 60)
    
    # Example 1: Async data stream
    print("\n1. Async Data Stream")
    print("-" * 40)
    
    stream = AsyncDataStream(["Sensor A", "Sensor B", "Sensor C"])
    
    async for data_point in stream.stream_all_sources():
        print(f"  {data_point.source}: {data_point.value:.2f}")
        # Process first few points
        if data_point.value > 90:
            print(f"    ⚠️ High value detected!")
    
    # Example 2: Batch processing
    print("\n2. Batch Processing")
    print("-" * 40)
    
    processor = AsyncBatchProcessor(batch_size=5, timeout=0.5)
    
    # Add items
    for i in range(12):
        await processor.add_item(i)
        await asyncio.sleep(0.1)
    
    # Process batches
    async for batch in processor.run():
        print(f"  Processed batch: {batch}")

if __name__ == "__main__":
    asyncio.run(demonstrate_async_iterators())
```

---

## Concurrent HTTP Requests

### Exercise 5: Building an Async HTTP Client

**Objective:** Build a robust async HTTP client with connection pooling and retries.

**Task:** Create an async HTTP client library:

```python
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
from urllib.parse import urljoin

class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class RequestOptions:
    """Options for HTTP requests."""
    method: HTTPMethod
    url: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    json_data: Optional[Any] = None
    timeout: float = 30.0
    retries: int = 3
    retry_delay: float = 1.0

@dataclass
class Response:
    """HTTP response wrapper."""
    status_code: int
    headers: Dict[str, str]
    content: bytes
    url: str
    elapsed_time: float
    
    @property
    def text(self) -> str:
        """Decode response content."""
        return self.content.decode('utf-8', errors='replace')
    
    def json(self) -> Any:
        """Parse JSON response."""
        import json
        return json.loads(self.content)

class AsyncHttpClient:
    """Robust async HTTP client with retry logic."""
    
    def __init__(self, max_connections: int = 100, timeout: float = 30.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=10
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.total_time = 0.0
    
    async def __aenter__(self) -> 'AsyncHttpClient':
        """Enter context manager."""
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if self.session:
            await self.session.close()
    
    async def request(self, options: RequestOptions) -> Response:
        """Execute HTTP request with retry logic."""
        if not self.session:
            raise RuntimeError("Client must be used as context manager")
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(options.retries):
            try:
                # Make request
                async with self.session.request(
                    method=options.method.value,
                    url=options.url,
                    headers=options.headers,
                    params=options.params,
                    json=options.json_data
                ) as response:
                    content = await response.read()
                    elapsed = time.time() - start_time
                    
                    self.request_count += 1
                    self.total_time += elapsed
                    
                    return Response(
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=content,
                        url=str(response.url),
                        elapsed_time=elapsed
                    )
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                if attempt < options.retries - 1:
                    await asyncio.sleep(options.retry_delay * (attempt + 1))
                    continue
                raise
        
        raise last_exception
    
    # Convenience methods
    async def get(self, url: str, **kwargs) -> Response:
        """GET request."""
        options = RequestOptions(method=HTTPMethod.GET, url=url, **kwargs)
        return await self.request(options)
    
    async def post(self, url: str, **kwargs) -> Response:
        """POST request."""
        options = RequestOptions(method=HTTPMethod.POST, url=url, **kwargs)
        return await self.request(options)

# Rate-limited HTTP client
class RateLimitedHttpClient(AsyncHttpClient):
    """HTTP client with rate limiting."""
    
    def __init__(self, requests_per_second: float = 10, **kwargs):
        super().__init__(**kwargs)
        self.rate_limit = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()
    
    async def request(self, options: RequestOptions) -> Response:
        """Execute request with rate limiting."""
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            self.last_request_time = time.time()
        
        return await super().request(options)

# Concurrent request manager
class ConcurrentRequestManager:
    """Manage multiple concurrent HTTP requests."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch(self, client: AsyncHttpClient, url: str) -> Response:
        """Fetch a URL with semaphore control."""
        async with self.semaphore:
            return await client.get(url)
    
    async def fetch_all(self, client: AsyncHttpClient, urls: List[str]) -> List[Response]:
        """Fetch multiple URLs concurrently."""
        tasks = [self.fetch(client, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Usage demonstration
async def demonstrate_http_client():
    """Demonstrate the async HTTP client."""
    print("\n" + "=" * 60)
    print("ASYNC HTTP CLIENT DEMO")
    print("=" * 60)
    
    async with AsyncHttpClient() as client:
        # Single request
        print("\n1. Single Request")
        print("-" * 40)
        
        try:
            response = await client.get(
                url="https://jsonplaceholder.typicode.com/posts/1",
                headers={"Accept": "application/json"}
            )
            print(f"  Status: {response.status_code}")
            print(f"  Time: {response.elapsed_time:.3f}s")
            print(f"  Content: {response.text[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
        
        # Concurrent requests
        print("\n2. Concurrent Requests")
        print("-" * 40)
        
        urls = [
            "https://jsonplaceholder.typicode.com/posts/1",
            "https://jsonplaceholder.typicode.com/posts/2",
            "https://jsonplaceholder.typicode.com/posts/3",
            "https://jsonplaceholder.typicode.com/posts/4",
            "https://jsonplaceholder.typicode.com/posts/5",
        ]
        
        manager = ConcurrentRequestManager(max_concurrent=3)
        start = time.time()
        results = await manager.fetch_all(client, urls)
        elapsed = time.time() - start
        
        successful = sum(1 for r in results if isinstance(r, Response))
        print(f"  Completed: {successful}/{len(urls)} requests")
        print(f"  Time: {elapsed:.3f}s")

if __name__ == "__main__":
    asyncio.run(demonstrate_http_client())
```

---

## Database Operations with Async

### Exercise 6: Async Database Operations

**Objective:** Implement async database operations with connection pooling.

**Task:** Build an async database wrapper:

```python
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool

@dataclass
class User:
    """User model."""
    id: int
    name: str
    email: str
    created_at: str

class AsyncDatabase:
    """Async database wrapper with connection pooling."""
    
    def __init__(self, dsn: str, pool_size: int = 10):
        self.dsn = dsn
        self.pool: Optional[Pool] = None
        self.pool_size = pool_size
    
    async def connect(self) -> None:
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(
            self.dsn,
            min_size=2,
            max_size=self.pool_size,
            command_timeout=30
        )
        print(f"  Connected to database")
    
    async def disconnect(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()
            print(f"  Disconnected from database")
    
    async def __aenter__(self) -> 'AsyncDatabase':
        """Enter context manager."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        await self.disconnect()
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status."""
        async with self.pool.acquire() as connection:
            await connection.execute(query, *args)
            return "OK"
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row."""
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch all rows."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def fetch_val(self, query: str, *args) -> Any:
        """Fetch a single value."""
        async with self.pool.acquire() as connection:
            return await connection.fetchval(query, *args)
    
    # User operations
    async def create_users_table(self) -> None:
        """Create users table."""
        await self.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    async def insert_user(self, name: str, email: str) -> int:
        """Insert a new user and return ID."""
        return await self.fetch_val(
            "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
            name, email
        )
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        row = await self.fetch_one(
            "SELECT id, name, email, created_at FROM users WHERE id = $1",
            user_id
        )
        if row:
            return User(**row)
        return None
    
    async def get_all_users(self) -> List[User]:
        """Get all users."""
        rows = await self.fetch_all(
            "SELECT id, name, email, created_at FROM users ORDER BY id"
        )
        return [User(**row) for row in rows]
    
    async def update_user(self, user_id: int, name: str = None, email: str = None) -> bool:
        """Update user."""
        updates = []
        params = []
        param_count = 0
        
        if name is not None:
            param_count += 1
            updates.append(f"name = ${param_count}")
            params.append(name)
        
        if email is not None:
            param_count += 1
            updates.append(f"email = ${param_count}")
            params.append(email)
        
        if not updates:
            return False
        
        param_count += 1
        params.append(user_id)
        
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ${param_count}"
        await self.execute(query, *params)
        return True
    
    async def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        result = await self.execute(
            "DELETE FROM users WHERE id = $1", user_id
        )
        return "DELETE" in result

# Transaction management
class Transaction:
    """Async transaction context manager."""
    
    def __init__(self, pool: Pool):
        self.pool = pool
        self.connection = None
    
    async def __aenter__(self) -> 'Transaction':
        """Start transaction."""
        self.connection = await self.pool.acquire()
        await self.connection.execute("BEGIN")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Commit or rollback transaction."""
        if exc_type is None:
            await self.connection.execute("COMMIT")
        else:
            await self.connection.execute("ROLLBACK")
        await self.pool.release(self.connection)

# Batch operations
class BatchDatabaseOperations:
    """Efficient batch database operations."""
    
    def __init__(self, db: AsyncDatabase):
        self.db = db
    
    async def batch_insert_users(self, users: List[Dict[str, str]]) -> List[int]:
        """Insert multiple users efficiently."""
        async with self.db.pool.acquire() as connection:
            # Use COPY for bulk insert (most efficient)
            data = [(u['name'], u['email']) for u in users]
            await connection.executemany(
                "INSERT INTO users (name, email) VALUES ($1, $2)",
                data
            )
            # Return IDs (simplified)
            return list(range(1, len(users) + 1))
    
    async def batch_update_users(self, updates: List[Dict]) -> int:
        """Update multiple users in a single query."""
        if not updates:
            return 0
        
        # Build parameterized query
        set_clauses = []
        params = []
        param_count = 0
        
        for i, update in enumerate(updates):
            for key, value in update.get('updates', {}).items():
                param_count += 1
                set_clauses.append(f"WHEN id = ${param_count} THEN ${param_count + 1}")
                params.extend([update['id'], value])
        
        if not set_clauses:
            return 0
        
        query = f"""
            UPDATE users SET
                name = CASE {' '.join([c for c in set_clauses if 'name' in c])}
                      ELSE name END,
                email = CASE {' '.join([c for c in set_clauses if 'email' in c])}
                      ELSE email END
            WHERE id IN ({', '.join([str(u['id']) for u in updates])})
        """
        
        await self.db.execute(query, *params)
        return len(updates)

# Mock database for demonstration
class MockAsyncDatabase:
    """Mock async database for environments without PostgreSQL."""
    
    def __init__(self):
        self.users: Dict[int, User] = {}
        self.next_id = 1
        self.lock = asyncio.Lock()
    
    async def connect(self) -> None:
        print("  Mock database connected")
    
    async def disconnect(self) -> None:
        print("  Mock database disconnected")
    
    async def __aenter__(self) -> 'MockAsyncDatabase':
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
    
    async def create_users_table(self) -> None:
        pass
    
    async def insert_user(self, name: str, email: str) -> int:
        async with self.lock:
            user_id = self.next_id
            self.next_id += 1
            self.users[user_id] = User(
                id=user_id,
                name=name,
                email=email,
                created_at="2024-01-01"
            )
            return user_id
    
    async def get_user(self, user_id: int) -> Optional[User]:
        return self.users.get(user_id)
    
    async def get_all_users(self) -> List[User]:
        return list(self.users.values())

# Demonstration
async def demonstrate_database_operations():
    """Demonstrate async database operations."""
    print("\n" + "=" * 60)
    print("ASYNC DATABASE OPERATIONS DEMO")
    print("=" * 60)
    
    # Use mock database for demonstration
    db = MockAsyncDatabase()
    
    async with db:
        # Create table
        print("\n1. Creating users table...")
        await db.create_users_table()
        
        # Insert users
        print("\n2. Inserting users...")
        user_data = [
            ("Alice Johnson", "alice@example.com"),
            ("Bob Smith", "bob@example.com"),
            ("Carol Davis", "carol@example.com"),
        ]
        
        for name, email in user_data:
            user_id = await db.insert_user(name, email)
            print(f"  Created user {user_id}: {name}")
        
        # Query users
        print("\n3. Querying users...")
        users = await db.get_all_users()
        for user in users:
            print(f"  - {user.name} ({user.email})")
        
        # Get single user
        print("\n4. Getting single user...")
        user = await db.get_user(1)
        if user:
            print(f"  Found: {user.name}")

if __name__ == "__main__":
    asyncio.run(demonstrate_database_operations())
```

---

## Error Handling and Resilience

### Exercise 7: Async Error Handling Patterns

**Objective:** Implement robust error handling and resilience patterns.

**Task:** Build a resilient async task system with proper error handling:

```python
import asyncio
import time
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

@dataclass
class TaskResult:
    """Result of an async task."""
    task_id: str
    state: TaskState
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    duration: float = 0.0
    completed_at: Optional[datetime] = None

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple = (Exception,)

class AsyncTask:
    """A resilient async task with retry logic."""
    
    def __init__(
        self,
        task_id: str,
        coro_func: Callable[[], Any],
        retry_config: Optional[RetryConfig] = None
    ):
        self.task_id = task_id
        self.coro_func = coro_func
        self.retry_config = retry_config or RetryConfig()
        self.state = TaskState.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
    
    async def run(self) -> TaskResult:
        """Run the task with retry logic."""
        start_time = time.time()
        self.state = TaskState.RUNNING
        attempts = 0
        last_error = None
        
        while attempts < self.retry_config.max_attempts:
            attempts += 1
            
            try:
                # Execute the task
                self.result = await asyncio.create_task(self.coro_func())
                self.state = TaskState.SUCCESS
                
                return TaskResult(
                    task_id=self.task_id,
                    state=TaskState.SUCCESS,
                    result=self.result,
                    attempts=attempts,
                    duration=time.time() - start_time,
                    completed_at=datetime.now()
                )
                
            except self.retry_config.retryable_exceptions as e:
                last_error = str(e)
                self.state = TaskState.RETRYING
                
                if attempts < self.retry_config.max_attempts:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.retry_config.initial_delay * 
                        (self.retry_config.exponential_base ** (attempts - 1)),
                        self.retry_config.max_delay
                    )
                    
                    # Add jitter
                    delay += random.uniform(0, delay * 0.1)
                    
                    print(f"  [RETRY] {self.task_id}: {last_error}")
                    print(f"          Retrying in {delay:.2f}s (attempt {attempts + 1})")
                    
                    await asyncio.sleep(delay)
                else:
                    self.state = TaskState.FAILED
                    self.error = last_error
        
        return TaskResult(
            task_id=self.task_id,
            state=TaskState.FAILED,
            error=last_error,
            attempts=attempts,
            duration=time.time() - start_time,
            completed_at=datetime.now()
        )

# Circuit breaker pattern
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0

class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
    
    async def call(self, coro) -> Any:
        """Execute coroutine through circuit breaker."""
        now = datetime.now()
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self.next_attempt_time and now >= self.next_attempt_time:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.failure_count = 0
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is open. Retry after {self.next_attempt_time}"
                )
        
        try:
            result = await coro
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                print(f"  [CIRCUIT] Closed (recovered)")
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = datetime.now() + self.config.timeout
            print(f"  [CIRCUIT] Opened (too many failures)")

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

# Task orchestrator with resilience
class ResilientTaskOrchestrator:
    """Orchestrate tasks with resilience patterns."""
    
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks: Dict[str, AsyncTask] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def add_task(
        self,
        task_id: str,
        coro_func: Callable[[], Any],
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ) -> None:
        """Add a task to the orchestrator."""
        self.tasks[task_id] = AsyncTask(task_id, coro_func, retry_config)
        self.circuit_breakers[task_id] = circuit_breaker or CircuitBreaker()
    
    async def run_all(self) -> Dict[str, TaskResult]:
        """Run all tasks with resilience patterns."""
        results = {}
        
        async def run_with_resilience(task_id: str, task: AsyncTask) -> TaskResult:
            async with self.semaphore:
                circuit = self.circuit_breakers[task_id]
                
                try:
                    result = await circuit.call(task.run)
                    results[task_id] = result
                    return result
                except Exception as e:
                    results[task_id] = TaskResult(
                        task_id=task_id,
                        state=TaskState.FAILED,
                        error=str(e)
                    )
                    return results[task_id]
        
        # Run all tasks concurrently
        coroutines = [
            run_with_resilience(task_id, task)
            for task_id, task in self.tasks.items()
        ]
        
        await asyncio.gather(*coroutines)
        return results

# Demonstration
async def demonstrate_error_handling():
    """Demonstrate error handling patterns."""
    print("\n" + "=" * 60)
    print("ASYNC ERROR HANDLING DEMO")
    print("=" * 60)
    
    # Example 1: Retry with exponential backoff
    print("\n1. Retry with Exponential Backoff")
    print("-" * 40)
    
    attempt_count = 0
    
    async def unreliable_task():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count < 3:
            raise ValueError(f"Attempt {attempt_count} failed")
        
        return "Success!"
    
    task = AsyncTask(
        task_id="unreliable_task",
        coro_func=unreliable_task,
        retry_config=RetryConfig(
            max_attempts=3,
            initial_delay=0.1
        )
    )
    
    result = await task.run()
    print(f"  Task ID: {result.task_id}")
    print(f"  State: {result.state.value}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Result: {result.result}")
    
    # Example 2: Circuit breaker
    print("\n2. Circuit Breaker Pattern")
    print("-" * 40)
    
    failure_count = 0
    
    async def failing_service():
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 3:
            raise ConnectionError("Service unavailable")
        return "Service recovered"
    
    circuit = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.5
        )
    )
    
    for i in range(5):
        try:
            result = await circuit.call(failing_service())
            print(f"  Call {i + 1}: Success - {result}")
        except CircuitBreakerOpenError as e:
            print(f"  Call {i + 1}: Blocked - {e}")
        except Exception as e:
            print(f"  Call {i + 1}: Failed - {e}")
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(demonstrate_error_handling())
```

---

## Real-World Async Applications

### Exercise 8: Building an Async Chat Server

**Objective:** Apply async programming concepts to build a real-world application.

**Task:** Build a simple async chat server with WebSocket support:

```python
import asyncio
import json
from typing import Dict, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class MessageType(Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    MESSAGE = "message"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

@dataclass
class User:
    """Chat user."""
    user_id: str
    username: str
    connected_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

@dataclass
class ChatMessage:
    """Chat message."""
    message_type: MessageType
    sender_id: Optional[str] = None
    content: Optional[str] = None
    room_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        return json.dumps({
            'type': self.message_type.value,
            'sender_id': self.sender_id,
            'content': self.content,
            'room_id': self.room_id,
            'timestamp': self.timestamp.isoformat()
        })

class AsyncChatServer:
    """Async chat server with rooms and private messaging."""
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.users: Dict[str, User] = {}
        self.connections: Dict[str, asyncio.Queue] = {}
        self.rooms: Dict[str, Set[str]] = {"lobby": set()}
        self.server: Optional[asyncio.Server] = None
    
    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection."""
        addr = writer.get_extra_info('peername')
        print(f"\n[NEW CONNECTION] {addr}")
        
        user_id = f"user_{len(self.users) + 1}"
        queue: asyncio.Queue = asyncio.Queue()
        self.connections[user_id] = queue
        
        try:
            # Handle client loop
            async def read_messages():
                """Read messages from client."""
                while True:
                    try:
                        data = await reader.readline()
                        if not data:
                            break
                        
                        message = json.loads(data.decode())
                        await self.process_message(user_id, message)
                        
                    except json.JSONDecodeError:
                        await self.send_error(user_id, "Invalid message format")
                    except Exception as e:
                        print(f"[ERROR] {e}")
                        break
            
            async def write_messages():
                """Send messages to client."""
                while True:
                    try:
                        message = await queue.get()
                        writer.write((message + "\n").encode())
                        await writer.drain()
                    except Exception as e:
                        break
            
            # Run reader and writer concurrently
            await asyncio.gather(read_messages(), write_messages())
            
        except Exception as e:
            print(f"[ERROR] Client error: {e}")
        finally:
            print(f"[DISCONNECT] {addr}")
            await self.disconnect_user(user_id)
            writer.close()
            await writer.wait_closed()
    
    async def process_message(self, user_id: str, message: dict) -> None:
        """Process incoming message."""
        msg_type = message.get('type')
        
        if msg_type == MessageType.CONNECT.value:
            await self.handle_connect(user_id, message)
        elif msg_type == MessageType.MESSAGE.value:
            await self.handle_chat_message(user_id, message)
        elif msg_type == MessageType.PING.value:
            await self.send_pong(user_id)
        else:
            await self.send_error(user_id, f"Unknown message type: {msg_type}")
    
    async def handle_connect(self, user_id: str, message: dict) -> None:
        """Handle user connection."""
        username = message.get('username', f"Guest_{user_id}")
        
        self.users[user_id] = User(user_id=user_id, username=username)
        
        # Send welcome message
        welcome = ChatMessage(
            message_type=MessageType.CONNECT,
            content=json.dumps({
                'user_id': user_id,
                'username': username,
                'rooms': list(self.rooms.keys())
            })
        )
        await self.send_to_user(user_id, welcome)
        
        # Notify others
        join_msg = ChatMessage(
            message_type=MessageType.MESSAGE,
            sender_id=user_id,
            content=f"{username} joined the chat",
            room_id="lobby"
        )
        await self.broadcast(join_msg, exclude=user_id)
        
        print(f"[JOIN] {username} (ID: {user_id})")
    
    async def handle_chat_message(self, user_id: str, message: dict) -> None:
        """Handle chat message."""
        if user_id not in self.users:
            await self.send_error(user_id, "Not connected")
            return
        
        user = self.users[user_id]
        content = message.get('content', '')
        room_id = message.get('room_id', 'lobby')
        
        # Validate room
        if room_id not in self.rooms:
            await self.send_error(user_id, f"Unknown room: {room_id}")
            return
        
        # Create and broadcast message
        chat_msg = ChatMessage(
            message_type=MessageType.MESSAGE,
            sender_id=user_id,
            content=content,
            room_id=room_id
        )
        await self.broadcast(chat_msg, room_id=room_id)
        
        # Update user activity
        user.last_active = datetime.now()
    
    async def broadcast(
        self,
        message: ChatMessage,
        room_id: Optional[str] = None,
        exclude: Optional[str] = None
    ) -> None:
        """Broadcast message to users."""
        room = room_id or "lobby"
        
        if room not in self.rooms:
            return
        
        for user_id in self.rooms[room]:
            if user_id != exclude:
                await self.send_to_user(user_id, message)
    
    async def send_to_user(self, user_id: str, message: ChatMessage) -> None:
        """Send message to specific user."""
        if user_id in self.connections:
            await self.connections[user_id].put(message.to_json())
    
    async def send_error(self, user_id: str, error_message: str) -> None:
        """Send error message to user."""
        error = ChatMessage(
            message_type=MessageType.ERROR,
            content=error_message
        )
        await self.send_to_user(user_id, error)
    
    async def send_pong(self, user_id: str) -> None:
        """Send pong response."""
        pong = ChatMessage(message_type=MessageType.PONG)
        await self.send_to_user(user_id, pong)
    
    async def disconnect_user(self, user_id: str) -> None:
        """Handle user disconnection."""
        if user_id in self.users:
            user = self.users[user_id]
            
            # Remove from rooms
            for room in self.rooms.values():
                room.discard(user_id)
            
            # Notify others
            leave_msg = ChatMessage(
                message_type=MessageType.MESSAGE,
                sender_id=user_id,
                content=f"{user.username} left the chat",
                room_id="lobby"
            )
            await self.broadcast(leave_msg, exclude=user_id)
            
            # Cleanup
            del self.users[user_id]
            if user_id in self.connections:
                await self.connections[user_id].put(None)  # Signal to stop
                del self.connections[user_id]
    
    async def start(self) -> None:
        """Start the chat server."""
        self.server = await asyncio.start_server(
            self.handle_client,
            self.host,
            self.port
        )
        
        addr = self.server.sockets[0].getsockname()
        print(f"\n{'=' * 60}")
        print(f"CHAT SERVER STARTED")
        print(f"Listening on {addr}")
        print(f"Press Ctrl+C to stop")
        print(f"{'=' * 60}\n")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self) -> None:
        """Stop the chat server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("\n[Server stopped]")

# Chat client for testing
class AsyncChatClient:
    """Async chat client for testing."""
    
    def __init__(self, host: str = "localhost", port: int = 8888):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.running = False
    
    async def connect(self, username: str) -> None:
        """Connect to chat server."""
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port
        )
        
        # Send connect message
        connect_msg = json.dumps({
            'type': MessageType.CONNECT.value,
            'username': username
        })
        self.writer.write((connect_msg + "\n").encode())
        await self.writer.drain()
        
        self.running = True
        
        # Start message receiver
        asyncio.create_task(self.receive_messages())
    
    async def receive_messages(self) -> None:
        """Receive messages from server."""
        while self.running:
            try:
                data = await self.reader.readline()
                if not data:
                    break
                
                message = json.loads(data.decode())
                print(f"\n[Received] {message}")
                
            except Exception as e:
                break
    
    async def send_message(self, content: str, room_id: str = "lobby") -> None:
        """Send message to server."""
        message = json.dumps({
            'type': MessageType.MESSAGE.value,
            'content': content,
            'room_id': room_id
        })
        self.writer.write((message + "\n").encode())
        await self.writer.drain()
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        self.running = False
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

# Demonstration
async def demonstrate_chat_server():
    """Demonstrate the async chat server."""
    print("\n" + "=" * 60)
    print("ASYNC CHAT SERVER DEMO")
    print("=" * 60)
    
    # Note: This is a conceptual demonstration
    # In practice, you would run the server in a separate process
    
    print("\nChat Server Features:")
    print("  ✓ Async TCP server with WebSocket-like protocol")
    print("  ✓ Multiple concurrent connections")
    print("  ✓ Room-based messaging")
    print("  ✓ Heartbeat/Ping-Pong for connection health")
    print("  ✓ Automatic disconnection handling")
    
    print("\nTo test the server:")
    print("  1. Run the server: asyncio.run(server.start())")
    print("  2. Connect clients using AsyncChatClient")
    print("  3. Send messages between clients")
    print("  4. Disconnect and observe cleanup")

if __name__ == "__main__":
    asyncio.run(demonstrate_chat_server())
```

### Exercise 9: Building an Async Data Pipeline

**Objective:** Build a production-ready async data processing pipeline.

**Task:** Create a scalable data pipeline with async producers, processors, and consumers:

```python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import hashlib

class PipelineStage(Enum):
    SOURCE = "source"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    SINK = "sink"

@dataclass
class PipelineConfig:
    """Configuration for a pipeline stage."""
    stage_type: PipelineStage
    name: str
    max_queue_size: int = 1000
    batch_size: int = 100
    batch_timeout: float = 1.0
    workers: int = 1
    error_policy: str = "fail"  # fail, skip, retry

@dataclass
class DataItem:
    """Data item flowing through pipeline."""
    item_id: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    
    @property
    def key(self) -> str:
        """Generate cache key from data."""
        content = json.dumps(self.data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

class AsyncPipeline:
    """Async data processing pipeline."""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: Dict[str, 'PipelineStage'] = {}
        self.queues: Dict[str, asyncio.Queue] = {}
        self.tasks: List[asyncio.Task] = []
        self.running = False
        self.metrics: Dict[str, Dict] = {}
    
    def add_stage(
        self,
        config: PipelineConfig,
        processor: Callable[[DataItem], Any]
    ) -> 'AsyncPipeline':
        """Add a processing stage."""
        self.stages[config.name] = PipelineStage(
            config=config,
            processor=processor,
            stage_type=config.stage_type
        )
        self.queues[config.name] = asyncio.Queue(maxsize=config.max_queue_size)
        self.metrics[config.name] = {
            'processed': 0,
            'errors': 0,
            'avg_process_time': 0
        }
        return self
    
    async def _process_item(
        self,
        stage_name: str,
        item: DataItem
    ) -> Optional[DataItem]:
        """Process a single item through a stage."""
        stage = self.stages[stage_name]
        start_time = time.time()
        
        try:
            # Process the item
            result = await stage.processor(item)
            self.metrics[stage_name]['processed'] += 1
            
            # Update timing
            elapsed = time.time() - start_time
            avg = self.metrics[stage_name]['avg_process_time']
            self.metrics[stage_name]['avg_process_time'] = (
                avg * 0.9 + elapsed * 0.1
            )
            
            return result
            
        except Exception as e:
            self.metrics[stage_name]['errors'] += 1
            
            if stage.config.error_policy == "fail":
                raise
            elif stage.config.error_policy == "skip":
                return None
            elif stage.config.error_policy == "retry":
                if item.retries < 3:
                    item.retries += 1
                    return item
                return None
            return None
    
    async def _run_stage(self, stage_name: str) -> None:
        """Run a processing stage."""
        stage = self.stages[stage_name]
        input_queue = self.queues[stage_name]
        
        # Get output queue if not sink
        output_queue = None
        stage_names = list(self.stages.keys())
        if stage.config.stage_type != PipelineStage.SINK:
            stage_idx = stage_names.index(stage_name)
            if stage_idx + 1 < len(stage_names):
                output_queue = self.queues[stage_names[stage_idx + 1]]
        
        while self.running:
            try:
                # Collect batch
                batch: List[DataItem] = []
                
                # Get first item with timeout
                try:
                    item = await asyncio.wait_for(
                        input_queue.get(),
                        timeout=stage.config.batch_timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    if not batch:
                        continue
                
                # Collect more items
                while len(batch) < stage.config.batch_size:
                    try:
                        item = await asyncio.wait_for(
                            input_queue.get(),
                            timeout=0.1
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                for item in batch:
                    if stage.config.stage_type == PipelineStage.FILTER:
                        # Filter stage: only pass if result is truthy
                        result = await self._process_item(stage_name, item)
                        if result and output_queue:
                            await output_queue.put(result)
                    elif stage.config.stage_type == PipelineStage.SINK:
                        # Sink stage: no output
                        await self._process_item(stage_name, item)
                    else:
                        # Transform/aggregate stages
                        result = await self._process_item(stage_name, item)
                        if result and output_queue:
                            await output_queue.put(result)
                
            except Exception as e:
                print(f"[ERROR] Stage {stage_name}: {e}")
                await asyncio.sleep(1)
    
    async def start(self) -> None:
        """Start the pipeline."""
        self.running = True
        
        # Start all stages
        for stage_name in self.stages:
            for _ in range(self.stages[stage_name].config.workers):
                task = asyncio.create_task(self._run_stage(stage_name))
                self.tasks.append(task)
        
        print(f"[PIPELINE] '{self.name}' started with {len(self.stages)} stages")
    
    async def stop(self) -> None:
        """Stop the pipeline."""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        
        print(f"[PIPELINE] '{self.name}' stopped")
    
    async def add_item(self, item: DataItem) -> None:
        """Add item to pipeline input."""
        first_stage = list(self.stages.keys())[0]
        await self.queues[first_stage].put(item)
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get pipeline metrics."""
        return self.metrics

# Example pipeline implementations
async def source_processor(item: DataItem) -> DataItem:
    """Source processor: generate or fetch data."""
    # In real implementation, this would fetch from API, database, etc.
    return item

async def transform_processor(item: DataItem) -> DataItem:
    """Transform processor: modify data."""
    # Example: Normalize data
    if 'name' in item.data:
        item.data['name'] = item.data['name'].strip().lower()
    if 'value' in item.data:
        item.data['value'] = float(item.data['value'])
    return item

async def filter_processor(item: DataItem) -> DataItem:
    """Filter processor: filter data based on criteria."""
    # Example: Only pass valid items
    required_fields = ['name', 'value']
    if not all(field in item.data for field in required_fields):
        return None  # Will be filtered out
    return item

async def aggregate_processor(items: List[DataItem]) -> Dict:
    """Aggregate processor: combine multiple items."""
    values = [item.data.get('value', 0) for item in items]
    return {
        'count': len(items),
        'sum': sum(values),
        'avg': sum(values) / len(values) if values else 0
    }

async def sink_processor(item: DataItem) -> None:
    """Sink processor: output data."""
    # In real implementation, this would write to database, file, etc.
    print(f"[SINK] Processed item {item.item_id}")

# Demonstration
async def demonstrate_pipeline():
    """Demonstrate the data pipeline."""
    print("\n" + "=" * 60)
    print("ASYNC DATA PIPELINE DEMO")
    print("=" * 60)
    
    # Create pipeline
    pipeline = AsyncPipeline("Data Processing Pipeline")
    
    # Add stages
    pipeline.add_stage(
        PipelineConfig(
            stage_type=PipelineStage.TRANSFORM,
            name="normalize",
            batch_size=10,
            workers=2
        ),
        transform_processor
    )
    
    pipeline.add_stage(
        PipelineConfig(
            stage_type=PipelineStage.FILTER,
            name="validate",
            batch_size=10
        ),
        filter_processor
    )
    
    pipeline.add_stage(
        PipelineConfig(
            stage_type=PipelineStage.SINK,
            name="output",
            batch_size=10
        ),
        sink_processor
    )
    
    # Start pipeline
    await pipeline.start()
    
    # Add some test data
    test_data = [
        {"name": "  Alice  ", "value": "100"},
        {"name": "Bob", "value": "invalid"},
        {"name": "  Carol  ", "value": "300"},
        {"name": "Dave", "value": "400"},
    ]
    
    print("\nAdding test data...")
    for i, data in enumerate(test_data):
        item = DataItem(
            item_id=f"item_{i}",
            data=data,
            source="test"
        )
        await pipeline.add_item(item)
        await asyncio.sleep(0.1)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Show metrics
    print("\nPipeline Metrics:")
    for stage_name, metrics in pipeline.get_metrics().items():
        print(f"  {stage_name}: {metrics}")
    
    # Stop pipeline
    await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(demonstrate_pipeline())
```

---

## Best Practices for Async Programming

### Guidelines for Writing Async Code

1. **Use Async for I/O-Bound Operations**
   - Network requests and responses
   - Database queries
   - File I/O operations
   - External API calls

2. **Avoid Async for CPU-Bound Operations**
   - Use multiprocessing for CPU-intensive tasks
   - Don't block the event loop with heavy computations
   - Consider using run_in_executor for occasional CPU work

3. **Proper Resource Management**
   - Always use async context managers
   - Clean up resources when done
   - Handle disconnections gracefully

4. **Error Handling**
   - Implement proper exception handling
   - Use retry logic with exponential backoff
   - Consider circuit breaker patterns for external services

5. **Concurrency Control**
   - Use semaphores to limit concurrent operations
   - Implement rate limiting for external APIs
   - Balance concurrency with resource usage

### Common Anti-Patterns to Avoid

1. **Never await in a non-async function**
2. **Don't create unnecessary tasks without awaiting**
3. **Avoid mixing sync and async code inappropriately**
4. **Don't forget to handle cancellation**
5. **Avoid memory leaks by cleaning up tasks and references**

### Summary

This practice guide has covered essential async programming concepts:

- **Event Loop Mechanics** - Understanding how async execution works
- **Coroutines and Tasks** - Creating and managing async tasks
- **Context Managers** - Resource management with async context
- **Iterators and Generators** - Streaming data efficiently
- **HTTP Operations** - Building async clients
- **Database Operations** - Async database access
- **Error Handling** - Resilience patterns
- **Real Applications** - Chat server and data pipeline

Practice these patterns to build efficient, scalable Python applications.
