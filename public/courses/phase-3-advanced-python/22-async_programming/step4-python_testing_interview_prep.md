# Async Programming & Production Python - Interview Preparation Guide

## Table of Contents

1. [Core Concepts Questions](#core-concepts)
2. [Technical Deep Dive](#technical-deep-dive)
3. [Production Scenarios](#production-scenarios)
4. [Code Challenges](#code-challenges)
5. [System Design Questions](#system-design)
6. [Best Practices](#best-practices)
7. [Common Mistakes](#common-mistakes)

## Core Concepts Questions {#core-concepts}

### Q1: What is the difference between synchronous and asynchronous programming?

**Answer:**
Synchronous programming executes tasks sequentially - each operation must complete before the next one starts. Asynchronous programming allows multiple operations to run concurrently without blocking execution.

**Example:**

```python
# Synchronous
import requests
def sync_example():
    response1 = requests.get("https://api1.com")  # Blocks here
    response2 = requests.get("https://api2.com")  # Waits for response1
    return response1, response2

# Asynchronous
import aiohttp
import asyncio
async def async_example():
    async with aiohttp.ClientSession() as session:
        task1 = session.get("https://api1.com")
        task2 = session.get("https://api2.com")
        response1, response2 = await asyncio.gather(task1, task2)
        return response1, response2
```

### Q2: Explain the event loop and how it works.

**Answer:**
The event loop is the core mechanism of async programming that:

- Manages the execution of async tasks
- Handles I/O operations without blocking
- Coordinates between coroutines
- Provides timing and scheduling

**Key concepts:**

- Single-threaded execution model
- Non-blocking I/O operations
- Task scheduling and switching
- Callback registration and execution

### Q3: What are coroutines and how do they differ from regular functions?

**Answer:**
Coroutines are special functions that can be paused and resumed. Defined with `async def`, they:

- Use `await` to pause execution
- Return control to the event loop
- Can be by awaited- Support other coroutines
  cancellation and timeout

```python
async def coroutine_example():
    print("Starting")
    await asyncio.sleep(1)  # Pauses here
    print("Continuing")
    return "Completed"

# Regular function
def regular_function():
    print("Regular function")
    time.sleep(1)  # Blocks execution
    return "Done"
```

### Q4: What is the difference between tasks and coroutines?

**Answer:**

- **Coroutines**: Functions that can be paused and resumed (created with `async def`)
- **Tasks**: Wrapped coroutines that are scheduled for execution (created with `asyncio.create_task()`)

```python
async def my_coroutine():
    await asyncio.sleep(1)
    return "Result"

# Coroutine (not yet running)
coro = my_coroutine()

# Task (scheduled to run)
task = asyncio.create_task(my_coroutine())
```

## Technical Deep Dive {#technical-deep-dive}

### Q5: How do you handle multiple concurrent HTTP requests?

**Answer:**

```python
import aiohttp
import asyncio

async def concurrent_http_requests():
    urls = [
        "https://api1.com/data",
        "https://api2.com/users",
        "https://api3.com/posts"
    ]

    async with aiohttp.ClientSession() as session:
        # Method 1: asyncio.gather
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        data = [await r.json() for r in responses]

        # Method 2: Semaphore for rate limiting
        semaphore = asyncio.Semaphore(5)

        async def limited_request(url):
            async with semaphore:
                async with session.get(url) as response:
                    return await response.json()

        limited_tasks = [limited_request(url) for url in urls]
        return await asyncio.gather(*limited_tasks)
```

### Q6: How do you implement a connection pool for async operations?

**Answer:**

```python
import asyncpg
import asyncio

class AsyncConnectionPool:
    def __init__(self, connection_string, min_size=5, max_size=20):
        self.pool = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            self.connection_string,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

    async def execute_query(self, query, *args):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)

    async def execute_transaction(self, queries):
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                results = []
                for query in queries:
                    result = await connection.execute(query)
                    results.append(result)
                return results
```

### Q7: What are the different ways to handle timeouts in async code?

**Answer:**

```python
import asyncio

# Method 1: asyncio.wait_for
async def with_wait_for():
    try:
        result = await asyncio.wait_for(
            long_running_operation(),
            timeout=10.0
        )
        return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None

# Method 2: asyncio.wait with timeout
async def with_wait_timeout():
    done, pending = await asyncio.wait(
        [long_running_operation()],
        timeout=5.0
    )

    if pending:
        for task in pending:
            task.cancel()
        return "Timed out"

    return list(done)[0].result()

# Method 3: Asyncio.wait_for with custom exception
async def with_custom_timeout():
    try:
        return await asyncio.wait_for(
            operation(),
            timeout=asyncio.timeout(30)  # Python 3.11+
        )
    except asyncio.TimeoutError:
        logger.error("Custom timeout handling")
        raise TimeoutException("Operation took too long")
```

### Q8: How do you implement a circuit breaker pattern in async code?

**Answer:**

```python
import asyncio
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class AsyncCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time < self.timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN

        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## Production Scenarios {#production-scenarios}

### Q9: How would you design an async web scraper that processes thousands of pages?

**Answer:**

```python
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Set, List

class AsyncWebScraper:
    def __init__(self, base_url: str, max_concurrent: int = 10):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.visited: Set[str] = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def scrape_url(self, url: str) -> List[str]:
        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Extract links
                    links = []
                    for link in soup.find_all('a', href=True):
                        absolute_url = urljoin(url, link['href'])
                        parsed = urlparse(absolute_url)

                        # Only scrape same domain
                        if parsed.netloc == self.domain and absolute_url not in self.visited:
                            links.append(absolute_url)

                    return links

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                return []

    async def scrape_site(self, max_pages: int = 1000) -> List[str]:
        queue = asyncio.Queue()
        await queue.put(self.base_url)
        results = []

        while not queue.empty() and len(self.visited) < max_pages:
            current_url = await queue.get()

            if current_url in self.visited:
                continue

            self.visited.add(current_url)

            # Scrape current page
            links = await self.scrape_url(current_url)
            results.append(current_url)

            # Add new links to queue
            for link in links:
                if link not in self.visited:
                    await queue.put(link)

        return results

# Usage
async def main():
    async with AsyncWebScraper("https://example.com") as scraper:
        scraped_urls = await scraper.scrape_site(max_pages=100)
        print(f"Scraped {len(scraped_urls)} URLs")
```

### Q10: How do you implement real-time data processing with async?

**Answer:**

```python
import asyncio
import json
from typing import Any, Callable

class AsyncDataProcessor:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.workers = []
        self.processors = []

    async def start(self):
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)

        # Start processor tasks
        processor = asyncio.create_task(self._processor())
        self.processors.append(processor)

    async def _worker(self, worker_id: int):
        while True:
            try:
                # Get data from input
                data = await self.input_queue.get()

                # Process data
                processed_data = await self._process_data(data)

                # Put to output queue
                await self.output_queue.put(processed_data)

                self.input_queue.task_done()

            except Exception as e:
                print(f"Worker {worker_id} error: {e}")

    async def _processor(self):
        while True:
            try:
                # Get processed data
                data = await self.output_queue.get()

                # Store or forward data
                await self._store_data(data)

                self.output_queue.task_done()

            except Exception as e:
                print(f"Processor error: {e}")

    async def _process_data(self, data: Any) -> Any:
        # Simulate processing time
        await asyncio.sleep(0.1)
        return {"processed": True, "data": data}

    async def _store_data(self, data: Any):
        # Simulate storage operation
        await asyncio.sleep(0.05)
        print(f"Stored: {data}")

    async def add_data(self, data: Any):
        await self.input_queue.put(data)

    async def get_result(self) -> Any:
        return await self.output_queue.get()

    async def stop(self):
        # Cancel all tasks
        for task in self.workers + self.processors:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.workers, *self.processors, return_exceptions=True)

# Usage
async def main():
    processor = AsyncDataProcessor(max_workers=3)
    await processor.start()

    # Add data
    for i in range(10):
        await processor.add_data(f"data_{i}")

    # Process and collect results
    results = []
    for _ in range(10):
        result = await processor.get_result()
        results.append(result)

    await processor.stop()
    print(f"Processed {len(results)} items")
```

### Q11: How do you handle graceful shutdown in a long-running async application?

**Answer:**

```python
import asyncio
import signal
import logging
from typing import List

class AsyncApplication:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.running_tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger(__name__)

    async def setup_signal_handlers(self):
        loop = asyncio.get_event_loop()

        # Handle termination signals
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                self.shutdown_event.set
            )

    async def start_background_task(self, task_func, *args, **kwargs):
        """Start a background task that respects shutdown"""
        async def wrapped_task():
            try:
                await task_func(*args, **kwargs)
            except asyncio.CancelledError:
                self.logger.info("Task cancelled during shutdown")
                raise
            except Exception as e:
                self.logger.error(f"Task error: {e}")

        task = asyncio.create_task(wrapped_task())
        self.running_tasks.append(task)
        return task

    async def shutdown(self):
        self.logger.info("Starting graceful shutdown...")

        # Signal shutdown to all tasks
        self.shutdown_event.set()

        # Cancel all running tasks
        for task in self.running_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete or timeout
        if self.running_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.running_tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some tasks didn't complete within timeout")

        # Perform cleanup
        await self.cleanup_resources()
        self.logger.info("Shutdown complete")

    async def cleanup_resources(self):
        """Override this method to cleanup specific resources"""
        self.logger.info("Cleaning up resources...")

    async def run(self):
        """Main application loop"""
        await self.setup_signal_handlers()

        try:
            while not self.shutdown_event.is_set():
                await self.process_requests()

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            await self.shutdown()

# Example usage
class MyApplication(AsyncApplication):
    async def process_requests(self):
        # Main application logic
        await asyncio.sleep(1)

    async def cleanup_resources(self):
        # Close database connections, file handles, etc.
        await self.close_database_connections()
        await self.close_file_handles()

async def main():
    app = MyApplication()
    await app.run()

# Run the application
# asyncio.run(main())
```

## Code Challenges {#code-challenges}

### Challenge 1: Async Rate Limiter

```python
class AsyncRateLimiter:
    """Rate limiter using sliding window algorithm"""

    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    async def acquire(self):
        """Acquire permission to make a request"""
        now = asyncio.get_event_loop().time()

        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests
                        if now - req_time < self.time_window]

        # Check if we've exceeded the limit
        if len(self.requests) >= self.max_requests:
            # Calculate sleep time until oldest request expires
            oldest_request = min(self.requests)
            sleep_time = self.time_window - (now - oldest_request)

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                return await self.acquire()  # Retry after sleeping

        # Record this request
        self.requests.append(now)

# Test the rate limiter
async def test_rate_limiter():
    limiter = AsyncRateLimiter(max_requests=3, time_window=1.0)

    # Should allow 3 requests quickly
    for i in range(3):
        await limiter.acquire()
        print(f"Request {i+1} allowed at {asyncio.get_event_loop().time()}")

    # 4th request should be delayed
    await limiter.acquire()
    print(f"Request 4 allowed at {asyncio.get_event_loop().time()}")
```

### Challenge 2: Async File Downloader

```python
import aiohttp
import aiofiles
import os
from pathlib import Path

class AsyncFileDownloader:
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def download_file(self, url: str, filepath: str):
        """Download a single file"""
        async with self.semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            # Ensure directory exists
                            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

                            # Write file asynchronously
                            async with aiofiles.open(filepath, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    await f.write(chunk)

                            print(f"Downloaded: {filepath}")
                            return filepath
                        else:
                            raise Exception(f"HTTP {response.status}: {url}")

            except Exception as e:
                print(f"Failed to download {url}: {e}")
                return None

    async def download_multiple(self, urls_and_paths: list):
        """Download multiple files concurrently"""
        tasks = [
            self.download_file(url, filepath)
            for url, filepath in urls_and_paths
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Test the downloader
async def test_downloader():
    downloader = AsyncFileDownloader(max_concurrent=3)

    files_to_download = [
        ("https://httpbin.org/json", "downloads/httpbin.json"),
        ("https://httpbin.org/uuid", "downloads/httpbin.uuid"),
        ("https://httpbin.org/user-agent", "downloads/httpbin.agent"),
    ]

    results = await downloader.download_multiple(files_to_download)
    successful = [r for r in results if r is not None]
    print(f"Successfully downloaded {len(successful)} files")
```

### Challenge 3: Async Task Scheduler

```python
import asyncio
from datetime import datetime, timedelta
from typing import Callable, Any, Dict

class AsyncTaskScheduler:
    def __init__(self):
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}
        self.recurring_tasks: Dict[str, asyncio.Task] = {}

    async def schedule_once(self, task_id: str, func: Callable, delay: float, *args, **kwargs):
        """Schedule a task to run once after a delay"""

        async def delayed_task():
            await asyncio.sleep(delay)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"Task {task_id} failed: {e}")
                raise

        task = asyncio.create_task(delayed_task(), name=task_id)
        self.scheduled_tasks[task_id] = task
        return task

    async def schedule_recurring(self, task_id: str, func: Callable, interval: float, *args, **kwargs):
        """Schedule a task to run repeatedly"""

        async def recurring_task():
            while True:
                try:
                    await func(*args, **kwargs)
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Recurring task {task_id} failed: {e}")
                    await asyncio.sleep(interval)  # Continue despite errors

        task = asyncio.create_task(recurring_task(), name=task_id)
        self.recurring_tasks[task_id] = task
        return task

    async def cancel_task(self, task_id: str):
        """Cancel a scheduled task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].cancel()
            del self.scheduled_tasks[task_id]

        if task_id in self.recurring_tasks:
            self.recurring_tasks[task_id].cancel()
            del self.recurring_tasks[task_id]

    async def cancel_all(self):
        """Cancel all scheduled tasks"""
        for task in list(self.scheduled_tasks.values()) + list(self.recurring_tasks.values()):
            task.cancel()
        self.scheduled_tasks.clear()
        self.recurring_tasks.clear()

# Test the scheduler
async def test_scheduler():
    scheduler = AsyncTaskScheduler()

    # Schedule one-time task
    await scheduler.schedule_once(
        "one_time_task",
        lambda: print("One-time task executed!"),
        delay=2.0
    )

    # Schedule recurring task
    await scheduler.schedule_recurring(
        "recurring_task",
        lambda: print("Recurring task executed!"),
        interval=1.0
    )

    # Let it run for 5 seconds
    await asyncio.sleep(5)

    # Cancel all tasks
    await scheduler.cancel_all()
    print("All tasks cancelled")
```

## System Design Questions {#system-design}

### Q12: Design an async microservices architecture

**Answer:**

```python
# Service Discovery and Load Balancing
class AsyncServiceRegistry:
    def __init__(self):
        self.services: Dict[str, List[str]] = {}
        self.health_checks: Dict[str, asyncio.Task] = {}

    async def register_service(self, service_name: str, service_url: str):
        """Register a service instance"""
        if service_name not in self.services:
            self.services[service_name] = []

        self.services[service_name].append(service_url)

        # Start health check
        health_task = asyncio.create_task(
            self._health_check(service_name, service_url)
        )
        self.health_checks[f"{service_name}:{service_url}"] = health_task

    async def _health_check(self, service_name: str, service_url: str):
        """Continuously check service health"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{service_url}/health") as response:
                        if response.status != 200:
                            await self._remove_unhealthy_service(service_name, service_url)

            except Exception:
                await self._remove_unhealthy_service(service_name, service_url)

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _remove_unhealthy_service(self, service_name: str, service_url: str):
        """Remove unhealthy service from registry"""
        if service_name in self.services:
            if service_url in self.services[service_name]:
                self.services[service_name].remove(service_url)
                print(f"Removed unhealthy service: {service_name} at {service_url}")

# Async API Gateway
class AsyncAPIGateway:
    def __init__(self, service_registry: AsyncServiceRegistry):
        self.service_registry = service_registry
        self.rate_limiters: Dict[str, AsyncRateLimiter] = {}

    async def route_request(self, path: str, method: str, data: Any = None):
        """Route request to appropriate service"""
        service_name = self._extract_service_name(path)

        # Get healthy service instances
        services = self.service_registry.services.get(service_name, [])
        if not services:
            raise ServiceNotFoundError(f"No healthy instances for {service_name}")

        # Load balance (simple round-robin)
        service_url = services[hash(path) % len(services)]

        # Rate limiting
        if service_name not in self.rate_limiters:
            self.rate_limiters[service_name] = AsyncRateLimiter(100, 60)  # 100 req/min

        await self.rate_limiters[service_name].acquire()

        # Forward request
        async with aiohttp.ClientSession() as session:
            url = f"{service_url}{path}"

            if method == "GET":
                async with session.get(url) as response:
                    return await response.json()
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()

    def _extract_service_name(self, path: str) -> str:
        """Extract service name from path"""
        parts = path.strip('/').split('/')
        return parts[0] if parts else 'unknown'

# Service Implementation
class AsyncUserService:
    async def get_user(self, user_id: int):
        """Get user by ID"""
        # Simulate database query
        await asyncio.sleep(0.1)
        return {"id": user_id, "name": f"User {user_id}"}

    async def create_user(self, user_data: dict):
        """Create new user"""
        # Simulate database insertion
        await asyncio.sleep(0.2)
        user_data["id"] = hash(str(user_data)) % 10000
        return user_data
```

### Q13: Design an async real-time chat system

**Answer:**

```python
import asyncio
import json
from typing import Dict, Set
from datetime import datetime

class AsyncChatRoom:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.participants: Set[asyncio.Queue] = set()
        self.message_history: list = []

    async def join(self, queue: asyncio.Queue):
        """Add participant to room"""
        self.participants.add(queue)
        await self.broadcast_system_message(f"User joined room {self.room_id}")

    async def leave(self, queue: asyncio.Queue):
        """Remove participant from room"""
        if queue in self.participants:
            self.participants.remove(queue)
            await self.broadcast_system_message(f"User left room {self.room_id}")

    async def send_message(self, sender: str, message: str):
        """Send message to all participants"""
        timestamp = datetime.now().isoformat()
        chat_message = {
            "type": "chat",
            "sender": sender,
            "message": message,
            "timestamp": timestamp,
            "room_id": self.room_id
        }

        # Store in history
        self.message_history.append(chat_message)

        # Broadcast to all participants
        await self.broadcast_message(chat_message)

    async def broadcast_message(self, message: dict):
        """Send message to all participants"""
        if self.participants:
            # Remove disconnected participants
            active_participants = set()

            for queue in self.participants:
                try:
                    # Try to send message (non-blocking)
                    queue.put_nowait(message)
                    active_participants.add(queue)
                except asyncio.QueueFull:
                    # Participant is slow, skip this message
                    pass
                except:
                    # Participant disconnected
                    pass

            self.participants = active_participants

    async def broadcast_system_message(self, message: str):
        """Send system message"""
        system_msg = {
            "type": "system",
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "room_id": self.room_id
        }
        await self.broadcast_message(system_msg)

class AsyncChatServer:
    def __init__(self):
        self.rooms: Dict[str, AsyncChatRoom] = {}
        self.user_queues: Dict[str, asyncio.Queue] = {}

    async def handle_user_connection(self, user_id: str, websocket):
        """Handle new user connection"""
        user_queue = asyncio.Queue(maxsize=100)
        self.user_queues[user_id] = user_queue

        try:
            # Send welcome message
            welcome_msg = {
                "type": "welcome",
                "user_id": user_id,
                "message": f"Welcome, {user_id}!"
            }
            await websocket.send(json.dumps(welcome_msg))

            # Handle user messages
            async for message_text in websocket:
                try:
                    message_data = json.loads(message_text)
                    await self.handle_user_message(user_id, message_data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))

        except Exception as e:
            print(f"Error handling user {user_id}: {e}")
        finally:
            await self.handle_user_disconnect(user_id)

    async def handle_user_message(self, user_id: str, message_data: dict):
        """Handle incoming message from user"""
        msg_type = message_data.get("type")

        if msg_type == "join_room":
            room_id = message_data["room_id"]
            await self.join_room(user_id, room_id)

        elif msg_type == "leave_room":
            room_id = message_data["room_id"]
            await self.leave_room(user_id, room_id)

        elif msg_type == "chat":
            room_id = message_data["room_id"]
            message = message_data["message"]

            if room_id in self.rooms:
                await self.rooms[room_id].send_message(user_id, message)

        elif msg_type == "get_history":
            room_id = message_data["room_id"]
            await self.send_message_to_user(user_id, {
                "type": "history",
                "room_id": room_id,
                "messages": self.rooms[room_id].message_history[-50:]  # Last 50 messages
            })

    async def join_room(self, user_id: str, room_id: str):
        """Add user to chat room"""
        if room_id not in self.rooms:
            self.rooms[room_id] = AsyncChatRoom(room_id)

        user_queue = self.user_queues[user_id]
        await self.rooms[room_id].join(user_queue)

        await self.send_message_to_user(user_id, {
            "type": "room_joined",
            "room_id": room_id
        })

    async def leave_room(self, user_id: str, room_id: str):
        """Remove user from chat room"""
        if room_id in self.rooms:
            user_queue = self.user_queues[user_id]
            await self.rooms[room_id].leave(user_queue)

        await self.send_message_to_user(user_id, {
            "type": "room_left",
            "room_id": room_id
        })

    async def send_message_to_user(self, user_id: str, message: dict):
        """Send message to specific user"""
        if user_id in self.user_queues:
            try:
                self.user_queues[user_id].put_nowait(message)
            except asyncio.QueueFull:
                print(f"User {user_id} queue is full, dropping message")

    async def handle_user_disconnect(self, user_id: str):
        """Handle user disconnection"""
        if user_id in self.user_queues:
            # Remove from all rooms
            for room in self.rooms.values():
                await room.leave(self.user_queues[user_id])

            # Clean up
            del self.user_queues[user_id]
```

## Best Practices {#best-practices}

### Q14: What are the key best practices for async Python in production?

**Answer:**

1. **Resource Management**
   - Always use context managers for cleanup
   - Implement proper connection pooling
   - Set appropriate timeouts

2. **Error Handling**
   - Implement comprehensive exception handling
   - Use circuit breakers for external services
   - Add retry logic with exponential backoff

3. **Performance**
   - Use async libraries designed for async (aiohttp, asyncpg)
   - Avoid blocking operations (time.sleep, blocking I/O)
   - Monitor memory usage and implement garbage collection

4. **Monitoring**
   - Add structured logging
   - Implement metrics collection
   - Monitor event loop performance

5. **Testing**
   - Use pytest-asyncio for testing
   - Mock async operations properly
   - Test timeout and error scenarios

```python
# Production-ready async application template
import asyncio
import logging
import structlog
from contextlib import asynccontextmanager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
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

class ProductionAsyncApp:
    def __init__(self):
        self.logger = structlog.get_logger()
        self.shutdown_event = asyncio.Event()

    @asynccontextmanager
    async def lifespan(self):
        """Application lifespan management"""
        # Startup
        self.logger.info("Starting application")
        await self.initialize_resources()

        try:
            yield
        finally:
            # Shutdown
            self.logger.info("Shutting down application")
            await self.cleanup_resources()

    async def initialize_resources(self):
        """Initialize connections, pools, etc."""
        self.logger.info("Initializing resources")
        # Initialize database connections, HTTP sessions, etc.

    async def cleanup_resources(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up resources")
        # Close connections, cancel tasks, etc.

    async def run_with_monitoring(self):
        """Run application with monitoring"""
        async with self.lifespan():
            while not self.shutdown_event.is_set():
                try:
                    await self.process_requests()
                except Exception as e:
                    self.logger.error("Error processing requests", error=str(e))
                    await asyncio.sleep(1)  # Brief pause before retry
```

## Common Mistakes {#common-mistakes}

### Q15: What are common async programming mistakes to avoid?

**Answer:**

1. **Blocking Operations in Async Code**

```python
# ❌ Wrong - blocks the event loop
async def bad_example():
    import time
    time.sleep(5)  # This blocks!
    return "Done"

# ✅ Correct - non-blocking
async def good_example():
    await asyncio.sleep(5)  # Yields control
    return "Done"
```

2. **Not Using Async Libraries**

```python
# ❌ Wrong - blocking HTTP library
import requests
async def bad_http():
    response = requests.get("https://api.example.com")
    return response.json()

# ✅ Correct - async HTTP library
import aiohttp
async def good_http():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as response:
            return await response.json()
```

3. **Missing Exception Handling**

```python
# ❌ Wrong - no error handling
async def bad_error_handling():
    result = await risky_operation()
    return result

# ✅ Correct - comprehensive error handling
async def good_error_handling():
    try:
        result = await risky_operation()
        return result
    except SpecificError as e:
        logger.error("Specific error occurred", error=str(e))
        return default_value
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise
```

4. **Not Managing Resources Properly**

```python
# ❌ Wrong - resource leaks
async def bad_resource_management():
    session = aiohttp.ClientSession()
    response = await session.get("https://example.com")
    data = await response.json()
    # Session never closed!

# ✅ Correct - proper resource management
async def good_resource_management():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://example.com") as response:
            return await response.json()
```

5. **Improper Task Management**

```python
# ❌ Wrong - tasks never awaited
async def bad_task_management():
    task = asyncio.create_task(long_operation())
    return "Done"  # Task is abandoned!

# ✅ Correct - proper task management
async def good_task_management():
    task = asyncio.create_task(long_operation())
    result = await task  # Wait for completion
    return result

# Or for background tasks
async def background_task_management():
    task = asyncio.create_task(background_operation())
    # Store task reference for later cancellation if needed
    return "Started background task"
```

### Q16: How do you debug async code effectively?

**Answer:**

1. **Enable Debug Mode**

```python
import asyncio

# Enable asyncio debug mode
asyncio.get_event_loop().set_debug(True)

# This will show detailed error messages for:
# - Tasks that were never awaited
# - Operations that took too long
# - Resource leaks
```

2. **Add Debug Logging**

```python
import functools
import logging

def debug_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Failed {func.__name__}: {e}")
            raise
    return wrapper

@debug_async
async def my_async_function():
    await asyncio.sleep(1)
    return "Success"
```

3. **Use Async-Aware Debugging Tools**

```python
# Memory profiling
import tracemalloc

async def profile_async_function():
    tracemalloc.start()
    try:
        result = await some_async_operation()
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Memory usage: {current / 1024 / 1024:.1f} MB")
        return result
    finally:
        tracemalloc.stop()
```

4. **Monitor Event Loop Performance**

```python
import asyncio
import time

class AsyncProfiler:
    def __init__(self):
        self.operations = []

    async def profile_operation(self, operation_name: str, operation):
        start_time = time.time()
        start_loop = asyncio.get_event_loop()

        try:
            result = await operation()
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time

            self.operations.append({
                'name': operation_name,
                'duration': duration,
                'timestamp': end_time
            })

            logger.info(f"Operation {operation_name} took {duration:.3f}s")
```

This comprehensive interview preparation guide covers all essential aspects of async programming and production Python development. Study these concepts thoroughly to excel in technical interviews!
