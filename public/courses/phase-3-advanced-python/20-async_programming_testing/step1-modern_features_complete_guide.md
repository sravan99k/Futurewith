---
title: "Modern Python Features: Python 3.8+ Advanced Techniques"
level: "Advanced"
time: "150 mins"
prereq: "python_fundamentals_complete_guide.md"
tags: ["python", "modern", "pattern-matching", "async", "typing", "dataclasses"]
---

# 🚀 Modern Python Features: Code Like a 2025 Pythonista

_Master the Latest Python Features for Professional Development_

---

## 📘 **VERSION & UPDATE INFO**

**📘 Version 2.1 — Updated: November 2025**  
_Cutting-edge Python features for modern software development_

**🔴 Advanced**  
_Essential for senior developers, system architects, and performance-critical applications_

**🏢 Used in:** High-performance systems, concurrent applications, type-safe codebases, AI/ML pipelines  
**🧰 Popular Tools:** Python 3.8+, mypy, asyncio, uvloop, pydantic, modern IDEs

**🔗 Cross-reference:** Connect with `python_problem_solving_mindset_complete_guide.md` and `python_libraries_complete_guide.md`

---

**💼 Career Paths:** Senior Python Developer, Tech Lead, System Architect, Performance Engineer  
**🎯 Master Level:** Utilize modern Python features for cleaner, faster, and safer code

**🎯 Learning Navigation Guide**  
**If you score < 70%** → Focus on one feature at a time and practice with real examples  
**If you score ≥ 80%** → Experiment with combining multiple features in complex projects

---

## 🎯 **Pattern Matching (PEP 636) - Structural Pattern Matching**

### **Understanding Structural Pattern Matching**

```python
# Traditional approach vs Pattern Matching
from typing import Union, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Old way - lots of if-elif-else
def process_message_old(message_type: str, data: Any) -> str:
    if message_type == "user":
        if "name" in data and "email" in data:
            return f"Processing user: {data['name']}"
        else:
            return "Invalid user data"
    elif message_type == "order":
        if "items" in data and "total" in data:
            return f"Processing order with {len(data['items'])} items"
        else:
            return "Invalid order data"
    elif message_type == "notification":
        if "title" in data and "message" in data:
            return f"Notification: {data['title']}"
        else:
            return "Invalid notification"
    else:
        return "Unknown message type"

# Modern approach - Pattern Matching (Python 3.10+)
def process_message_new(message: Dict[str, Any]) -> str:
    match message:
        case {"type": "user", "name": str(name), "email": str(email)}:
            return f"Processing user: {name} ({email})"
        case {"type": "order", "items": items, "total": float(total)} if total > 0:
            return f"Processing order: {len(items)} items, ${total:.2f}"
        case {"type": "notification", "title": str(title), "message": str(message)}:
            return f"Alert: {title} - {message}"
        case {"type": "error", "code": int(code), "details": str(details)}:
            return f"Error {code}: {details}"
        case {"type": unknown_type, **rest}:
            return f"Unknown message type: {unknown_type}"
        case _:
            return "Invalid message format"

# Test the functions
test_messages = [
    {"type": "user", "name": "Alice", "email": "alice@example.com"},
    {"type": "order", "items": ["book", "pen"], "total": 25.50},
    {"type": "notification", "title": "System Alert", "message": "High CPU usage"},
    {"type": "error", "code": 404, "details": "Resource not found"},
    {"type": "unknown", "extra": "data"}
]

print("🔄 Pattern Matching vs Traditional Approach")
for msg in test_messages:
    result = process_message_new(msg)
    print(f"  {result}")
```

### **Advanced Pattern Matching Patterns**

```python
from typing import Tuple, Optional
import re

# 1. OR Patterns (match any of several values)
def handle_http_status(status_code: int) -> str:
    match status_code:
        case 200 | 201 | 202:
            return "Success"
        case 400 | 401 | 403 | 404:
            return "Client Error"
        case 500 | 502 | 503 | 504:
            return "Server Error"
        case code if 100 <= code < 600:
            return f"HTTP {code}"
        case _:
            return "Invalid status code"

# 2. AS Pattern (capture and use)
def parse_config(config: Dict[str, Any]) -> str:
    match config:
        case {"database": {"host": host, "port": port} as db_config, **rest}:
            return f"Database: {host}:{port} with extra config: {len(rest)} keys"
        case {"cache": {"type": "redis", "host": str(host)} as cache_config}:
            return f"Redis cache configured: {host}"
        case _:
            return "Unknown configuration"

# 3. Sequence Patterns
def analyze_data_structure(data: Any) -> str:
    match data:
        case []:
            return "Empty list"
        case [x]:
            return f"Single element list: {x}"
        case [x, y]:
            return f"Two element list: {x}, {y}"
        case [first, *middle, last]:
            return f"List with {first}...{last}, {len(middle)} in between"
        case tuple() as t:
            return f"Tuple with {len(t)} elements: {t}"
        case _ if isinstance(data, (list, tuple)):
            return f"Collection with {len(data)} elements"

# 4. Mapping Patterns with guards
def process_api_response(response: Dict[str, Any]) -> str:
    match response:
        case {"status": "success", "data": {"users": users, "count": count}} if count == len(users):
            return f"Success: {count} users returned"
        case {"status": "success", "data": {"items": items}} if len(items) > 100:
            return f"Large dataset: {len(items)} items (paginated recommended)"
        case {"status": "error", "message": str(msg), "code": int(code)}:
            return f"API Error {code}: {msg}"
        case {"status": status, **rest}:
            return f"Unexpected status: {status} with extra data"

# 5. Class Patterns with __match_args__
@dataclass
class Point:
    x: float
    y: float
    color: str = "black"

@dataclass
class Circle:
    center: Point
    radius: float

@dataclass
class Rectangle:
    top_left: Point
    bottom_right: Point

def describe_shape(shape: Union[Point, Circle, Rectangle]) -> str:
    match shape:
        case Point(x=0, y=0):
            return f"Origin point in {shape.color}"
        case Point(x, y, color="red"):
            return f"Red point at ({x}, {y})"
        case Circle(center=Point(x=0, y=0), radius=r):
            return f"Circle centered at origin with radius {r}"
        case Circle(center, radius) if radius > 10:
            return f"Large circle: {center} with radius {radius}"
        case Rectangle(top_left=Point(x1, y1), bottom_right=Point(x2, y2)):
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            return f"Rectangle: {width}×{height}"
        case _:
            return f"Unknown shape: {type(shape).__name__}"

# Test advanced patterns
def demo_advanced_patterns():
    """Demonstrate advanced pattern matching"""
    print("🎯 Advanced Pattern Matching Demo")

    # HTTP Status
    for code in [200, 404, 500, 999]:
        print(f"  {code}: {handle_http_status(code)}")

    # Config parsing
    configs = [
        {"database": {"host": "localhost", "port": 5432}, "cache": {"type": "redis", "host": "redis"}},
        {"cache": {"type": "redis", "host": "cache.local"}},
        {"unknown": "config"}
    ]
    for config in configs:
        print(f"  Config: {parse_config(config)}")

    # Data structures
    test_data = [[], [42], [1, 2, 3], (1, 2, 3, 4, 5)]
    for data in test_data:
        print(f"  {data}: {analyze_data_structure(data)}")

    # API responses
    responses = [
        {"status": "success", "data": {"users": ["alice", "bob"], "count": 2}},
        {"status": "error", "message": "Unauthorized", "code": 401},
        {"status": "unknown", "extra": "data"}
    ]
    for response in responses:
        print(f"  Response: {process_api_response(response)}")

    # Shapes
    shapes = [
        Point(0, 0),
        Point(10, 20, "red"),
        Circle(Point(0, 0), 15),
        Rectangle(Point(0, 10), Point(5, 0))
    ]
    for shape in shapes:
        print(f"  Shape: {describe_shape(shape)}")

demo_advanced_patterns()
```

---

## ⚡ **AsyncIO and Modern Concurrency**

### **Understanding Async/Await Basics**

```python
import asyncio
import aiohttp
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import aiofiles

# 1. Basic Async Function
async def fetch_data_async(delay: int, data: str) -> str:
    """Simulate async I/O operation"""
    print(f"  📡 Starting fetch for {data} (delay: {delay}s)")
    await asyncio.sleep(delay)  # Non-blocking sleep
    result = f"✅ Completed: {data} after {delay}s"
    print(f"  {result}")
    return result

async def demo_basic_async():
    """Demonstrate basic async operations"""
    print("⚡ Basic Async/Await Demo")

    # Sequential execution (slow)
    print("\n🔄 Sequential execution:")
    start = time.time()
    results = []
    for i in range(3):
        result = await fetch_data_async(1, f"Task {i+1}")
        results.append(result)
    sequential_time = time.time() - start
    print(f"  ⏱️ Total time: {sequential_time:.2f}s")

    # Concurrent execution (fast)
    print("\n🚀 Concurrent execution:")
    start = time.time()
    tasks = [fetch_data_async(1, f"Task {i+1}") for i in range(3)]
    results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    print(f"  ⏱️ Total time: {concurrent_time:.2f}s")
    print(f"  🚀 Speedup: {sequential_time/concurrent_time:.1f}x")

# Run the demo
# asyncio.run(demo_basic_async())
```

### **Advanced Async Patterns**

```python
# 2. Async Context Managers
class AsyncDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None

    async def __aenter__(self):
        self.connection = sqlite3.connect(self.db_path)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

    async def execute_query(self, query: str) -> List[Dict]:
        """Execute query asynchronously"""
        cursor = self.connection.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

# Usage with async context manager
async def async_database_demo():
    print("💾 Async Database Operations")

    # Simulate database operations
    async with AsyncDatabase(":memory:") as db:
        # Simulate async operations
        await asyncio.sleep(0.1)  # Simulate connection
        print("  📊 Database connected")

        # In real implementation, you would have actual queries
        await asyncio.sleep(0.1)
        print("  🔍 Query executed")

        await asyncio.sleep(0.1)
        print("  📈 Results retrieved")

# 3. Async Generators and Comprehensions
async def async_data_generator(count: int) -> str:
    """Generate data asynchronously"""
    for i in range(count):
        await asyncio.sleep(0.1)  # Simulate data processing
        yield f"Data chunk {i+1}"

async def process_streaming_data():
    """Process streaming data with async generators"""
    print("🌊 Streaming Data Processing")

    # Async generator comprehension
    processed_data = [
        item.upper()
        async for item in async_data_generator(5)
    ]

    print("  📦 Processed items:")
    for item in processed_data:
        print(f"    {item}")

# 4. Async Context Patterns
class RateLimiter:
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    async def __aenter__(self):
        await self._acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass  # Cleanup if needed

    async def _acquire(self):
        """Acquire a rate limit slot"""
        now = time.time()
        # Remove old calls outside time window
        self.calls = [call_time for call_time in self.calls
                     if now - call_time < self.time_window]

        if len(self.calls) >= self.max_calls:
            # Calculate wait time
            wait_time = self.time_window - (now - self.calls[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                await self._acquire()  # Retry
        else:
            self.calls.append(now)

# 5. Async Web Scraping with Rate Limiting
async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch URL asynchronously"""
    async with session.get(url) as response:
        return await response.text()

async def scrape_websites_with_rate_limiting(urls: List[str]):
    """Scrape multiple websites with rate limiting"""
    print("🕷️ Web Scraping with Rate Limiting")

    connector = aiohttp.TCPConnector(limit=10)  # Connection pool
    timeout = aiohttp.ClientTimeout(total=30)

    async with RateLimiter(max_calls=5, time_window=1.0):  # 5 calls per second
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    print(f"  ❌ {url}: Error - {result}")
                else:
                    print(f"  ✅ {url}: {len(result)} characters")

# 6. Async Task Management
class TaskManager:
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tasks = []

    async def run_with_limit(self, coro):
        """Run coroutine with concurrency limit"""
        async with self.semaphore:
            return await coro

    async def add_task(self, coro):
        """Add a task to the manager"""
        task = asyncio.create_task(self.run_with_limit(coro))
        self.tasks.append(task)
        return task

    async def wait_all(self):
        """Wait for all tasks to complete"""
        return await asyncio.gather(*self.tasks, return_exceptions=True)

# Demo async task management
async def task_management_demo():
    print("🎛️ Async Task Management")

    async def worker(task_id: int, duration: float):
        print(f"  🔄 Task {task_id} started")
        await asyncio.sleep(duration)
        print(f"  ✅ Task {task_id} completed")
        return f"Result from task {task_id}"

    manager = TaskManager(max_concurrent=3)

    # Add multiple tasks
    for i in range(5):
        await manager.add_task(worker(i, 1.0))

    # Wait for all to complete
    results = await manager.wait_all()

    print(f"  📊 All tasks completed: {len([r for r in results if not isinstance(r, Exception)])} successful")
```

### **Modern Async Patterns for High Performance**

```python
# 7. Async Context Manager for Resource Pooling
class AsyncConnectionPool:
    def __init__(self, create_connection, max_size: int = 10):
        self.create_connection = create_connection
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.connections = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup connections
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                await self._close_connection(conn)
            except asyncio.QueueEmpty:
                break

    async def get_connection(self):
        """Get connection from pool"""
        try:
            return self.pool.get_nowait()
        except asyncio.QueueEmpty:
            if self.connections < self.max_size:
                self.connections += 1
                return await self.create_connection()
            else:
                # Wait for connection to be returned
                return await self.pool.get()

    async def return_connection(self, conn):
        """Return connection to pool"""
        try:
            self.pool.put_nowait(conn)
        except asyncio.QueueFull:
            await self._close_connection(conn)

    async def _close_connection(self, conn):
        """Close connection (mock implementation)"""
        self.connections -= 1
        # In real implementation, close actual connection

# 8. Async File Operations
async def process_large_file(file_path: str):
    """Process large file asynchronously"""
    print("📄 Async File Processing")

    async with aiofiles.open(file_path, mode='r') as file:
        # Process file line by line asynchronously
        processed_lines = []
        async for line in file:
            # Process line asynchronously
            await asyncio.sleep(0.001)  # Simulate processing
            processed_line = line.strip().upper()
            processed_lines.append(processed_line)

            # Process in chunks
            if len(processed_lines) >= 1000:
                yield processed_lines
                processed_lines = []

        # Yield remaining lines
        if processed_lines:
            yield processed_lines

# 9. Error Handling in Async Code
async def robust_async_operation(operation_id: int, should_fail: bool = False):
    """Demonstrate error handling in async operations"""
    try:
        await asyncio.sleep(0.1)  # Simulate async work

        if should_fail:
            raise ValueError(f"Operation {operation_id} failed intentionally")

        return f"Success: operation {operation_id}"

    except ValueError as e:
        print(f"  ⚠️ Operation {operation_id} error: {e}")
        raise  # Re-raise to be handled by caller
    except Exception as e:
        print(f"  ❌ Unexpected error in operation {operation_id}: {e}")
        return None

async def async_error_handling_demo():
    """Demonstrate comprehensive async error handling"""
    print("🛡️ Async Error Handling")

    # Create tasks with different failure scenarios
    tasks = []
    for i in range(5):
        should_fail = i == 2  # Make 3rd task fail
        task = robust_async_operation(i, should_fail)
        tasks.append(task)

    # Run with error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Task {i}: Exception - {result}")
        elif result is None:
            print(f"  Task {i}: Failed with None return")
        else:
            print(f"  Task {i}: {result}")

# Complete async demonstration
async def complete_async_demo():
    """Complete demonstration of modern async patterns"""
    print("🚀 Complete Async/Await Demo")
    print("=" * 50)

    # Run all async demos
    await demo_basic_async()
    await async_database_demo()
    await process_streaming_data()
    await task_management_demo()
    await async_error_handling_demo()

    # Simulate web scraping
    print("\n🌐 Simulated Web Scraping:")
    test_urls = [
        "https://api.example.com/data1",
        "https://api.example.com/data2",
        "https://api.example.com/data3"
    ]
    # In a real scenario, you would use actual URLs
    print("  📡 Would scrape:", test_urls)
    print("  🚀 With rate limiting and error handling")

# Run the complete demo
# asyncio.run(complete_async_demo())
```

---

## 🔒 **Advanced Typing & Type Safety**

### **Modern Type Annotations**

```python
from typing import (
    TypeVar, Generic, Protocol, Union, Optional, Dict, List,
    Tuple, Set, FrozenSet, Callable, Awaitable, Iterator,
    Type, Any, Literal, TypedDict, NamedTuple, Final
)
from dataclasses import dataclass
from enum import Enum, Flag, IntEnum
import sys

# 1. Type Variables and Generic Types
T = TypeVar('T')
U = TypeVar('U', bound=int)  # Bounded type variable
K = TypeVar('K', str, int)  # Constrained type variable

class Container(Generic[T]):
    """Generic container with type safety"""
    def __init__(self, items: Optional[List[T]] = None):
        self.items = items or []

    def add(self, item: T) -> None:
        self.items.append(item)

    def get_all(self) -> List[T]:
        return self.items.copy()

    def get_first(self) -> Optional[T]:
        return self.items[0] if self.items else None

# 2. Protocol for Duck Typing
class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def __init__(self, radius: float):
        self.radius = radius

    def draw(self) -> str:
        return f"Circle with radius {self.radius}"

class Square:
    def __init__(self, side: float):
        self.side = side

    def draw(self) -> str:
        return f"Square with side {self.side}"

def render_shapes(shapes: List[Drawable]) -> List[str]:
    """Render any drawable shapes"""
    return [shape.draw() for shape in shapes]

# 3. Advanced Type Annotations
from typing import overload, runtime_checkable

@runtime_checkable
class JsonSerializable(Protocol):
    """Protocol for JSON-serializable objects"""
    def to_json(self) -> str: ...

@dataclass
class User:
    name: str
    email: str
    age: int

    def to_json(self) -> str:
        import json
        return json.dumps({
            "name": self.name,
            "email": self.email,
            "age": self.age
        })

# Function with multiple type signatures
@overload
def process_data(data: List[int]) -> int: ...

@overload
def process_data(data: List[str]) -> str: ...

@overload
def process_data(data: List[float]) -> float: ...

def process_data(data: List[Union[int, str, float]]) -> Union[int, str, float]:
    """Process data based on its type"""
    if not data:
        raise ValueError("Empty list provided")

    if isinstance(data[0], int):
        return sum(data)  # int
    elif isinstance(data[0], str):
        return "".join(data)  # str
    elif isinstance(data[0], float):
        return sum(data) / len(data)  # float
    else:
        raise TypeError(f"Unsupported type: {type(data[0])}")

# 4. TypedDict for Structured Data
class ConfigDict(TypedDict):
    host: str
    port: int
    debug: bool
    database_url: str
    api_key: NotRequired[str]  # Optional field

class UserInfo(TypedDict, total=False):  # Fields are optional by default
    name: str
    email: str
    preferences: Dict[str, Any]

# 5. NamedTuple for Lightweight Data Classes
from collections import namedtuple

class Coordinate(NamedTuple):
    x: float
    y: float
    z: Optional[float] = None

class RGBColor(NamedTuple):
    red: int
    green: int
    blue: int

    def to_hex(self) -> str:
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"

# 6. Literal Types for Specific Values
def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]) -> None:
    """Set log level with literal type"""
    print(f"Setting log level to: {level}")

def create_connection(
    driver: Literal["sqlite", "mysql", "postgresql"],
    host: str,
    port: int = 5432
) -> str:
    """Create connection string based on driver"""
    return f"{driver}://{host}:{port}"

# 7. Final and ReadOnly Types
from typing import Final

class MathConstants:
    PI: Final[float] = 3.14159265359
    E: Final[float] = 2.71828182846
    GOLDEN_RATIO: Final[float] = 1.61803398875

# 8. NewType for Type Safety
from typing import NewType

UserId = NewType('UserId', int)
SessionToken = NewType('SessionToken', str)

def get_user_by_id(user_id: UserId) -> str:
    """Get user with type-safe ID"""
    return f"User {user_id}"

def authenticate_session(token: SessionToken) -> UserId:
    """Authenticate and return user ID"""
    # In real implementation, validate token
    return UserId(123)

# 9. Self-Referencing Types
class TreeNode:
    def __init__(self, value: T, children: Optional[List[TreeNode[T]]] = None):
        self.value = value
        self.children = children or []

    def add_child(self, child: 'TreeNode[T]') -> None:
        self.children.append(child)

    def find_by_value(self, value: T) -> Optional['TreeNode[T]']:
        if self.value == value:
            return self
        for child in self.children:
            result = child.find_by_value(value)
            if result:
                return result
        return None

# 10. Type Guards
def is_string_list(value: List[Any]) -> TypeGuard[List[str]]:
    """Type guard to narrow List[Any] to List[str]"""
    return all(isinstance(item, str) for item in value)

def is_numeric_list(value: List[Any]) -> TypeGuard[List[float]]:
    """Type guard to narrow List[Any] to List[float]"""
    return all(isinstance(item, (int, float)) for item in value)

def safe_process_data(data: List[Any]) -> float:
    """Process data with type guards"""
    if is_string_list(data):
        return len("".join(data))  # String processing
    elif is_numeric_list(data):
        return sum(data) / len(data)  # Numeric processing
    else:
        raise TypeError("Unsupported data type")

# Demo Type Safety
def demo_advanced_typing():
    """Demonstrate advanced typing features"""
    print("🔒 Advanced Type Safety Demo")
    print("=" * 40)

    # Generic containers
    int_container = Container[int]([1, 2, 3])
    str_container = Container[str](["hello", "world"])

    print(f"  Int container: {int_container.get_all()}")
    print(f"  String container: {str_container.get_all()}")

    # Protocol usage
    shapes: List[Drawable] = [Circle(5.0), Square(3.0)]
    rendered = render_shapes(shapes)
    print(f"  Rendered shapes: {rendered}")

    # Overloaded function
    numbers = [1, 2, 3, 4, 5]
    result = process_data(numbers)
    print(f"  Sum of numbers: {result}")

    # NamedTuple
    coord = Coordinate(1.0, 2.0, 3.0)
    color = RGBColor(255, 128, 0)
    print(f"  Coordinate: {coord}")
    print(f"  Color: {color} -> {color.to_hex()}")

    # Literal types
    set_log_level("DEBUG")
    conn_str = create_connection("postgresql", "localhost")
    print(f"  Connection: {conn_str}")

    # NewType usage
    user_id = UserId(42)
    session_token = SessionToken("abc123")
    user = get_user_by_id(user_id)
    print(f"  User: {user}")
    print(f"  Session: {session_token}")

    # Type guards
    mixed_data = [1, 2, 3, 4, 5]
    safe_result = safe_process_data(mixed_data)
    print(f"  Safe processed data: {safe_result}")

demo_advanced_typing()
```

---

## 📦 **Modern Data Classes & Pydantic**

### **Enhanced Dataclasses**

```python
from dataclasses import dataclass, field, asdict, astuple, replace
from dataclasses import InitVar, make_dataclass
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
import json

# 1. Advanced Dataclass Features
@dataclass
class Config:
    """Enhanced configuration with validation"""
    name: str
    version: str = "1.0.0"
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # Post-init validation
    def __post_init__(self):
        if not self.name:
            raise ValueError("Name cannot be empty")
        if not self.version.count('.') == 2:
            raise ValueError("Version must follow semantic versioning (x.y.z)")

    # Custom methods
    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        self.settings[key] = value

# 2. Complex Field Types
@dataclass
class User:
    """User model with complex field types"""
    id: int
    name: str
    email: str
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Immutable after creation
    immutable_id: int = field(init=False, compare=True)
    # Computed field (not stored)
    display_name: str = field(init=False)

    def __post_init__(self):
        self.immutable_id = self.id
        self.display_name = f"{self.name} ({self.email})"

# 3. Field with Factory Functions
@dataclass
class LogEntry:
    """Log entry with dynamic field generation"""
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = field(default="INFO")
    message: str = ""
    # Factory function for request ID
    request_id: str = field(default_factory=lambda: f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # Field with repr=False (don't show in repr)
    sensitive_data: str = field(default="", repr=False)
    # Field with compare=False (don't include in comparison)
    created_at: datetime = field(default_factory=datetime.now, compare=False)

# 4. Dataclass with Custom Methods
@dataclass
class Order:
    """Order with business logic methods"""
    order_id: str
    items: List[Dict[str, Any]]
    total_amount: float
    currency: str = "USD"
    status: str = "pending"

    def add_item(self, item: Dict[str, Any]) -> None:
        """Add item to order"""
        self.items.append(item)
        self._recalculate_total()

    def remove_item(self, item_id: str) -> bool:
        """Remove item from order"""
        for i, item in enumerate(self.items):
            if item.get('id') == item_id:
                del self.items[i]
                self._recalculate_total()
                return True
        return False

    def _recalculate_total(self) -> None:
        """Recalculate total amount"""
        self.total_amount = sum(item.get('price', 0) * item.get('quantity', 1)
                              for item in self.items)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create Order from dictionary"""
        return cls(**data)

    def copy_with_updates(self, **updates) -> 'Order':
        """Create copy with updates"""
        return replace(self, **updates)

# 5. Dynamic Dataclass Creation
def create_dataclass_from_dict(name: str, data: Dict[str, Any]) -> type:
    """Dynamically create a dataclass from a dictionary"""
    fields = []
    for key, value in data.items():
        if isinstance(value, list):
            field_type = List[Any]
        elif isinstance(value, dict):
            field_type = Dict[str, Any]
        elif isinstance(value, str):
            field_type = str
        elif isinstance(value, int):
            field_type = int
        elif isinstance(value, float):
            field_type = float
        else:
            field_type = Any

        fields.append((key, field_type, field(default=value)))

    return make_dataclass(name, fields)

# 6. Dataclass Inheritance
@dataclass
class BaseEntity:
    """Base entity with common fields"""
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1

    def mark_updated(self) -> None:
        self.updated_at = datetime.now()
        self.version += 1

@dataclass
class Product(BaseEntity):
    """Product entity inheriting from BaseEntity"""
    name: str
    price: float
    category: str
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if not self.name.strip():
            raise ValueError("Product name cannot be empty")

@dataclass
class DigitalProduct(Product):
    """Digital product with additional fields"""
    file_size: int  # in bytes
    download_url: str
    license_key: Optional[str] = None

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

# Demo Enhanced Dataclasses
def demo_enhanced_dataclasses():
    """Demonstrate enhanced dataclass features"""
    print("📦 Enhanced Dataclasses Demo")
    print("=" * 40)

    # Advanced config
    try:
        config = Config("my_app", "1.0.0")
        config.set_setting("debug", True)
        config.set_setting("max_connections", 100)
        print(f"  ✅ Config: {config.name} v{config.version}")
        print(f"     Settings: {config.settings}")
    except ValueError as e:
        print(f"  ❌ Config error: {e}")

    # User with complex fields
    user = User(
        id=1,
        name="Alice Johnson",
        email="alice@example.com",
        roles=["admin", "user"],
        metadata={"department": "engineering", "level": "senior"}
    )
    print(f"  👤 User: {user.display_name}")
    print(f"     Roles: {user.roles}")
    print(f"     Metadata: {user.metadata}")

    # Order with business logic
    order = Order(
        order_id="ORD-001",
        items=[
            {"id": "item1", "name": "Laptop", "price": 999.99, "quantity": 1},
            {"id": "item2", "name": "Mouse", "price": 29.99, "quantity": 2}
        ],
        total_amount=0  # Will be calculated
    )
    print(f"  📦 Order: {order.order_id}")
    print(f"     Total: ${order.total_amount:.2f}")

    order.add_item({"id": "item3", "name": "Keyboard", "price": 79.99, "quantity": 1})
    print(f"     After adding item: ${order.total_amount:.2f}")

    # Product with inheritance
    product = Product(
        id="PROD-001",
        name="Python Course",
        price=199.99,
        category="education",
        tags=["programming", "python", "beginner"]
    )
    print(f"  🛍️ Product: {product.name} (${product.price})")
    print(f"     Category: {product.category}")
    print(f"     Tags: {product.tags}")

    # Digital product with property
    digital = DigitalProduct(
        id="DIG-001",
        name="Python Handbook",
        price=49.99,
        category="ebook",
        file_size=5 * 1024 * 1024,  # 5 MB
        download_url="https://example.com/download"
    )
    print(f"  📱 Digital: {digital.name}")
    print(f"     File size: {digital.file_size_mb:.1f} MB")
    print(f"     Download: {digital.download_url}")

    # Dynamic dataclass creation
    sample_data = {"name": "Course", "duration": 120, "level": "beginner", "price": 199.99}
    DynamicCourse = create_dataclass_from_dict("DynamicCourse", sample_data)
    course = DynamicCourse()
    print(f"  🔧 Dynamic class: {course}")

    # Dataclass conversion
    order_dict = order.to_dict()
    print(f"  🔄 Order as dict: {order_dict}")

    order_json = order.to_json()
    print(f"  📄 Order as JSON: {order_json[:50]}...")

demo_enhanced_dataclasses()
```

### **Pydantic for Data Validation and Serialization**

```python
# Pydantic is a popular data validation library (install with: pip install pydantic)
# Note: This is conceptual code - requires pydantic installation

"""
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.datetime_parse import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import re

# Note: Commented out to avoid dependency - run with: pip install pydantic

# 1. Basic Pydantic Model
class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    age: int = Field(..., ge=0, le=150)
    is_active: bool = True
    tags: List[str] = []

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "John Doe",
                "email": "john@example.com",
                "age": 30,
                "is_active": True,
                "tags": ["developer", "python"]
            }
        }

# 2. Model with Validators
class EmailUser(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

    @validator('email')
    def email_must_be_valid(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v

    @validator('password')
    def password_must_be_strong(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain uppercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain a digit')
        return v

    @root_validator
    def passwords_match(cls, values):
        password = values.get('password')
        confirm_password = values.get('confirm_password')
        if password != confirm_password:
            raise ValueError('Passwords do not match')
        return values

# 3. Complex Data Structure
class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"

class Company(BaseModel):
    name: str
    address: Address
    employees: List[User]
    founded: datetime

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Company name cannot be empty')
        return v.strip()

# 4. Enum for Status Values
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: int = Field(1, ge=1, le=5)

    @validator('due_date')
    def due_date_must_be_future(cls, v):
        if v and v < datetime.now():
            raise ValueError('Due date must be in the future')
        return v

# 5. Config and Customization
class ConfigurableModel(BaseModel):
    name: str
    value: int

    class Config:
        # Use enum values instead of enum objects
        use_enum_values = True
        # Validate assignment
        validate_assignment = True
        # Allow population by field name
        allow_population_by_field_name = True
        # JSON schema customization
        schema_extra = {
            "example": {
                "name": "test",
                "value": 42
            }
        }

# 6. Generic Model
class ResponseModel(BaseModel, Generic[T]):
    success: bool
    data: T
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "data": {"key": "value"},
                "message": "Operation successful"
            }
        }

# Demo Pydantic features
def demo_pydantic_features():
    # Note: This demo would require pydantic installation
    print("🔍 Pydantic Data Validation Demo")
    print("=" * 40)

    # Basic validation
    try:
        user = User(
            id=1,
            name="John Doe",
            email="john@example.com",
            age=30,
            tags=["python", "developer"]
        )
        print(f"  ✅ Valid user: {user.name} ({user.email})")
    except ValueError as e:
        print(f"  ❌ Validation error: {e}")

    # Email validation
    try:
        email_user = EmailUser(
            username="johndoe",
            email="invalid-email",
            password="weak",
            confirm_password="strong"
        )
    except ValueError as e:
        print(f"  ❌ Email validation error: {e}")

    # Complex structure
    address = Address(
        street="123 Main St",
        city="New York",
        state="NY",
        zip_code="10001"
    )

    user1 = User(id=1, name="Alice", email="alice@example.com", age=25)
    user2 = User(id=2, name="Bob", email="bob@example.com", age=30)

    company = Company(
        name="Tech Corp",
        address=address,
        employees=[user1, user2],
        founded=datetime.now()
    )

    print(f"  🏢 Company: {company.name}")
    print(f"     Employees: {len(company.employees)}")

    # Task management
    task = Task(
        id="task-001",
        title="Implement feature X",
        description="Add new functionality",
        priority=3
    )

    print(f"  📋 Task: {task.title} (Priority: {task.priority})")

    # JSON serialization
    user_dict = user.dict()
    user_json = user.json()

    print(f"  📄 User as dict: {user_dict}")
    print(f"  📄 User as JSON: {user_json}")

# This would be the actual demo call:
# demo_pydantic_features()
"""

print("📋 Pydantic Demo (Conceptual - Install pydantic to run)")
print("pip install pydantic  # Install pydantic for full features")
```

---

## 🔄 **Performance Optimization Techniques**

### **Memory Management and Profiling**

```python
import sys
import time
import tracemalloc
import cProfile
import pstats
from functools import wraps
from typing import Any, Dict, List, Callable
import memory_profiler
import psutil
import gc

# 1. Memory Profiling Decorator
def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()

        # Record starting memory
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Record ending memory
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Get memory snapshot
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"📊 Memory Profile for {func.__name__}:")
        print(f"  ⏱️  Execution time: {end_time - start_time:.4f} seconds")
        print(f"  💾 Memory used: {end_memory - start_memory:.2f} MB")
        print(f"  📈 Peak memory: {peak / 1024 / 1024:.2f} MB")
        print(f"  🔍 Current memory: {current / 1024 / 1024:.2f} MB")

        return result
    return wrapper

# 2. Function Performance Profiler
def profile_performance(func: Callable) -> Callable:
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Profile the function
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # Get statistics
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')

        print(f"⚡ Performance Profile for {func.__name__}:")
        stats.print_stats(5)  # Top 5 functions

        return result
    return wrapper

# 3. Memory-Efficient Data Structures
class MemoryEfficientList:
    """Memory-efficient list implementation using generators"""
    def __init__(self, generator_func: Callable):
        self.generator_func = generator_func
        self._cache = None
        self._cached = False

    def __iter__(self):
        if not self._cached:
            self._cache = list(self.generator_func())
            self._cached = True
        return iter(self._cache)

    def __len__(self):
        if not self._cached:
            self._cache = list(self.generator_func())
            self._cached = True
        return len(self._cache)

    def __getitem__(self, index):
        if not self._cached:
            self._cache = list(self.generator_func())
            self._cached = True
        return self._cache[index]

def fibonacci_generator(n: int):
    """Generate Fibonacci numbers"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 4. Lazy Loading and Caching
class LazyProperty:
    """Lazy property decorator that computes value only when needed"""
    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value

class ExpensiveComputation:
    """Class with expensive computations"""
    def __init__(self, data_size: int):
        self.data_size = data_size
        self.computed_data = None

    @LazyProperty
    def expensive_result(self):
        """This will only be computed once when first accessed"""
        print(f"  🧮 Computing expensive result for data size {self.data_size}")
        # Simulate expensive computation
        time.sleep(0.1)
        result = sum(i**2 for i in range(self.data_size))
        return result

    def clear_cache(self):
        """Clear cached lazy properties"""
        if hasattr(self, 'expensive_result'):
            delattr(self, 'expensive_result')

# 5. Object Pool Pattern
class ObjectPool:
    """Object pool to reuse expensive objects"""
    def __init__(self, create_func: Callable, reset_func: Callable, max_size: int = 10):
        self.create_func = create_func
        self.reset_func = reset_func
        self.max_size = max_size
        self._pool = []
        self._created = 0

    def acquire(self):
        """Get object from pool or create new one"""
        if self._pool:
            obj = self._pool.pop()
            self.reset_func(obj)
            return obj
        else:
            self._created += 1
            return self.create_func()

    def release(self, obj):
        """Return object to pool"""
        if len(self._pool) < self.max_size:
            self._pool.append(obj)
        # else: object is discarded

    def stats(self):
        """Get pool statistics"""
        return {
            'pool_size': len(self._pool),
            'total_created': self._created,
            'max_size': self.max_size
        }

# Example object for pooling
class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.is_connected = False

    def connect(self):
        self.is_connected = True
        print(f"  🔗 Connected to {self.connection_string}")

    def disconnect(self):
        self.is_connected = False
        print(f"  ❌ Disconnected from {self.connection_string}")

    def execute_query(self, query: str):
        if not self.is_connected:
            self.connect()
        return f"Executed: {query}"

# 6. Memory Leak Detection
def find_memory_leaks():
    """Detect potential memory leaks"""
    print("🔍 Memory Leak Detection")

    # Check garbage collection
    gc.collect()
    objects = gc.get_objects()
    print(f"  📊 Total objects in memory: {len(objects)}")

    # Check for duplicate objects
    object_counts = {}
    for obj in objects:
        obj_type = type(obj).__name__
        object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

    # Show types with most instances
    print("  🏆 Most common object types:")
    for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {obj_type}: {count}")

    return object_counts

# 7. Garbage Collection Tuning
def optimize_garbage_collection():
    """Optimize garbage collection settings"""
    print("🗑️ Garbage Collection Optimization")

    # Check current settings
    print(f"  Current GC thresholds: {gc.get_threshold()}")
    print(f"  GC counts: {gc.get_count()}")

    # Tune thresholds for better performance
    # Lower thresholds = more frequent GC, potentially better for memory
    # Higher thresholds = less frequent GC, potentially better for speed
    gc.set_threshold(700, 10, 10)
    print(f"  New GC thresholds: {gc.get_threshold()}")

    # Force garbage collection
    collected = gc.collect()
    print(f"  🧹 Garbage collected: {collected} objects")

# Demo performance optimization
@profile_memory
@profile_performance
def memory_intensive_function(size: int):
    """Function that uses significant memory"""
    # Create large data structures
    data = []
    for i in range(size):
        data.append([j for j in range(100)])  # Nested lists
    return sum(sum(row) for row in data)

@profile_memory
def lazy_loading_demo():
    """Demonstrate lazy loading"""
    print("⏳ Lazy Loading Demo")

    # Create expensive computation
    comp = ExpensiveComputation(10000)

    print("  First access (computes):")
    result1 = comp.expensive_result
    print(f"    Result: {result1}")

    print("  Second access (cached):")
    result2 = comp.expensive_result
    print(f"    Result: {result2}")

    # Clear cache and access again
    comp.clear_cache()
    print("  After clearing cache (recomputes):")
    result3 = comp.expensive_result

@profile_memory
def object_pool_demo():
    """Demonstrate object pooling"""
    print("🏊 Object Pool Demo")

    # Create object pool
    def create_connection():
        return DatabaseConnection("postgresql://localhost:5432/mydb")

    def reset_connection(conn):
        conn.disconnect()

    pool = ObjectPool(create_connection, reset_connection, max_size=5)

    # Use objects from pool
    connections = []
    for i in range(8):
        conn = pool.acquire()
        result = conn.execute_query("SELECT 1")
        connections.append(conn)

    # Release objects back to pool
    for conn in connections:
        pool.release(conn)

    print(f"  Pool statistics: {pool.stats()}")

def performance_optimization_demo():
    """Complete performance optimization demonstration"""
    print("🚀 Performance Optimization Demo")
    print("=" * 50)

    # Memory intensive function
    print("📊 Memory Intensive Function:")
    result = memory_intensive_function(1000)
    print(f"  Result: {result}")

    # Lazy loading
    print("\n" + "="*30)
    lazy_loading_demo()

    # Object pooling
    print("\n" + "="*30)
    object_pool_demo()

    # Memory leak detection
    print("\n" + "="*30)
    find_memory_leaks()

    # GC optimization
    print("\n" + "="*30)
    optimize_garbage_collection()

# Run performance demo
# performance_optimization_demo()
print("💡 Run performance_optimization_demo() to see profiling in action")
```

### **Advanced Caching Strategies**

```python
from functools import lru_cache, wraps, cache
from typing import Hashable, Any, Dict, Optional
import time
import hashlib

# 1. Advanced Caching Decorator
def advanced_cache(maxsize: int = 128, typed: bool = False, ttl: float = None):
    """Advanced caching decorator with TTL support"""
    def decorator(func):
        cache_dict = {}
        cache_timestamps = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            if typed:
                key = args, tuple(sorted(kwargs.items()))
            else:
                key = args, tuple(sorted((k, v) for k, v in kwargs.items() if isinstance(v, Hashable)))

            # Check cache
            if key in cache_dict:
                # Check TTL
                if ttl is None or time.time() - cache_timestamps[key] < ttl:
                    print(f"  💾 Cache hit for {func.__name__}")
                    return cache_dict[key]
                else:
                    print(f"  ⏰ Cache expired for {func.__name__}")

            # Compute and cache
            result = func(*args, **kwargs)
            cache_dict[key] = result
            cache_timestamps[key] = time.time()

            # Manage cache size
            if len(cache_dict) > maxsize:
                # Remove oldest entry
                oldest_key = min(cache_timestamps, key=cache_timestamps.get)
                del cache_dict[oldest_key]
                del cache_timestamps[oldest_key]

            return result

        wrapper.cache_info = lambda: {
            'hits': getattr(wrapper, '_hits', 0),
            'misses': getattr(wrapper, '_misses', 0)
        }

        return wrapper
    return decorator

# 2. Memoization with Disk Cache
import pickle
import os
from pathlib import Path

class DiskCache:
    """File-based cache for expensive computations"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        # Create safe filename
        safe_key = hashlib.md5(str(key).encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data, timestamp = pickle.load(f)

                # Check if cache is still valid (24 hours default)
                if time.time() - timestamp < 86400:  # 24 hours
                    return data
                else:
                    cache_file.unlink()  # Remove expired cache
            except (pickle.PickleError, EOFError):
                pass
        return None

    def set(self, key: str, value: Any) -> None:
        """Cache value"""
        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump((value, time.time()), f)
        except pickle.PickleError:
            pass  # Skip if can't pickle

    def clear(self) -> None:
        """Clear all cached files"""
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()

    def size(self) -> int:
        """Get cache size"""
        return len(list(self.cache_dir.glob("*.cache")))

disk_cache = DiskCache()

def disk_cached(func):
    """Decorator for disk-based caching"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key
        key = str((args, tuple(sorted(kwargs.items()))))

        # Check disk cache
        cached_result = disk_cache.get(key)
        if cached_result is not None:
            return cached_result

        # Compute result
        result = func(*args, **kwargs)

        # Cache result
        disk_cache.set(key, result)

        return result
    return wrapper

# 3. Function-specific caches
@advanced_cache(maxsize=256, ttl=60)  # Cache for 60 seconds
def expensive_computation(x: int, y: int) -> int:
    """Expensive computation that benefits from caching"""
    print(f"  🧮 Computing: {x} + {y} (expensive)")
    time.sleep(0.1)  # Simulate expensive work
    return x + y

@disk_cached
def fibonacci_disk_cached(n: int) -> int:
    """Fibonacci with disk caching"""
    if n <= 1:
        return n
    print(f"  📝 Computing fibonacci({n})")
    time.sleep(0.01)  # Simulate computation
    return fibonacci_disk_cached(n-1) + fibonacci_disk_cached(n-2)

@cache  # Simple built-in cache
def factorial_cached(n: int) -> int:
    """Factorial with simple caching"""
    if n <= 1:
        return 1
    print(f"  🔢 Computing factorial({n})")
    return n * factorial_cached(n-1)

# 4. Cache-aware algorithms
class FibonacciSequence:
    """Fibonacci sequence with intelligent caching"""

    def __init__(self):
        self.cache = {0: 0, 1: 1}
        self.computations = 0

    def fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number with caching"""
        self.computations += 1

        if n in self.cache:
            return self.cache[n]

        # Compute recursively with caching
        if n <= 1:
            return n

        self.cache[n] = self.fibonacci(n-1) + self.fibonacci(n-2)
        return self.cache[n]

    def get_cache_stats(self) -> Dict[str, int]:
        """Get caching statistics"""
        cache_size = len(self.cache)
        total_computations = self.computations
        return {
            'cache_size': cache_size,
            'total_computations': total_computations,
            'cache_hits': total_computations - cache_size,
            'efficiency': (total_computations - cache_size) / total_computations if total_computations > 0 else 0
        }

# 5. LRU Cache with Custom Replacement Policy
class CustomLRUCache:
    """Custom LRU cache with custom eviction policy"""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []  # Track access order
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            # Cache hit
            self.hit_count += 1
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        else:
            # Cache miss
            self.miss_count += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new
            if len(self.cache) >= self.maxsize:
                # Evict least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.maxsize
        }

# Demo advanced caching
def demo_advanced_caching():
    """Demonstrate advanced caching strategies"""
    print("💾 Advanced Caching Demo")
    print("=" * 40)

    # TTL cache demo
    print("⏰ TTL Cache Demo (60 second expiration):")
    print("  First call (cache miss):")
    result1 = expensive_computation(10, 20)
    print(f"  Result: {result1}")

    print("  Second call (cache hit):")
    result2 = expensive_computation(10, 20)
    print(f"  Result: {result2}")

    # Disk cache demo
    print("\n💽 Disk Cache Demo:")
    for i in [5, 10, 5, 10]:  # Some duplicates
        print(f"  fibonacci({i}) = {fibonacci_disk_cached(i)}")

    print(f"  📊 Disk cache size: {disk_cache.size()} files")

    # Built-in cache demo
    print("\n🔢 Built-in Cache Demo:")
    for i in [5, 10, 5, 10]:
        result = factorial_cached(i)
        print(f"  factorial({i}) = {result}")

    # Custom cache demo
    print("\n🎛️ Custom LRU Cache Demo:")
    lru_cache = CustomLRUCache(maxsize=3)

    # Add some items
    lru_cache.put("item1", "value1")
    lru_cache.put("item2", "value2")
    lru_cache.put("item3", "value3")

    print(f"  Initial cache: {list(lru_cache.cache.keys())}")

    # Access item1 (updates LRU order)
    lru_cache.get("item1")

    # Add new item (should evict item2 - least recently used)
    lru_cache.put("item4", "value4")

    print(f"  After adding item4: {list(lru_cache.cache.keys())}")
    print(f"  Cache stats: {lru_cache.stats()}")

    # Fibonacci with intelligent caching
    print("\n🧮 Fibonacci with Intelligent Caching:")
    fib_seq = FibonacciSequence()

    numbers = [10, 15, 10, 5, 20, 15]
    for n in numbers:
        result = fib_seq.fibonacci(n)
        print(f"  fibonacci({n}) = {result}")

    stats = fib_seq.get_cache_stats()
    print(f"  Cache efficiency: {stats['efficiency']:.2%}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Total computations: {stats['total_computations']}")

# Run caching demo
# demo_advanced_caching()
print("💡 Run demo_advanced_caching() to see caching strategies in action")
```

---

## 🎉 **Congratulations!**

You've mastered **Modern Python Features** for cutting-edge development!

### **What You've Accomplished:**

✅ **Pattern Matching** - Structural pattern matching for cleaner code  
✅ **Async/Await** - Modern concurrency for high-performance applications  
✅ **Advanced Typing** - Type safety with modern Python type system  
✅ **Modern Data Classes** - Enhanced dataclasses and Pydantic validation  
✅ **Performance Optimization** - Memory management, profiling, and caching

### **Your Modern Python Skills:**

🎯 **Code Quality** - Write cleaner, more maintainable code  
🎯 **Performance** - Optimize for speed and memory efficiency  
🎯 **Type Safety** - Catch errors at development time  
🎯 **Concurrency** - Build scalable applications  
🎯 **Best Practices** - Use cutting-edge Python features professionally

### **Next Steps:**

🚀 **Build Projects** - Apply modern features in real applications  
🚀 **Performance Testing** - Profile and optimize your code  
🚀 **Type Checking** - Integrate mypy in your development workflow  
🚀 **Async Frameworks** - Explore FastAPI, aiohttp, asyncio libraries

**🔗 Continue Your Journey:** Move to `python_automation_projects_complete_guide.md` for practical automation applications!

---

## _Modern Python isn't just about new features—it's about writing better, faster, and more maintainable code!_ 🐍⚡✨

## 🔍 COMMON CONFUSIONS & MISTAKES

### 1. Pattern Matching Overuse

**❌ Mistake:** Using pattern matching for simple conditional logic that should use if-elif-else
**✅ Solution:** Use pattern matching for complex data structure matching, not simple value comparisons

```python
# Overusing pattern matching - NOT RECOMMENDED
def check_value(x):
    match x:
        case 1:
            return "one"
        case 2:
            return "two"
        case _:
            return "other"

# Better approach for simple cases
def check_value_better(x):
    return {
        1: "one",
        2: "two"
    }.get(x, "other")
```

### 2. Async/Await Misconceptions

**❌ Mistake:** Using async/await for CPU-bound tasks or not properly handling async contexts
**✅ Solution:** Use async/await for I/O-bound operations and ensure proper event loop management

```python
import asyncio
import time

# WRONG - Using async for CPU-bound work
async def cpu_bound_task():
    # This won't benefit from async
    result = sum(range(1000000))
    return result

# CORRECT - Use async for I/O operations
async def io_bound_task():
    # This benefits from async
    await asyncio.sleep(1)  # Simulates I/O
    return "Task completed"

# CORRECT - Use regular functions for CPU work
def cpu_intensive_calculation():
    return sum(range(1000000))  # CPU-bound
```

### 3. Type Annotation Misuse

**❌ Mistake:** Over-annotating everything with complex types or ignoring type checking benefits
**✅ Solution:** Use type annotations strategically for complex logic and interfaces

```python
from typing import List, Dict, Optional, Union, Protocol
from dataclasses import dataclass

# Over-annotating simple functions
def simple_sum(a: int, b: int) -> int:  # Type hints unnecessary here
    return a + b

# Good use of type hints for complex data
@dataclass
class UserProfile:
    name: str
    email: str
    preferences: Dict[str, Union[str, int, bool]]

    def get_preference(self, key: str) -> Optional[Union[str, int, bool]]:
        return self.preferences.get(key)

# Type hint for function interfaces
class DataProcessor(Protocol):
    def process(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        ...
```

### 4. Dataclass Misunderstanding

**❌ Mistake:** Using dataclasses for mutable data that should be handled differently, or not leveraging dataclass features
**✅ Solution:** Understand when to use @dataclass, @field, and different dataclass options

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# Good use of dataclass
@dataclass
class Order:
    id: int
    customer_id: int
    items: List[str] = field(default_factory=list)
    total: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def add_item(self, item: str, price: float):
        self.items.append(item)
        self.total += price

# When NOT to use dataclass
class Configuration:
    def __init__(self, settings: dict):
        for key, value in settings.items():
            setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)
```

### 5. Walrus Operator Misuse

**❌ Mistake:** Using the walrus operator (:=) in contexts where it reduces readability
**✅ Solution:** Use walrus operator for reducing code duplication and improving readability

```python
# Good use of walrus operator
if (n := len(data)) > 10:
    print(f"Processing {n} items")

# Bad use of walrus operator (harder to read)
if (data_list := get_data()) is not None and (processed := process(data_list)) is not None:
    return processed

# Better without walrus
data_list = get_data()
if data_list is not None:
    processed = process(data_list)
    if processed is not None:
        return processed
```

### 6. Dictionary Merge and Update Confusion

**❌ Mistake:** Not understanding the difference between | merge and |= update operators
**✅ Solution:** Use the correct operator for your use case

```python
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}

# Merge creates new dict
merged = dict1 | dict2  # {'a': 1, 'b': 3, 'c': 4}
print(f"Original dict1: {dict1}")  # unchanged

# Update modifies in place
dict1 |= dict2  # dict1 becomes {'a': 1, 'b': 3, 'c': 4}
print(f"Modified dict1: {dict1}")  # changed
```

### 7. f-string Formatting Complexity

**❌ Mistake:** Overcomplicating f-strings or not using them effectively
**✅ Solution:** Use f-strings for readability but break complex expressions into variables

```python
# Overcomplicated f-string
message = f"User {user.get('name', 'Unknown')} has {len([order for order in orders if order.get('status') == 'active'])} active orders with total value ${sum(order.get('amount', 0) for order in orders if order.get('status') == 'active'):.2f}"

# Better approach
active_orders = [order for order in orders if order.get('status') == 'active']
total_value = sum(order.get('amount', 0) for order in active_orders)
user_name = user.get('name', 'Unknown')
order_count = len(active_orders)

message = f"User {user_name} has {order_count} active orders with total value ${total_value:.2f}"
```

### 8. Type Union vs UnionType Confusion

**❌ Mistake:** Using Union when Literal or TypeGuard would be more appropriate
**✅ Solution:** Choose the right type construct for your use case

```python
from typing import Union, Literal, TypeGuard
from typing_extensions import TypeGuard

# Using Union for specific values
Status = Union[Literal["pending"], Literal["processing"], Literal["completed"], Literal["failed"]]

def set_status(status: Status):
    pass

# Using TypeGuard for runtime type checking
def is_string_list(value: list) -> TypeGuard[list[str]]:
    return all(isinstance(item, str) for item in value)

# Example usage
data = ["hello", "world", 123]  # Mixed types
if is_string_list(data):
    # Type narrowing - Python knows data is list[str] here
    result = " ".join(data)  # Safe
```

---

## 📝 MICRO-QUIZ (80% MASTERY REQUIRED)

**Instructions:** Answer all questions. You need 5/6 correct (80%) to pass.

### Question 1: Pattern Matching Best Practices

When should you use pattern matching instead of traditional if-elif-else statements?
a) Always use pattern matching for better performance
b) Only for simple value comparisons
c) For complex data structure matching and destructuring
d) Never use pattern matching in professional code

**Correct Answer:** c) For complex data structure matching and destructuring

### Question 2: Async/Await Usage

What type of operations benefit most from async/await?
a) CPU-intensive calculations
b) File I/O operations and network requests
c) Mathematical computations
d) String manipulation

**Correct Answer:** b) File I/O operations and network requests

### Question 3: Type Annotations

What is the primary benefit of using type annotations in Python?
a) Faster code execution
b) Runtime error prevention and better IDE support
c) Reduced memory usage
d) Automatic code optimization

**Correct Answer:** b) Runtime error prevention and better IDE support

### Question 4: Dataclass Selection

When is @dataclass most beneficial?
a) For all Python classes
b) For classes with simple data and default values
c) For complex inheritance hierarchies
d) For performance-critical applications

**Correct Answer:** b) For classes with simple data and default values

### Question 5: Walrus Operator

What is the main use case for the walrus operator (:=)?
a) Always prefer it over regular assignment
b) Reducing code duplication in expressions and conditions
c) Making code more confusing
d) Only for lambda functions

**Correct Answer:** b) Reducing code duplication in expressions and conditions

### Question 6: Dictionary Operations

What is the key difference between dict | merge and dict |= update?
a) There is no difference
b) | creates a new dictionary, |= modifies in place
c) |= creates a new dictionary, | modifies in place
d) | only works with compatible types, |= works with all types

**Correct Answer:** b) | creates a new dictionary, |= modifies in place

---

## 🤔 REFLECTION PROMPTS

### 1. Concept Understanding

How would you explain the evolution of Python's type system to someone who learned Python before type hints were common? What examples would illustrate the benefits of modern typing?

**Reflection Focus:** Consider both the learning curve and practical benefits. Think about when type hints provide value and when they might be overkill.

### 2. Real-World Application

Consider a large codebase you work with regularly. How could modern Python features (pattern matching, async/await, dataclasses) improve the maintainability and performance of that code?

**Reflection Focus:** Apply modern Python concepts to existing code. Consider refactoring strategies and migration approaches.

### 3. Future Evolution

How do you think Python's feature set will continue to evolve? What modern features do you find most compelling, and what features would you like to see in future Python versions?

**Reflection Focus:** Consider language design trends, community needs, and technological changes. Think about both opportunities and potential drawbacks.

---

## ⚡ MINI SPRINT PROJECT (30-40 minutes)

### Project: Modern API Response Handler

Build a modern API response handler that demonstrates multiple Python 3.8+ features working together.

**Objective:** Create a production-ready API response handler using modern Python features.

**Time Investment:** 30-40 minutes
**Difficulty Level:** Intermediate to Advanced
**Skills Practiced:** Pattern matching, type annotations, dataclasses, async/await, error handling

### Step-by-Step Implementation

**Step 1: Modern Data Models (10 minutes)**

```python
# api_models.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Literal
from datetime import datetime
from enum import Enum

class ResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"

@dataclass
class ErrorDetails:
    code: str
    message: str
    field: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ApiResponse:
    status: ResponseStatus
    data: Optional[Dict[str, Any]] = None
    errors: List[ErrorDetails] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def is_success(self) -> bool:
        return self.status == ResponseStatus.SUCCESS

    def is_error(self) -> bool:
        return self.status == ResponseStatus.ERROR

    def add_error(self, code: str, message: str, field: Optional[str] = None):
        error = ErrorDetails(code=code, message=message, field=field)
        self.errors.append(error)
        self.status = ResponseStatus.ERROR

    def add_success_data(self, key: str, value: Any):
        if self.data is None:
            self.data = {}
        self.data[key] = value
```

**Step 2: Pattern Matching Response Processor (12 minutes)**

```python
# response_processor.py
from typing import Any, Protocol
from api_models import ApiResponse, ResponseStatus

class DataProcessor(Protocol):
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...

class ModernResponseProcessor:
    def __init__(self):
        self.processors: Dict[str, DataProcessor] = {}
        self.default_processor: DataProcessor = DefaultProcessor()

    def register_processor(self, data_type: str, processor: DataProcessor):
        """Register a processor for specific data types"""
        self.processors[data_type] = processor

    def process_response(self, response: ApiResponse) -> ApiResponse:
        """Process API response using pattern matching"""
        match response:
            case ApiResponse(status=ResponseStatus.SUCCESS, data=data) if data:
                return self._process_success_response(data, response)
            case ApiResponse(status=ResponseStatus.ERROR, errors=errors):
                return self._process_error_response(errors, response)
            case ApiResponse(status=ResponseStatus.PENDING):
                return self._process_pending_response(response)
            case _:
                return self._process_unknown_response(response)

    def _process_success_response(self, data: Dict[str, Any], response: ApiResponse) -> ApiResponse:
        """Process successful response with data"""
        # Pattern matching on data structure
        match data:
            case {"type": data_type, "content": content} if data_type in self.processors:
                # Use registered processor
                processor = self.processors[data_type]
                processed_data = processor.process(content)
                response.data = {"type": data_type, "content": processed_data}
            case {"results": results, "total": total}:
                # Handle paginated results
                processed_results = [
                    self._process_single_result(result) for result in results
                ]
                response.data = {
                    "results": processed_results,
                    "total": total,
                    "count": len(processed_results)
                }
            case {"user": user_data, "settings": settings}:
                # Handle user data
                response.data = {
                    "user": self._sanitize_user_data(user_data),
                    "settings": settings
                }
            case _:
                # Use default processing
                processed_data = self.default_processor.process(data)
                response.data = processed_data

        return response

    def _process_error_response(self, errors: List[Any], response: ApiResponse) -> ApiResponse:
        """Process error response"""
        # Pattern matching on error patterns
        for error in errors:
            match error:
                case ErrorDetails(code="validation", field=field) if field:
                    response.metadata["validation_errors"] = True
                case ErrorDetails(code="auth", message=message):
                    response.metadata["requires_auth"] = True
                case ErrorDetails(code="rate_limit"):
                    response.metadata["retry_after"] = 300  # 5 minutes

        return response

    def _process_pending_response(self, response: ApiResponse) -> ApiResponse:
        """Process pending response"""
        response.metadata["polling_url"] = f"/jobs/{response.metadata.get('job_id')}/status"
        response.metadata["estimated_completion"] = self._estimate_completion(response)
        return response

    def _process_unknown_response(self, response: ApiResponse) -> ApiResponse:
        """Process unknown response structure"""
        response.add_error("unknown_format", "Unable to process response format")
        return response

    def _process_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual result using pattern matching"""
        match result:
            case {"id": id_val, "name": name, "type": "user"}:
                return {"id": id_val, "name": name, "processed": True}
            case {"id": id_val, "data": data, "status": "active"}:
                return {"id": id_val, "data": data, "active": True}
            case _:
                # Add processing metadata
                result["processed"] = True
                return result

    def _sanitize_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize user data"""
        # Remove sensitive information
        sensitive_fields = ["password", "ssn", "credit_card"]
        sanitized = user_data.copy()

        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"

        return sanitized

    def _estimate_completion(self, response: ApiResponse) -> str:
        """Estimate completion time"""
        # Simple estimation based on job type
        job_type = response.metadata.get("job_type", "general")

        estimations = {
            "file_upload": "2-5 minutes",
            "data_processing": "5-15 minutes",
            "report_generation": "10-30 minutes",
            "general": "5-10 minutes"
        }

        return estimations.get(job_type, "Unknown")

class DefaultProcessor:
    """Default data processor for unknown formats"""
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Add processing metadata
        processed = data.copy()
        processed["processed_at"] = datetime.now().isoformat()
        processed["processor"] = "default"
        return processed
```

**Step 3: Async API Client (10 minutes)**

```python
# api_client.py
import asyncio
import aiohttp
from typing import Optional
from api_models import ApiResponse
from response_processor import ModernResponseProcessor

class ModernApiClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.processor = ModernResponseProcessor()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get(self, endpoint: str, **kwargs) -> ApiResponse:
        """Make GET request"""
        return await self._request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> ApiResponse:
        """Make POST request"""
        json_data = data if data else {}
        return await self._request("POST", endpoint, json=json_data, **kwargs)

    async def _request(self, method: str, endpoint: str, **kwargs) -> ApiResponse:
        """Make HTTP request and process response"""
        if not self.session:
            raise RuntimeError("Client must be used as async context manager")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            async with self.session.request(method, url, **kwargs) as response:
                response_data = await response.json()

                # Create response model
                if response.status == 200:
                    api_response = ApiResponse(status=ResponseStatus.SUCCESS, data=response_data)
                elif response.status == 202:
                    api_response = ApiResponse(status=ResponseStatus.PENDING, metadata=response_data)
                else:
                    api_response = ApiResponse(status=ResponseStatus.ERROR)
                    # Extract error information
                    if isinstance(response_data, dict) and "error" in response_data:
                        error_info = response_data["error"]
                        api_response.add_error(
                            error_info.get("code", "unknown"),
                            error_info.get("message", "Unknown error")
                        )

                # Process the response
                return self.processor.process_response(api_response)

        except aiohttp.ClientTimeout:
            error_response = ApiResponse(status=ResponseStatus.ERROR)
            error_response.add_error("timeout", f"Request timed out after {self.timeout}s")
            return error_response
        except aiohttp.ClientError as e:
            error_response = ApiResponse(status=ResponseStatus.ERROR)
            error_response.add_error("network_error", str(e))
            return error_response
        except Exception as e:
            error_response = ApiResponse(status=ResponseStatus.ERROR)
            error_response.add_error("processing_error", str(e))
            return error_response
```

**Step 4: Example Usage and Testing (8 minutes)**

```python
# main.py
import asyncio
from api_client import ModernApiClient
from api_models import ApiResponse, ResponseStatus

async def demonstrate_modern_api_client():
    """Demonstrate modern API client with pattern matching"""

    print("🚀 Modern API Client Demo")
    print("=" * 40)

    # Simulated API responses for demonstration
    sample_responses = [
        # Success response with user data
        ApiResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "user": {
                    "id": 123,
                    "name": "John Doe",
                    "email": "john@example.com",
                    "password": "secret123"  # Should be redacted
                },
                "settings": {"theme": "dark", "notifications": True}
            }
        ),
        # Success response with paginated results
        ApiResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "results": [
                    {"id": 1, "name": "Item 1", "type": "user"},
                    {"id": 2, "name": "Item 2", "data": {"value": 42}, "status": "active"}
                ],
                "total": 150
            }
        ),
        # Error response
        ApiResponse(
            status=ResponseStatus.ERROR,
            errors=[]
        ),
        # Pending response
        ApiResponse(
            status=ResponseStatus.PENDING,
            metadata={"job_id": "job123", "job_type": "data_processing"}
        )
    ]

    # Process each response
    processor = ModernResponseProcessor()

    for i, response in enumerate(sample_responses, 1):
        print(f"\n📋 Response {i}: {response.status.value}")
        print("-" * 30)

        processed = processor.process_response(response)

        # Show processing results
        if processed.is_success():
            print("✅ Status: Success")
            if processed.data:
                print(f"📊 Data: {processed.data}")
        elif processed.is_error():
            print("❌ Status: Error")
            if processed.errors:
                print(f"🚨 Errors: {len(processed.errors)} found")

        if processed.metadata:
            print(f"📝 Metadata: {processed.metadata}")

    # Demonstrate async API client (simulation)
    print(f"\n🌐 Async API Client Simulation")
    print("-" * 35)

    # Simulate API calls
    async with ModernApiClient("https://api.example.com") as client:
        # This would make real HTTP requests in production
        print("✅ API client initialized")
        print("🔄 Ready for async operations")

    print(f"\n✅ Modern Python features demonstration complete!")
    print("🎯 Features demonstrated:")
    print("  • Pattern matching for response processing")
    print("  • Dataclasses for type-safe data models")
    print("  • Async/await for non-blocking operations")
    print("  • Type annotations throughout")
    print("  • Modern error handling patterns")

# Test the system
if __name__ == "__main__":
    asyncio.run(demonstrate_modern_api_client())
```

### Success Criteria

- [ ] Successfully demonstrates pattern matching for complex data structures
- [ ] Uses modern type annotations effectively throughout
- [ ] Implements dataclasses with proper field handling
- [ ] Shows async/await patterns for I/O operations
- [ ] Provides comprehensive error handling with modern patterns
- [ ] Maintains clean, readable code using modern Python features

### Test Your Implementation

1. Run the main demo: `python main.py`
2. Test different response types and patterns
3. Examine the pattern matching logic in different scenarios
4. Try adding new response processors
5. Experiment with the async client patterns

### Quick Extensions (if time permits)

- Add more complex pattern matching scenarios
- Implement additional modern Python features (f-strings with =, etc.)
- Create a web interface to test the API client
- Add comprehensive type checking with mypy
- Implement more sophisticated error recovery patterns
- Add caching and performance optimizations

---

## 🏗️ FULL PROJECT EXTENSION (6-10 hours)

### Project: Modern Python Web Framework

Build a web framework that showcases modern Python features in a practical, production-ready application.

**Objective:** Create a web framework demonstrating advanced use of modern Python features including async/await, pattern matching, advanced typing, and modern data classes.

**Time Investment:** 6-10 hours
**Difficulty Level:** Advanced
**Skills Practiced:** Framework design, async programming, pattern matching, type system, modern Python architecture

### Phase 1: Modern Router System (2-3 hours)

**Features to Implement:**

- Pattern matching for route handling
- Type-safe request/response models
- Modern error handling and validation
- Async route processing

### Phase 2: Middleware and Interceptors (2-3 hours)

**Features to Implement:**

- Async middleware pipeline
- Request/response processing with pattern matching
- Modern logging and monitoring
- Security and authentication middleware

### Phase 3: Database Integration (1-2 hours)

**Features to Implement:**

- Async database operations
- Modern data models with validation
- Connection pooling and optimization
- Migration system

### Phase 4: Testing and Documentation (1-2 hours)

**Features to Implement:**

- Comprehensive type checking with mypy
- Modern testing patterns with pytest
- API documentation generation
- Performance monitoring

### Success Criteria

- [ ] Complete web framework with modern Python features
- [ ] Pattern matching for routing and request handling
- [ ] Full async/await support for high performance
- [ ] Comprehensive type safety throughout
- [ ] Modern error handling and validation
- [ ] Production-ready testing and documentation

### Advanced Extensions

- **GraphQL Integration:** Add GraphQL support with pattern matching
- **WebSocket Support:** Real-time communication with async patterns
- **Plugin System:** Extensible architecture with modern Python patterns
- **Performance Optimization:** Caching, connection pooling, monitoring
- **Deployment:** Containerization and cloud deployment

## This project serves as a comprehensive demonstration of modern Python development skills, suitable for senior developer positions, technical architecture roles, or open-source framework development.

## 🤝 Common Confusions & Misconceptions

### 1. Modern vs. Legacy Feature Confusion

**Misconception:** "New Python features are just syntactic sugar and don't provide real benefits."
**Reality:** Modern Python features improve code readability, performance, safety, and developer productivity in meaningful ways.
**Solution:** Learn the rationale behind modern features and practice applying them in appropriate contexts rather than avoiding them.

### 2. Backward Compatibility Assumption

**Misconception:** "I can use all modern Python features without considering compatibility with older versions."
**Reality:** Modern features may not be available in older Python versions, and professional development requires compatibility planning.
**Solution:** Understand feature availability across Python versions and plan for compatibility in production environments.

### 3. Type System Misunderstanding

**Misconception:** "Type hints are optional and don't actually enforce type safety."
**Reality:** While optional, type hints improve code maintainability, IDE support, and can catch errors early when used with proper tools.
**Solution:** Use type hints to improve code documentation, IDE support, and error detection, especially in larger projects.

### 4. Async/Await Complexity Avoidance

**Misconception:** "Async programming is too complex for most applications and should be avoided."
**Reality:** Async programming enables significant performance improvements for I/O-bound applications and is essential for modern web services.
**Solution:** Learn async basics for I/O-intensive applications and understand when it's appropriate to use versus traditional synchronous code.

### 5. Pattern Matching Overuse

**Misconception:** "Pattern matching should replace all if/else statements for better code."
**Reality:** Pattern matching excels for complex conditional logic and data structure matching, but simple conditions are better with traditional if/else.
**Solution:** Use pattern matching where it provides clear advantages over traditional conditionals, not as a universal replacement.

### 6. Modern Feature Adoption Rush

**Misconception:** "I should immediately use all new Python features as soon as they're available."
**Reality:** Professional development requires careful consideration of team knowledge, project requirements, and compatibility needs.
**Solution:** Adopt new features thoughtfully, considering team readiness, project timeline, and long-term maintenance implications.

### 7. Performance Assumption

**Misconception:** "Modern Python features are always faster than traditional approaches."
**Reality:** Some modern features improve readability and maintainability, while performance benefits vary by use case and implementation.
**Solution:** Focus on the benefits each feature provides (readability, safety, maintainability) and measure performance when it matters.

### 8. Learning Progression Neglect

**Misconception:** "I can skip learning modern features and still be considered a competent Python programmer."
**Reality:** Modern features represent current Python best practices and are increasingly expected in professional development.
**Solution:** Understand modern features to stay current with Python development standards and best practices.

---

## 🧠 Micro-Quiz: Test Your Modern Python Skills

### Question 1: Type Hints Usage

**What's the main benefit of using type hints in Python code?**
A) They make code run faster
B) They improve code documentation, IDE support, and error detection
C) They are required for all Python code
D) They prevent all runtime errors

**Correct Answer:** B - Type hints primarily improve documentation, IDE support, and can help catch errors early when used with type checkers.

### Question 2: Async Programming Application

**When is async/await most beneficial in Python applications?**
A) For CPU-intensive calculations
B) For I/O-bound operations like web requests and database queries
C) For simple mathematical operations
D) For string processing

**Correct Answer:** B - Async/await is most beneficial for I/O-bound operations where programs wait for external resources.

### Question 3: Pattern Matching Advantage

**What's a situation where pattern matching provides clear advantages over traditional conditionals?**
A) Simple boolean checks
B) Complex nested data structure analysis with multiple conditions
C) Single variable comparisons
D) Basic arithmetic operations

**Correct Answer:** B - Pattern matching excels at analyzing complex nested data structures with multiple conditions and destructuring.

### Question 4: Modern Feature Adoption

**What's the best approach when considering new Python features for a project?**
A) Use all new features immediately to show expertise
B) Ignore new features to maintain compatibility
C) Evaluate features based on project needs, team knowledge, and long-term benefits
D) Only use features that improve performance

**Correct Answer:** C - Thoughtful adoption considers project requirements, team readiness, and long-term maintenance implications.

### Question 5: Data Classes Usage

**When are Python dataclasses most beneficial?**
A) For simple data storage only
B) For classes that primarily store data with minimal custom behavior
C) For complex business logic implementation
D) For single-value variables

**Correct Answer:** B - Dataclasses are most beneficial for data-focused classes that would otherwise require lots of boilerplate code.

### Question 6: Version Compatibility

**What's important when using modern Python features in production code?**
A) Using only the latest Python version
B) Understanding feature availability across Python versions and planning compatibility
C) Avoiding all new features forever
D) Using features only in personal projects

**Correct Answer:** B - Production code requires understanding feature availability and planning for compatibility across different Python versions.

---

## 💭 Reflection Prompts

### 1. Modern Development Evolution

"Reflect on how modern Python features represent the evolution of programming practices toward greater expressiveness, safety, and developer productivity. How does this evolution parallel other areas of technological advancement? What does this reveal about the importance of staying current with evolving tools and practices?"

### 2. Code Readability and Maintainability

"Consider how modern Python features like type hints and pattern matching improve code readability and maintainability. How does this focus on developer experience influence the quality and longevity of software systems? What does this teach about the relationship between tool design and development effectiveness?"

### 3. Professional Development and Continuous Learning

"Think about how learning modern Python features reflects the broader need for continuous learning in technology. How does staying current with evolving best practices contribute to professional effectiveness? What does this reveal about the mindset required for long-term success in technology careers?"

---

## 🚀 Mini Sprint Project (1-3 hours)

### Modern Python Features Integration Demo

**Objective:** Create a demonstration project that showcases modern Python features working together in a practical, real-world application.

**Task Breakdown:**

1. **Feature Selection Planning (30 minutes):** Choose 3-4 modern Python features (type hints, dataclasses, pattern matching, async) and design a project that demonstrates their integration
2. **Core Implementation (75 minutes):** Build the project using selected modern features with proper integration and complementary functionality
3. **Comparison and Analysis (30 minutes):** Compare modern feature implementation with traditional approaches and document the benefits and trade-offs
4. **Testing and Validation (30 minutes):** Test the implementation using modern testing approaches and validate feature integration
5. **Documentation and Examples (15 minutes):** Create documentation showing modern feature usage patterns and best practices

**Success Criteria:**

- Working application demonstrating integration of multiple modern Python features
- Shows practical benefits of modern features in real-world scenarios
- Includes comparison with traditional approaches and documentation of trade-offs
- Demonstrates proper use of type hints, modern syntax, and contemporary development practices
- Provides foundation for understanding how modern features scale to larger applications

---

## 🏗️ Full Project Extension (10-25 hours)

### Comprehensive Modern Python Development Platform

**Objective:** Build a sophisticated development platform that demonstrates mastery of modern Python features, contemporary development practices, and enterprise-level software engineering through advanced system development.

**Extended Scope:**

#### Phase 1: Modern Python Architecture Design (2-3 hours)

- **Advanced Feature Integration Strategy:** Plan comprehensive integration of modern Python features (3.8+) for optimal developer experience and system performance
- **Contemporary Development Practices:** Design development workflows using modern Python tools, testing frameworks, and quality assurance practices
- **Performance and Scalability Planning:** Design systems leveraging modern Python features for improved performance, scalability, and maintainability
- **Enterprise Modern Python Standards:** Establish standards and guidelines for modern Python development in enterprise environments

#### Phase 2: Core Modern Features Implementation (3-4 hours)

- **Advanced Type System Integration:** Implement comprehensive type hints, custom types, and type checking throughout the system for improved reliability
- **Pattern Matching and Modern Syntax:** Build systems using pattern matching, walrus operators, f-strings, and other modern Python syntax features
- **Async/Await and Concurrency:** Implement modern concurrency patterns using async/await, context managers, and other asyncio features
- **Modern Data Structures:** Utilize dataclasses, enums, typing modules, and other modern data structure features

#### Phase 3: Advanced Development Tools Integration (3-4 hours)

- **Modern Testing and Quality Assurance:** Build comprehensive testing using pytest, coverage tools, and modern testing patterns
- **Code Quality and Linting:** Implement modern code quality tools including mypy, black, isort, and other contemporary development tools
- **Documentation and Type Generation:** Create comprehensive documentation using modern tools and automatic type hint documentation generation
- **Performance Monitoring and Profiling:** Implement modern performance monitoring using contemporary Python profiling and optimization tools

#### Phase 4: Contemporary Architecture Patterns (2-3 hours)

- **Modern Web Framework Integration:** Build web services using modern Python web frameworks with async support and contemporary patterns
- **Microservices and API Design:** Implement modern microservices architecture using contemporary Python patterns and service integration
- **Data Processing and Analytics:** Create modern data processing systems using contemporary Python libraries and patterns
- **Container and Cloud Integration:** Implement modern deployment strategies using contemporary containerization and cloud-native patterns

#### Phase 5: Professional Development Practices (2-3 hours)

- **Modern CI/CD Pipeline:** Build contemporary continuous integration and deployment using modern Python development tools and practices
- **Code Review and Collaboration:** Implement modern code review processes and collaborative development using contemporary version control patterns
- **Performance Optimization and Monitoring:** Create systems for performance monitoring, optimization, and capacity planning using modern Python tools
- **Security and Compliance Integration:** Implement modern security practices and compliance using contemporary Python security tools and patterns

#### Phase 6: Community and Professional Advancement (1-2 hours)

- **Open Source and Community Contribution:** Plan contributions to modern Python libraries, tools, and frameworks using contemporary development practices
- **Professional Mentoring and Training:** Create systems for mentoring and training others in modern Python development practices
- **Technology Leadership and Standards:** Plan for technology leadership roles in establishing and maintaining modern Python development standards
- **Long-term Evolution and Innovation:** Design strategies for ongoing evolution with Python ecosystem development and emerging technologies

**Extended Deliverables:**

- Complete modern Python development platform demonstrating mastery of contemporary Python features and development practices
- Professional-grade system utilizing modern Python capabilities, contemporary development tools, and enterprise-level quality assurance
- Advanced implementation of modern Python features including type systems, async programming, and contemporary syntax patterns
- Comprehensive testing, documentation, and quality assurance systems using modern Python development tools and practices
- Professional development workflow and standards for modern Python development in enterprise environments
- Professional mentoring and community contribution plan for advancing modern Python development practices

**Impact Goals:**

- Demonstrate mastery of modern Python features and contemporary development practices through sophisticated system development
- Build portfolio showcase of advanced Python capabilities including modern syntax, type systems, and contemporary development workflows
- Develop systematic approach to adopting and implementing modern Python features for improved development effectiveness
- Create reusable frameworks and methodologies for modern Python development and contemporary software engineering practices
- Establish foundation for leadership roles in modern Python development, technology standards, and professional mentoring
- Show integration of modern Python skills with contemporary development practices, enterprise requirements, and professional software engineering
- Contribute to Python community advancement through demonstrated mastery of modern development concepts and best practices

---

_Your mastery of modern Python features represents a crucial milestone in professional Python development. These contemporary capabilities position you at the forefront of Python evolution and enable you to write more expressive, maintainable, and efficient code. The systematic adoption and application of modern features not only improves your immediate development effectiveness but also prepares you for the future of Python and software development. Each modern feature you master becomes a tool for creating better software, more efficient workflows, and more innovative solutions._
