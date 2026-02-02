# Modern Python Features Cheat Sheet

## Type Hints

### Basic Type Annotations
```python
# Variable annotations
name: str = "Alice"
age: int = 25
height: float = 5.9
is_active: bool = True

# Collection types
from typing import List, Dict, Set, Tuple

numbers: List[int] = [1, 2, 3]
scores: Dict[str, int] = {"Alice": 90, "Bob": 85}
unique_items: Set[int] = {1, 2, 3}
coordinates: Tuple[int, int] = (10, 20)

# Optional types
from typing import Optional

maybe_value: Optional[int] = None  # Can be int or None

# Union types
from typing import Union

result: Union[int, str] = "error"  # Can be int or str

# Literal types
from typing import Literal

Direction: Literal["north", "south", "east", "west"] = "north"

# TypedDict
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: Optional[str]

user: User = {"name": "Alice", "age": 30}
```

### Callable Types
```python
from typing import Callable

# Function that takes int and returns str
processor: Callable[[int], str] = lambda x: str(x)

# Function with multiple arguments
def add(a: int, b: int) -> int:
    return a + b

callback: Callable[[int, int], int] = add
```

### Generics
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value

int_container = Container(42)  # type is Container[int]
str_container = Container("hello")  # type is Container[str]
```

## Structural Pattern Matching (match/case)

### Basic Pattern Matching
```python
def process_response(status: int) -> str:
    match status:
        case 200:
            return "Success"
        case 404:
            return "Not Found"
        case 500:
            return "Server Error"
        case _:
            return "Unknown Status"

# Guard conditions
def classify_number(n: int) -> str:
    match n:
        case n if n < 0:
            return "Negative"
        case 0:
            return "Zero"
        case n if 0 < n < 10:
            return "Single digit"
        case _:
            return "Multiple digits"

# Multiple values
def handle_color(color: str) -> str:
    match color.lower():
        case "red" | "r":
            return "Color is red"
        case "blue" | "b":
            return "Color is blue"
        case _:
            return "Unknown color"
```

### Destructuring Patterns
```python
# Tuple destructuring
def point_operation(point: tuple[int, int]) -> str:
    match point:
        case (0, 0):
            return "Origin"
        case (x, 0):
            return f"X-axis at {x}"
        case (0, y):
            return f"Y-axis at {y}"
        case (x, y):
            return f"Point at ({x}, {y})"

# List destructuring
def sum_list(nums: list[int]) -> int:
    match nums:
        case []:
            return 0
        case [first, *rest]:
            return first + sum_list(rest)

# Class destructuring
class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

def describe_point(p: Point) -> str:
    match p:
        case Point(x=0, y=0):
            return "Origin point"
        case Point(x=x, y=0):
            return f"Point on X-axis: {x}"
        case Point(x=0, y=y):
            return f"Point on Y-axis: {y}"
        case Point(x=x, y=y):
            return f"Point at ({x}, {y})"
```

### Pattern Matching with Sequences
```python
# Nested patterns
def parse_command(cmd: list[str]) -> str:
    match cmd:
        case ["ls", *files]:
            return f"Listing {len(files)} files"
        case ["cd", path]:
            return f"Changed to {path}"
        case ["rm", "-r", *paths]:
            return f"Removing {len(paths)} paths recursively"
        case _:
            return "Unknown command"

# Dictionary patterns
def handle_request(req: dict) -> str:
    match req:
        case {"action": "login", "user": user, "pass": pass}:
            return f"Login attempt for {user}"
        case {"action": "logout"}:
            return "User logged out"
        case {"action": action, **rest}:
            return f"Action '{action}' with extra data"
```

## Assignment Expression (Walrus Operator)

### Basic Usage
```python
# Before walrus
n = 10
if n > 5:
    print(f"{n} is greater than 5")

# With walrus
if (n := 10) > 5:
    print(f"{n} is greater than 5")

# Reuse expensive computations
data = [1, 2, 3, 4, 5]
if (avg := sum(data) / len(data)) > 3:
    print(f"Average is {avg}, which is above threshold")

# While loops
cache = {}
def get_value(key: str) -> str:
    if key not in cache:
        # Expensive computation
        cache[key] = f"Computed value for {key}"
    return cache[key]

# With walrus
while (line := input("Enter value: ")) != "quit":
    print(f"You entered: {line}")
```

### List Comprehensions
```python
# Without walrus
values = [1, 2, 3, 4, 5]
squares = [x * x for x in values if x > 2]

# With walrus - computing intermediate values
data = [("name", "Alice"), ("age", 30), ("name", "Bob")]
# Extract names of people over 25
result = [(name, age) for _, (name, age) in [(entry := ("Alice", 25)), (entry2 := ("Bob", 35))] if age > 30]
```

## f-strings Enhancements

### Self-Documenting Expressions
```python
# Python 3.8+
name = "Alice"
age = 30
print(f"{name=} {age=}")  # Output: name='Alice' age=30

# Debugging
def calculate(a: int, b: int) -> int:
    result = a + b
    print(f"{a=}+{b=} {result=}")
    return result

calculate(5, 3)
# Output: a=5+b=3 result=8
```

### Debugging
```python
# Quick variable inspection
values = [1, 2, 3]
print(f"{values=}")  # values=[1, 2, 3]

# Expression debugging
x = 10
y = 20
print(f"{x * y = }")  # x * y = 200
```

## Decorators

### Common Decorators
```python
from functools import lru_cache, wraps, partial

# Caching decorator
def memoize(func):
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

# Timing decorator
import time
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Property decorator
class Temperature:
    def __init__(self, celsius: float):
        self._celsius = celsius
    
    @property
    def celsius(self) -> float:
        return self._celsius
    
    @celsius.setter
    def celsius(self, value: float):
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self) -> float:
        return self._celsius * 9/5 + 32
```

## Context Managers

### Basic Context Manager
```python
from contextlib import contextmanager

class DatabaseConnection:
    def __enter__(self):
        print("Connecting to database...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection...")
        if exc_type:
            print(f"Error occurred: {exc_val}")
        return False  # Don't suppress exceptions

# Using contextmanager decorator
@contextmanager
def file_manager(filename: str, mode: str = 'r'):
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()

# Usage
with file_manager("test.txt", "w") as f:
    f.write("Hello, World!")
```

## Data Classes

### Basic Data Class
```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: int
    y: int
    label: str = "origin"

# With default factory
@dataclass
class Student:
    name: str
    grades: list[int] = field(default_factory=list)
    
    @property
    def average(self) -> float:
        return sum(self.grades) / len(self.grades) if self.grades else 0

# Frozen (immutable) data class
@dataclass(frozen=True)
class ImmutablePoint:
    x: int
    y: int
```

## Enum

### Enum Usage
```python
from enum import Enum, auto

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# Auto-numbering
class Status(Enum):
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()

# Using enums
def get_color_name(color: Color) -> str:
    match color:
        case Color.RED:
            return "Red"
        case Color.GREEN:
            return "Green"
        case Color.BLUE:
            return "Blue"
```

## Summary Table

| Feature | Syntax | Use Case |
|---------|--------|----------|
| Type Hints | `var: Type` | Static analysis, IDE support |
| Pattern Matching | `match x: case ...` | Complex conditional logic |
| Walrus Operator | `(x := value)` | Reuse expensive computations |
| f-string Debug | `f"{x=}"` | Quick debugging output |
| Decorators | `@decorator` | Code reuse, cross-cutting concerns |
| Context Managers | `with ...:` | Resource management |
| Data Classes | `@dataclass` | Boilerplate reduction |
| Enums | `class Enum:` | Named constants |

## Best Practices

1. **Use type hints** for better IDE support and maintainability
2. **Prefer pattern matching** over complex if/elif chains
3. **Use walrus operator** sparingly for readability
4. **Leverage data classes** for simple data structures
5. **Use context managers** for resource cleanup
6. **Type annotate** public APIs for better documentation
