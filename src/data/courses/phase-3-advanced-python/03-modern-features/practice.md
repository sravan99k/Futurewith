# Modern Python Features Practice

## Exercise 1: Type Hints Refactoring

Refactor the following code to add proper type hints:

```python
# Original code without type hints
def process_data(data):
    results = []
    for item in data:
        processed = item * 2
        if processed > 10:
            results.append(processed)
    return results

# Your task: Add type hints and improve the function
def process_data_improved(data: list[int]) -> list[int]:
    # Your implementation here
    pass
```

**Solution:**
```python
from typing import List

def process_data_improved(data: list[int]) -> list[int]:
    return [item * 2 for item in data if item * 2 > 10]
```

## Exercise 2: Pattern Matching Challenge

Use pattern matching to create a command parser:

```python
def parse_command(parts):
    """
    Parse commands in format:
    - "add <number1> <number2>"
    - "multiply <number> <factor>"
    - "repeat <text> <count>"
    - "exit"
    
    Returns a dict with parsed command info
    """
    pass

# Test cases
assert parse_command(["add", "5", "3"]) == {"cmd": "add", "a": 5, "b": 3}
assert parse_command(["multiply", "10", "2"]) == {"cmd": "multiply", "num": 10, "factor": 2}
assert parse_command(["repeat", "hello", "3"]) == {"cmd": "repeat", "text": "hello", "count": 3}
assert parse_command(["exit"]) == {"cmd": "exit"}
```

**Solution:**
```python
def parse_command(parts: list[str]) -> dict:
    match parts:
        case ["add", a, b]:
            return {"cmd": "add", "a": int(a), "b": int(b)}
        case ["multiply", num, factor]:
            return {"cmd": "multiply", "num": int(num), "factor": int(factor)}
        case ["repeat", text, count]:
            return {"cmd": "repeat", "text": text, "count": int(count)}
        case ["exit"]:
            return {"cmd": "exit"}
        case _:
            return {"cmd": "unknown"}
```

## Exercise 3: Walrus Operator

Rewrite the following code using the walrus operator:

```python
# Without walrus operator
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = []
for n in numbers:
    result = is_prime(n)
    if result:
        primes.append(n)

# Refactor using walrus operator
# Your code here
```

**Solution:**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

primes = [n for n in numbers if (is_prime(n) or (result := False))]
# Or more clearly:
primes = [n for n in numbers if (result := is_prime(n))]
```

## Exercise 4: Data Class Implementation

Create a data class for a `Rectangle` with the following requirements:

```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float
    color: str = "white"
    
    @property
    def area(self) -> float:
        pass
    
    @property
    def perimeter(self) -> float:
        pass
    
    def scale(self, factor: float) -> None:
        pass

# Test
rect = Rectangle(5, 3, "blue")
assert rect.area == 15
assert rect.perimeter == 16
rect.scale(2)
assert rect.width == 10
assert rect.height == 6
```

## Exercise 5: Context Manager

Create a context manager that logs execution time:

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    """Context manager that measures execution time"""
    # Your implementation here
    yield
    # Print elapsed time

# Usage test
with timer("List creation"):
    large_list = list(range(1000000))

# Expected output:
# [timer] List creation: 0.0456 seconds
```

## Exercise 6: Complex Pattern Matching

Use pattern matching to parse JSON-like structures:

```python
def parse_json_like(data) -> str:
    """
    Parse JSON-like structures:
    - {"type": "number", "value": n} -> f"Number: {n}"
    - {"type": "string", "value": s} -> f"String: {s}"
    - {"type": "array", "items": [...]} -> f"Array with {len(items)} items"
    - {"type": "object", "pairs": [...]} -> f"Object with {len(pairs)} pairs"
    - {"type": "null"} -> "Null value"
    """
    pass

# Test cases
assert parse_json_like({"type": "number", "value": 42}) == "Number: 42"
assert parse_json_like({"type": "string", "value": "hello"}) == "String: hello"
assert parse_json_like({"type": "array", "items": [1, 2, 3]}) == "Array with 3 items"
assert parse_json_like({"type": "null"}) == "Null value"
```

## Exercise 7: Generic Class

Implement a generic `Stack` class:

```python
from typing import Generic, TypeVar, List

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        pass
    
    def pop(self) -> T:
        pass
    
    def peek(self) -> T:
        pass
    
    def is_empty(self) -> bool:
        pass
    
    def size(self) -> int:
        pass

# Test
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
assert stack.pop() == 3
assert stack.peek() == 2
assert stack.size() == 2
```

## Exercise 8: Decorator Implementation

Create a decorator that caches function results with a max size:

```python
from functools import lru_cache

def bounded_cache(max_size: int = 128):
    """Decorator that limits LRU cache size"""
    def decorator(func):
        cached = lru_cache(maxsize=max_size)(func)
        return cached
    return decorator

# Test
@bounded_cache(max_size=3)
def expensive_computation(n: int) -> int:
    print(f"Computing {n}")
    return n * 2

expensive_computation(1)  # Should print "Computing 1"
expensive_computation(1)  # Should use cache, no print
expensive_computation(2)  # Should print "Computing 2"
expensive_computation(3)  # Should print "Computing 3"
expensive_computation(4)  # Should print "Computing 4" and evict 1
expensive_computation(1)  # Should print "Computing 1" again (was evicted)
```

## Exercise 9: Advanced Type Hints

Create a function with complex type hints:

```python
from typing import TypeVar, Callable, Union, List

T = TypeVar('T')
U = TypeVar('U')

def transform_items(
    items: List[T],
    transform: Callable[[T], U],
    filter_func: Callable[[U], bool]
) -> List[U]:
    """
    Apply transform to each item, then filter using filter_func
    
    Args:
        items: List of items to transform
        transform: Function to apply to each item
        filter_func: Function to filter transformed items
    
    Returns:
        Filtered list of transformed items
    """
    pass

# Test
result = transform_items(
    [1, 2, 3, 4, 5],
    lambda x: x * 2,  # Transform: double each number
    lambda x: x > 5   # Filter: keep only > 5
)
assert result == [6, 8, 10]  # 2*3=6, 2*4=8, 2*5=10 (1*2=2, 2*2=4 filtered out)
```

## Exercise 10: Enum and Pattern Matching

Create a state machine using Enum and pattern matching:

```python
from enum import Enum, auto

class State(Enum):
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()

class PaymentStatus(Enum):
    UNPAID = auto()
    PAID = auto()
    REFUNDED = auto()

def describe_status(order_state: State, payment_state: PaymentStatus) -> str:
    """
    Describe the status of an order based on order and payment states
    """
    match (order_state, payment_state):
        case (State.PENDING, PaymentStatus.UNPAID):
            return "Order placed, awaiting payment"
        case (State.PENDING, PaymentStatus.PAID):
            return "Payment received, preparing order"
        case (State.PROCESSING, PaymentStatus.PAID):
            return "Order is being processed"
        case (State.COMPLETED, PaymentStatus.PAID):
            return "Order completed successfully"
        case (State.FAILED, _):
            return "Order failed"
        case (State.COMPLETED, PaymentStatus.REFUNDED):
            return "Order refunded"
        case _:
            return "Unknown status"

# Test
assert describe_status(State.PENDING, PaymentStatus.UNPAID) == "Order placed, awaiting payment"
assert describe_status(State.COMPLETED, PaymentStatus.PAID) == "Order completed successfully"
```

## Challenge Exercise: Build a Simple Interpreter

Use pattern matching to create a simple expression evaluator:

```python
from typing import Union

Number = Union[int, float]

def evaluate(expr) -> Number:
    """
    Evaluate expressions:
    - Numbers return themselves
    - ("add", a, b) evaluates both and returns a + b
    - ("sub", a, b) evaluates both and returns a - b
    - ("mul", a, b) evaluates both and returns a * b
    - ("div", a, b) evaluates both and returns a / b
    - ("neg", a) evaluates and returns -a
    """
    pass

# Test
assert evaluate(5) == 5
assert evaluate(("add", 2, 3)) == 5
assert evaluate(("mul", ("add", 1, 2), 3)) == 9  # (1+2)*3 = 9
assert evaluate(("neg", ("add", 5, 5))) == -10
```

---

## Solutions

### Exercise 1 Solution
```python
def process_data_improved(data: list[int]) -> list[int]:
    return [item * 2 for item in data if item * 2 > 10]
```

### Exercise 4 Solution
```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float
    color: str = "white"
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)
    
    def scale(self, factor: float) -> None:
        self.width *= factor
        self.height *= factor
```

### Exercise 5 Solution
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[timer] {name}: {elapsed:.4f} seconds")
```

### Exercise 6 Solution
```python
def parse_json_like(data: dict) -> str:
    match data:
        case {"type": "number", "value": n}:
            return f"Number: {n}"
        case {"type": "string", "value": s}:
            return f"String: {s}"
        case {"type": "array", "items": items}:
            return f"Array with {len(items)} items"
        case {"type": "object", "pairs": pairs}:
            return f"Object with {len(pairs)} pairs"
        case {"type": "null"}:
            return "Null value"
        case _:
            return "Unknown type"
```

### Exercise 7 Solution
```python
from typing import Generic, TypeVar, List

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items.pop()
    
    def peek(self) -> T:
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def size(self) -> int:
        return len(self._items)
```

### Exercise 10 Solution
Already provided in the exercise.

### Challenge Solution
```python
from typing import Union

Number = Union[int, float]

def evaluate(expr) -> Number:
    match expr:
        case int() | float():
            return expr
        case ("add", a, b):
            return evaluate(a) + evaluate(b)
        case ("sub", a, b):
            return evaluate(a) - evaluate(b)
        case ("mul", a, b):
            return evaluate(a) * evaluate(b)
        case ("div", a, b):
            return evaluate(a) / evaluate(b)
        case ("neg", a):
            return -evaluate(a)
        case _:
            raise ValueError(f"Unknown expression: {expr}")
```
