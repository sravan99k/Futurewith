# Modern Python Features Interview Preparation

## Interview Questions

### Question 1: Type Hints and Static Analysis

**Q: How do type hints help in Python development? Can you give an example of when they catch a bug?**

**A:** Type hints provide several benefits:
- **IDE Support:** Better autocompletion and code suggestions
- **Static Analysis:** Tools like mypy can catch type-related errors
- **Documentation:** Self-documenting code for function signatures
- **Refactoring Safety:** Easier to change code with confidence

**Example catching a bug:**
```python
from typing import List

def calculate_total(prices: List[int]) -> int:
    return sum(prices)

# Without type hints, this might work but be wrong:
# calculate_total(["10", "20", "30"])  # Would fail at runtime

# With type hints, mypy would catch:
# calculate_total(["10", "20", "30"])  # Error: List[int] expected, got List[str]
```

---

### Question 2: Pattern Matching

**Q: How is Python's pattern matching different from switch/case in other languages?**

**A:** Python's pattern matching (3.10+) is more powerful:

1. **Destructuring:** Extract values from data structures
2. **Guard Conditions:** Add if conditions to patterns
3. **Multiple Values:** Match multiple values with |
4. **Class Patterns:** Match against object attributes

**Example:**
```python
def process(point):
    match point:
        case (0, 0):
            return "Origin"
        case (x, 0):
            return f"X-axis at {x}"
        case (0, y):
            return f"Y-axis at {y}"
        case (x, y) if x == y:
            return f"Diagonal at ({x}, {y})"
        case _:
            return "Other"

# More powerful than simple switch!
```

---

### Question 3: Walrus Operator

**Q: When should you use the walrus operator? When should you avoid it?**

**A:** Use walrus operator when:
- **Reusing expensive computations** in conditions
- **While loops** where you need to read a value and test it
- **List comprehensions** with complex conditions

**Avoid when:**
- **Readability suffers** - if it makes code harder to understand
- **Simple cases** where regular assignment is clearer
- **Debugging** - walrus can make stack traces harder to follow

**Good use:**
```python
# Without walrus - compute twice
if data := expensive_api_call():
    process(data)

# With walrus - compute once
```

**Avoid:**
```python
# Confusing
if (x := 10) > 5:  # Just use x = 10!
    print(x)
```

---

### Question 4: Data Classes

**Q: What are the advantages of using dataclasses over regular classes?**

**A:** Data classes provide:
- **Less boilerplate** - auto-generates `__init__`, `__repr__`, `__eq__`
- **Default values** - easy field defaults
- **Field metadata** - with `field()` for additional options
- **Immutability** - with `frozen=True`
- **Auto methods** - `__repr__`, `__eq__`, ordering with `order=True`

**Example:**
```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: int = 18  # Default value
    
    @property
    def is_adult(self) -> bool:
        return self.age >= 18

# Auto-generates: __init__, __repr__, __eq__
user = User("Alice", "alice@example.com", 25)
print(user)  # User(name='Alice', email='alice@example.com', age=25)
```

---

### Question 5: Context Managers

**Q: Explain the context manager protocol. When would you use `@contextmanager` decorator?**

**A:** Context managers implement `__enter__` and `__exit__`:

```python
class Manager:
    def __enter__(self):
        # Setup code
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code
        pass

with Manager() as m:
    # Use m
    pass
```

**Use `@contextmanager` when:**
- Your manager is primarily a function with setup/cleanup
- You want a simpler alternative to a full class
- You're converting a generator-based approach

**Example:**
```python
from contextlib import contextmanager

@contextmanager
def file_lock(filename: str):
    lockfile = f"{filename}.lock"
    open(lockfile, 'w').close()  # Create lock
    try:
        yield  # Critical section
    finally:
        os.remove(lockfile)  # Cleanup
```

---

### Question 6: Decorators

**Q: What is the difference between `@functools.wraps` and `functools.lru_cache`?**

**A:** They serve different purposes:

**`@wraps`:** Preserves metadata when creating decorators
```python
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

**`@lru_cache`:** Caches function results
```python
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

---

### Question 7: Generics

**Q: How do you create a generic class in Python? Why would you use TypeVar?**

**A:** `TypeVar` allows type-safe generics:

```python
from typing import Generic, TypeVar

T = TypeVar('T')
U = TypeVar('U')

class Pair(Generic[T, U]):
    def __init__(self, first: T, second: U):
        self.first = first
        self.second = second
    
    def swap(self) -> tuple[U, T]:
        return (self.second, self.first)

# Usage with type hints
pair1: Pair[int, str] = Pair(1, "hello")
pair2: Pair[str, int] = Pair("world", 2)
```

---

### Question 8: Enum

**Q: How does Enum differ from regular class attributes or constants?**

**A:** Enum provides:
- **Named constants** with type safety
- **Iteration** over all members
- **Value access** via `.value`
- **Identity checks** with `is`/`is not`

```python
from enum import Enum

class Status(Enum):
    PENDING = 1
    APPROVED = 2
    REJECTED = 3

# Comparison
Status.PENDING == Status.PENDING  # True (same object)
Status.PENDING == 1  # False (different types)
Status.PENDING.value == 1  # True

# Iteration
for status in Status:
    print(status.name, status.value)
```

---

### Question 9: Advanced Pattern Matching

**Q: Can you pattern match against custom classes? How does it work?**

**A:** Yes, with class patterns:

```python
class Point:
    __match_args__ = ('x', 'y')  # Specify pattern order
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

def describe_point(p: Point) -> str:
    match p:
        case Point(x, y) if x == y:
            return f"Diagonal point at ({x}, {y})"
        case Point(x=0, y=0):
            return "Origin"
        case Point(x=0, y):
            return f"On Y-axis at y={y}"
        case Point(x, y):
            return f"Point at ({x}, {y})"
```

---

### Question 10: Performance Considerations

**Q: Do type hints affect runtime performance?**

**A:** **No, type hints are ignored at runtime.** They exist solely for:
- Static type checkers (mypy, pyright)
- IDE support
- Documentation

**However,** using modern features like pattern matching has some overhead. The performance benefit comes from writing more efficient code with better patterns, not from the features themselves.

---

## Coding Interview Tasks

### Task 1: Implement a Typed Function

```python
# Implement this function with proper type hints
def filter_and_transform(items, threshold, func):
    """Filter items above threshold, then apply func"""
    pass

# Should work with type checker
# filter_and_transform([1, 2, 3, 4, 5], 2, lambda x: x*2)  # [6, 8, 10]
```

**Evaluation Criteria:**
- Proper type hints (List[int], Callable, etc.)
- Handle edge cases (empty list, None)
- Type safety

---

### Task 2: Pattern Matching Calculator

```python
# Implement a calculator using pattern matching
def calculate(expr):
    """
    Supported operations:
    - ("add", a, b)
    - ("sub", a, b)  
    - ("mul", a, b)
    - ("div", a, b)
    """
    pass

# calculate(("mul", ("add", 1, 2), 3))  # Should return 9
```

**Evaluation Criteria:**
- Correct pattern matching
- Nested expression handling
- Error handling (division by zero)

---

### Task 3: Decorator Factory

```python
# Create a decorator that takes parameters
def repeat(times: int):
    """Decorator that repeats function execution times times"""
    pass

@repeat(3)
def greet(name: str) -> str:
    return f"Hello, {name}!"

# greet("Alice") should return "Hello, Alice! Hello, Alice! Hello, Alice!"
```

**Evaluation Criteria:**
- Proper decorator syntax
- Preserving function metadata
- Type hints

---

## Best Practices Summary

| Feature | Best Practice |
|---------|---------------|
| Type Hints | Use for public APIs and complex functions |
| Pattern Matching | Use for complex conditionals, avoid for simple if/else |
| Walrus Operator | Use for expensive computations in conditions |
| Data Classes | Use for simple data containers |
| Context Managers | Use for resource management |
| Decorators | Use for cross-cutting concerns |
| Generics | Use for type-safe collections |
| Enum | Use for related constants |

---

## Common Mistakes to Avoid

1. **Over-using type hints** in private functions
2. **Complex nested patterns** that hurt readability
3. **Walrus in list comprehensions** when simple comprehension is clearer
4. **Mutable defaults** in function signatures (use None instead)
5. **Forgetting `@wraps`** when creating decorators
6. **Ignoring pattern matching failures** (use wildcard `_` case)
