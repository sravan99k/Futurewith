# Performance Optimization Cheat Sheet

## Profiling Tools

### cProfile - Function-Level Profiling
```python
import cProfile
import pstats
from io import StringIO

# Basic profiling
def slow_function():
    total = 0
    for i in range(10000):
        total += i
    return total

profiler = cProfile.Profile()
profiler.enable()
result = slow_function()
profiler.disable()

# Print results
stream = StringIO()
stats = pstats.Stats(profiler, stream=stream)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
print(stream.getvalue())

# Save to file
profiler.dump_stats("profile.prof")

# Analyze with pstats
stats = pstats.Stats("profile.prof")
stats.sort_stats('tottime')  # Sort by total time
stats.print_stats(10)
```

### line_profiler - Line-by-Line Profiling
```bash
pip install line_profiler

# Add @profile decorator to function
@profile
def my_function():
    for i in range(10000):
        pass

# Run profiler
kernprof -l -v script.py
```

### memory_profiler - Memory Profiling
```bash
pip install memory_profiler

@profile
def memory_intensive():
    data = [i for i in range(1000000)]
    return sum(data)

# Run memory profile
python -m memory_profiler script.py
```

### timeit - Timing Small Code Blocks
```python
import timeit

# Measure execution time
time = timeit.timeit(
    '[x**2 for x in range(1000)]',
    number=1000
)
print(f"Average time: {time/1000:.6f} seconds")

# Compare two implementations
time1 = timeit.timeit('sum(x*x for x in range(1000))', number=1000)
time2 = timeit.timeit('sum([x*x for x in range(1000)])', number=1000)
print(f"Generator: {time1:.6f}s, List comp: {time2:.6f}s")
```

## Optimization Techniques

### 1. Algorithm Optimization

```python
# O(n²) - Slow for large n
def find_duplicates_slow(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                duplicates.append(items[i])
    return duplicates

# O(n) - Much faster
def find_duplicates_fast(items):
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)
```

### 2. List vs Generator

```python
# List comprehension - creates full list in memory
squares = [x*x for x in range(1000000)]

# Generator - yields one at a time, memory efficient
squares_gen = (x*x for x in range(1000000))
```

### 3. Using Built-ins

```python
# Slow - Python loop
total = 0
for x in data:
    total += x

# Fast - C implementation
total = sum(data)

# Slow
result = []
for x in data:
    result.append(x * 2)

# Fast
result = list(map(lambda x: x * 2, data))

# Even faster (when possible)
result = [x * 2 for x in data]
```

### 4. Caching with lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Without cache: O(2^n)
# With cache: O(n)
```

### 5. Local Variable Access

```python
# Slower - global lookup each iteration
def slow_loop(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result

# Faster - local variable
def fast_loop(data):
    result = []
    append = result.append  # Local reference
    for item in data:
        append(item * 2)
    return result
```

### 6. String Concatenation

```python
# Slow - creates new string each time
def slow_concat():
    result = ""
    for i in range(1000):
        result += str(i)
    return result

# Fast - list + join
def fast_concat():
    parts = []
    for i in range(1000):
        parts.append(str(i))
    return "".join(parts)
```

### 7. Using itertools

```python
import itertools

# Infinite iterators
for i in itertools.count(10, 2):  # 10, 12, 14, ...
    if i > 100:
        break

# Combinations
for combo in itertools.combinations([1, 2, 3, 4], 2):
    print(combo)  # (1, 2), (1, 3), ...

# Permutations
for perm in itertools.permutations([1, 2, 3], 2):
    print(perm)  # (1, 2), (1, 3), (2, 1), ...

# Chaining
for item in itertools.chain([1, 2], [3, 4], [5, 6]):
    print(item)  # 1, 2, 3, 4, 5, 6
```

## NumPy for Numerical Operations

```python
import numpy as np

# Create large arrays
arr = np.arange(1000000)

# Vectorized operations (much faster than loops)
arr_squared = arr ** 2
arr_sum = np.sum(arr)
arr_mean = np.mean(arr)

# 2D operations
matrix = np.random.rand(1000, 1000)
row_sums = matrix.sum(axis=1)
col_sums = matrix.sum(axis=0)
```

## Multiprocessing for CPU-Bound Tasks

```python
from multiprocessing import Pool

def process_item(item):
    return item * 2

if __name__ == '__main__':
    data = list(range(1000000))
    
    with Pool(processes=4) as pool:
        results = pool.map(process_item, data)
    
    # Results collected when all done
```

## Asyncio for I/O-Bound Tasks

```python
import asyncio

async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

async def fetch_all(urls: list[str]) -> list[dict]:
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)
```

## Memory Optimization

### Using __slots__
```python
# Regular class - uses dict for attributes
class Regular:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# With slots - no dict, less memory
class Optimized:
    __slots__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

### Using array module
```python
from array import array

# List of integers - overhead per item
numbers_list = [1, 2, 3, 4, 5]

# Array - more memory efficient
numbers_array = array('I', [1, 2, 3, 4, 5])  # unsigned int
```

### Generators for Lazy Evaluation
```python
# Creates full list in memory
def get_numbers():
    return [i for i in range(1000000)]

# Generator - memory efficient
def get_numbers_gen():
    for i in range(1000000):
        yield i
```

## Quick Reference

| Operation | Slow Approach | Fast Approach |
|-----------|---------------|---------------|
| Loop sum | Python loop | `sum()` built-in |
| Map/filter | Python loop | `map()`, `filter()` |
| String concat | `+=` in loop | `join()` |
| Membership test | List | Set |
| Unique items | List check | Set |
| File reading | One by one | Buffering |
| HTTP requests | Sequential | Async/aiohttp |
| CPU intensive | Sequential | Multiprocessing |
| Array ops | Python list | NumPy |
| Sorting | Custom sort | `sorted()` / `list.sort()` |

## Profiling Checklist

1. **Profile first** - Don't optimize without data
2. **Identify bottleneck** - Find the slowest part
3. **Measure** - Quantify improvement
4. **Iterate** - Try different approaches
5. **Benchmark** - Use timeit for comparisons

## Common Performance Anti-patterns

```python
# ❌ Wrong - Multiple list iterations
result = [x for x in data if x > 0]
count = len(result)
avg = sum(result) / count

# ✅ Right - Single iteration
total = 0
count = 0
result = []
for x in data:
    if x > 0:
        result.append(x)
        total += x
        count += 1
avg = total / count if count > 0 else 0

# ❌ Wrong - Recreating objects
for i in range(1000):
    obj = ExpensiveClass()

# ✅ Right - Reuse or lazy init
obj = None
for i in range(1000):
    if obj is None:
        obj = ExpensiveClass()
```

## Benchmarking Template

```python
import timeit
import statistics

def benchmark(func, args, runs=100):
    times = []
    for _ in range(runs):
        start = timeit.default_timer()
        func(*args)
        times.append(timeit.default_timer() - start)
    
    return {
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times)
    }

# Usage
result = benchmark(my_function, (arg1, arg2))
print(f"Mean: {result['mean']:.6f}s")
print(f"Median: {result['median']:.6f}s")
```
