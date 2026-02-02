# Performance Optimization Practice

This section provides hands-on exercises to reinforce your understanding of Python performance optimization techniques. Complete each exercise to build practical skills in profiling, optimizing, and benchmarking Python code.

---

## Exercise 1: Profiling Basics

**Objective:** Learn to use profiling tools to identify performance bottlenecks.

**Task:** Given the following code, use `cProfile` to identify which functions consume the most time:

```python
import time
import random
from typing import List

def slow_calculation(n: int) -> int:
    """A computationally intensive function."""
    result = 0
    for i in range(n):
        result += i * i
    return result

def process_data(data: List[int]) -> int:
    """Process a list of numbers."""
    total = 0
    for item in data:
        total += slow_calculation(item)
    return total

def main():
    """Main function with multiple operations."""
    data = [random.randint(100, 1000) for _ in range(100)]
    
    # First pass
    result1 = process_data(data)
    time.sleep(0.1)  # Simulate I/O
    
    # Second pass with larger data
    large_data = [random.randint(500, 2000) for _ in range(200)]
    result2 = process_data(large_data)
    time.sleep(0.1)
    
    # Third pass
    result3 = process_data(data)
    
    return result1 + result2 + result3

if __name__ == "__main__":
    main()
```

**Requirements:**

1. Run the script with `cProfile` and save the output to a file
2. Analyze the output to identify the top 5 functions by cumulative time
3. Suggest optimizations for the slowest functions

**Solution Template:**

```python
import cProfile
import pstats
import io

def profile_function(func):
    """Decorator to profile a function."""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Print results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)
        print(s.getvalue())
        
        return result
    return wrapper

# Apply the decorator and run
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    
    # Save to file
    with open('profile_results.txt', 'w') as f:
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        f.write(s.getvalue())
```

**Deliverable:** Write a brief report identifying the bottlenecks and proposing optimizations.

---

## Exercise 2: Algorithm Optimization

**Objective:** Improve algorithmic efficiency from O(n²) to O(n).

**Task:** Optimize the following duplicate detection algorithm:

```python
def find_duplicates_slow(numbers: List[int]) -> List[int]:
    """Find duplicates using nested loops - O(n²) complexity."""
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    return duplicates

# Test data
test_numbers = [1, 3, 5, 7, 9, 3, 2, 4, 6, 8, 1, 10, 12, 11, 5]
```

**Requirements:**

1. Implement an optimized version using a set for O(n) complexity
2. Benchmark both implementations with `timeit`
3. Test with datasets of varying sizes (100, 1000, 10000 elements)
4. Create a visualization comparing the performance

**Starter Code:**

```python
import timeit
import random
from typing import List

def find_duplicates_optimized(numbers: List[int]) -> List[int]:
    """Your optimized implementation here."""
    # TODO: Implement using a set for O(n) complexity
    pass

def benchmark_implementation(func, data_generator, sizes):
    """Benchmark a function across different data sizes."""
    results = []
    for size in sizes:
        data = data_generator(size)
        time_taken = timeit.timeit(
            lambda: func(data),
            number=10
        )
        results.append((size, time_taken))
    return results

# Generate test data
def generate_test_data(size: int) -> List[int]:
    """Generate test data with some duplicates."""
    return [random.randint(1, size // 2) for _ in range(size)]

# Run benchmarks
sizes = [100, 500, 1000, 2000]
print("Benchmarking slow implementation...")
slow_results = benchmark_implementation(find_duplicates_slow, generate_test_data, sizes)

print("Benchmarking optimized implementation...")
fast_results = benchmark_implementation(find_duplicates_optimized, generate_test_data, sizes)
```

**Deliverable:** Submit both implementations with benchmark results and analysis.

---

## Exercise 3: Memory Profiling

**Objective:** Identify and fix memory leaks and excessive memory usage.

**Task:** The following code has memory issues. Identify and fix them:

```python
class DataProcessor:
    def __init__(self):
        self.data_store = []
        self.cache = {}
    
    def process_large_dataset(self, iterations: int = 100):
        """Process data in chunks."""
        for i in range(iterations):
            # Simulate processing
            chunk = [j for j in range(10000)]
            self.data_store.extend(chunk)
            self.cache[i] = chunk  # Always growing cache
        return sum(self.data_store)
    
    def get_statistics(self):
        """Calculate statistics."""
        return {
            'total_items': len(self.data_store),
            'cache_size': len(self.cache),
            'memory_estimate': len(self.data_store) * 28 + len(self.cache) * 100
        }

# Usage pattern that causes memory issues
processor = DataProcessor()
result = processor.process_large_dataset(50)
print(processor.get_statistics())
```

**Requirements:**

1. Use `memory_profiler` to profile memory usage
2. Identify the memory leak sources
3. Implement fixes using:
   - Weak references for cache
   - Generators instead of lists
   - Proper cleanup methods
4. Verify the fixes reduce memory consumption

**Starter Code:**

```python
from memory_profiler import memory_usage
import gc

def profile_memory_usage(func, *args, **kwargs):
    """Profile memory usage of a function."""
    gc.collect()
    mem_before = memory_usage()[0]
    
    result = func(*args, **kwargs)
    
    mem_after = memory_usage()[0]
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after: {mem_after:.2f} MB")
    print(f"Peak usage: {max(memory_usage(func=lambda: func(*args, **kwargs))):.2f} MB")
    
    return result

class OptimizedDataProcessor:
    """Fixed version of DataProcessor."""
    def __init__(self):
        self.data_store = []
        self.cache = {}
    
    def process_large_dataset(self, iterations: int = 100):
        """Process data with proper memory management."""
        # TODO: Implement generator-based processing
        pass
    
    def clear_cache(self):
        """Clear old cache entries."""
        # TODO: Implement cache size limit
        pass
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()
```

**Deliverable:** Provide the optimized class with detailed comments explaining each memory optimization.

---

## Exercise 4: Caching Implementation

**Objective:** Implement an effective caching strategy using `functools.lru_cache` and understand when to use it.

**Task:** Implement a caching solution for a recursive Fibonacci function and analyze the performance impact:

```python
import time
from functools import lru_cache

def fibonacci(n: int) -> int:
    """Basic recursive Fibonacci - very slow for large n."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test without caching
start = time.time()
result = fibonacci(35)
print(f"Result: {result}")
print(f"Time without cache: {time.time() - start:.4f}s")

# TODO: Implement with LRU cache
@lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    pass

start = time.time()
result = fibonacci_cached(35)
print(f"Result with cache: {result}")
print(f"Time with cache: {time.time() - start:.4f}s")
```

**Requirements:**

1. Complete the cached version of Fibonacci
2. Measure and compare execution times for n = 35, 40, 45
3. Visualize the performance improvement
4. Discuss the memory vs speed tradeoff

**Advanced Challenge:** Implement a custom cache with a TTL (Time-To-Live) feature:

```python
import time
from typing import Any, Callable, Optional

class TTLCache:
    """Cache with time-to-live functionality."""
    
    def __init__(self, ttl_seconds: float = 60):
        self.ttl = ttl_seconds
        self.cache = {}
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache if not expired."""
        pass
    
    def set(self, key: Any, value: Any):
        """Set value in cache with timestamp."""
        pass
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply TTL caching to a function."""
        pass
```

**Deliverable:** Submit the TTL cache implementation with test cases.

---

## Exercise 5: Concurrency Optimization

**Objective:** Optimize I/O-bound and CPU-bound tasks using threading and multiprocessing.

**Task:** Compare performance of sequential, threaded, and multiprocessed approaches for the following tasks:

**Part A: I/O-Bound Task (API Calls)**

```python
import time
import requests
from typing import List

def fetch_url(url: str) -> dict:
    """Fetch data from a URL."""
    response = requests.get(url, timeout=10)
    return {
        'url': url,
        'status': response.status_code,
        'content_length': len(response.content)
    }

urls = [
    'https://httpbin.org/get',
    'https://httpbin.org/headers',
    'https://httpbin.org/ip',
    'https://httpbin.org/user-agent',
    'https://httpbin.org/cookies',
] * 10  # 50 total requests

def fetch_sequential(urls: List[str]) -> List[dict]:
    """Fetch URLs sequentially."""
    results = []
    for url in urls:
        results.append(fetch_url(url))
    return results

# TODO: Implement threaded version
def fetch_threading(urls: List[str]) -> List[dict]:
    pass

# TODO: Implement async version
import asyncio

async def fetch_async(url: str, session) -> dict:
    pass
```

**Part B: CPU-Bound Task (Data Processing)**

```python
import math
from typing import List

def process_number(n: int) -> float:
    """CPU-intensive computation."""
    result = 0
    for i in range(n):
        result += math.sqrt(i) * math.sin(i) * math.cos(i)
    return result

numbers = [100000, 200000, 300000, 400000, 500000]

def process_sequential(numbers: List[int]) -> List[float]:
    """Process numbers sequentially."""
    return [process_number(n) for n in numbers]

# TODO: Implement multiprocessing version
from multiprocessing import Pool

def process_multiprocessing(numbers: List[int]) -> List[float]:
    pass
```

**Requirements:**

1. Implement both threading and multiprocessing versions
2. Benchmark all approaches for I/O-bound and CPU-bound tasks
3. Create comparison tables and visualizations
4. Explain when to use each approach

**Deliverable:** Complete code with benchmark results and analysis.

---

## Exercise 6: NumPy Optimization

**Objective:** Leverage NumPy for efficient numerical operations.

**Task:** Optimize the following pure Python code using NumPy:

```python
import random
import time
from typing import List

def calculate_stats_python(data: List[List[float]]) -> dict:
    """Calculate statistics using pure Python."""
    n_rows = len(data)
    n_cols = len(data[0]) if data else 0
    
    # Calculate column means
    col_means = []
    for col in range(n_cols):
        col_sum = 0
        for row in range(n_rows):
            col_sum += data[row][col]
        col_means.append(col_sum / n_rows)
    
    # Calculate row sums
    row_sums = []
    for row in range(n_rows):
        row_sum = 0
        for col in range(n_cols):
            row_sum += data[row][col]
        row_sums.append(row_sum)
    
    # Calculate standard deviation
    col_stds = []
    for col in range(n_cols):
        col_sum = 0
        col_sq_sum = 0
        for row in range(n_rows):
            val = data[row][col]
            col_sum += val
            col_sq_sum += val * val
        mean = col_sum / n_rows
        variance = (col_sq_sum / n_rows) - (mean * mean)
        col_stds.append(math.sqrt(variance))
    
    return {
        'col_means': col_means,
        'row_sums': row_sums,
        'col_stds': col_stds
    }

# Generate test data
data = [[random.random() for _ in range(100)] for _ in range(1000)]
```

**Requirements:**

1. Implement the same functionality using NumPy
2. Benchmark both implementations with various data sizes
3. Create a performance comparison table
4. Explain NumPy's performance advantages

**Deliverable:** Optimized NumPy implementation with benchmark results.

---

## Exercise 7: JIT Compilation with Numba

**Objective:** Use Numba for JIT compilation to speed up Python code.

**Task:** Optimize computationally intensive functions using Numba:

```python
import numpy as np
from numba import jit, njit, prange
import time

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive matrix multiplication."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C

def monte_carlo_pi(n_points: int) -> float:
    """Estimate Pi using Monte Carlo method."""
    n_inside = 0
    for _ in range(n_points):
        x = np.random.random()
        y = np.random.random()
        if x * x + y * y <= 1:
            n_inside += 1
    return 4 * n_inside / n_points

def mandelbrot(width: int, height: int, max_iter: int = 100) -> np.ndarray:
    """Generate Mandelbrot set."""
    img = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            c = complex((x - width/2) / (width/4), (y - height/2) / (height/4))
            z = 0
            for i in range(max_iter):
                if abs(z) > 2:
                    img[y, x] = i
                    break
                z = z * z + c
    return img
```

**Requirements:**

1. Apply `@jit` and `@njit` decorators to each function
2. Compare performance before and after JIT compilation
3. Experiment with different Numba options (`nopython=True`, `parallel=True`, `cache=True`)
4. Create a performance comparison report

**Bonus Challenge:** Implement a parallel version of matrix multiplication using `prange`:

```python
@njit(parallel=True)
def matrix_multiply_parallel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Parallel matrix multiplication with Numba."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = np.zeros((n, m))
    
    for i in prange(n):
        for j in range(m):
            for k in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C
```

**Deliverable:** Complete code with detailed performance analysis.

---

## Exercise 8: String Concatenation Optimization

**Objective:** Understand string concatenation performance implications.

**Task:** Compare different string concatenation methods:

```python
import time
from typing import List

def concatenate_plus(strings: List[str]) -> str:
    """Concatenate using + operator."""
    result = ""
    for s in strings:
        result += s
    return result

def concatenate_join(strings: List[str]) -> str:
    """Concatenate using join method."""
    return "".join(strings)

def concatenate_list(strings: List[str]) -> str:
    """Concatenate using list append."""
    parts = []
    for s in strings:
        parts.append(s)
    return "".join(parts)

# Test with different string lengths and counts
test_cases = [
    (100, 10),      # 100 chars, 10 strings
    (1000, 100),    # 1000 chars, 100 strings
    (10000, 500),   # 10000 chars, 500 strings
]

def generate_test_data(num_strings: int, chars_per_string: int) -> List[str]:
    return ["x" * chars_per_string for _ in range(num_strings)]

# Benchmark
for num_strings, chars_per_string in test_cases:
    data = generate_test_data(num_strings, chars_per_string)
    
    print(f"\nTest: {num_strings} strings × {chars_per_string} chars")
    
    start = time.time()
    result = concatenate_plus(data)
    print(f"  + operator: {time.time() - start:.4f}s")
    
    start = time.time()
    result = concatenate_join(data)
    print(f"  join():     {time.time() - start:.4f}s")
    
    start = time.time()
    result = concatenate_list(data)
    print(f"  list join:  {time.time() - start:.4f}s")
```

**Requirements:**

1. Explain why `+` operator is slow for repeated concatenation
2. Test with more data sizes and types
3. Compare memory usage of each method
4. Discuss when each method is appropriate

**Advanced Challenge:** Implement a `StringBuilder` class similar to Java's:

```python
class StringBuilder:
    """Efficient string concatenation class."""
    
    def __init__(self):
        self._parts = []
        self._length = 0
    
    def append(self, s: str) -> 'StringBuilder':
        """Append a string."""
        pass
    
    def append_line(self, s: str) -> 'StringBuilder':
        """Append a string with newline."""
        pass
    
    def to_string(self) -> str:
        """Convert to final string."""
        pass
    
    def __len__(self) -> int:
        """Return total length."""
        pass
```

**Deliverable:** Complete analysis with recommendations for each use case.

---

## Exercise 9: Generator and Iterator Optimization

**Objective:** Use generators to reduce memory usage for large datasets.

**Task:** Compare memory usage of list comprehensions vs generators:

```python
import sys
from typing import Generator, List

def process_with_list(data: List[int]) -> int:
    """Process using list comprehension - loads everything into memory."""
    squared = [x * x for x in data]  # Creates full list in memory
    return sum(squared)

def process_with_generator(data: List[int]) -> int:
    """Processing using generator - memory efficient."""
    squared = (x * x for x in data)  # Creates generator
    return sum(squared)

# Test memory usage
data = list(range(1000000))

print(f"Data size: {sys.getsizeof(data)} bytes")
print(f"List result: {process_with_list(data)}")
print(f"Generator result: {process_with_generator(data)}")

# Compare memory profiles
from memory_profiler import profile

@profile
def test_list_comprehension():
    result = process_with_list(data)
    return result

@profile
def test_generator():
    result = process_with_generator(data)
    return result
```

**Requirements:**

1. Profile memory usage of both approaches
2. Create a large dataset test (10M+ elements)
3. Implement a pipeline using generators for processing large files
4. Discuss when generators should be used

**Practical Challenge:** Process a large CSV file efficiently:

```python
import csv
from typing import Dict, Any

def process_large_csv_file(filepath: str) -> Generator[Dict[str, Any], None, None]:
    """
    Process a large CSV file row by row using a generator.
    
    This approach should:
    - Read the file line by line
    - Transform data efficiently
    - Yield results instead of storing all in memory
    """
    pass

# Example usage
for row in process_large_csv_file('large_file.csv'):
    # Process each row individually
    print(row)
```

**Deliverable:** Generator implementation with memory profiling results.

---

## Exercise 10: Comprehensive Optimization Project

**Objective:** Apply multiple optimization techniques to a real-world scenario.

**Task:** Optimize a data analysis pipeline:

```python
import pandas as pd
import numpy as np
from typing import List, Dict
import time

class DataAnalysisPipeline:
    """Original unoptimized data analysis pipeline."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV."""
        return pd.read_csv(self.data_path)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values (slow approach)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('unknown')
            else:
                df[col] = df[col].fillna(0)
        
        return df
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate various statistics."""
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].tolist()  # Convert to Python list
            stats[col] = {
                'mean': sum(col_data) / len(col_data),  # Manual calculation
                'max': max(col_data),
                'min': min(col_data),
                'std': np.std(col_data)  # But using NumPy here
            }
        return stats
    
    def filter_and_aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and aggregate data."""
        filtered = []
        for idx, row in df.iterrows():  # Slow row-by-row iteration
            if row['value'] > 100:
                filtered.append(row)
        
        result = pd.DataFrame(filtered)
        if not result.empty:
            result = result.groupby('category').agg({
                'value': ['sum', 'mean', 'count']
            })
        
        return result
    
    def run(self) -> Dict:
        """Execute the full pipeline."""
        print("Loading data...")
        start = time.time()
        df = self.load_data()
        print(f"Load time: {time.time() - start:.2f}s")
        
        print("Cleaning data...")
        df = self.clean_data(df)
        
        print("Calculating statistics...")
        stats = self.calculate_statistics(df)
        
        print("Filtering and aggregating...")
        result = self.filter_and_aggregate(df)
        
        return {
            'statistics': stats,
            'result': result,
            'row_count': len(df)
        }
```

**Requirements:**

1. Identify at least 5 optimization opportunities
2. Implement optimizations using:
   - Vectorized operations (Pandas/NumPy)
   - Efficient data types (category, int32, float32)
   - Chunked processing for large files
   - Caching for repeated operations
   - Parallel processing where appropriate
3. Create before/after benchmarks
4. Document all optimizations

**Optimization Checklist:**

- [ ] Replace `iterrows()` with vectorized operations
- [ ] Use `fillna()` with dictionaries for cleaner code
- [ ] Convert object columns to category type
- [ ] Use appropriate numeric types (float32 vs float64)
- [ ] Implement chunked reading for large files
- [ ] Use `query()` for filtering
- [ ] Consider using `swifter` for parallel apply
- [ ] Optimize groupby operations

**Deliverable:** Complete optimized pipeline with benchmark comparisons.

---

## Bonus Exercises

### Bonus Exercise A: asyncio Implementation

Implement an async web scraper that:
- Fetches multiple URLs concurrently
- Implements retry logic with exponential backoff
- Uses semaphores to limit concurrent requests
- Tracks progress with tqdm

### Bonus Exercise B: Cython Basics

Convert a computationally intensive Python function to Cython:
- Create a `.pyx` file
- Define static types
- Compare performance with pure Python and Numba

### Bonus Exercise C: Memory View and Buffer Protocol

Work with Python's buffer protocol:
- Implement a custom array type using `__buffer__`
- Compare with NumPy arrays
- Understand zero-copy operations

---

## Submission Guidelines

For each exercise:
1. Provide complete, runnable code
2. Include benchmark results
3. Explain the optimization techniques used
4. Discuss trade-offs and limitations
5. Provide visualizations where applicable

**Expected Time per Exercise:**
- Exercise 1: 20-30 minutes
- Exercise 2: 30-45 minutes
- Exercise 3: 30-45 minutes
- Exercise 4: 30-45 minutes
- Exercise 5: 45-60 minutes
- Exercise 6: 20-30 minutes
- Exercise 7: 30-45 minutes
- Exercise 8: 20-30 minutes
- Exercise 9: 20-30 minutes
- Exercise 10: 60-90 minutes
