# Performance Optimization Practice Exercises

This comprehensive practice guide provides hands-on exercises to reinforce your understanding of Python performance optimization techniques. Complete each exercise to build practical skills in profiling, optimizing, and benchmarking Python code.

## Table of Contents

1. [Profiling Basics](#profiling-basics)
2. [Algorithm Optimization](#algorithm-optimization)
3. [Memory Optimization](#memory-optimization)
4. [I/O Optimization](#io-optimization)
5. [Concurrency Optimization](#concurrency-optimization)
6. [Real-World Optimization Projects](#real-world-optimization-projects)

---

## Profiling Basics

### Exercise 1: Identifying Bottlenecks with cProfile

**Objective:** Learn to use profiling tools to identify performance bottlenecks in Python code.

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
    # Profile this code
    import cProfile
    import pstats
    from io import StringIO
    
    profiler = cProfile.Profile()
    profiler.enable()
    result = main()
    profiler.disable()
    
    # Analyze results
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(15)  # Show top 15 functions
    
    print("\n--- Analysis Questions ---")
    print("1. Which function takes the most cumulative time?")
    print("2. How many times is slow_calculation called?")
    print("3. What percentage of time is spent in process_data vs slow_calculation?")
    print("4. How would you optimize this code?")
```

**Expected Output:** You should identify that `slow_calculation` is called many times and takes the most time.

**Solution Approach:**
```python
# Optimized version using vectorization
import numpy as np

def optimized_slow_calculation(n: int) -> int:
    """Vectorized calculation for better performance."""
    arr = np.arange(n)
    return int(np.sum(arr * arr))

def optimized_process_data(data: List[int]) -> int:
    """Optimized data processing using numpy."""
    return sum(optimized_slow_calculation(item) for item in data)
```

### Exercise 2: Line-by-Line Profiling with line_profiler

**Objective:** Use line_profiler to identify which specific lines of code are slow.

**Task:** Install line_profiler and profile the following code at the line level:

```python
# profiled_code.py
import random
from typing import List

@profile  # This decorator is recognized by kernprof.py
def process_large_dataset(data_size: int = 10000) -> dict:
    """Process a large dataset with multiple operations."""
    
    # Generate data
    data = []
    for i in range(data_size):
        data.append(random.random() * random.randint(1, 100))
    
    # Filter data
    filtered = []
    for value in data:
        if value > 50:
            filtered.append(value)
    
    # Calculate statistics
    total = 0
    count = 0
    for value in filtered:
        total += value
        count += 1
    
    average = total / count if count > 0 else 0
    
    # Sort data
    sorted_data = sorted(filtered)
    
    # Find percentiles
    p50 = sorted_data[len(sorted_data) // 2]
    p90 = sorted_data[int(len(sorted_data) * 0.9)]
    p99 = sorted_data[int(len(sorted_data) * 0.99)]
    
    return {
        'count': count,
        'average': average,
        'p50': p50,
        'p90': p90,
        'p99': p99
    }

if __name__ == "__main__":
    result = process_large_dataset()
    print(f"Processed {result['count']} items")
    print(f"Average: {result['average']:.2f}")
```

**Instructions:**
1. Install line_profiler: `pip install line-profiler`
2. Run: `kernprof.py -l -v profiled_code.py`
3. Analyze which lines take the most time

**Optimization Challenge:** Rewrite the code to be more efficient while maintaining the same functionality.

### Exercise 3: Memory Profiling

**Objective:** Identify memory usage patterns using memory_profiler.

**Task:** Profile memory usage of the following code and identify memory leaks:

```python
# memory_profile_demo.py
import random
from typing import List
import gc

class DataProcessor:
    """A data processor that may have memory issues."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
        self.cache = {}
    
    def process_batch(self, batch_size: int) -> List[float]:
        """Process a batch of data."""
        results = []
        for i in range(batch_size):
            # Simulate some processing
            value = random.random() * 1000
            result = self.heavy_computation(value)
            results.append(result)
        return results
    
    def heavy_computation(self, value: float) -> float:
        """Heavy computation that creates intermediate objects."""
        intermediate = []
        for i in range(100):
            intermediate.append(value * i)
        
        result = sum(intermediate) / len(intermediate)
        return result
    
    def process_with_caching(self, values: List[float]) -> List[float]:
        """Process values with caching."""
        results = []
        for value in values:
            if value not in self.cache:
                self.cache[value] = self.heavy_computation(value)
            results.append(self.cache[value])
        return results

def main():
    """Main function that demonstrates memory usage."""
    processor = DataProcessor("TestProcessor")
    
    print("Starting memory-intensive operations...")
    
    # Process multiple batches
    for batch_num in range(5):
        print(f"\nProcessing batch {batch_num + 1}...")
        results = processor.process_batch(10000)
        print(f"  Batch {batch_num + 1} complete: {len(results)} results")
        
        # Force garbage collection
        gc.collect()
        
        # Simulate some work
        import time
        time.sleep(0.1)
    
    print(f"\nCache size: {len(processor.cache)} items")

if __name__ == "__main__":
    # Profile memory usage
    from memory_profiler import profile
    
    @profile
    def profiled_main():
        main()
    
    profiled_main()
```

**Analysis Questions:**
1. Which operations use the most memory?
2. Is there a memory leak? Why or why not?
3. How could we reduce memory usage?
4. When is forcing garbage collection beneficial?

---

## Algorithm Optimization

### Exercise 4: Big O Analysis and Optimization

**Objective:** Identify algorithmic inefficiencies and optimize them.

**Task:** Analyze the following functions for time complexity and optimize them:

```python
# algorithm_analysis.py
from typing import List, Set

def find_duplicates_slow(data: List[int]) -> Set[int]:
    """Find duplicate values in a list - O(n²) complexity."""
    duplicates = set()
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] == data[j]:
                duplicates.add(data[i])
    return duplicates

def find_duplicates_optimized(data: List[int]) -> Set[int]:
    """Find duplicate values in a list - O(n) complexity."""
    # Your code here
    pass

# Test the functions
if __name__ == "__main__":
    import random
    import time
    
    # Create test data with duplicates
    test_data = list(range(1000)) + list(range(500))  # 1500 items, 500 duplicates
    
    # Test slow version
    start = time.time()
    duplicates_slow = find_duplicates_slow(test_data)
    slow_time = time.time() - start
    
    # Test optimized version
    start = time.time()
    duplicates_fast = find_duplicates_optimized(test_data)
    fast_time = time.time() - start
    
    print(f"Slow version: {len(duplicates_slow)} duplicates found in {slow_time:.4f}s")
    print(f"Fast version: {len(duplicates_fast)} duplicates found in {fast_time:.4f}s")
    print(f"Speedup: {slow_time/fast_time:.2f}x")
    print(f"Results match: {duplicates_slow == duplicates_fast}")
```

**Additional Functions to Optimize:**

```python
def fibonacci_slow(n: int) -> int:
    """Calculate nth Fibonacci number - exponential time."""
    if n <= 1:
        return n
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)

def fibonacci_memoized(n: int, memo: dict = None) -> int:
    """Calculate nth Fibonacci number with memoization."""
    # Your code here
    pass

def is_palindrome_slow(s: str) -> bool:
    """Check if string is palindrome - O(n) with extra space."""
    return s == s[::-1]

def is_palindrome_optimized(s: str) -> bool:
    """Check if string is palindrome - O(n) with O(1) space."""
    # Your code here
    pass

def find_max_slow(data: List[int]) -> int:
    """Find maximum value - O(n) but with multiple passes."""
    if not data:
        return None
    
    max_val = data[0]
    for num in data:
        if num > max_val:
            max_val = num
    
    # Additional unnecessary pass
    for num in data:
        if num > max_val:
            max_val = num
    
    return max_val

def find_max_optimized(data: List[int]) -> int:
    """Find maximum value - O(n) single pass."""
    # Your code here
    pass
```

### Exercise 5: Data Structure Optimization

**Objective:** Choose the right data structure for optimal performance.

**Task:** Optimize the following code by selecting appropriate data structures:

```python
from typing import List, Dict, Set
from collections import defaultdict

class SearchEngine:
    """A simple search engine with optimization opportunities."""
    
    def __init__(self):
        self.documents = {}  # doc_id -> content
        self.inverted_index = {}  # word -> set of doc_ids
    
    def add_document(self, doc_id: int, content: str) -> None:
        """Add a document to the search index."""
        self.documents[doc_id] = content
        
        # Build inverted index
        words = content.lower().split()
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(doc_id)
    
    def search(self, query: str) -> List[int]:
        """Search for documents matching the query."""
        words = query.lower().split()
        
        if not words:
            return []
        
        # Get documents containing all words
        result_sets = []
        for word in words:
            if word in self.inverted_index:
                result_sets.append(self.inverted_index[word])
        
        if not result_sets:
            return []
        
        # Intersection of all result sets
        result = result_sets[0]
        for s in result_sets[1:]:
            result = result.intersection(s)
        
        return sorted(list(result))
    
    def search_optimized(self, query: str) -> List[int]:
        """Optimized search using better data structures."""
        # Your optimization here
        pass

# Performance test
if __name__ == "__main__":
    import random
    import string
    import time
    
    engine = SearchEngine()
    
    # Add 10000 documents
    print("Adding 10,000 documents...")
    for i in range(10000):
        # Generate random document with 50-100 words
        num_words = random.randint(50, 100)
        words = [''.join(random.choices(string.ascii_lowercase, k=5)) 
                 for _ in range(num_words)]
        content = ' '.join(words)
        engine.add_document(i, content)
    
    print(f"Index contains {len(engine.inverted_index)} unique words")
    
    # Test search performance
    test_queries = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "theta iota kappa"
    ]
    
    for query in test_queries:
        start = time.time()
        results = engine.search(query)
        search_time = time.time() - start
        print(f"Query '{query}': {len(results)} results in {search_time:.4f}s")
```

### Exercise 6: Caching and Memoization

**Objective:** Implement caching strategies to improve performance.

**Task:** Create an optimized version of the following function with intelligent caching:

```python
from functools import lru_cache
from typing import Dict, List, Tuple
import time

def expensive_computation(n: int, m: int) -> int:
    """An expensive computation that could benefit from caching."""
    # Simulate expensive work
    time.sleep(0.01)  # 10ms delay
    
    result = 0
    for i in range(n):
        for j in range(m):
            result += i * j
    
    return result

def process_data_with_caching(data: List[Tuple[int, int]]) -> List[int]:
    """Process data using caching for expensive computations."""
    # Your implementation here
    pass

# Advanced caching challenge
class MemoizedFibonacci:
    """Fibonacci calculator with various caching strategies."""
    
    def __init__(self):
        self.cache = {}
        self.call_count = 0
    
    def fib_recursive(self, n: int) -> int:
        """Recursive Fibonacci without caching."""
        self.call_count += 1
        if n <= 1:
            return n
        return self.fib_recursive(n - 1) + self.fib_recursive(n - 2)
    
    def fib_memoized(self, n: int) -> int:
        """Fibonacci with manual memoization."""
        # Your code here
        pass
    
    def fib_lru_cache(self, n: int) -> int:
        """Fibonacci using functools.lru_cache."""
        # Your code here
        pass
    
    def fib_iterative(self, n: int) -> int:
        """Fibonacci using iterative approach."""
        # Your code here
        pass
    
    def compare_methods(self, n: int) -> Dict[str, Tuple[int, int]]:
        """Compare different Fibonacci implementations."""
        results = {}
        
        # Test recursive
        self.call_count = 0
        start = time.time()
        result = self.fib_recursive(n)
        recursive_time = time.time() - start
        results['recursive'] = (result, self.call_count)
        
        # Test memoized
        start = time.time()
        result = self.fib_memoized(n)
        memoized_time = time.time() - start
        results['memoized'] = (result, 0)  # Count doesn't apply
        
        # Test lru_cache
        start = time.time()
        result = self.fib_lru_cache(n)
        lru_time = time.time() - start
        results['lru_cache'] = (result, 0)
        
        # Test iterative
        start = time.time()
        result = self.fib_iterative(n)
        iterative_time = time.time() - start
        results['iterative'] = (result, 0)
        
        return results

if __name__ == "__main__":
    # Test caching
    print("Testing expensive computation caching...")
    
    data = [(10, 20), (15, 25), (10, 20), (20, 30), (10, 20)]
    results = process_data_with_caching(data)
    print(f"Results: {results}")
    
    # Test Fibonacci methods
    print("\nComparing Fibonacci methods (n=30)...")
    fib = MemoizedFibonacci()
    results = fib.compare_methods(30)
    
    for method, (result, count) in results.items():
        print(f"{method}: {result} (time varies)")
```

---

## Memory Optimization

### Exercise 7: Generator and Iterator Patterns

**Objective:** Use generators to reduce memory usage for large datasets.

**Task:** Convert the following memory-intensive code to use generators:

```python
from typing import List, Generator
import random

def generate_large_dataset(size: int) -> List[int]:
    """Generate a large dataset - uses a lot of memory."""
    return [random.randint(0, 1000000) for _ in range(size)]

def generate_large_dataset_optimized(size: int) -> Generator[int, None, None]:
    """Generate a large dataset using a generator - memory efficient."""
    # Your code here
    pass

def process_data_traditional(data: List[int]) -> int:
    """Process all data at once - high memory usage."""
    filtered = [x for x in data if x > 500000]
    squared = [x * x for x in filtered]
    total = sum(squared)
    return total

def process_data_optimized(data_generator: Generator[int, None, None]) -> int:
    """Process data as it arrives - low memory usage."""
    # Your code here
    pass

# Memory comparison
if __name__ == "__main__":
    import sys
    
    print("Memory Usage Comparison")
    print("=" * 50)
    
    # Test traditional approach
    print("\nTraditional approach:")
    large_data = generate_large_dataset(1000000)
    data_size = sys.getsizeof(large_data)
    print(f"  List size: {data_size / 1024 / 1024:.2f} MB")
    
    result = process_data_traditional(large_data)
    print(f"  Result: {result}")
    
    # Test optimized approach
    print("\nOptimized approach:")
    data_gen = generate_large_dataset_optimized(1000000)
    gen_size = sys.getsizeof(data_gen)
    print(f"  Generator size: {gen_size / 1024:.2f} KB")
    
    result = process_data_optimized(data_gen)
    print(f"  Result: {result}")
    
    print("\nMemory saved:", f"{(data_size - gen_size) / 1024 / 1024:.2f} MB")
```

### Exercise 8: Object Pooling and Reuse

**Objective:** Implement object pooling to reduce memory allocation overhead.

**Task:** Create an object pool for expensive-to-create objects:

```python
from typing import Dict, List, Optional
import time
import random

class DatabaseConnection:
    """Simulated database connection."""
    
    def __init__(self, connection_id: int):
        self.connection_id = connection_id
        self.created_at = time.time()
        self.queries_executed = 0
    
    def execute_query(self, query: str) -> str:
        """Execute a database query."""
        self.queries_executed += 1
        return f"Result for '{query}' from connection {self.connection_id}"
    
    def close(self):
        """Close the connection."""
        print(f"Closing connection {self.connection_id}")

class ConnectionPool:
    """A connection pool for database connections."""
    
    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.available: List[DatabaseConnection] = []
        self.in_use: Dict[int, DatabaseConnection] = {}
        self.connection_counter = 0
        
        # Pre-create connections
        for _ in range(pool_size):
            self.connection_counter += 1
            self.available.append(DatabaseConnection(self.connection_counter))
    
    def acquire(self) -> DatabaseConnection:
        """Acquire a connection from the pool."""
        # Your code here
        pass
    
    def release(self, connection: DatabaseConnection) -> None:
        """Release a connection back to the pool."""
        # Your code here
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        # Your code here
        pass

class ExpensiveObject:
    """An expensive object to create."""
    
    instances_created = 0
    
    def __init__(self, data_size: int = 1000):
        ExpensiveObject.instances_created += 1
        self.id = ExpensiveObject.instances_created
        self.data = [random.random() for _ in range(data_size)]
        time.sleep(0.01)  # Simulate expensive initialization
    
    def process(self) -> float:
        """Process the data."""
        return sum(self.data) / len(self.data)

class ObjectPool:
    """Generic object pool for expensive objects."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: List[ExpensiveObject] = []
    
    def get_object(self) -> ExpensiveObject:
        """Get an object from the pool or create new one."""
        # Your code here
        pass
    
    def return_object(self, obj: ExpensiveObject) -> None:
        """Return an object to the pool."""
        # Your code here
        pass

if __name__ == "__main__":
    print("Object Pooling Demo")
    print("=" * 50)
    
    # Test connection pool
    print("\n1. Database Connection Pool")
    
    with ConnectionPool(pool_size=3) as pool:
        # Acquire connections
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        
        print(f"  Using {len(pool.in_use)} connections")
        
        # Use connections
        print(f"  {conn1.execute_query('SELECT * FROM users')}")
        print(f"  {conn2.execute_query('SELECT * FROM orders')}")
        
        # Release connections
        pool.release(conn1)
        pool.release(conn2)
        print(f"  Available after release: {len(pool.available)}")
    
    # Test object pool
    print("\n2. Expensive Object Pool")
    
    pool = ObjectPool(max_size=3)
    
    start = time.time()
    objects = [pool.get_object() for _ in range(5)]
    creation_time = time.time() - start
    
    print(f"  Created 5 objects in {creation_time:.3f}s")
    print(f"  Pool size: {len(pool.pool)}")
    print(f"  Total instances created: {ExpensiveObject.instances_created}")
    
    # Return objects to pool
    for obj in objects[:3]:
        pool.return_object(obj)
    
    print(f"  Pool size after return: {len(pool.pool)}")
    
    # Get more objects (should reuse from pool)
    start = time.time()
    more_objects = [pool.get_object() for _ in range(3)]
    reuse_time = time.time() - start
    
    print(f"  Reused 3 objects in {reuse_time:.3f}s")
    print(f"  Total instances created: {ExpensiveObject.instances_created}")
```

### Exercise 9: Memory-Efficient Data Types

**Objective:** Use appropriate data types to minimize memory usage.

**Task:** Optimize memory usage by choosing the right data structures:

```python
from typing import List, Set, Dict
from array import array
from dataclasses import dataclass
import sys

class MemoryInefficientExample:
    """Demonstrates memory-inefficient patterns."""
    
    def __init__(self):
        # Using list where set would be better
        self.unique_ids_list = []
        
        # Using dict where tuple would be better
        self.coordinates = []
        
        # Using int where bool would be better
        self.flags = []
        
        # Using list of ints where array would be better
        self.large_numbers = []

class MemoryEfficientExample:
    """Demonstrates memory-efficient patterns."""
    
    def __init__(self):
        # Use set for unique items
        self.unique_ids_set = set()
        
        # Use tuples for immutable coordinates
        self.coordinates = []
        
        # Use bool instead of int for flags
        self.flags = []
        
        # Use array for large number sequences
        self.large_numbers = array('I')  # Unsigned int array

@dataclass
class UserProfile:
    """Memory-efficient user profile."""
    user_id: int
    name: str
    age: int
    is_active: bool
    tags: tuple  # Use tuple instead of list for immutable data

def compare_memory_usage():
    """Compare memory usage of different approaches."""
    
    print("Memory Usage Comparison")
    print("=" * 60)
    
    # List vs Set
    print("\n1. List vs Set for unique IDs")
    
    data = list(range(10000))
    
    list_version = list(data)
    set_version = set(data)
    
    print(f"  List: {sys.getsizeof(list_version) / 1024:.2f} KB")
    print(f"  Set: {sys.getsizeof(set_version) / 1024:.2f} KB")
    print(f"  Memory saved: {(sys.getsizeof(list_version) - sys.getsizeof(set_version)) / 1024:.2f} KB")
    
    # List vs Array
    print("\n2. List vs Array for numbers")
    
    numbers_list = [i for i in range(100000)]
    numbers_array = array('I', range(100000))
    
    print(f"  List: {sys.getsizeof(numbers_list) / 1024 / 1024:.2f} MB")
    print(f"  Array: {sys.getsizeof(numbers_array) / 1024 / 1024:.2f} MB")
    print(f"  Memory saved: {(sys.getsizeof(numbers_list) - sys.getsizeof(numbers_array)) / 1024 / 1024:.2f} MB")
    
    # String concatenation optimization
    print("\n3. String concatenation methods")
    
    parts = ["word"] * 1000
    
    # Inefficient
    start = time.time()
    result1 = ""
    for part in parts:
        result1 += part
    concat_time = time.time() - start
    
    # Efficient
    start = time.time()
    result2 = "".join(parts)
    join_time = time.time() - start
    
    print(f"  Concatenation: {concat_time:.4f}s")
    print(f"  Join: {join_time:.4f}s")
    print(f"  Speedup: {concat_time/join_time:.2f}x")
    
    # Custom class optimization
    print("\n4. Regular class vs __slots__")
    
    class RegularClass:
        __slots__ = ()  # Empty, but prevents __dict__
        pass
    
    regular = RegularClass()
    regular.x = 1
    regular.y = 2
    regular.z = 3
    
    class SlotsClass:
        __slots__ = ('x', 'y', 'z')
    
    slots = SlotsClass()
    slots.x = 1
    slots.y = 2
    slots.z = 3
    
    print(f"  Regular class instance: {sys.getsizeof(RegularClass()) + sys.getsizeof(RegularClass().__dict__)} bytes")
    print(f"  __slots__ class instance: {sys.getsizeof(SlotsClass())} bytes")

if __name__ == "__main__":
    import time
    compare_memory_usage()
```

---

## I/O Optimization

### Exercise 10: File I/O Optimization

**Objective:** Optimize file reading and writing operations.

**Task:** Optimize the following file processing code:

```python
import os
import time
from typing import List, Generator

class FileProcessor:
    """File processor with optimization opportunities."""
    
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
    
    def process_file_inefficient(self) -> None:
        """Process file inefficiently."""
        with open(self.input_file, 'r') as f:
            lines = f.readlines()  # Read all at once
        
        processed = []
        for line in lines:
            processed.append(line.strip().upper())
        
        with open(self.output_file, 'w') as f:
            for line in processed:
                f.write(line + '\n')
    
    def process_file_optimized(self) -> None:
        """Process file with I/O optimization."""
        # Your optimized implementation here
        pass
    
    def process_large_file_line_by_line(self) -> Generator[str, None, None]:
        """Process large file line by line using generator."""
        # Your implementation here
        pass

def create_test_file(filename: str, num_lines: int = 10000) -> None:
    """Create a test file with random data."""
    import random
    import string
    
    with open(filename, 'w') as f:
        for _ in range(num_lines):
            line = ''.join(random.choices(string.ascii_lowercase, k=50))
            f.write(line + '\n')

if __name__ == "__main__":
    # Create test file
    input_file = "test_input.txt"
    output_file = "test_output.txt"
    
    print("Creating test file with 100,000 lines...")
    create_test_file(input_file, 100000)
    
    # Test inefficient version
    processor = FileProcessor(input_file, output_file)
    
    print("\nTesting inefficient version...")
    start = time.time()
    processor.process_file_inefficient()
    inefficient_time = time.time() - start
    print(f"  Inefficient version: {inefficient_time:.3f}s")
    
    print("\nTesting optimized version...")
    start = time.time()
    processor.process_file_optimized()
    optimized_time = time.time() - start
    print(f"  Optimized version: {optimized_time:.3f}s")
    
    print(f"\nSpeedup: {inefficient_time/optimized_time:.2f}x")
    
    # Cleanup
    os.remove(input_file)
    os.remove(output_file)
```

### Exercise 11: Batch Processing and Buffering

**Objective:** Use batching and buffering to optimize I/O operations.

**Task:** Implement batch processing for database operations:

```python
import time
from typing import List, Dict
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class UserRecord:
    """A user record for database operations."""
    user_id: int
    name: str
    email: str
    created_at: str

class BatchProcessor:
    """Batch processor for database operations."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.individual_inserts = 0
        self.batch_inserts = 0
    
    def insert_individual(self, records: List[UserRecord]) -> None:
        """Insert records one at a time (slow)."""
        for record in records:
            # Simulate individual insert
            self.individual_inserts += 1
            time.sleep(0.001)  # Simulate DB latency
    
    def insert_batch(self, records: List[UserRecord]) -> None:
        """Insert records in batches (fast)."""
        # Your implementation here
        pass
    
    def process_transactions(self, transactions: List[Dict]) -> Dict:
        """Process financial transactions in batches."""
        # Track balances
        balances = defaultdict(float)
        
        # Process in batches
        # Your implementation here
        pass

class BufferedWriter:
    """Buffered file writer for efficient I/O."""
    
    def __init__(self, filename: str, buffer_size: int = 8192):
        self.filename = filename
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffered_bytes = 0
    
    def write(self, data: str) -> None:
        """Write data to buffer."""
        # Your implementation here
        pass
    
    def flush(self) -> None:
        """Flush buffer to file."""
        # Your implementation here
        pass
    
    def close(self) -> None:
        """Close writer and flush remaining data."""
        # Your implementation here
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    print("Batch Processing Demo")
    print("=" * 50)
    
    # Create test records
    records = [
        UserRecord(i, f"User {i}", f"user{i}@example.com", "2024-01-01")
        for i in range(1000)
    ]
    
    # Test individual inserts
    processor = BatchProcessor()
    
    print("\n1. Individual inserts (1000 records)")
    start = time.time()
    processor.insert_individual(records)
    individual_time = time.time() - start
    print(f"  Time: {individual_time:.2f}s")
    
    # Test batch inserts
    print("\n2. Batch inserts (1000 records in batches of 100)")
    processor.individual_inserts = 0
    processor.batch_inserts = 0
    
    start = time.time()
    processor.insert_batch(records)
    batch_time = time.time() - start
    print(f"  Time: {batch_time:.2f}s")
    print(f"  Batch operations: {processor.batch_inserts}")
    
    print(f"\nSpeedup: {individual_time/batch_time:.2f}x")
    
    # Test buffered writer
    print("\n3. Buffered writer performance")
    
    test_data = ["line " + str(i) + "\n" for i in range(10000)]
    
    # Unbuffered
    start = time.time()
    with open("unbuffered.txt", 'w') as f:
        for line in test_data:
            f.write(line)
    unbuffered_time = time.time() - start
    
    # Buffered
    start = time.time()
    with BufferedWriter("buffered.txt") as writer:
        for line in test_data:
            writer.write(line)
    buffered_time = time.time() - start
    
    print(f"  Unbuffered: {unbuffered_time:.3f}s")
    print(f"  Buffered: {buffered_time:.3f}s")
    
    # Cleanup
    import os
    for filename in ["unbuffered.txt", "buffered.txt"]:
        if os.path.exists(filename):
            os.remove(filename)
```

---

## Concurrency Optimization

### Exercise 12: Parallel Processing with multiprocessing

**Objective:** Use multiprocessing to parallelize CPU-bound tasks.

**Task:** Parallelize the following data processing tasks:

```python
import time
import random
from typing import List
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor

def process_item(item: int) -> int:
    """Process a single item (CPU-bound operation)."""
    # Simulate CPU-bound work
    result = 0
    for i in range(10000):
        result += item * i
    return result

def process_chunk(chunk: List[int]) -> int:
    """Process a chunk of items."""
    return sum(process_item(item) for item in chunk)

def process_data_sequential(data: List[int]) -> int:
    """Process data sequentially."""
    total = 0
    for item in data:
        total += process_item(item)
    return total

def process_data_parallel(data: List[int], num_processes: int = None) -> int:
    """Process data in parallel using multiprocessing."""
    # Your implementation here
    pass

def process_data_concurrent(data: List[int], max_workers: int = None) -> int:
    """Process data using ProcessPoolExecutor."""
    # Your implementation here
    pass

if __name__ == "__main__":
    print("Parallel Processing Demo")
    print("=" * 50)
    
    # Create test data
    data = [random.randint(1, 100) for _ in range(100)]
    
    # Test sequential processing
    print("\n1. Sequential processing")
    start = time.time()
    result_sequential = process_data_sequential(data)
    sequential_time = time.time() - start
    print(f"  Time: {sequential_time:.3f}s")
    print(f"  Result: {result_sequential}")
    
    # Test parallel processing
    print(f"\n2. Parallel processing ({cpu_count()} CPUs)")
    start = time.time()
    result_parallel = process_data_parallel(data)
    parallel_time = time.time() - start
    print(f"  Time: {parallel_time:.3f}s")
    print(f"  Result: {result_parallel}")
    
    print(f"\nSpeedup: {sequential_time/parallel_time:.2f}x")
    
    # Test with different process counts
    print("\n3. Scaling with different process counts:")
    for num_processes in [2, 4, 8]:
        start = time.time()
        process_data_parallel(data, num_processes)
        elapsed = time.time() - start
        print(f"  {num_processes} processes: {elapsed:.3f}s")
```

### Exercise 13: Async I/O Operations

**Objective:** Use async programming for I/O-bound operations.

**Task:** Convert the following synchronous code to async:

```python
import asyncio
import time
from typing import List
import aiohttp

async def fetch_data_sync(url: str) -> dict:
    """Fetch data from URL synchronously."""
    import urllib.request
    import json
    
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    return data

async def fetch_data_async(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch data from URL asynchronously."""
    # Your implementation here
    pass

async def fetch_all_data_async(urls: List[str]) -> List[dict]:
    """Fetch data from multiple URLs concurrently."""
    # Your implementation here
    pass

def fetch_all_data_sync(urls: List[str]) -> List[dict]:
    """Fetch data from multiple URLs sequentially."""
    results = []
    for url in urls:
        results.append(asyncio.run(fetch_data_sync(url)))
    return results

if __name__ == "__main__":
    # Test with sample URLs (using placeholder URLs)
    test_urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3",
    ]
    
    print("Async I/O Demo")
    print("=" * 50)
    
    # Note: These URLs may not work in offline environment
    # This is a conceptual demonstration
    
    print("\nFor production use:")
    print("1. Use aiohttp for async HTTP requests")
    print("2. Use asyncio.gather() for concurrent requests")
    print("3. Implement proper error handling")
    print("4. Add rate limiting and retries")
```

---

## Real-World Optimization Projects

### Project 1: Log File Analyzer

**Objective:** Build an optimized log file analyzer.

**Task Description:** Create a log analyzer that processes large log files efficiently:

```python
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Generator, Tuple
import time

@dataclass
class LogEntry:
    """Parsed log entry."""
    timestamp: str
    level: str
    message: str
    source: str

class LogAnalyzer:
    """Optimized log file analyzer."""
    
    LOG_PATTERN = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+'
        r'(\w+)\s+'
        r'\[(\w+)\]\s+'
        r'(.+)'
    )
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.entries: List[LogEntry] = []
        self.stats = defaultdict(int)
    
    def parse_line(self, line: str) -> LogEntry:
        """Parse a single log line."""
        match = self.LOG_PATTERN.match(line)
        if match:
            return LogEntry(
                timestamp=match.group(1),
                level=match.group(2),
                source=match.group(3),
                message=match.group(4)
            )
        return None
    
    def process_inefficient(self) -> Dict:
        """Process log file inefficiently - loads all into memory."""
        with open(self.log_file, 'r') as f:
            self.entries = [self.parse_line(line) 
                          for line in f.readlines() 
                          if self.parse_line(line)]
        
        for entry in self.entries:
            self.stats[f"level:{entry.level}"] += 1
            self.stats[f"source:{entry.source}"] += 1
        
        return dict(self.stats)
    
    def process_optimized(self) -> Dict:
        """Process log file efficiently - streaming approach."""
        # Your implementation here
        pass
    
    def find_slow_requests(self, threshold_ms: float = 1000) -> List[Tuple[str, float]]:
        """Find slow requests from log entries."""
        slow_requests = []
        
        for entry in self.entries:
            if "request_time" in entry.message.lower():
                # Parse request time from message
                # Your implementation here
                pass
        
        return slow_requests
    
    def get_error_summary(self) -> Counter:
        """Get summary of error messages."""
        return Counter(
            entry.message.split('Error:')[-1].strip()
            for entry in self.entries
            if entry.level == 'ERROR'
        )

# Usage example
if __name__ == "__main__":
    # Create sample log file
    sample_logs = [
        "2024-01-15 10:30:45 INFO [user-service] User logged in successfully",
        "2024-01-15 10:30:46 ERROR [payment-service] Payment failed: insufficient funds",
        "2024-01-15 10:30:47 WARNING [cache-service] Cache miss for key: user_123",
        "2024-01-15 10:30:48 INFO [user-service] Request processed in 45ms",
        "2024-01-15 10:30:49 ERROR [database-service] Connection timeout after 5000ms",
    ]
    
    with open("sample.log", 'w') as f:
        f.write('\n'.join(sample_logs))
    
    # Test analyzer
    analyzer = LogAnalyzer("sample.log")
    
    print("Log Analyzer Demo")
    print("=" * 50)
    
    # Process logs
    stats = analyzer.process_inefficient()
    print(f"Stats: {stats}")
    
    # Cleanup
    import os
    os.remove("sample.log")
```

### Project 2: Image Processing Pipeline

**Objective:** Build an optimized image processing pipeline.

**Task Description:** Create an image processing system with optimization for large batches:

```python
from typing import List, Tuple
from dataclasses import dataclass
from PIL import Image
import os
import time
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ImageInfo:
    """Image metadata."""
    filename: str
    width: int
    height: int
    format: str
    size_bytes: int

class ImageProcessor:
    """Optimized image processor."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processed_count = 0
        self.total_time = 0
    
    def process_single_image(self, filename: str, target_size: Tuple[int, int] = (800, 600)) -> ImageInfo:
        """Process a single image."""
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, f"processed_{filename}")
        
        # Open and resize image
        with Image.open(input_path) as img:
            img.thumbnail(target_size)
            img.save(output_path, quality=85)
            
            return ImageInfo(
                filename=filename,
                width=img.width,
                height=img.height,
                format=img.format,
                size_bytes=os.path.getsize(output_path)
            )
    
    def process_images_sequential(self, filenames: List[str]) -> List[ImageInfo]:
        """Process images sequentially."""
        results = []
        for filename in filenames:
            start = time.time()
            result = self.process_single_image(filename)
            self.total_time += time.time() - start
            results.append(result)
            self.processed_count += 1
        return results
    
    def process_images_parallel(self, filenames: List[str], max_workers: int = 4) -> List[ImageInfo]:
        """Process images in parallel using threading."""
        # Your implementation here
        pass
    
    def generate_thumbnails(self, filenames: List[str], size: Tuple[int, int] = (200, 150)) -> None:
        """Generate thumbnails for all images."""
        # Your implementation here
        pass

if __name__ == "__main__":
    # Create test images
    from PIL import Image
    
    print("Image Processing Pipeline Demo")
    print("=" * 50)
    
    # Create test directory
    os.makedirs("test_images", exist_ok=True)
    os.makedirs("output_images", exist_ok=True)
    
    # Create test images
    for i in range(10):
        img = Image.new('RGB', (1920, 1080), color=(i * 25, i * 25, 255))
        img.save(f"test_images/image_{i}.jpg")
    
    filenames = [f"image_{i}.jpg" for i in range(10)]
    
    # Initialize processor
    processor = ImageProcessor("test_images", "output_images")
    
    # Process sequentially
    print("\n1. Sequential processing:")
    start = time.time()
    results = processor.process_images_sequential(filenames)
    sequential_time = time.time() - start
    print(f"  Time: {sequential_time:.2f}s")
    print(f"  Processed: {processor.processed_count} images")
    
    # Process in parallel
    print("\n2. Parallel processing:")
    processor2 = ImageProcessor("test_images", "output_images")
    start = time.time()
    results = processor2.process_images_parallel(filenames, max_workers=4)
    parallel_time = time.time() - start
    print(f"  Time: {parallel_time:.2f}s")
    print(f"  Processed: {processor2.processed_count} images")
    
    print(f"\nSpeedup: {sequential_time/parallel_time:.2f}x")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_images")
    shutil.rmtree("output_images")
```

---

## Best Practices and Tips

### Performance Optimization Checklist

When optimizing Python code, follow these best practices:

1. **Measure First, Optimize Second**
   - Always profile before optimizing
   - Use cProfile for CPU profiling
   - Use memory_profiler for memory analysis
   - Identify the actual bottlenecks

2. **Choose the Right Algorithm**
   - Consider Big O complexity
   - Use appropriate data structures
   - Implement caching where beneficial

3. **Use Built-in Functions and Libraries**
   - NumPy for numerical operations
   - itertools and collections for efficient iterators
   - functools for caching and partial functions

4. **Optimize I/O Operations**
   - Use buffering for file operations
   - Batch database operations
   - Use async I/O for network operations

5. **Consider Concurrency**
   - Use multiprocessing for CPU-bound tasks
   - Use asyncio for I/O-bound tasks
   - Use threading for blocking I/O

6. **Memory Management**
   - Use generators for large datasets
   - Implement object pooling for expensive objects
   - Use __slots__ for memory-efficient classes

### Common Performance Anti-Patterns

Avoid these common mistakes:

1. **Premature Optimization**
   - Don't optimize without profiling
   - Focus on clear code first
   - Measure actual impact

2. **Unnecessary Object Creation**
   - Reuse objects when possible
   - Use immutable data types
   - Implement object pooling

3. **Inefficient String Operations**
   - Use string formatting over concatenation
   - Use join() for multiple strings
   - Avoid repeated string concatenation in loops

4. **Inefficient Data Structure Usage**
   - Use sets for membership testing
   - Use dictionaries for key-value lookups
   - Use arrays for numerical data

### Summary

This practice guide has covered essential performance optimization techniques in Python:

- **Profiling** tools help identify bottlenecks
- **Algorithm optimization** improves time complexity
- **Memory optimization** reduces resource usage
- **I/O optimization** improves throughput
- **Concurrency** enables parallel processing

Practice these techniques regularly, and always profile your code before and after optimization to ensure meaningful improvements.
