# Heaps Quick Reference Cheatsheet

## Heap Properties

### Min-Heap

```
Parent <= Children
Root is the minimum element
```

### Max-Heap

```
Parent >= Children
Root is the maximum element
```

### Array Representation

```
For index i:
- Parent: (i-1) // 2
- Left child: 2*i + 1
- Right child: 2*i + 2
```

## Time Complexities

| Operation       | Complexity | Description                |
| --------------- | ---------- | -------------------------- |
| Build Heap      | O(n)       | Build from array           |
| Insert          | O(log n)   | Add element and heapify up |
| Extract Min/Max | O(log n)   | Remove and heapify down    |
| Peek            | O(1)       | Get root element           |
| Delete          | O(log n)   | Remove arbitrary element   |

## Python heapq Library

### Basic Operations

```python
import heapq

# Min-heap operations
heap = [3, 1, 4, 1, 5, 9]
heapq.heapify(heap)  # Convert to heap
heapq.heappush(heap, 2)  # Insert
min_val = heapq.heappop(heap)  # Extract min

# Max-heap (use negative values)
max_heap = [-x for x in [3, 1, 4, 1, 5, 9]]
heapq.heapify(max_heap)
max_val = -heapq.heappop(max_heap)
```

### Common Patterns

#### 1. K Largest Elements

```python
def k_largest(arr, k):
    import heapq
    if k >= len(arr):
        return sorted(arr, reverse=True)

    return heapq.nlargest(k, arr)  # Built-in function
```

#### 2. K Smallest Elements

```python
def k_smallest(arr, k):
    import heapq
    return heapq.nsmallest(k, arr)  # Built-in function
```

#### 3. Merge K Lists

```python
def merge_k_lists(lists):
    import heapq

    min_heap = []
    result = []

    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0, lst))

    while min_heap:
        val, list_idx, elem_idx, lst = heapq.heappop(min_heap)
        result.append(val)

        if elem_idx + 1 < len(lst):
            heapq.heappush(min_heap, (lst[elem_idx + 1], list_idx, elem_idx + 1, lst))

    return result
```

#### 4. Top K Frequent Elements

```python
def top_k_frequent(nums, k):
    from collections import Counter
    import heapq

    freq = Counter(nums)
    max_heap = [(-count, num) for num, count in freq.items()]
    heapq.heapify(max_heap)

    return [heapq.heappop(max_heap)[1] for _ in range(k)]
```

#### 5. Median Finder

```python
class MedianFinder:
    def __init__(self):
        self.lower = []  # max-heap (negate values)
        self.upper = []  # min-heap

    def add_num(self, num):
        if not self.lower or num <= -self.lower[0]:
            heapq.heappush(self.lower, -num)
        else:
            heapq.heappush(self.upper, num)

        # Balance
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def find_median(self):
        if len(self.lower) > len(self.upper):
            return -self.lower[0]
        elif len(self.upper) > len(self.lower):
            return self.upper[0]
        else:
            return (-self.lower[0] + self.upper[0]) / 2
```

## Heap Sort Implementation

```python
def heap_sort(arr):
    n = len(arr)

    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

## Binary Heap Properties

### Complete Binary Tree

- All levels completely filled except possibly last level
- Last level filled from left to right

### Heap Order Property

- **Min-heap**: Parent ≤ children
- **Max-heap**: Parent ≥ children

## Applications Quick Reference

| Use Case             | Pattern            | Complexity             |
| -------------------- | ------------------ | ---------------------- |
| Dijkstra's Algorithm | Min-heap           | O((V+E) log V)         |
| Prim's MST           | Min-heap           | O(E log V)             |
| Huffman Coding       | Max-heap           | O(n log n)             |
| Median in Stream     | Two heaps          | O(log n) per operation |
| K Largest            | Min-heap of size k | O(n log k)             |
| Top K Frequent       | Max-heap           | O(n log k)             |
| Merge K Lists        | Min-heap           | O(n log k)             |

## Common Interview Patterns

### 1. Sliding Window Maximum

```python
def max_sliding_window(nums, k):
    from collections import deque

    dq = deque()
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements
        while dq and nums[dq[-1]] <= num:
            dq.pop()

        dq.append(i)

        # Add to result
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### 2. IPO (Leetcode 502)

```python
def find_maximized_capital(k, w, profits, capital):
    import heapq

    min_capital = []
    max_profit = []

    projects = list(zip(capital, profits))
    projects.sort()

    i = 0

    for _ in range(k):
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(max_profit, -projects[i][1])
            i += 1

        if not max_profit:
            break

        w += -heapq.heappop(max_profit)

    return w
```

### 3. K Closest Points to Origin

```python
def k_closest(points, k):
    import heapq

    def distance_squared(point):
        return point[0]**2 + point[1]**2

    max_heap = []

    for i, (x, y) in enumerate(points):
        dist = distance_squared((x, y))
        if i < k:
            heapq.heappush(max_heap, (-dist, x, y))
        elif dist < -max_heap[0][0]:
            heapq.heapreplace(max_heap, (-dist, x, y))

    return [(x, y) for _, x, y in max_heap]
```

## Common Mistakes to Avoid

1. **Not handling empty heap** before operations
2. **Forgetting to heapify** after building from array
3. **Using wrong heap type** (max vs min)
4. **Index calculation errors** in array representation
5. **Memory leaks** in priority queues with deletion
6. **Not balancing heaps** in median finder

## Quick Debugging Tips

```python
# Check if array is valid heap
def is_heap(arr, is_min=True):
    n = len(arr)
    for i in range(n):
        left = 2*i + 1
        right = 2*i + 2

        if left < n:
            if is_min and arr[i] > arr[left]:
                return False
            if not is_min and arr[i] < arr[left]:
                return False

        if right < n:
            if is_min and arr[i] > arr[right]:
                return False
            if not is_min and arr[i] < arr[right]:
                return False

    return True

# Convert between min and max heap
def min_to_max_heap(arr):
    return [-x for x in arr]

def max_to_min_heap(arr):
    return [-x for x in arr]
```
