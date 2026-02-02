# Heaps Practice Exercises

## Basic Implementation Exercises

### 1. Min-Heap Implementation

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def insert(self, value):
        # TODO: Implement insertion with heapify
        pass

    def extract_min(self):
        # TODO: Implement extraction of minimum element
        pass

    def heapify(self, i):
        # TODO: Implement heapify-down operation
        pass
```

**Solution:**

```python
class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def insert(self, value):
        self.heap.append(value)
        self.heapify_up(len(self.heap) - 1)

    def heapify_up(self, i):
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.heap[i], self.heap[self.parent(i)] = self.heap[self.parent(i)], self.heap[i]
            i = self.parent(i)

    def extract_min(self):
        if not self.heap:
            return None

        min_val = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()

        if self.heap:
            self.heapify_down(0)

        return min_val

    def heapify_down(self, i):
        smallest = i
        left = self.left_child(i)
        right = self.right_child(i)

        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left

        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right

        if smallest != i:
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            self.heapify_down(smallest)
```

### 2. Priority Queue Implementation

```python
class PriorityQueue:
    def __init__(self):
        # Use a min-heap
        self.heap = MinHeap()

    def enqueue(self, item, priority):
        # TODO: Implement enqueue with priority
        pass

    def dequeue(self):
        # TODO: Implement dequeue operation
        pass

    def peek(self):
        # TODO: Return the highest priority item without removing
        pass

    def is_empty(self):
        # TODO: Check if queue is empty
        pass
```

### 3. K Largest Elements

```python
def find_k_largest_elements(arr, k):
    """
    Find the k largest elements in an array using a min-heap
    Time: O(n log k)
    """
    # TODO: Implement using min-heap of size k
    pass

# Test with [3, 1, 4, 1, 5, 9, 2, 6], k=3
# Expected: [9, 6, 5] (or any order of top 3)
```

**Solution:**

```python
def find_k_largest_elements(arr, k):
    import heapq

    # Edge cases
    if k <= 0 or not arr:
        return []

    if k >= len(arr):
        return sorted(arr, reverse=True)

    # Use min-heap of size k
    min_heap = arr[:k]
    heapq.heapify(min_heap)

    for num in arr[k:]:
        if num > min_heap[0]:
            heapq.heapreplace(min_heap, num)

    return sorted(min_heap, reverse=True)
```

### 4. Merge K Sorted Lists

```python
def merge_k_sorted_lists(lists):
    """
    Merge k sorted linked lists into one sorted list
    """
    # TODO: Use min-heap to efficiently merge
    pass
```

**Solution:**

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_sorted_lists(lists):
    # Edge case
    if not lists or not any(lists):
        return None

    min_heap = []

    # Initialize heap with first element of each list
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(min_heap, (node.val, i, node))

    dummy = ListNode(0)
    current = dummy

    while min_heap:
        val, list_idx, node = heapq.heappop(min_heap)
        current.next = ListNode(val)
        current = current.next

        if node.next:
            heapq.heappush(min_heap, (node.next.val, list_idx, node.next))

    return dummy.next
```

### 5. Top K Frequent Elements

```python
def top_k_frequent_elements(nums, k):
    """
    Find the k most frequent elements in an array
    """
    # TODO: Use heap to efficiently find top k
    pass
```

**Solution:**

```python
from collections import Counter
import heapq

def top_k_frequent_elements(nums, k):
    # Count frequencies
    freq_map = Counter(nums)

    # Create max-heap (negate frequencies for min-heap behavior)
    max_heap = [(-freq, num) for num, freq in freq_map.items()]
    heapq.heapify(max_heap)

    # Extract top k
    result = []
    for _ in range(k):
        if max_heap:
            result.append(heapq.heappop(max_heap)[1])

    return result
```

### 6. Median Finder

```python
class MedianFinder:
    def __init__(self):
        # TODO: Use two heaps - max-heap for lower half, min-heap for upper half
        pass

    def add_num(self, num):
        # TODO: Add number and maintain heap balance
        pass

    def find_median(self):
        # TODO: Calculate median from heaps
        pass
```

**Solution:**

```python
import heapq

class MedianFinder:
    def __init__(self):
        # max-heap for lower half (negate values for heapq)
        self.lower = []  # max-heap (negated values)
        # min-heap for upper half
        self.upper = []  # min-heap

    def add_num(self, num):
        if not self.lower or num <= -self.lower[0]:
            heapq.heappush(self.lower, -num)
        else:
            heapq.heappush(self.upper, num)

        # Balance the heaps
        if len(self.lower) > len(self.upper) + 1:
            val = -heapq.heappop(self.lower)
            heapq.heappush(self.upper, val)
        elif len(self.upper) > len(self.lower):
            val = heapq.heappop(self.upper)
            heapq.heappush(self.lower, -val)

    def find_median(self):
        if len(self.lower) > len(self.upper):
            return -self.lower[0]
        elif len(self.upper) > len(self.lower):
            return self.upper[0]
        else:
            return (-self.lower[0] + self.upper[0]) / 2
```

### 7. Heap Sort

```python
def heap_sort(arr):
    """
    Implement heap sort algorithm
    """
    # TODO: Build max-heap and sort the array
    pass
```

**Solution:**

```python
def heap_sort(arr):
    n = len(arr)

    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move current root to end
        heapify(arr, i, 0)  # Call max heapify on reduced heap

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    # If left child exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # If right child exists and is greater than largest so far
    if right < n and arr[right] > arr[largest]:
        largest = right

    # If largest is not root
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)
```

### 8. Sliding Window Maximum

```python
def sliding_window_maximum(nums, k):
    """
    Find maximum in each sliding window of size k
    """
    # TODO: Use deque (deque is faster than heap for this)
    pass
```

**Solution:**

```python
from collections import deque

def sliding_window_maximum(nums, k):
    if not nums or k == 0:
        return []

    if k == 1:
        return nums

    result = []
    dq = deque()  # Store indices

    for i in range(len(nums)):
        # Remove elements outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements from back
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # Add current maximum to result
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

## Advanced Exercises

### 9. Design a Streaming Median

```python
class StreamingMedian:
    def __init__(self):
        # TODO: Implement streaming median data structure
        pass

    def add_number(self, num):
        # TODO: Add number and update median
        pass

    def get_median(self):
        # TODO: Get current median
        pass
```

### 10. Lazy Deletion in Priority Queue

```python
class LazyPriorityQueue:
    def __init__(self):
        # TODO: Implement priority queue with lazy deletion
        self.heap = []
        self.deleted = set()
        self.time = 0

    def insert(self, item, priority):
        # TODO: Insert item with priority
        pass

    def delete(self, item):
        # TODO: Mark item for deletion (lazy deletion)
        pass

    def extract_min(self):
        # TODO: Extract minimum, skipping deleted items
        pass
```

## Time Complexity Analysis

| Operation   | Min-Heap | Max-Heap | Priority Queue |
| ----------- | -------- | -------- | -------------- |
| Insert      | O(log n) | O(log n) | O(log n)       |
| Extract Min | O(log n) | O(log n) | O(log n)       |
| Peek        | O(1)     | O(1)     | O(1)           |
| Delete      | O(log n) | O(log n) | O(log n)       |
| Build Heap  | O(n)     | O(n)     | O(n)           |

## Space Complexity

- **Time**: O(n) for storing n elements
- **Space**: O(n) auxiliary space for heap storage

## Common Applications

1. **Dijkstra's Algorithm**: Finding shortest paths
2. **Prim's Algorithm**: Finding minimum spanning tree
3. **A\* Search**: Pathfinding with heuristic
4. **Event Simulation**: Priority-based event processing
5. **Resource Allocation**: Task scheduling
6. **Data Compression**: Huffman coding

## Interview Questions

### Question 1: Find the Median

**Problem**: Design a data structure that supports addNum and findMedian operations.
**Answer**: Use two heaps - max-heap for lower half, min-heap for upper half.

### Question 2: Merge K Sorted Arrays

**Problem**: Merge k sorted arrays into one sorted array.
**Answer**: Use min-heap of size k, extract minimum and add next element.

### Question 3: Top K Frequent Words

**Problem**: Find the k most frequent words in a document.
**Answer**: Count frequencies, use max-heap with custom comparator.
