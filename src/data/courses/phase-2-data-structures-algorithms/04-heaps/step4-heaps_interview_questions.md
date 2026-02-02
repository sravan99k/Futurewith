# Heaps Interview Questions

## Basic Level Questions

### 1. Implement a Min-Heap

**Question**: Implement a min-heap from scratch supporting insert, extractMin, and getMin operations.

**Answer**:

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

    def get_min(self):
        return self.heap[0] if self.heap else None

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0
```

**Time Complexity**:

- Insert: O(log n)
- Extract Min: O(log n)
- Get Min: O(1)

### 2. Find K Largest Elements

**Question**: Given an array and integer k, return the k largest elements in the array.

**Answer**:

```python
import heapq

def find_k_largest_elements(arr, k):
    if k <= 0 or not arr:
        return []

    if k >= len(arr):
        return sorted(arr, reverse=True)

    # Method 1: Using heapq.nlargest
    return heapq.nlargest(k, arr)

def find_k_largest_manual(arr, k):
    # Method 2: Manual heap approach
    min_heap = arr[:k]
    heapq.heapify(min_heap)

    for num in arr[k:]:
        if num > min_heap[0]:
            heapq.heapreplace(min_heap, num)

    return sorted(min_heap, reverse=True)
```

**Time Complexity**: O(n log k) for manual approach, O(n log k) for built-in

### 3. Merge K Sorted Lists

**Question**: Merge k sorted linked lists into one sorted linked list.

**Answer**:

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists):
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

**Time Complexity**: O(n log k) where n is total nodes

## Intermediate Level Questions

### 4. Design Median Finder

**Question**: Design a data structure that supports addNum and findMedian operations.

**Answer**:

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lower = []  # max-heap (negate values)
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

**Follow-up**: How to handle large data streams?
**Answer**: Use lazy deletion or maintain window of recent elements.

### 5. Top K Frequent Words

**Question**: Given a list of words and integer k, return the k most frequent words.

**Answer**:

```python
from collections import Counter
import heapq

def top_k_frequent(words, k):
    count = Counter(words)
    # Use max-heap (negate frequency for min-heap behavior)
    max_heap = [(-freq, word) for word, freq in count.items()]
    heapq.heapify(max_heap)

    result = []
    for _ in range(k):
        freq, word = heapq.heappop(max_heap)
        result.append(word)

    return result

# Alternative with sorting for tie-breaking
def top_k_frequent_with_ties(words, k):
    count = Counter(words)
    # Sort by frequency (descending) then by word (ascending)
    sorted_words = sorted(count.keys(), key=lambda x: (-count[x], x))
    return sorted_words[:k]
```

**Time Complexity**: O(n log n) for building heap, O(k log n) for extraction

### 6. Sliding Window Maximum

**Question**: Find maximum in each sliding window of size k.

**Answer**:

```python
from collections import deque

def max_sliding_window(nums, k):
    if not nums or k == 0:
        return []

    if k == 1:
        return nums

    dq = deque()  # Store indices
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements from back
        while dq and nums[dq[-1]] <= num:
            dq.pop()

        dq.append(i)

        # Add current maximum to result
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

**Time Complexity**: O(n) - each element enters and leaves deque once

## Advanced Level Questions

### 7. IPO (Leetcode 502)

**Question**: You have initial capital w. You can select up to k distinct projects. Each project requires some capital and yields some profit. Find maximum capital after k projects.

**Answer**:

```python
import heapq

def find_maximized_capital(k, w, profits, capital):
    # Sort projects by required capital
    projects = list(zip(capital, profits))
    projects.sort()

    min_capital = []
    max_profit = []

    i = 0  # Index for projects

    for _ in range(k):
        # Add all projects that we can afford
        while i < len(projects) and projects[i][0] <= w:
            heapq.heappush(max_profit, -projects[i][1])
            i += 1

        # If we can't afford any more projects, break
        if not max_profit:
            break

        # Select the most profitable project
        w += -heapq.heappop(max_profit)

    return w
```

**Time Complexity**: O((n + k) log n) where n is number of projects

### 8. Merge Sorted Arrays

**Question**: Given m sorted arrays, merge them into one sorted array.

**Answer**:

```python
import heapq

def merge_arrays(arrays):
    if not arrays:
        return []

    min_heap = []
    result = []

    # Initialize heap with first element of each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(min_heap, (arr[0], i, 0))

    while min_heap:
        val, arr_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # Add next element from same array
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, arr_idx, elem_idx + 1))

    return result
```

**Follow-up**: What if you have infinite data streams?
**Answer**: Use circular buffer or file-based merging with external sorting.

### 9. K Closest Points to Origin

**Question**: Given n points, find the k points closest to origin.

**Answer**:

```python
import heapq

def k_closest_points(points, k):
    def distance_squared(point):
        return point[0]**2 + point[1]**2

    max_heap = []  # max-heap of size k

    for i, (x, y) in enumerate(points):
        dist = distance_squared((x, y))
        if i < k:
            heapq.heappush(max_heap, (-dist, x, y))
        elif dist < -max_heap[0][0]:
            heapq.heapreplace(max_heap, (-dist, x, y))

    return [(x, y) for _, x, y in max_heap]
```

**Alternative approach**:

```python
def k_closest_points_alternative(points, k):
    # Sort all points by distance (O(n log n))
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    return points[:k]
```

### 10. Design a Heap with Delete

**Question**: Design a heap that supports add, getMin, and delete operations.

**Answer**:

```python
import heapq

class DeletableMinHeap:
    def __init__(self):
        self.heap = []
        self.deleted = set()
        self.deleted_count = 0

    def add(self, val):
        heapq.heappush(self.heap, val)

    def delete(self, val):
        if val in self.heap:
            self.deleted.add(val)
            self.deleted_count += 1

    def get_min(self):
        self._clean()
        return self.heap[0] if self.heap else None

    def extract_min(self):
        self._clean()
        return heapq.heappop(self.heap) if self.heap else None

    def _clean(self):
        # Remove deleted elements lazily
        while self.heap and self.deleted_count > 0:
            val = heapq.heappop(self.heap)
            if val not in self.deleted:
                heapq.heappush(self.heap, val)
            else:
                self.deleted.remove(val)
                self.deleted_count -= 1
```

**Time Complexity**: O(log n) for add, O(log n) for getMin after cleanup

## System Design Questions

### 11. Design Task Scheduler

**Question**: Design a task scheduler that prioritizes tasks by priority and deadline.

**Answer**:

```python
import heapq
from datetime import datetime, timedelta

class TaskScheduler:
    def __init__(self):
        self.tasks = []  # (priority, deadline, task_id, task)
        self.task_id = 0

    def add_task(self, task, priority, deadline):
        heapq.heappush(self.tasks, (-priority, deadline, self.task_id, task))
        self.task_id += 1

    def get_next_task(self):
        if not self.tasks:
            return None

        # Get highest priority task
        priority, deadline, task_id, task = heapq.heappop(self.tasks)
        return task

    def schedule_tasks(self, time_limit):
        scheduled = []
        current_time = datetime.now()

        while self.tasks and len(scheduled) < 100:  # Limit to prevent infinite loop
            priority, deadline, task_id, task = heapq.heappop(self.tasks)

            if deadline > current_time:
                scheduled.append((task, current_time))
                current_time += timedelta(minutes=30)  # Assume 30 min tasks
            else:
                # Task missed deadline, skip or handle differently
                continue

        return scheduled
```

### 12. Design Rate Limiter

**Question**: Design a rate limiter that uses heap to track requests.

**Answer**:

```python
import heapq
import time

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.request_heap = []  # min-heap of timestamps

    def allow_request(self, user_id):
        current_time = time.time()

        # Remove old requests outside time window
        while self.request_heap and self.request_heap[0] < current_time - self.time_window:
            heapq.heappop(self.request_heap)

        # Check if limit reached
        if len(self.request_heap) >= self.max_requests:
            return False, None

        # Allow request
        heapq.heappush(self.request_heap, current_time)
        return True, current_time
```

## Common Follow-up Questions

### Q1: How would you handle duplicates in heaps?

**A**: Use a combination of (value, count) tuples or maintain a frequency map.

### Q2: What if the data is too large to fit in memory?

**A**: Use external sorting and merge k-way or stream processing with window-based heaps.

### Q3: How to implement a median with lazy deletion?

**A**: Maintain a deletion counter and skip marked elements during operations.

### Q4: What's the space complexity of median finder?

**A**: O(1) space (only 2 heaps) vs O(n) for storing all numbers.

### Q5: How to implement a max-heap efficiently?

**A**: Use negative values with min-heap or implement custom max-heap comparison.

## Tips for Interview Success

1. **Start with basic operations** before building complex solutions
2. **Consider edge cases** like empty heaps, single elements
3. **Think about space-time tradeoffs** - when to use heap vs other data structures
4. **Practice common patterns** like k-largest, median, sliding window
5. **Understand heap invariants** and how to maintain them
6. **Consider lazy deletion** for delete operations
7. **Think about streaming** scenarios and how to handle large data
