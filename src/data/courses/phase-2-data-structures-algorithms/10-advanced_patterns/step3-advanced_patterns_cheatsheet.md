---
title: "Advanced Patterns - Quick Reference Cheatsheet"
level: "Advanced"
estimated_time: "45 minutes review"
tags: ["advanced", "patterns", "cheatsheet", "optimization", "templates"]
---

# Advanced Patterns - Quick Reference Cheatsheet

## 🎯 Pattern Selection Guide

### Quick Decision Tree

```python
def choose_pattern(problem_type):
    if "two elements" in problem_type and "sorted array" in problem_type:
        return "Two Pointers"
    elif "subarray/substring" in problem_type:
        return "Sliding Window"
    elif "next greater/smaller" in problem_type:
        return "Monotonic Stack"
    elif "range maximum/minimum" in problem_type:
        return "Monotonic Queue"
    elif "connectivity/groups" in problem_type:
        return "Union-Find"
    elif "XOR/bit operations" in problem_type:
        return "Bit Manipulation"
    elif "optimization with constraints" in problem_type:
        return "Binary Search on Answer"
    else:
        return "Consider combining patterns"
```

---

## 🔥 Essential Pattern Templates

### 1. Two Pointers Templates

#### Convergent Two Pointers

```python
def two_pointers_convergent(arr, target):
    """
    Template for problems like Two Sum, Three Sum.
    Use when array is sorted and looking for pairs.

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1

    return [-1, -1]

# Three Sum pattern
def three_sum_template(nums, target=0):
    """Template for three sum problems."""
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue  # Skip duplicates

        left, right = i + 1, len(nums) - 1
        current_target = target - nums[i]

        while left < right:
            if nums[left] + nums[right] == current_target:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif nums[left] + nums[right] < current_target:
                left += 1
            else:
                right -= 1

    return result
```

#### Fast-Slow Pointers (Floyd's)

```python
def floyd_cycle_detection(head):
    """
    Template for cycle detection in linked lists.

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return None

    slow = fast = head

    # Phase 1: Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow

# Array cycle detection (for problems like "Find Duplicate Number")
def find_duplicate_array(nums):
    """Use array as implicit linked list."""
    slow = fast = nums[0]

    # Find intersection point
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Find cycle start
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow
```

---

### 2. Sliding Window Templates

#### Fixed Size Window

```python
def sliding_window_fixed(arr, k):
    """
    Template for fixed size window problems.

    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return []

    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# With deque for max/min tracking
from collections import deque

def sliding_window_maximum(arr, k):
    """Template for sliding window maximum."""
    dq = deque()  # Store indices
    result = []

    for i in range(len(arr)):
        # Remove out of window elements
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements (monotonic decreasing)
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(arr[dq[0]])

    return result
```

#### Variable Size Window

```python
def sliding_window_variable(s, condition_func):
    """
    Template for variable size window problems.

    Time: O(n), Space: depends on condition
    """
    left = 0
    max_length = 0
    window_state = {}  # Track window state

    for right in range(len(s)):
        # Expand window
        char = s[right]
        window_state = update_state(window_state, char, 1)

        # Contract window while invalid
        while not condition_func(window_state):
            left_char = s[left]
            window_state = update_state(window_state, left_char, -1)
            left += 1

        # Update result
        max_length = max(max_length, right - left + 1)

    return max_length

# Specific example: Longest substring without repeating characters
def longest_unique_substring(s):
    """Longest substring without repeating characters."""
    char_index = {}
    left = 0
    max_length = 0

    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1

        char_index[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

---

### 3. Monotonic Stack Templates

#### Basic Monotonic Stack

```python
def next_greater_element(arr):
    """
    Template for next greater element problems.

    Time: O(n), Space: O(n)
    """
    stack = []  # Store indices
    result = [-1] * len(arr)

    for i in range(len(arr)):
        while stack and arr[stack[-1]] < arr[i]:
            index = stack.pop()
            result[index] = arr[i]
        stack.append(i)

    return result

def previous_smaller_element(arr):
    """Find previous smaller element for each position."""
    stack = []
    result = [-1] * len(arr)

    for i in range(len(arr)):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()

        if stack:
            result[i] = stack[-1]

        stack.append(i)

    return result
```

#### Advanced: Largest Rectangle

```python
def largest_rectangle_histogram(heights):
    """
    Template for largest rectangle problems.

    Time: O(n), Space: O(n)
    """
    stack = []
    max_area = 0

    for i in range(len(heights) + 1):
        current_height = heights[i] if i < len(heights) else 0

        while stack and heights[stack[-1]] > current_height:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area

def trapping_rainwater(height):
    """Trapping rainwater using monotonic stack."""
    stack = []
    trapped = 0

    for current in range(len(height)):
        while stack and height[current] > height[stack[-1]]:
            top = stack.pop()

            if not stack:
                break

            distance = current - stack[-1] - 1
            bounded_height = (min(height[current], height[stack[-1]]) -
                             height[top])
            trapped += distance * bounded_height

        stack.append(current)

    return trapped
```

---

### 4. Union-Find Template

```python
class UnionFind:
    """
    Optimized Union-Find with path compression and union by rank.

    Time: O(α(n)) per operation, Space: O(n)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union by rank."""
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.components -= 1
        return True

    def connected(self, x, y):
        """Check if connected."""
        return self.find(x) == self.find(y)

    def component_count(self):
        """Get number of components."""
        return self.components

# String-based Union-Find
class StringUnionFind:
    """Union-Find for string keys."""
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True
```

---

### 5. Binary Search on Answer Template

```python
def binary_search_answer(left, right, condition_func, find_min=True):
    """
    Template for binary search on answer space.

    Args:
        left, right: Search bounds
        condition_func: Returns True if answer is valid
        find_min: True for leftmost valid, False for rightmost valid

    Time: O(log(right - left) × condition_complexity)
    """
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if condition_func(mid):
            result = mid
            if find_min:
                right = mid - 1  # Look for smaller valid answer
            else:
                left = mid + 1   # Look for larger valid answer
        else:
            if find_min:
                left = mid + 1   # Need larger value
            else:
                right = mid - 1  # Need smaller value

    return result

# Specific applications
def min_capacity_ship_packages(weights, days):
    """Find minimum ship capacity."""
    def can_ship(capacity):
        current_weight = 0
        days_used = 1

        for weight in weights:
            if weight > capacity:
                return False

            if current_weight + weight > capacity:
                days_used += 1
                current_weight = weight
            else:
                current_weight += weight

        return days_used <= days

    return binary_search_answer(max(weights), sum(weights), can_ship, True)
```

---

### 6. Bit Manipulation Templates

#### Core Bit Operations

```python
class BitOps:
    """Essential bit manipulation operations."""

    @staticmethod
    def get_bit(num, i):
        """Get ith bit (0-indexed from right)."""
        return (num & (1 << i)) != 0

    @staticmethod
    def set_bit(num, i):
        """Set ith bit to 1."""
        return num | (1 << i)

    @staticmethod
    def clear_bit(num, i):
        """Clear ith bit to 0."""
        return num & ~(1 << i)

    @staticmethod
    def toggle_bit(num, i):
        """Toggle ith bit."""
        return num ^ (1 << i)

    @staticmethod
    def count_set_bits(num):
        """Count number of 1s (Brian Kernighan's algorithm)."""
        count = 0
        while num:
            num &= num - 1  # Remove rightmost set bit
            count += 1
        return count

    @staticmethod
    def is_power_of_two(num):
        """Check if number is power of 2."""
        return num > 0 and (num & (num - 1)) == 0

# XOR patterns
def find_single_number(nums):
    """Single number (others appear twice)."""
    result = 0
    for num in nums:
        result ^= num
    return result

def find_two_single_numbers(nums):
    """Two single numbers (others appear twice)."""
    xor = 0
    for num in nums:
        xor ^= num

    # Find rightmost set bit
    rightmost_bit = xor & (-xor)

    num1 = num2 = 0
    for num in nums:
        if num & rightmost_bit:
            num1 ^= num
        else:
            num2 ^= num

    return [num1, num2]

def maximum_xor_two_numbers(nums):
    """Maximum XOR of any two numbers."""
    max_xor = 0
    mask = 0

    for i in range(30, -1, -1):
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}
        temp = max_xor | (1 << i)

        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break

    return max_xor
```

---

### 7. Mathematical Algorithm Templates

#### Fast Exponentiation

```python
def fast_power(base, exp, mod=None):
    """
    Fast exponentiation using binary method.

    Time: O(log exp), Space: O(1)
    """
    result = 1
    base = base % mod if mod else base

    while exp > 0:
        if exp & 1:
            result = (result * base) % mod if mod else result * base
        exp >>= 1
        base = (base * base) % mod if mod else base * base

    return result

def matrix_multiply(A, B, mod=None):
    """Matrix multiplication with optional modulo."""
    n = len(A)
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                product = A[i][k] * B[k][j]
                result[i][j] = (result[i][j] + product) % mod if mod else result[i][j] + product

    return result

def matrix_power(matrix, n, mod=None):
    """Fast matrix exponentiation."""
    size = len(matrix)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    while n > 0:
        if n & 1:
            result = matrix_multiply(result, matrix, mod)
        matrix = matrix_multiply(matrix, matrix, mod)
        n >>= 1

    return result

def fibonacci_fast(n):
    """Fibonacci using matrix exponentiation."""
    if n <= 1:
        return n

    fib_matrix = [[1, 1], [1, 0]]
    result = matrix_power(fib_matrix, n - 1)
    return result[0][0]
```

#### Number Theory

```python
def gcd(a, b):
    """Greatest Common Divisor."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least Common Multiple."""
    return (a * b) // gcd(a, b)

def sieve_of_eratosthenes(n):
    """Find all primes up to n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(2, n + 1) if is_prime[i]]

def extended_gcd(a, b):
    """Extended Euclidean Algorithm."""
    if a == 0:
        return b, 0, 1

    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1

    return gcd, x, y
```

---

## ⚡ Pattern Combinations

### Multi-Pattern Solutions

```python
# Sliding Window + Monotonic Queue
def sliding_window_median(nums, k):
    """Median in sliding window using balanced heaps."""
    import heapq
    from collections import defaultdict

    max_heap = []  # Left half (negated)
    min_heap = []  # Right half
    balance = 0
    hash_table = defaultdict(int)
    result = []

    def add_number(num):
        nonlocal balance
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
            balance += 1
        else:
            heapq.heappush(min_heap, num)
            balance -= 1
        rebalance()

    def remove_number(num):
        nonlocal balance
        hash_table[num] += 1
        if num <= -max_heap[0]:
            balance -= 1
        else:
            balance += 1
        rebalance()

    def rebalance():
        nonlocal balance
        if balance > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
            balance -= 2
        elif balance < -1:
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
            balance += 2

        prune_heaps()

    def prune_heaps():
        while max_heap and hash_table[-max_heap[0]] > 0:
            hash_table[-max_heap[0]] -= 1
            heapq.heappop(max_heap)

        while min_heap and hash_table[min_heap[0]] > 0:
            hash_table[min_heap[0]] -= 1
            heapq.heappop(min_heap)

    def get_median():
        if k & 1:
            return float(-max_heap[0])
        else:
            return (-max_heap[0] + min_heap[0]) / 2.0

    for i, num in enumerate(nums):
        add_number(num)
        if i >= k:
            remove_number(nums[i - k])
        if i >= k - 1:
            result.append(get_median())

    return result
```

---

## 🚨 Common Patterns & Pitfalls

### Pattern Recognition Checklist

- **Two Pointers**: ✅ Sorted array ✅ Find pairs/triplets ✅ Cycle detection
- **Sliding Window**: ✅ Subarray/substring ✅ Optimization ✅ Fixed/variable size
- **Monotonic Stack**: ✅ Next greater/smaller ✅ Histogram problems ✅ O(n) requirement
- **Union-Find**: ✅ Connectivity ✅ Dynamic grouping ✅ MST problems
- **Binary Search**: ✅ Sorted search space ✅ Optimization problems ✅ "Minimum/maximum X such that..."

### Common Mistakes

#### ❌ Two Pointers Errors

```python
# WRONG: Not handling duplicates in Three Sum
def three_sum_wrong(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        left, right = i + 1, len(nums) - 1
        while left < right:
            # Missing duplicate handling
            if nums[i] + nums[left] + nums[right] == 0:
                result.append([nums[i], nums[left], nums[right]])
            left += 1
            right -= 1

# CORRECT: Handle duplicates
def three_sum_correct(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue  # Skip duplicates for first element
        # ... rest with proper duplicate handling
```

#### ❌ Sliding Window Errors

```python
# WRONG: Not updating window state correctly
def sliding_window_wrong(s):
    left = 0
    for right in range(len(s)):
        # Add s[right] to window
        while invalid_condition():
            # Forgot to remove s[left] from window state
            left += 1

# CORRECT: Maintain window state properly
def sliding_window_correct(s):
    left = 0
    window_state = {}
    for right in range(len(s)):
        add_to_window(s[right])
        while invalid_condition():
            remove_from_window(s[left])
            left += 1
```

---

## 🎯 Complexity Quick Reference

| Pattern                 | Time Complexity       | Space Complexity | Best Use Cases                  |
| ----------------------- | --------------------- | ---------------- | ------------------------------- |
| Two Pointers            | O(n) - O(n²)          | O(1)             | Sorted arrays, cycle detection  |
| Sliding Window          | O(n)                  | O(1) - O(k)      | Subarray/substring optimization |
| Monotonic Stack         | O(n)                  | O(n)             | Next greater/smaller problems   |
| Union-Find              | O(α(n)) per op        | O(n)             | Dynamic connectivity            |
| Binary Search on Answer | O(log(range) × check) | O(1)             | Optimization problems           |
| Bit Manipulation        | O(1) - O(n)           | O(1)             | XOR, subset problems            |
| Matrix Exponentiation   | O(k³ log n)           | O(k²)            | Linear recurrences              |

---

## 💡 Interview Strategy

### Pattern Selection in 30 Seconds

1. **Read problem twice** - identify key characteristics
2. **Look for keywords**: pairs, subarray, next greater, connectivity, optimization
3. **Check constraints**: sorted array, window size, bit operations
4. **Choose primary pattern** - start with most obvious
5. **Consider combinations** - many hard problems use 2+ patterns

### Implementation Order

1. **Basic solution** - get correctness first
2. **Add optimizations** - apply pattern optimizations
3. **Handle edge cases** - empty input, single element, etc.
4. **Test with examples** - verify with given test cases

### Time Management

- **5 min**: Pattern recognition and approach
- **15 min**: Core implementation
- **5 min**: Edge cases and testing

---

_This cheatsheet covers the most critical advanced patterns for interviews. Master these templates and you'll recognize 90% of advanced algorithm problems!_
