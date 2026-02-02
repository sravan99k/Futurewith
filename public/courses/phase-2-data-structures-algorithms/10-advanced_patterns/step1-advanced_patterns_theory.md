---
title: "Advanced Patterns - Theory and Fundamentals"
level: "Advanced"
estimated_time: "8-10 hours"
prerequisites:
  [
    Basic algorithms,
    Arrays,
    Strings,
    Trees,
    Graphs,
    Time complexity analysis,
    Problem-solving techniques,
  ]
skills_gained:
  [
    Two-pointer techniques,
    Sliding window algorithms,
    Monotonic stacks,
    Advanced optimization,
    Pattern recognition,
    Problem decomposition,
    Interview-ready skills,
  ]
success_criteria:
  [
    "Apply two-pointer techniques to optimize O(n²) to O(n) problems",
    "Implement sliding window for subarray/substring optimization",
    "Use monotonic stacks for next greater/smaller element problems",
    "Recognize and combine multiple patterns for complex solutions",
    "Analyze when to apply each advanced pattern optimally",
    "Solve FAANG-level problems using advanced pattern combinations",
  ]
tags:
  [
    "advanced",
    "patterns",
    "algorithms",
    "optimization",
    "two-pointers",
    "sliding-window",
  ]
version: 1.0
last_updated: 2025-11-11
---

# Advanced Patterns - Theory and Fundamentals

## Learning Goals

By the end of this comprehensive guide, you will be able to:

- Master two-pointer techniques for O(n) optimization of O(n²) problems
- Implement sliding window algorithms for subarray and substring problems
- Apply monotonic stacks and queues for next greater/smaller element analysis
- Combine multiple patterns to solve complex multi-step problems
- Recognize pattern signatures in unfamiliar problem statements
- Analyze time and space complexity of advanced pattern implementations
- Develop algorithmic intuition for choosing optimal patterns
- Apply advanced patterns to real-world optimization challenges

## TL;DR

Advanced patterns are optimized algorithmic techniques that transform inefficient solutions into efficient ones. Two-pointer techniques use multiple indices to avoid nested loops, sliding windows maintain dynamic ranges for sequence problems, and monotonic structures maintain order relationships. These patterns can reduce O(n²) solutions to O(n) and O(n³) to O(n²), making the difference between acceptable and unacceptable solutions.

## Why This Matters

These patterns separate good programmers from great engineers. FAANG companies specifically design interview questions to test advanced pattern recognition - problems that seem impossible to candidates unfamiliar with these techniques become trivial once you recognize the underlying pattern. Beyond interviews, these patterns power the optimizations in every high-performance system: database query optimization, machine learning feature engineering, and real-time data processing all rely on these advanced techniques.

## Common Confusions & Mistakes

- **Confusion: "When to use which pattern"** — Two-pointers work when you need to traverse from both ends, sliding window when you need contiguous subsequences, monotonic stacks when you need next greater/smaller elements.

- **Confusion: "Pattern vs Algorithm"** — Patterns are mental frameworks for recognizing problem types, algorithms are the specific implementations; patterns help you choose the right algorithm.

- **Confusion: "Combination of patterns"** — Real problems often require multiple patterns combined; don't force a single pattern when multiple are needed.

- **Quick Debug Tip:** For pattern problems, first identify the input constraints and operation requirements, then match the pattern to the problem signature rather than trying random approaches.

- **Performance Pitfall:** Always verify that your pattern application actually improves complexity; incorrect pattern application can make solutions worse than brute force.

- **Implementation Error:** For sliding windows, be careful with window boundaries and off-by-one errors; always test with edge cases (empty windows, single elements).

- **Pattern Recognition:** Don't try to memorize patterns by heart; learn to recognize the underlying problem structure that suggests each pattern.

- **Complexity Analysis:** Pattern applications can create subtle time complexity changes; always verify your actual runtime matches your theoretical analysis.

## Micro-Quiz (80% mastery required)

1. **Q:** What type of problems are best solved with two-pointer technique? **A:** Problems involving sorted arrays, subarray/substring analysis, or when you need to find pairs/triples that satisfy conditions.

2. **Q:** How do you know when to use a sliding window vs two-pointers? **A:** Sliding window for contiguous subsequences with variable length, two-pointers for problems that need to process from both ends or when you need fixed-size windows.

3. **Q:** What makes a stack "monotonic" and when is it useful? **A:** A monotonic stack maintains elements in increasing or decreasing order; useful for next greater/smaller element problems and range maximum queries.

4. **Q:** How do you combine multiple patterns effectively? **A:** Identify the primary pattern that addresses the main problem, then use secondary patterns for sub-problems or optimization steps.

5. **Q:** What's the key difference between expanding and contracting windows? **A:** Expanding windows (like in maximum subarray) grow the window, contracting windows (like in minimum window substring) shrink the window while maintaining constraints.

## Reflection Prompts

- **Pattern Recognition:** How would you develop the ability to quickly identify which pattern applies to a new algorithmic problem?

- **Optimization Strategy:** What process would you use to systematically optimize a brute force solution using advanced patterns?

- **Real-world Application:** How do the patterns you learned apply to performance optimization in real software systems you use daily?

_Master patterns that transform complexity! 🎯_

---

## 🔍 Pattern Overview

Advanced patterns represent the evolution of basic algorithmic thinking into sophisticated, optimized approaches. They combine fundamental concepts with clever insights to achieve dramatic performance improvements.

### Core Advanced Patterns

1. **Two Pointers**: Converging and diverging pointer strategies
2. **Sliding Window**: Dynamic window management for sequence problems
3. **Monotonic Stack/Queue**: Maintaining order for optimization
4. **Union-Find**: Efficient disjoint set operations
5. **Topological Sort**: Advanced dependency resolution
6. **Binary Search on Answer**: Search space transformation
7. **Bit Manipulation**: Efficient operations on binary representations
8. **Mathematical Algorithms**: Number theory and combinatorics
9. **Advanced Tree Patterns**: Specialized tree algorithms
10. **State Machine**: Complex state transition modeling

---

## 🎯 Pattern 1: Two Pointers

### Concept

Use two pointers moving through data structure to solve problems efficiently that would otherwise require nested loops.

### Types of Two Pointers

#### 1. Convergent Two Pointers

```python
def two_sum_sorted(arr, target):
    """
    Find two numbers that sum to target in sorted array.
    Classic convergent two pointers pattern.

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum

    return [-1, -1]

# Advanced: Three Sum with two pointers
def three_sum(nums):
    """
    Find all unique triplets that sum to zero.
    Combines sorting with two pointers.

    Time: O(n²), Space: O(1)
    """
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, n - 1
        target = -nums[i]

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return result
```

#### 2. Fast-Slow Pointers (Floyd's Algorithm)

```python
def detect_cycle(head):
    """
    Detect cycle in linked list using Floyd's tortoise and hare.

    Time: O(n), Space: O(1)
    """
    if not head or not head.next:
        return False

    slow = fast = head

    # Phase 1: Detect if cycle exists
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            break
    else:
        return False  # No cycle

    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow  # Start of cycle

# Advanced application: Find duplicate number
def find_duplicate(nums):
    """
    Find duplicate in array of n+1 integers (1 to n).
    Uses array as implicit linked list.

    Time: O(n), Space: O(1)
    """
    # Phase 1: Detect cycle (duplicate creates cycle)
    slow = fast = nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Phase 2: Find cycle entrance (duplicate number)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow
```

#### 3. Divergent Two Pointers

```python
def expand_around_centers(s):
    """
    Find all palindromic substrings by expanding around centers.

    Time: O(n²), Space: O(1)
    """
    def expand_around_center(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    total = 0

    for i in range(len(s)):
        # Odd length palindromes
        total += expand_around_center(i, i)
        # Even length palindromes
        total += expand_around_center(i, i + 1)

    return total
```

---

## 🎯 Pattern 2: Sliding Window

### Concept

Maintain a window of elements and slide it across the data structure, adjusting window size based on conditions.

### Types of Sliding Window

#### 1. Fixed Size Window

```python
def max_sum_subarray(arr, k):
    """
    Find maximum sum of subarray of size k.
    Classic fixed window pattern.

    Time: O(n), Space: O(1)
    """
    if len(arr) < k:
        return -1

    # Calculate first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide window
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Advanced: Maximum of all subarrays of size k
def max_sliding_window(nums, k):
    """
    Find maximum element in each sliding window.
    Uses deque for optimization.

    Time: O(n), Space: O(k)
    """
    from collections import deque

    dq = deque()  # Store indices
    result = []

    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements (they'll never be max)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

#### 2. Variable Size Window

```python
def longest_substring_without_repeat(s):
    """
    Find longest substring without repeating characters.
    Classic variable window pattern.

    Time: O(n), Space: O(min(m,n)) where m = charset size
    """
    char_map = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        char = s[right]

        if char in char_map and char_map[char] >= left:
            # Character repeats in current window
            left = char_map[char] + 1

        char_map[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length

# Advanced: Minimum window substring
def min_window_substring(s, t):
    """
    Find minimum window in s that contains all characters of t.

    Time: O(|s| + |t|), Space: O(|s| + |t|)
    """
    from collections import Counter, defaultdict

    if not s or not t or len(s) < len(t):
        return ""

    # Count characters in t
    target_count = Counter(t)
    required = len(target_count)

    # Sliding window
    left = right = 0
    formed = 0  # Number of unique chars in window with desired frequency
    window_counts = defaultdict(int)

    # Result
    min_len = float('inf')
    min_left = 0

    while right < len(s):
        # Expand window
        char = s[right]
        window_counts[char] += 1

        if char in target_count and window_counts[char] == target_count[char]:
            formed += 1

        # Contract window
        while left <= right and formed == required:
            char = s[left]

            # Update result if smaller window found
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left

            # Remove from left
            window_counts[char] -= 1
            if char in target_count and window_counts[char] < target_count[char]:
                formed -= 1

            left += 1

        right += 1

    return s[min_left:min_left + min_len] if min_len != float('inf') else ""
```

---

## 🎯 Pattern 3: Monotonic Stack/Queue

### Concept

Maintain stack or queue in monotonic (increasing/decreasing) order to efficiently solve problems involving "next greater/smaller" elements.

#### Monotonic Stack Applications

```python
def next_greater_elements(arr):
    """
    Find next greater element for each array element.
    Classic monotonic stack pattern.

    Time: O(n), Space: O(n)
    """
    stack = []  # Indices
    result = [-1] * len(arr)

    for i in range(len(arr)):
        # Pop elements smaller than current
        while stack and arr[stack[-1]] < arr[i]:
            index = stack.pop()
            result[index] = arr[i]

        stack.append(i)

    return result

# Advanced: Largest rectangle in histogram
def largest_rectangle_histogram(heights):
    """
    Find largest rectangle area in histogram.
    Uses monotonic stack for optimization.

    Time: O(n), Space: O(n)
    """
    stack = []  # Indices
    max_area = 0

    for i in range(len(heights) + 1):  # +1 to process remaining stack
        current_height = heights[i] if i < len(heights) else 0

        while stack and heights[stack[-1]] > current_height:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area

# Advanced: Trapping rainwater
def trap_rainwater(height):
    """
    Calculate trapped rainwater using monotonic stack.

    Time: O(n), Space: O(n)
    """
    stack = []  # Indices
    trapped = 0

    for current in range(len(height)):
        while stack and height[current] > height[stack[-1]]:
            top = stack.pop()

            if not stack:
                break

            distance = current - stack[-1] - 1
            bounded_height = min(height[current], height[stack[-1]]) - height[top]
            trapped += distance * bounded_height

        stack.append(current)

    return trapped
```

#### Monotonic Queue (Deque) Applications

```python
def sliding_window_maximum_optimized(nums, k):
    """
    Sliding window maximum using monotonic deque.
    More efficient than heap-based approach.

    Time: O(n), Space: O(k)
    """
    from collections import deque

    dq = deque()  # Store indices in decreasing order of values
    result = []

    for i in range(len(nums)):
        # Remove indices outside window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Maintain monotonic decreasing order
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # Add maximum to result
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## 🎯 Pattern 4: Union-Find (Disjoint Set Union)

### Concept

Efficiently handle dynamic connectivity queries and union operations on disjoint sets.

#### Basic Implementation

```python
class UnionFind:
    """
    Union-Find with path compression and union by rank.

    Time: O(α(n)) amortized per operation where α is inverse Ackermann
    Space: O(n)
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union by rank."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already connected

        # Union by rank
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
        """Check if two elements are in same component."""
        return self.find(x) == self.find(y)

    def count_components(self):
        """Return number of connected components."""
        return self.components

# Advanced applications
def number_of_islands(grid):
    """
    Count islands using Union-Find.
    Alternative to DFS/BFS approach.

    Time: O(m×n×α(m×n)), Space: O(m×n)
    """
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)

    def get_index(r, c):
        return r * cols + c

    # Count initial water cells
    water_cells = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '0':
                water_cells += 1
            else:
                # Connect to adjacent land cells
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        grid[nr][nc] == '1'):
                        uf.union(get_index(r, c), get_index(nr, nc))

    return uf.count_components() - water_cells

# Kruskal's MST using Union-Find
def kruskal_mst(edges, n):
    """
    Find Minimum Spanning Tree using Kruskal's algorithm.

    Time: O(E log E), Space: O(V)
    """
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, weight in edges:
        if uf.union(u, v):  # Not in same component
            mst.append((u, v, weight))
            total_weight += weight

            if len(mst) == n - 1:  # MST complete
                break

    return mst, total_weight
```

---

## 🎯 Pattern 5: Advanced Binary Search

### Concept

Binary search on answer space rather than traditional sorted arrays.

#### Binary Search on Answer

```python
def search_answer_space(condition_func, left, right):
    """
    Generic binary search on answer space.
    Find leftmost/rightmost value satisfying condition.

    Time: O(log(right - left) × condition_complexity)
    """
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if condition_func(mid):
            result = mid
            right = mid - 1  # Search for smaller valid answer
        else:
            left = mid + 1

    return result

# Application: Capacity to ship packages
def ship_within_days(weights, D):
    """
    Find minimum ship capacity to ship all packages within D days.

    Time: O(n × log(sum_weights)), Space: O(1)
    """
    def can_ship_with_capacity(capacity):
        days_needed = 1
        current_load = 0

        for weight in weights:
            if weight > capacity:
                return False  # Single package exceeds capacity

            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight

        return days_needed <= D

    left = max(weights)  # Minimum possible capacity
    right = sum(weights)  # Maximum possible capacity

    return search_answer_space(can_ship_with_capacity, left, right)

# Advanced: Kth smallest in multiplication table
def kth_smallest_multiplication(m, n, k):
    """
    Find kth smallest number in m×n multiplication table.

    Time: O(m × log(m×n)), Space: O(1)
    """
    def count_less_equal(target):
        count = 0
        for i in range(1, m + 1):
            count += min(target // i, n)
        return count

    left, right = 1, m * n

    while left < right:
        mid = left + (right - left) // 2

        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid

    return left
```

---

## 🎯 Pattern 6: Bit Manipulation

### Concept

Efficient operations using binary representations and bitwise operations.

#### Core Bit Operations

```python
class BitManipulation:
    """Collection of essential bit manipulation techniques."""

    @staticmethod
    def count_set_bits(n):
        """Count number of 1s in binary representation."""
        # Method 1: Brian Kernighan's algorithm
        count = 0
        while n:
            n &= n - 1  # Remove rightmost set bit
            count += 1
        return count

        # Method 2: Built-in
        # return bin(n).count('1')

    @staticmethod
    def is_power_of_two(n):
        """Check if n is power of 2."""
        return n > 0 and (n & (n - 1)) == 0

    @staticmethod
    def find_single_number(arr):
        """Find single number in array where all others appear twice."""
        result = 0
        for num in arr:
            result ^= num  # XOR cancels out duplicates
        return result

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
        """Set ith bit to 0."""
        return num & ~(1 << i)

    @staticmethod
    def toggle_bit(num, i):
        """Toggle ith bit."""
        return num ^ (1 << i)

# Advanced application: Maximum XOR of two numbers
def find_maximum_xor(nums):
    """
    Find maximum XOR of any two numbers in array.
    Uses bit manipulation trie concept.

    Time: O(n), Space: O(n)
    """
    max_xor = 0
    mask = 0

    # Process bits from MSB to LSB
    for i in range(30, -1, -1):
        mask |= (1 << i)  # Update mask to include current bit
        prefixes = {num & mask for num in nums}

        temp = max_xor | (1 << i)  # Try to set current bit in result

        # Check if this bit can be achieved
        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break

    return max_xor

# Subset generation using bit manipulation
def generate_all_subsets(arr):
    """
    Generate all subsets using bit manipulation.

    Time: O(n × 2^n), Space: O(1) excluding output
    """
    n = len(arr)
    subsets = []

    # Generate all numbers from 0 to 2^n - 1
    for i in range(1 << n):  # 2^n possible subsets
        subset = []

        for j in range(n):
            if i & (1 << j):  # Check if jth bit is set
                subset.append(arr[j])

        subsets.append(subset)

    return subsets
```

---

## 🎯 Pattern 7: Mathematical Algorithms

### Concept

Algorithms based on mathematical insights and number theory.

#### Number Theory

```python
def gcd(a, b):
    """Greatest Common Divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least Common Multiple."""
    return (a * b) // gcd(a, b)

def sieve_of_eratosthenes(n):
    """
    Find all prime numbers up to n.

    Time: O(n log log n), Space: O(n)
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples as composite
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(2, n + 1) if is_prime[i]]

def fast_power(base, exp, mod=None):
    """
    Fast exponentiation using binary exponentiation.

    Time: O(log exp), Space: O(1)
    """
    result = 1
    base = base % mod if mod else base

    while exp > 0:
        if exp & 1:  # If exp is odd
            result = (result * base) % mod if mod else result * base

        exp >>= 1  # exp = exp // 2
        base = (base * base) % mod if mod else base * base

    return result

# Advanced: Matrix exponentiation
def matrix_multiply(A, B, mod=None):
    """Multiply two matrices with optional modulo."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        raise ValueError("Matrix dimensions don't match")

    result = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                product = A[i][k] * B[k][j]
                result[i][j] = (result[i][j] + product) % mod if mod else result[i][j] + product

    return result

def matrix_power(matrix, exp, mod=None):
    """
    Fast matrix exponentiation.
    Useful for solving linear recurrences.

    Time: O(n³ log exp), Space: O(n²)
    """
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # Identity matrix

    while exp > 0:
        if exp & 1:
            result = matrix_multiply(result, matrix, mod)
        matrix = matrix_multiply(matrix, matrix, mod)
        exp >>= 1

    return result

# Application: nth Fibonacci number in O(log n)
def fibonacci_fast(n):
    """
    Calculate nth Fibonacci number using matrix exponentiation.

    Time: O(log n), Space: O(1)
    """
    if n <= 1:
        return n

    # Fibonacci transition matrix
    fib_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(fib_matrix, n - 1)

    return result_matrix[0][0]
```

---

## 🎯 Pattern 8: Advanced Tree Patterns

### Concept

Sophisticated tree algorithms beyond basic traversals.

#### Tree Diameter and Path Problems

```python
def tree_diameter(root):
    """
    Find diameter of binary tree (longest path between any two nodes).

    Time: O(n), Space: O(h) where h is height
    """
    max_diameter = 0

    def max_depth(node):
        nonlocal max_diameter

        if not node:
            return 0

        left_depth = max_depth(node.left)
        right_depth = max_depth(node.right)

        # Update diameter (path through current node)
        max_diameter = max(max_diameter, left_depth + right_depth)

        return 1 + max(left_depth, right_depth)

    max_depth(root)
    return max_diameter

def binary_tree_maximum_path_sum(root):
    """
    Find maximum path sum in binary tree.
    Path can start and end at any nodes.

    Time: O(n), Space: O(h)
    """
    max_sum = float('-inf')

    def max_gain(node):
        nonlocal max_sum

        if not node:
            return 0

        # Maximum gain from left and right subtrees
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        # Maximum path sum through current node
        current_max = node.val + left_gain + right_gain
        max_sum = max(max_sum, current_max)

        # Return maximum gain if we include current node in path
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum

# Lowest Common Ancestor with parent pointers
def lca_with_parent(p, q):
    """
    Find LCA when nodes have parent pointers.

    Time: O(h), Space: O(1)
    """
    # Get depths
    def get_depth(node):
        depth = 0
        while node.parent:
            node = node.parent
            depth += 1
        return depth

    depth_p = get_depth(p)
    depth_q = get_depth(q)

    # Bring both nodes to same level
    while depth_p > depth_q:
        p = p.parent
        depth_p -= 1

    while depth_q > depth_p:
        q = q.parent
        depth_q -= 1

    # Move both up until they meet
    while p != q:
        p = p.parent
        q = q.parent

    return p
```

---

## 🚀 Pattern Combinations

### Multi-Pattern Solutions

Advanced problems often require combining multiple patterns:

```python
def sliding_window_median(nums, k):
    """
    Find median of all sliding windows of size k.
    Combines sliding window + balanced data structure.

    Time: O(n log k), Space: O(k)
    """
    import heapq
    from collections import defaultdict

    max_heap = []  # For smaller half (negated values)
    min_heap = []  # For larger half
    balance = 0    # max_heap_size - min_heap_size
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

        # Rebalance if needed
        if balance > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
            balance -= 2
        elif balance < -1:
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
            balance += 2

    def remove_number(num):
        nonlocal balance
        hash_table[num] += 1

        if num <= -max_heap[0]:
            balance -= 1
            if num == -max_heap[0]:
                prune_heap(max_heap, True)
        else:
            balance += 1
            if num == min_heap[0]:
                prune_heap(min_heap, False)

        # Rebalance
        if balance > 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
            balance -= 2
            prune_heap(max_heap, True)
        elif balance < -1:
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
            balance += 2
            prune_heap(min_heap, False)

    def prune_heap(heap, is_max_heap):
        while heap:
            top = -heap[0] if is_max_heap else heap[0]
            if hash_table[top] > 0:
                hash_table[top] -= 1
                heapq.heappop(heap)
            else:
                break

    def get_median():
        if k & 1:  # Odd k
            return float(-max_heap[0])
        else:  # Even k
            return (-max_heap[0] + min_heap[0]) / 2

    # Process each window
    for i, num in enumerate(nums):
        add_number(num)

        if i >= k:
            remove_number(nums[i - k])

        if i >= k - 1:
            result.append(get_median())

    return result
```

---

## 📊 Pattern Selection Guide

### Decision Framework

```python
def choose_pattern(problem_characteristics):
    """
    Decision tree for pattern selection based on problem characteristics.
    """

    if problem_characteristics.has_sorted_array:
        if problem_characteristics.needs_two_elements:
            return "Two Pointers"
        elif problem_characteristics.search_space:
            return "Binary Search"

    if problem_characteristics.has_sequence:
        if problem_characteristics.fixed_window:
            return "Sliding Window (Fixed)"
        elif problem_characteristics.variable_window:
            return "Sliding Window (Variable)"

    if problem_characteristics.needs_next_greater_smaller:
        return "Monotonic Stack"

    if problem_characteristics.needs_maximum_in_window:
        return "Monotonic Queue"

    if problem_characteristics.has_connectivity:
        return "Union-Find"

    if problem_characteristics.has_bits:
        return "Bit Manipulation"

    if problem_characteristics.has_cycles:
        return "Floyd's Algorithm"

    if problem_characteristics.needs_optimization_space:
        return "Binary Search on Answer"

    return "Consider combining multiple patterns"
```

---

## 🎯 Summary

### When to Use Each Pattern

| Pattern                 | Best For                      | Time Complexity        | Space Complexity |
| ----------------------- | ----------------------------- | ---------------------- | ---------------- |
| Two Pointers            | Sorted arrays, pairs/triplets | O(n) - O(n²)           | O(1)             |
| Sliding Window          | Subarray/substring problems   | O(n)                   | O(1) - O(k)      |
| Monotonic Stack         | Next greater/smaller problems | O(n)                   | O(n)             |
| Union-Find              | Connectivity, MST, components | O(α(n)) per op         | O(n)             |
| Binary Search on Answer | Optimization problems         | O(log(answer) × check) | O(1)             |
| Bit Manipulation        | Set operations, XOR problems  | O(1) - O(n)            | O(1)             |
| Math Algorithms         | Number theory, combinatorics  | Varies                 | Varies           |

### Key Insights

1. **Pattern Recognition**: Most complex problems combine 2-3 patterns
2. **Optimization Focus**: Advanced patterns prioritize time complexity optimization
3. **Space-Time Tradeoffs**: Often sacrifice space for time improvements
4. **Problem Transformation**: Many patterns involve viewing problems differently

---

## 🏃‍♂️ Mini Sprint Project: Advanced Pattern Problem Solver

**Time Required:** 30-45 minutes  
**Difficulty:** Advanced  
**Skills Practiced:** Pattern recognition, two-pointer techniques, sliding window, complexity optimization

### Project Overview

Build a comprehensive problem solver that identifies and applies the optimal advanced pattern to solve algorithmic challenges efficiently.

### Core Requirements

1. **Pattern Detection Engine**
   - Analyze problem characteristics (sorted arrays, sequences, constraints)
   - Automatically suggest the most appropriate pattern
   - Provide reasoning for pattern selection

2. **Multi-Pattern Implementation**
   - Implement solutions for all major patterns
   - Support pattern combination for complex problems
   - Handle edge cases and optimization trade-offs

3. **Performance Analysis**
   - Compare brute force vs optimized solutions
   - Validate time complexity improvements
   - Generate performance reports

### Starter Code

```python
from typing import List, Tuple, Any
import time

class PatternSolver:
    def __init__(self):
        self.patterns = {
            'two_pointers': TwoPointerSolver(),
            'sliding_window': SlidingWindowSolver(),
            'monotonic_stack': MonotonicStackSolver(),
            'binary_search': BinarySearchSolver()
        }

    def analyze_problem(self, problem_type: str, input_data: Any) -> dict:
        """Analyze problem to suggest optimal pattern"""
        analysis = {
            'suggested_pattern': None,
            'reasoning': '',
            'complexity': {'time': '?', 'space': '?'},
            'implementation_notes': []
        }

        # Pattern detection logic based on problem characteristics
        if self._is_sorted_array_problem(problem_type, input_data):
            analysis['suggested_pattern'] = 'two_pointers'
            analysis['complexity']['time'] = 'O(n) vs O(n²) brute force'

        elif self._is_subarray_problem(problem_type, input_data):
            analysis['suggested_pattern'] = 'sliding_window'
            analysis['complexity']['time'] = 'O(n) vs O(n²) brute force'

        return analysis

    def solve_with_pattern(self, problem: dict, pattern: str) -> Any:
        """Solve problem using specified pattern"""
        if pattern not in self.patterns:
            raise ValueError(f"Pattern {pattern} not implemented")

        solver = self.patterns[pattern]
        return solver.solve(problem)

class TwoPointerSolver:
    def solve(self, problem: dict) -> Any:
        """Implement two-pointer solution"""
        if problem['type'] == 'two_sum_sorted':
            return self._two_sum_sorted(problem['data'])
        elif problem['type'] == 'pair_with_sum':
            return self._pair_with_sum(problem['data'])
        # Add more two-pointer problems

    def _two_sum_sorted(self, arr: List[int], target: int) -> List[int]:
        """Two pointer solution for sorted array"""
        left, right = 0, len(arr) - 1
        while left < right:
            current_sum = arr[left] + arr[right]
            if current_sum == target:
                return [left, right]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        return []

class SlidingWindowSolver:
    def solve(self, problem: dict) -> Any:
        """Implement sliding window solution"""
        if problem['type'] == 'max_sum_subarray':
            return self._max_sum_subarray(problem['data'])
        elif problem['type'] == 'longest_substring':
            return self._longest_substring_no_repeat(problem['data'])

    def _max_sum_subarray(self, arr: List[int], k: int) -> int:
        """Sliding window for maximum subarray sum"""
        if len(arr) < k:
            return sum(arr)

        # Calculate sum of first window
        window_sum = sum(arr[:k])
        max_sum = window_sum

        # Slide the window
        for i in range(len(arr) - k):
            window_sum = window_sum - arr[i] + arr[i + k]
            max_sum = max(max_sum, window_sum)

        return max_sum

# Test the pattern solver
def test_pattern_solver():
    solver = PatternSolver()

    # Test problem analysis
    problem1 = {
        'type': 'two_sum_sorted',
        'data': ([1, 2, 3, 4, 6, 8, 11], 14)
    }

    analysis = solver.analyze_problem(problem1['type'], problem1['data'])
    print("Problem Analysis:", analysis)

    # Solve with suggested pattern
    solution = solver.solve_with_pattern(problem1, 'two_pointers')
    print("Solution:", solution)

    # Performance comparison
    start_time = time.time()
    brute_force_result = solver._brute_force_two_sum(problem1['data'])
    brute_force_time = time.time() - start_time

    start_time = time.time()
    optimized_result = solver.solve_with_pattern(problem1, 'two_pointers')
    optimized_time = time.time() - start_time

    print(f"Brute force: {brute_force_result} (Time: {brute_force_time:.6f}s)")
    print(f"Optimized: {optimized_result} (Time: {optimized_time:.6f}s)")

test_pattern_solver()
```

### Success Criteria

- [ ] Correctly identifies appropriate patterns for different problem types
- [ ] Implements all major patterns with O(n) or better complexity
- [ ] Provides meaningful analysis and reasoning for pattern selection
- [ ] Demonstrates significant performance improvements over brute force
- [ ] Handles edge cases and boundary conditions correctly

### Extension Challenges

1. **Automatic Pattern Combination** - Combine multiple patterns for complex problems
2. **Adaptive Learning** - Learn from solution performance to improve pattern selection
3. **Visualization** - Create visual representations of pattern application

---

## 🚀 Full Project Extension: FAANG-Ready Algorithm Platform

**Time Required:** 10-15 hours  
**Difficulty:** Expert  
**Skills Practiced:** Advanced patterns, system design, performance optimization, interview preparation

### Project Overview

Build a comprehensive algorithmic problem-solving platform that helps candidates prepare for FAANG interviews by providing pattern-based solutions, real-time performance analysis, and adaptive learning systems.

### Core Architecture

#### 1. Intelligent Pattern Recognition System

```python
class AlgorithmIntelligence:
    def __init__(self):
        self.pattern_classifier = PatternClassifier()
        self.solution_engine = MultiPatternSolver()
        self.performance_analyzer = PerformanceAnalyzer()
        self.learning_system = AdaptiveLearning()

    def analyze_problem_statement(self, problem: dict) -> dict:
        """Extract problem characteristics and classify pattern"""
        characteristics = {
            'input_structure': self._analyze_input_structure(problem),
            'operation_type': self._classify_operations(problem),
            'complexity_requirements': self._extract_complexity(problem),
            'constraint_patterns': self._identify_constraints(problem)
        }

        pattern_suggestions = self.pattern_classifier.predict(characteristics)
        return {
            'primary_pattern': pattern_suggestions[0],
            'alternative_patterns': pattern_suggestions[1:],
            'pattern_confidence': pattern_suggestions.confidence,
            'implementation_roadmap': self._generate_roadmap(pattern_suggestions)
        }

    def solve_with_explanation(self, problem: dict, pattern: str) -> dict:
        """Generate solution with step-by-step explanation"""
        solution = self.solution_engine.solve(problem, pattern)
        explanation = self._generate_explanation(problem, solution, pattern)
        complexity_analysis = self.performance_analyzer.analyze(solution)

        return {
            'solution': solution,
            'explanation': explanation,
            'complexity': complexity_analysis,
            'code': self._generate_code(solution, pattern),
            'alternatives': self._suggest_alternatives(problem)
        }
```

#### 2. Advanced Pattern Implementation

- **Multi-Pattern Combinations** - Problems requiring 2-3 patterns combined
- **Dynamic Pattern Selection** - Choose patterns based on input characteristics
- **Real-time Optimization** - Adapt solutions based on runtime performance
- **Space-Time Tradeoff Analysis** - Show different solution approaches

#### 3. Interview Simulation System

- **Timed Problem Solving** - FAANG-style interview timing constraints
- **Progressive Difficulty** - Adaptive difficulty based on user performance
- **Feedback System** - Detailed analysis of solution approach and improvements
- **Mock Interview Integration** - Combine with actual interview patterns

#### 4. Performance Analytics & Learning

- **Solution Analysis** - Compare multiple approaches for same problem
- **Pattern Mastery Tracking** - Track progress across different pattern types
- **Weakness Identification** - Identify and recommend focus areas
- **Adaptive Learning Path** - Personalized curriculum based on performance

### Implementation Phases

#### Phase 1: Pattern Recognition Engine (3-4 hours)

- Build comprehensive pattern classification system
- Implement NLP for problem statement analysis
- Create pattern combination algorithms

#### Phase 2: Solution Engine (3-4 hours)

- Implement all major patterns with optimizations
- Build solution explanation generation
- Create performance comparison tools

#### Phase 3: Interview Simulation (2-3 hours)

- Build timed problem-solving interface
- Create adaptive difficulty system
- Implement comprehensive feedback system

#### Phase 4: Learning & Analytics (2-4 hours)

- Build progress tracking and analytics
- Create adaptive learning path generation
- Implement weakness identification and recommendations

### Success Criteria

- [ ] Accurate pattern recognition for 95%+ of algorithmic problems
- [ ] Solution generation with clear explanations for all patterns
- [ ] Interview simulation with realistic timing and pressure
- [ ] Adaptive learning system that personalizes to individual needs
- [ ] Comprehensive performance analytics and progress tracking
- [ ] Real-time solution optimization and comparison
- [ ] Professional user interface with engaging interaction design

### Technical Stack Recommendations

- **Backend:** Python with FastAPI for high-performance algorithm processing
- **Machine Learning:** scikit-learn for pattern classification, NLP libraries
- **Frontend:** React with TypeScript for interactive problem-solving interface
- **Database:** PostgreSQL for user progress, Redis for caching
- **Real-time:** WebSockets for interactive coding environment
- **Analytics:** Custom analytics for pattern mastery tracking

### Learning Outcomes

This project demonstrates mastery of advanced algorithmic patterns in production interview preparation systems, showcasing ability to:

- Build intelligent algorithmic problem-solving systems
- Design adaptive learning platforms for technical skills
- Implement complex pattern recognition and classification
- Create engaging user experiences for technical learning
- Deploy and maintain high-performance learning systems
- Apply machine learning to optimize educational outcomes

---

_Continue to Advanced Patterns Practice to master these concepts through implementation!_
