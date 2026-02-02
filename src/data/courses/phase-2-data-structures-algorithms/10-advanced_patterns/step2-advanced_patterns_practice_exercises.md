---
title: "Advanced Patterns - Practice Problems"
level: "Advanced"
estimated_time: "10-12 hours"
tags: ["advanced", "patterns", "practice", "optimization", "interview"]
---

# Advanced Patterns - Practice Problems

## 🎯 Learning Objectives

- Master advanced algorithmic patterns through progressive practice
- Develop pattern recognition skills for complex problems
- Build intuition for combining multiple optimization techniques
- Prepare for senior-level technical interviews

## 🌟 Practice Philosophy

> "Advanced patterns are like chess grandmaster techniques - once you master them, you see the board differently. A problem that stumps others becomes a familiar pattern. The goal isn't just to solve, but to solve elegantly, efficiently, and with confidence."

---

## 🔥 Two Pointers Problems

### Problem 1: Container With Most Water

**LeetCode 11 - Classic Two Pointers Optimization**

```python
def max_area(height):
    """
    Find two lines that form container with most water.

    Strategy: Two pointers from ends, move pointer with smaller height
    Intuition: Moving smaller height might find larger area, moving larger won't

    Time: O(n), Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # Calculate current area
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height
        max_water = max(max_water, current_area)

        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water

# Why this greedy approach works:
# If we move the pointer with larger height, we decrease width
# but height is still limited by the smaller one, so area can only decrease
# Moving smaller height pointer gives chance to find larger height

# Test
height = [1,8,6,2,5,4,8,3,7]
print(max_area(height))  # 49
```

### Problem 2: 3Sum Closest

**Advanced Two Pointers with Optimization**

```python
def three_sum_closest(nums, target):
    """
    Find three numbers whose sum is closest to target.

    Strategy: Sort + fix one number + two pointers for remaining two
    Time: O(n²), Space: O(1)
    """
    nums.sort()
    n = len(nums)
    closest_sum = float('inf')

    for i in range(n - 2):
        left, right = i + 1, n - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            # Update closest if current is closer to target
            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum

            if current_sum == target:
                return target  # Exact match found
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return closest_sum

# Optimized version with early termination
def three_sum_closest_optimized(nums, target):
    """
    Optimized with pruning and early termination.
    """
    nums.sort()
    n = len(nums)
    closest_sum = nums[0] + nums[1] + nums[2]

    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, n - 1

        # Early termination: check boundaries
        min_sum = nums[i] + nums[left] + nums[left + 1]
        if min_sum > target:
            if abs(min_sum - target) < abs(closest_sum - target):
                closest_sum = min_sum
            break  # All remaining sums will be larger

        max_sum = nums[i] + nums[right - 1] + nums[right]
        if max_sum < target:
            if abs(max_sum - target) < abs(closest_sum - target):
                closest_sum = max_sum
            continue  # Move to next i

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            if abs(current_sum - target) < abs(closest_sum - target):
                closest_sum = current_sum

            if current_sum == target:
                return target
            elif current_sum < target:
                left += 1
                # Skip duplicates
                while left < right and nums[left] == nums[left-1]:
                    left += 1
            else:
                right -= 1
                # Skip duplicates
                while left < right and nums[right] == nums[right+1]:
                    right -= 1

    return closest_sum

# Test
nums = [-1, 2, 1, -4]
target = 1
print(three_sum_closest(nums, target))  # 2
```

### Problem 3: Trapping Rain Water (Two Pointers)

**Advanced Two Pointers Technique**

```python
def trap_rainwater_two_pointers(height):
    """
    Calculate trapped rainwater using two pointers approach.

    Key insight: Water level at any point is limited by max height
    on its left and right. We can process from both ends.

    Time: O(n), Space: O(1)
    """
    if not height or len(height) < 3:
        return 0

    left, right = 0, len(height) - 1
    left_max = right_max = 0
    trapped = 0

    while left < right:
        if height[left] < height[right]:
            # Process left side
            if height[left] >= left_max:
                left_max = height[left]
            else:
                trapped += left_max - height[left]
            left += 1
        else:
            # Process right side
            if height[right] >= right_max:
                right_max = height[right]
            else:
                trapped += right_max - height[right]
            right -= 1

    return trapped

# Why this works:
# If height[left] < height[right], then right_max > left_max
# So water at position 'left' is determined by left_max only
# We can safely process left side without knowing exact right_max

# Comparison with DP approach
def trap_rainwater_dp(height):
    """
    DP approach for comparison - O(n) space.
    """
    if not height:
        return 0

    n = len(height)
    left_max = [0] * n
    right_max = [0] * n

    # Fill left_max
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])

    # Fill right_max
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])

    # Calculate trapped water
    trapped = 0
    for i in range(n):
        trapped += min(left_max[i], right_max[i]) - height[i]

    return trapped

# Test
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trap_rainwater_two_pointers(height))  # 6
print(trap_rainwater_dp(height))           # 6
```

---

## 🔥 Sliding Window Problems

### Problem 4: Minimum Window Substring

**LeetCode 76 - Advanced Variable Window**

```python
def min_window_substring(s, t):
    """
    Find minimum window in s that contains all characters of t.

    Strategy: Expand window until valid, then contract to find minimum
    Time: O(|s| + |t|), Space: O(|s| + |t|)
    """
    if not s or not t or len(s) < len(t):
        return ""

    from collections import Counter, defaultdict

    # Count characters in t
    target_count = Counter(t)
    required_chars = len(target_count)

    # Sliding window
    left = right = 0
    formed_chars = 0  # Number of unique chars in window with correct frequency
    window_count = defaultdict(int)

    # Result tracking
    min_len = float('inf')
    min_left = 0

    while right < len(s):
        # Expand window by adding right character
        char = s[right]
        window_count[char] += 1

        # Check if current character frequency matches target
        if char in target_count and window_count[char] == target_count[char]:
            formed_chars += 1

        # Contract window if valid
        while left <= right and formed_chars == required_chars:
            # Update minimum window if smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_left = left

            # Remove left character
            left_char = s[left]
            window_count[left_char] -= 1
            if (left_char in target_count and
                window_count[left_char] < target_count[left_char]):
                formed_chars -= 1

            left += 1

        right += 1

    return s[min_left:min_left + min_len] if min_len != float('inf') else ""

# Optimized version with filtered string
def min_window_substring_optimized(s, t):
    """
    Optimization: Only consider characters that are in t.
    Reduces iterations when t is much smaller than s.
    """
    if not s or not t:
        return ""

    from collections import Counter, defaultdict

    target_count = Counter(t)
    required_chars = len(target_count)

    # Filter s to only include characters in t
    filtered_s = []
    for i, char in enumerate(s):
        if char in target_count:
            filtered_s.append((i, char))

    left = right = 0
    formed_chars = 0
    window_count = defaultdict(int)

    min_len = float('inf')
    min_left = 0

    while right < len(filtered_s):
        char = filtered_s[right][1]
        window_count[char] += 1

        if window_count[char] == target_count[char]:
            formed_chars += 1

        while left <= right and formed_chars == required_chars:
            # Get actual indices in original string
            start_idx = filtered_s[left][0]
            end_idx = filtered_s[right][0]

            if end_idx - start_idx + 1 < min_len:
                min_len = end_idx - start_idx + 1
                min_left = start_idx

            left_char = filtered_s[left][1]
            window_count[left_char] -= 1
            if window_count[left_char] < target_count[left_char]:
                formed_chars -= 1

            left += 1

        right += 1

    return s[min_left:min_left + min_len] if min_len != float('inf') else ""

# Test
s = "ADOBECODEBANC"
t = "ABC"
print(min_window_substring(s, t))  # "BANC"
```

### Problem 5: Sliding Window Maximum

**LeetCode 239 - Advanced Fixed Window with Deque**

```python
def max_sliding_window(nums, k):
    """
    Find maximum in each sliding window of size k.

    Strategy: Use deque to maintain potential maximums in decreasing order
    Time: O(n), Space: O(k)
    """
    if not nums or k == 0:
        return []

    from collections import deque

    dq = deque()  # Store indices
    result = []

    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove indices of smaller elements (they'll never be max)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # Add maximum to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Alternative: Segment Tree approach for comparison
class SegmentTree:
    """Segment tree for range maximum queries."""

    def __init__(self, arr):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 1, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2 * node, start, mid)
            self.build(arr, 2 * node + 1, mid + 1, end)
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, node, start, end, l, r):
        if r < start or end < l:
            return float('-inf')
        if l <= start and end <= r:
            return self.tree[node]

        mid = (start + end) // 2
        left_max = self.query(2 * node, start, mid, l, r)
        right_max = self.query(2 * node + 1, mid + 1, end, l, r)
        return max(left_max, right_max)

    def range_max(self, l, r):
        return self.query(1, 0, self.n - 1, l, r)

def max_sliding_window_segtree(nums, k):
    """
    Segment tree approach: O(n log n) time, O(n) space.
    Less efficient than deque but demonstrates alternative approach.
    """
    if not nums or k == 0:
        return []

    seg_tree = SegmentTree(nums)
    result = []

    for i in range(len(nums) - k + 1):
        max_val = seg_tree.range_max(i, i + k - 1)
        result.append(max_val)

    return result

# Test
nums = [1,3,-1,-3,5,3,6,7]
k = 3
print(max_sliding_window(nums, k))  # [3,3,5,5,6,7]
```

### Problem 6: Longest Substring with At Most K Distinct Characters

**Advanced Variable Window Pattern**

```python
def longest_substring_k_distinct(s, k):
    """
    Find longest substring with at most k distinct characters.

    Strategy: Variable sliding window with frequency tracking
    Time: O(n), Space: O(k)
    """
    if not s or k == 0:
        return 0

    from collections import defaultdict

    char_count = defaultdict(int)
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Expand window
        char_count[s[right]] += 1

        # Contract window if too many distinct characters
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        # Update maximum length
        max_length = max(max_length, right - left + 1)

    return max_length

# Follow-up: Exactly k distinct characters
def longest_substring_exactly_k_distinct(s, k):
    """
    Find longest substring with exactly k distinct characters.
    """
    def at_most_k_distinct(k):
        if k == 0:
            return 0

        char_count = defaultdict(int)
        left = 0
        max_len = 0

        for right in range(len(s)):
            char_count[s[right]] += 1

            while len(char_count) > k:
                char_count[s[left]] -= 1
                if char_count[s[left]] == 0:
                    del char_count[s[left]]
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len

    return at_most_k_distinct(k) - at_most_k_distinct(k - 1)

# Advanced: Longest substring with at most k distinct characters
# and each character appears at least twice
def longest_substring_k_distinct_min_freq(s, k, min_freq):
    """
    Find longest substring with at most k distinct characters
    where each character appears at least min_freq times.
    """
    if not s or k == 0:
        return 0

    from collections import defaultdict

    char_count = defaultdict(int)
    valid_chars = 0  # Characters with frequency >= min_freq
    left = 0
    max_length = 0

    for right in range(len(s)):
        char = s[right]
        char_count[char] += 1

        if char_count[char] == min_freq:
            valid_chars += 1

        # Contract window if too many distinct characters
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1

            if char_count[left_char] == min_freq - 1:
                valid_chars -= 1

            if char_count[left_char] == 0:
                del char_count[left_char]

            left += 1

        # Update maximum if all characters have sufficient frequency
        if valid_chars == len(char_count):
            max_length = max(max_length, right - left + 1)

    return max_length

# Test
s = "eceba"
k = 2
print(longest_substring_k_distinct(s, k))  # 3 ("ece")
```

---

## 🔥 Monotonic Stack/Queue Problems

### Problem 7: Largest Rectangle in Histogram

**LeetCode 84 - Classic Monotonic Stack**

```python
def largest_rectangle_area(heights):
    """
    Find largest rectangle area in histogram.

    Strategy: Monotonic increasing stack to find boundaries
    For each bar, find left and right boundaries where height >= current

    Time: O(n), Space: O(n)
    """
    stack = []  # Store indices
    max_area = 0

    # Process each bar and a dummy bar at end
    for i in range(len(heights) + 1):
        # Current height (0 for dummy bar)
        current_height = heights[i] if i < len(heights) else 0

        # Process all bars higher than current
        while stack and heights[stack[-1]] > current_height:
            height = heights[stack.pop()]

            # Width calculation
            width = i if not stack else i - stack[-1] - 1

            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area

# Alternative: Divide and conquer approach
def largest_rectangle_divide_conquer(heights):
    """
    Divide and conquer approach for comparison.

    Time: O(n log n) average, O(n²) worst case
    Space: O(log n) for recursion
    """
    def find_max_area(left, right):
        if left > right:
            return 0

        # Find minimum height in range
        min_idx = left
        for i in range(left + 1, right + 1):
            if heights[i] < heights[min_idx]:
                min_idx = i

        # Rectangle using minimum height
        area_with_min = heights[min_idx] * (right - left + 1)

        # Recursively check left and right parts
        area_left = find_max_area(left, min_idx - 1)
        area_right = find_max_area(min_idx + 1, right)

        return max(area_with_min, area_left, area_right)

    if not heights:
        return 0

    return find_max_area(0, len(heights) - 1)

# Optimized: Using segment tree for minimum queries
def largest_rectangle_segment_tree(heights):
    """
    Segment tree optimization for divide and conquer.

    Time: O(n log n), Space: O(n)
    """
    class MinSegmentTree:
        def __init__(self, arr):
            self.n = len(arr)
            self.arr = arr
            self.tree = [0] * (4 * self.n)
            self.build(1, 0, self.n - 1)

        def build(self, node, start, end):
            if start == end:
                self.tree[node] = start
            else:
                mid = (start + end) // 2
                self.build(2 * node, start, mid)
                self.build(2 * node + 1, mid + 1, end)

                left_idx = self.tree[2 * node]
                right_idx = self.tree[2 * node + 1]

                if self.arr[left_idx] <= self.arr[right_idx]:
                    self.tree[node] = left_idx
                else:
                    self.tree[node] = right_idx

        def query(self, node, start, end, l, r):
            if r < start or end < l:
                return -1

            if l <= start and end <= r:
                return self.tree[node]

            mid = (start + end) // 2
            left_idx = self.query(2 * node, start, mid, l, r)
            right_idx = self.query(2 * node + 1, mid + 1, end, l, r)

            if left_idx == -1:
                return right_idx
            if right_idx == -1:
                return left_idx

            return left_idx if self.arr[left_idx] <= self.arr[right_idx] else right_idx

        def range_min_idx(self, l, r):
            return self.query(1, 0, self.n - 1, l, r)

    if not heights:
        return 0

    seg_tree = MinSegmentTree(heights)

    def find_max_area(left, right):
        if left > right:
            return 0

        min_idx = seg_tree.range_min_idx(left, right)
        area_with_min = heights[min_idx] * (right - left + 1)

        area_left = find_max_area(left, min_idx - 1)
        area_right = find_max_area(min_idx + 1, right)

        return max(area_with_min, area_left, area_right)

    return find_max_area(0, len(heights) - 1)

# Test
heights = [2,1,5,6,2,3]
print(largest_rectangle_area(heights))  # 10
```

### Problem 8: Next Greater Element II

**LeetCode 503 - Circular Array with Monotonic Stack**

```python
def next_greater_elements_circular(nums):
    """
    Find next greater element in circular array.

    Strategy: Process array twice using monotonic decreasing stack
    Time: O(n), Space: O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices

    # Process array twice to handle circular nature
    for i in range(2 * n):
        current_idx = i % n

        # Pop elements smaller than current
        while stack and nums[stack[-1]] < nums[current_idx]:
            idx = stack.pop()
            result[idx] = nums[current_idx]

        # Only add to stack in first pass
        if i < n:
            stack.append(current_idx)

    return result

# Follow-up: Next greater element with wrap-around count
def next_greater_elements_with_distance(nums):
    """
    Return both next greater element and distance to it.
    """
    n = len(nums)
    result = [(-1, -1)] * n  # (value, distance)
    stack = []

    for i in range(2 * n):
        current_idx = i % n

        while stack and nums[stack[-1]] < nums[current_idx]:
            idx = stack.pop()
            distance = (i - idx) if i >= n else (i - idx)
            result[idx] = (nums[current_idx], distance)

        if i < n:
            stack.append(current_idx)

    return result

# Advanced: Temperature warmer days
def daily_temperatures(temperatures):
    """
    LeetCode 739 - Find how many days until warmer temperature.

    Time: O(n), Space: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store indices

    for i in range(n):
        # Pop days with lower temperature
        while stack and temperatures[stack[-1]] < temperatures[i]:
            idx = stack.pop()
            result[idx] = i - idx  # Days until warmer

        stack.append(i)

    return result

# Test
nums = [1,2,1]
print(next_greater_elements_circular(nums))  # [2,-1,2]

temps = [73,74,75,71,69,72,76,73]
print(daily_temperatures(temps))  # [1,1,4,2,1,1,0,0]
```

---

## 🔥 Union-Find Problems

### Problem 9: Number of Islands II

**LeetCode 305 - Dynamic Connectivity**

```python
def num_islands_online(m, n, positions):
    """
    Count islands as land is added dynamically.

    Strategy: Union-Find to track connected components
    Time: O(k × α(m×n)) where k = number of positions
    Space: O(m×n)
    """
    class UnionFind:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
            self.count = 0

        def find(self, x):
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

            self.count -= 1
            return True

        def add_island(self):
            self.count += 1

    uf = UnionFind(m * n)
    grid = [[0] * n for _ in range(m)]
    result = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def get_id(r, c):
        return r * n + c

    for r, c in positions:
        if grid[r][c] == 1:
            # Already land, no change
            result.append(uf.count)
            continue

        grid[r][c] = 1
        uf.add_island()

        # Check and union with adjacent land
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < m and 0 <= nc < n and
                grid[nr][nc] == 1):
                uf.union(get_id(r, c), get_id(nr, nc))

        result.append(uf.count)

    return result

# Test
m, n = 3, 3
positions = [[0,0],[0,1],[1,2],[2,1]]
print(num_islands_online(m, n, positions))  # [1,1,2,3]
```

### Problem 10: Accounts Merge

**LeetCode 721 - String-based Union-Find**

```python
def accounts_merge(accounts):
    """
    Merge accounts belonging to the same person.

    Strategy: Union-Find with email mapping
    Time: O(N × α(N)) where N = total emails
    Space: O(N)
    """
    class UnionFind:
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
                return

            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    uf = UnionFind()
    email_to_name = {}

    # Build union-find structure
    for account in accounts:
        name = account[0]
        first_email = account[1]

        for email in account[1:]:
            email_to_name[email] = name
            uf.union(first_email, email)

    # Group emails by root
    from collections import defaultdict
    root_to_emails = defaultdict(set)

    for email in email_to_name:
        root = uf.find(email)
        root_to_emails[root].add(email)

    # Build result
    result = []
    for emails in root_to_emails.values():
        name = email_to_name[next(iter(emails))]
        result.append([name] + sorted(emails))

    return result

# Alternative: DFS approach
def accounts_merge_dfs(accounts):
    """
    DFS approach for comparison.

    Time: O(N log N) for sorting, Space: O(N)
    """
    from collections import defaultdict

    # Build graph
    email_to_name = {}
    graph = defaultdict(set)

    for account in accounts:
        name = account[0]
        first_email = account[1]

        for email in account[1:]:
            email_to_name[email] = name
            graph[first_email].add(email)
            graph[email].add(first_email)

    visited = set()
    result = []

    def dfs(email, emails):
        visited.add(email)
        emails.add(email)

        for neighbor in graph[email]:
            if neighbor not in visited:
                dfs(neighbor, emails)

    for email in email_to_name:
        if email not in visited:
            emails = set()
            dfs(email, emails)

            name = email_to_name[email]
            result.append([name] + sorted(emails))

    return result

# Test
accounts = [
    ["John","johnsmith@mail.com","john_newyork@mail.com"],
    ["John","johnsmith@mail.com","john00@mail.com"],
    ["Mary","mary@mail.com"],
    ["John","johnnybravo@mail.com"]
]
print(accounts_merge(accounts))
```

---

## 🔥 Binary Search on Answer Problems

### Problem 11: Koko Eating Bananas

**LeetCode 875 - Search Space Transformation**

```python
def min_eating_speed(piles, h):
    """
    Find minimum eating speed to finish all bananas in h hours.

    Strategy: Binary search on eating speed (answer space)
    Time: O(n × log(max_pile)), Space: O(1)
    """
    def can_finish(speed):
        """Check if given speed allows finishing in h hours."""
        hours_needed = 0
        for pile in piles:
            hours_needed += (pile + speed - 1) // speed  # Ceiling division
        return hours_needed <= h

    # Binary search on speed
    left = 1  # Minimum possible speed
    right = max(piles)  # Maximum possible speed (eat largest pile in 1 hour)

    while left < right:
        mid = left + (right - left) // 2

        if can_finish(mid):
            right = mid  # Try slower speed
        else:
            left = mid + 1  # Need faster speed

    return left

# Advanced: Minimize maximum workload
def minimize_max_workload(jobs, k):
    """
    Assign jobs to k workers to minimize maximum workload.

    Strategy: Binary search on maximum workload
    Time: O(n × log(sum) × 2^n) worst case
    """
    def can_distribute(max_workload):
        """Check if jobs can be distributed with given max workload."""
        workers = [0] * k

        def backtrack(job_idx):
            if job_idx == len(jobs):
                return True

            # Try assigning current job to each worker
            for i in range(k):
                if workers[i] + jobs[job_idx] <= max_workload:
                    workers[i] += jobs[job_idx]

                    if backtrack(job_idx + 1):
                        return True

                    workers[i] -= jobs[job_idx]

                # Optimization: if worker is empty, skip other empty workers
                if workers[i] == 0:
                    break

            return False

        return backtrack(0)

    # Sort jobs in descending order for optimization
    jobs.sort(reverse=True)

    left = max(jobs)  # At least one job per worker
    right = sum(jobs)  # All jobs to one worker

    while left < right:
        mid = left + (right - left) // 2

        if can_distribute(mid):
            right = mid
        else:
            left = mid + 1

    return left

# Test
piles = [3,6,7,11]
h = 8
print(min_eating_speed(piles, h))  # 4
```

### Problem 12: Split Array Largest Sum

**LeetCode 410 - Complex Binary Search Application**

```python
def split_array(nums, k):
    """
    Split array into k subarrays to minimize largest sum.

    Strategy: Binary search on largest sum
    Time: O(n × log(sum)), Space: O(1)
    """
    def can_split(max_sum):
        """Check if array can be split into k parts with max_sum limit."""
        subarrays = 1
        current_sum = 0

        for num in nums:
            if num > max_sum:
                return False  # Single element exceeds limit

            if current_sum + num > max_sum:
                subarrays += 1
                current_sum = num
            else:
                current_sum += num

        return subarrays <= k

    left = max(nums)  # At least largest element
    right = sum(nums)  # All elements in one subarray

    while left < right:
        mid = left + (right - left) // 2

        if can_split(mid):
            right = mid
        else:
            left = mid + 1

    return left

# DP approach for comparison
def split_array_dp(nums, k):
    """
    Dynamic programming approach.

    Time: O(k × n²), Space: O(k × n)
    """
    n = len(nums)
    prefix_sum = [0]
    for num in nums:
        prefix_sum.append(prefix_sum[-1] + num)

    # dp[i][j] = minimum largest sum to split first i elements into j parts
    dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            for p in range(j - 1, i):
                subarray_sum = prefix_sum[i] - prefix_sum[p]
                dp[i][j] = min(dp[i][j], max(dp[p][j - 1], subarray_sum))

    return dp[n][k]

# Test
nums = [7,2,5,10,8]
k = 2
print(split_array(nums, k))  # 18
print(split_array_dp(nums, k))  # 18
```

---

## 🔥 Bit Manipulation Problems

### Problem 13: Single Number Series

**Advanced Bit Manipulation Patterns**

```python
def single_number_i(nums):
    """
    LeetCode 136 - Find single number (others appear twice).

    Strategy: XOR cancels out duplicates
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def single_number_ii(nums):
    """
    LeetCode 137 - Find single number (others appear thrice).

    Strategy: Count bits at each position
    Time: O(n), Space: O(1)
    """
    # Method 1: Bit counting
    result = 0
    for i in range(32):
        count = 0
        for num in nums:
            count += (num >> i) & 1

        # If count is not multiple of 3, single number has this bit
        if count % 3:
            result |= (1 << i)

    # Handle negative numbers (Python specific)
    if result >= 2**31:
        result -= 2**32

    return result

def single_number_ii_optimized(nums):
    """
    Optimized using finite state machine.

    Use two variables to represent states:
    - ones: bits appearing 1 time
    - twos: bits appearing 2 times
    - threes: bits appearing 3 times (reset to 0)
    """
    ones = twos = 0

    for num in nums:
        # Update twos: add bits that were in ones and now in num
        twos |= ones & num

        # Update ones: add new bits from num
        ones ^= num

        # Remove bits that appear 3 times
        threes = ones & twos
        ones &= ~threes
        twos &= ~threes

    return ones

def single_number_iii(nums):
    """
    LeetCode 260 - Find two single numbers (others appear twice).

    Strategy: XOR to separate into two groups
    Time: O(n), Space: O(1)
    """
    # XOR all numbers to get XOR of two single numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Find rightmost set bit (differentiator)
    rightmost_set_bit = xor_all & (-xor_all)

    # Separate numbers into two groups and XOR each group
    num1 = num2 = 0
    for num in nums:
        if num & rightmost_set_bit:
            num1 ^= num
        else:
            num2 ^= num

    return [num1, num2]

# Advanced: Missing number with bit manipulation
def missing_number_xor(nums):
    """
    Find missing number in range [0, n].

    Strategy: XOR with expected range
    Time: O(n), Space: O(1)
    """
    n = len(nums)
    result = n  # Start with n

    for i, num in enumerate(nums):
        result ^= i ^ num

    return result

# Test
print(single_number_i([2,2,1]))           # 1
print(single_number_ii([2,2,3,2]))        # 3
print(single_number_iii([1,2,1,3,2,5]))   # [3,5] or [5,3]
```

### Problem 14: Maximum XOR of Two Numbers

**LeetCode 421 - Advanced Bit Manipulation**

```python
def find_maximum_xor(nums):
    """
    Find maximum XOR of any two numbers in array.

    Strategy: Build answer bit by bit from MSB using prefix sets
    Time: O(n), Space: O(n)
    """
    max_xor = 0
    mask = 0

    # Process bits from MSB to LSB
    for i in range(30, -1, -1):
        mask |= (1 << i)  # Include current bit in mask

        # Get all prefixes with current mask
        prefixes = {num & mask for num in nums}

        # Try to set current bit in result
        temp = max_xor | (1 << i)

        # Check if this bit can be achieved
        # If temp = a ^ b, then a = temp ^ b
        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break

    return max_xor

# Alternative: Trie-based approach
class TrieNode:
    def __init__(self):
        self.children = {}

def find_maximum_xor_trie(nums):
    """
    Trie-based approach for maximum XOR.

    Time: O(n), Space: O(n)
    """
    root = TrieNode()

    # Insert all numbers into trie
    for num in nums:
        node = root
        for i in range(30, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]

    max_xor = 0

    # For each number, find maximum XOR
    for num in nums:
        node = root
        current_xor = 0

        for i in range(30, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for maximum XOR
            toggle_bit = 1 - bit

            if toggle_bit in node.children:
                current_xor |= (1 << i)
                node = node.children[toggle_bit]
            else:
                node = node.children[bit]

        max_xor = max(max_xor, current_xor)

    return max_xor

# Test
nums = [3, 10, 5, 25, 2, 8]
print(find_maximum_xor(nums))        # 28 (5 ^ 25 = 28)
print(find_maximum_xor_trie(nums))   # 28
```

---

## 🔥 Advanced Mathematical Problems

### Problem 15: Fast Exponentiation Applications

**Matrix Exponentiation for Recurrences**

```python
def matrix_multiply(A, B, mod=None):
    """Multiply two matrices with optional modulo."""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    result = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
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

def fibonacci_matrix(n):
    """
    Calculate nth Fibonacci number using matrix exponentiation.

    F(n) = [F(n), F(n-1)] = [F(1), F(0)] * [[1, 1], [1, 0]]^(n-1)

    Time: O(log n), Space: O(1)
    """
    if n <= 1:
        return n

    fib_matrix = [[1, 1], [1, 0]]
    result_matrix = matrix_power(fib_matrix, n - 1)

    return result_matrix[0][0]  # F(n)

# Advanced: Solve linear recurrence
def solve_linear_recurrence(coefficients, initial_values, n):
    """
    Solve linear recurrence of form:
    a(n) = c1*a(n-1) + c2*a(n-2) + ... + ck*a(n-k)

    Time: O(k³ log n), Space: O(k²)
    """
    k = len(coefficients)
    if n < k:
        return initial_values[n]

    # Build transition matrix
    # [a(n), a(n-1), ..., a(n-k+1)] = [a(n-1), a(n-2), ..., a(n-k)] * T
    transition = [[0] * k for _ in range(k)]

    # First row: coefficients
    for i in range(k):
        transition[0][i] = coefficients[i]

    # Identity part for shifting
    for i in range(1, k):
        transition[i][i-1] = 1

    # Apply matrix exponentiation
    result_matrix = matrix_power(transition, n - k + 1)

    # Calculate result
    result = 0
    for i in range(k):
        result += result_matrix[0][i] * initial_values[k - 1 - i]

    return result

# Test
print(fibonacci_matrix(10))  # 55

# Example: a(n) = 2*a(n-1) + 3*a(n-2), a(0)=1, a(1)=2
coefficients = [2, 3]  # [c1, c2]
initial_values = [1, 2]  # [a(0), a(1)]
print(solve_linear_recurrence(coefficients, initial_values, 5))
```

---

## 📊 Pattern Recognition Summary

### Problem-Pattern Mapping Guide

| Problem Characteristic              | Primary Pattern         | Secondary Pattern | Time Complexity       |
| ----------------------------------- | ----------------------- | ----------------- | --------------------- |
| Find pairs/triplets in sorted array | Two Pointers            |                   | O(n) - O(n²)          |
| Subarray/substring optimization     | Sliding Window          | Two Pointers      | O(n)                  |
| Next greater/smaller elements       | Monotonic Stack         |                   | O(n)                  |
| Range maximum in sliding window     | Monotonic Queue         | Segment Tree      | O(n)                  |
| Dynamic connectivity                | Union-Find              | DFS/BFS           | O(α(n))               |
| Optimization with constraints       | Binary Search on Answer |                   | O(log(range) × check) |
| XOR problems                        | Bit Manipulation        | Trie              | O(n)                  |
| Linear recurrences                  | Matrix Exponentiation   | DP                | O(k³ log n)           |

### Advanced Combinations

1. **Sliding Window + Monotonic Queue**: Range extremes in windows
2. **Union-Find + Binary Search**: Dynamic connectivity with optimization
3. **Bit Manipulation + Trie**: XOR optimization problems
4. **Two Pointers + Binary Search**: Complex search problems

---

_Continue to Advanced Patterns Cheatsheet for quick reference templates!_
