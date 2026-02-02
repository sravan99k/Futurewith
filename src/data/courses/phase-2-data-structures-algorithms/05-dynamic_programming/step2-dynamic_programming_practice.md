---
title: "Dynamic Programming Practice Problems - 100+ Questions"
level: "Beginner to Advanced"
difficulty: "Progressive (Easy → Medium → Hard)"
time: "Varies (10-60 minutes per question)"
tags: ["dsa", "dynamic-programming", "dp", "practice", "coding-interview"]
---

# ⚡ Dynamic Programming Practice Problems

_100+ Progressive Problems from Basics to Expert Level_

---

## 📊 Problem Difficulty Distribution

| Level         | Count       | Time/Problem | Focus                       |
| ------------- | ----------- | ------------ | --------------------------- |
| 🌱 **Easy**   | 30 problems | 10-20 min    | 1D DP, basic patterns       |
| ⚡ **Medium** | 40 problems | 20-40 min    | 2D DP, optimization         |
| 🔥 **Hard**   | 30 problems | 40-60 min    | Advanced DP, complex states |

---

## 🌱 EASY LEVEL (1-30) - Building DP Foundations

### **Problem 1: Fibonacci Numbers**

**Difficulty:** ⭐ Easy | **Time:** 10 minutes

```python
"""
Calculate nth Fibonacci number using DP.
F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)

Input: n = 5
Output: 5 (sequence: 0,1,1,2,3,5)
"""

# Solution 1: Top-down with memoization
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Solution 2: Bottom-up
def fibonacci_dp(n):
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# Solution 3: Space optimized
def fibonacci_optimized(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr

    return prev1

# Test cases
assert fibonacci_dp(0) == 0
assert fibonacci_dp(1) == 1
assert fibonacci_dp(5) == 5
assert fibonacci_dp(10) == 55
```

### **Problem 2: Climbing Stairs**

**Difficulty:** ⭐ Easy | **Time:** 10 minutes

```python
"""
You can climb 1 or 2 steps at a time.
How many distinct ways to climb n stairs?

Input: n = 3
Output: 3 (1+1+1, 1+2, 2+1)
"""

def climb_stairs(n):
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

# Space optimized
def climb_stairs_optimized(n):
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr

    return prev1

# Test cases
assert climb_stairs(1) == 1
assert climb_stairs(2) == 2
assert climb_stairs(3) == 3
assert climb_stairs(4) == 5
```

### **Problem 3: House Robber**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Rob houses but can't rob adjacent ones.
Return maximum money you can rob.

Input: [2,7,9,3,1]
Output: 12 (rob houses 0, 2, 4: 2+9+1=12)
"""

def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    return dp[-1]

# Space optimized
def rob_optimized(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2, prev1 = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr

    return prev1

# Test cases
assert rob([2,7,9,3,1]) == 12
assert rob([5,1,3,9]) == 14
assert rob([1]) == 1
```

### **Problem 4: Min Cost Climbing Stairs**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Each step has a cost. You can start from index 0 or 1.
Find minimum cost to reach the top.

Input: cost = [10,15,20]
Output: 15 (start at index 1, pay 15, step to top)
"""

def min_cost_climbing_stairs(cost):
    n = len(cost)

    # dp[i] = minimum cost to reach step i
    dp = [0] * (n + 1)

    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])

    return dp[n]

# Space optimized
def min_cost_optimized(cost):
    prev2 = prev1 = 0

    for i in range(2, len(cost) + 1):
        curr = min(prev1 + cost[i-1], prev2 + cost[i-2])
        prev2, prev1 = prev1, curr

    return prev1

# Test cases
assert min_cost_climbing_stairs([10,15,20]) == 15
assert min_cost_climbing_stairs([1,100,1,1,1,100,1,1,99,1]) == 6
```

### **Problem 5: Maximum Subarray (Kadane's Algorithm)**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Find the contiguous subarray with maximum sum.

Input: [-2,1,-3,4,-1,2,1,-5,4]
Output: 6 (subarray [4,-1,2,1])
"""

def max_subarray(nums):
    if not nums:
        return 0

    max_ending_here = max_so_far = nums[0]

    for i in range(1, len(nums)):
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

# DP approach
def max_subarray_dp(nums):
    dp = [0] * len(nums)
    dp[0] = nums[0]
    max_sum = nums[0]

    for i in range(1, len(nums)):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
        max_sum = max(max_sum, dp[i])

    return max_sum

# Test cases
assert max_subarray([-2,1,-3,4,-1,2,1,-5,4]) == 6
assert max_subarray([1]) == 1
assert max_subarray([5,4,-1,7,8]) == 23
```

### **Problem 6: Unique Paths**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Robot moves right or down in m×n grid.
How many unique paths to bottom-right?

Input: m = 3, n = 7
Output: 28
"""

def unique_paths(m, n):
    # dp[i][j] = number of paths to cell (i,j)
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# Space optimized (1D array)
def unique_paths_optimized(m, n):
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j-1]

    return dp[n-1]

# Test cases
assert unique_paths(3, 7) == 28
assert unique_paths(3, 2) == 3
assert unique_paths(7, 3) == 28
```

### **Problem 7: Unique Paths II (With Obstacles)**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Same as unique paths but some cells have obstacles (1).

Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
"""

def unique_paths_with_obstacles(obstacle_grid):
    if not obstacle_grid or obstacle_grid[0][0] == 1:
        return 0

    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1

    # Fill first row
    for j in range(1, n):
        if obstacle_grid[0][j] == 0:
            dp[0][j] = dp[0][j-1]

    # Fill first column
    for i in range(1, m):
        if obstacle_grid[i][0] == 0:
            dp[i][0] = dp[i-1][0]

    # Fill rest of the grid
    for i in range(1, m):
        for j in range(1, n):
            if obstacle_grid[i][j] == 0:
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# Test case
grid = [[0,0,0],[0,1,0],[0,0,0]]
assert unique_paths_with_obstacles(grid) == 2
```

### **Problem 8: Min Path Sum**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Find path with minimum sum from top-left to bottom-right.
Can only move right or down.

Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7 (path: 1→3→1→1→1)
"""

def min_path_sum(grid):
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])

    # Initialize first row
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]

    # Initialize first column
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]

    # Fill the rest
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])

    return grid[m-1][n-1]

# Without modifying input
def min_path_sum_clean(grid):
    m, n = len(grid), len(grid[0])
    dp = [[float('inf')] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue
            if i > 0:
                dp[i][j] = min(dp[i][j], dp[i-1][j] + grid[i][j])
            if j > 0:
                dp[i][j] = min(dp[i][j], dp[i][j-1] + grid[i][j])

    return dp[m-1][n-1]

# Test case
grid = [[1,3,1],[1,5,1],[4,2,1]]
assert min_path_sum_clean(grid) == 7
```

### **Problem 9: Decode Ways**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Decode string where 'A'=1, 'B'=2, ..., 'Z'=26.
How many ways to decode?

Input: "12"
Output: 2 ("AB" or "L")
"""

def num_decodings(s):
    if not s or s[0] == '0':
        return 0

    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1

    for i in range(2, n + 1):
        # Single digit
        if s[i-1] != '0':
            dp[i] += dp[i-1]

        # Two digits
        two_digit = int(s[i-2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i-2]

    return dp[n]

# Space optimized
def num_decodings_optimized(s):
    if not s or s[0] == '0':
        return 0

    prev2 = prev1 = 1

    for i in range(1, len(s)):
        curr = 0

        # Single digit
        if s[i] != '0':
            curr += prev1

        # Two digits
        if 10 <= int(s[i-1:i+1]) <= 26:
            curr += prev2

        prev2, prev1 = prev1, curr

    return prev1

# Test cases
assert num_decodings("12") == 2
assert num_decodings("226") == 3
assert num_decodings("06") == 0
```

### **Problem 10: Triangle**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Find minimum path sum from top to bottom of triangle.
Adjacent numbers on the row below.

Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
Output: 11 (path: 2+3+5+1=11)
"""

def minimum_total(triangle):
    if not triangle:
        return 0

    # Bottom-up approach
    dp = triangle[-1][:]

    for i in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[i])):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])

    return dp[0]

# Top-down approach
def minimum_total_topdown(triangle):
    n = len(triangle)
    dp = [[0] * len(triangle[i]) for i in range(n)]
    dp[0][0] = triangle[0][0]

    for i in range(1, n):
        for j in range(len(triangle[i])):
            dp[i][j] = triangle[i][j]

            if j == 0:
                dp[i][j] += dp[i-1][0]
            elif j == len(triangle[i]) - 1:
                dp[i][j] += dp[i-1][j-1]
            else:
                dp[i][j] += min(dp[i-1][j-1], dp[i-1][j])

    return min(dp[-1])

# Test case
triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
assert minimum_total(triangle) == 11
```

---

## ⚡ MEDIUM LEVEL (31-70) - Intermediate DP Patterns

### **Problem 31: Coin Change**

**Difficulty:** ⚡ Medium | **Time:** 25 minutes

```python
"""
Find minimum number of coins to make amount.
Each coin can be used unlimited times.

Input: coins = [1,3,4], amount = 6
Output: 2 (use two coins of 3)
"""

def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# With path reconstruction
def coin_change_with_path(coins, amount):
    dp = [float('inf')] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    if dp[amount] == float('inf'):
        return -1, []

    # Reconstruct path
    path = []
    curr = amount
    while curr > 0:
        coin = parent[curr]
        path.append(coin)
        curr -= coin

    return dp[amount], path

# Test cases
assert coin_change([1,3,4], 6) == 2
assert coin_change([2], 3) == -1
count, path = coin_change_with_path([1,3,4], 6)
assert count == 2
```

### **Problem 32: Coin Change 2 (Count Ways)**

**Difficulty:** ⚡ Medium | **Time:** 25 minutes

```python
"""
Count number of ways to make amount using coins.

Input: amount = 5, coins = [1,2,5]
Output: 4 (5, 2+2+1, 2+1+1+1, 1+1+1+1+1)
"""

def change(amount, coins):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

# 2D DP approach
def change_2d(amount, coins):
    n = len(coins)
    dp = [[0] * (amount + 1) for _ in range(n + 1)]

    # Base case: one way to make amount 0
    for i in range(n + 1):
        dp[i][0] = 1

    for i in range(1, n + 1):
        for j in range(1, amount + 1):
            # Don't use current coin
            dp[i][j] = dp[i-1][j]

            # Use current coin if possible
            if j >= coins[i-1]:
                dp[i][j] += dp[i][j - coins[i-1]]

    return dp[n][amount]

# Test cases
assert change(5, [1,2,5]) == 4
assert change(3, [2]) == 0
```

### **Problem 33: Longest Increasing Subsequence**

**Difficulty:** ⚡ Medium | **Time:** 30 minutes

```python
"""
Find length of longest increasing subsequence.

Input: [10,9,2,5,3,7,101,18]
Output: 4 (subsequence: 2,3,7,18 or 2,3,7,101)
"""

def length_of_lis(nums):
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# Optimized O(n log n) solution using binary search
def length_of_lis_optimized(nums):
    if not nums:
        return 0

    tails = []

    for num in nums:
        # Binary search for insertion point
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        # If num is larger than all elements, append
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num

    return len(tails)

# With actual subsequence reconstruction
def lis_with_sequence(nums):
    if not nums:
        return 0, []

    n = len(nums)
    dp = [1] * n
    parent = [-1] * n

    max_len = 1
    max_idx = 0

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

                if dp[i] > max_len:
                    max_len = dp[i]
                    max_idx = i

    # Reconstruct sequence
    sequence = []
    curr = max_idx
    while curr != -1:
        sequence.append(nums[curr])
        curr = parent[curr]

    sequence.reverse()
    return max_len, sequence

# Test cases
assert length_of_lis([10,9,2,5,3,7,101,18]) == 4
assert length_of_lis_optimized([0,1,0,3,2,3]) == 4
length, seq = lis_with_sequence([10,9,2,5,3,7,101,18])
assert length == 4
```

### **Problem 34: Edit Distance (Levenshtein)**

**Difficulty:** ⚡ Medium | **Time:** 30 minutes

```python
"""
Convert word1 to word2 using minimum operations:
insert, delete, replace.

Input: word1 = "horse", word2 = "ros"
Output: 3 (horse -> rorse -> rose -> ros)
"""

def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete
                    dp[i][j-1],    # Insert
                    dp[i-1][j-1]   # Replace
                )

    return dp[m][n]

# Space optimized
def min_distance_optimized(word1, word2):
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        curr = [i]
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                curr.append(prev[j-1])
            else:
                curr.append(1 + min(prev[j], curr[j-1], prev[j-1]))
        prev = curr

    return prev[n]

# Test cases
assert min_distance("horse", "ros") == 3
assert min_distance("intention", "execution") == 5
```

### **Problem 35: Longest Common Subsequence**

**Difficulty:** ⚡ Medium | **Time:** 25 minutes

```python
"""
Find length of longest common subsequence.

Input: text1 = "abcde", text2 = "ace"
Output: 3 (subsequence "ace")
"""

def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# With actual subsequence
def lcs_with_sequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Reconstruct LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return dp[m][n], ''.join(reversed(lcs))

# Test cases
assert longest_common_subsequence("abcde", "ace") == 3
length, seq = lcs_with_sequence("abcde", "ace")
assert length == 3 and seq == "ace"
```

---

## 🔥 HARD LEVEL (71-100) - Advanced DP Mastery

### **Problem 71: Regular Expression Matching**

**Difficulty:** 🔥 Hard | **Time:** 45 minutes

```python
"""
Implement regex matching with '.' and '*'.
'.' matches any single character
'*' matches zero or more of preceding character

Input: s = "aa", p = "a*"
Output: True
"""

def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        dp[0][j] = dp[0][j-2] and p[j-1] == '*'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Zero occurrence of preceding character
                dp[i][j] = dp[i][j-2]

                # One or more occurrences
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] |= dp[i-1][j]
            else:
                # Current characters match
                if p[j-1] == '.' or p[j-1] == s[i-1]:
                    dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# Test cases
assert is_match("aa", "a") == False
assert is_match("aa", "a*") == True
assert is_match("ab", ".*") == True
```

### **Problem 72: Wildcard Matching**

**Difficulty:** 🔥 Hard | **Time:** 40 minutes

```python
"""
Implement wildcard pattern matching with '?' and '*'.
'?' matches any single character
'*' matches any sequence of characters (including empty)

Input: s = "adceb", p = "*a*b*"
Output: True
"""

def is_match_wildcard(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # Handle patterns starting with *
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-1] or dp[i-1][j]
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# Test cases
assert is_match_wildcard("aa", "a") == False
assert is_match_wildcard("aa", "*") == True
assert is_match_wildcard("adceb", "*a*b*") == True
```

### **Problem 73: Burst Balloons**

**Difficulty:** 🔥 Hard | **Time:** 50 minutes

```python
"""
Burst balloons to maximize coins.
When you burst balloon i, you get nums[left]*nums[i]*nums[right] coins.

Input: [3,1,5,8]
Output: 167
"""

def max_coins(nums):
    # Add 1s at both ends
    nums = [1] + nums + [1]
    n = len(nums)

    # dp[i][j] = maximum coins from bursting balloons between i and j
    dp = [[0] * n for _ in range(n)]

    # Length of interval
    for length in range(2, n):
        for i in range(n - length):
            j = i + length

            # Try bursting each balloon k between i and j
            for k in range(i + 1, j):
                coins = nums[i] * nums[k] * nums[j]
                dp[i][j] = max(dp[i][j], dp[i][k] + coins + dp[k][j])

    return dp[0][n-1]

# Test case
assert max_coins([3,1,5,8]) == 167
```

### **Problem 74: Palindrome Partitioning II**

**Difficulty:** 🔥 Hard | **Time:** 45 minutes

```python
"""
Partition string into palindromes with minimum cuts.

Input: "aab"
Output: 1 (cut: "aa" | "b")
"""

def min_cut(s):
    n = len(s)

    # Precompute palindrome check
    is_palindrome = [[False] * n for _ in range(n)]

    for i in range(n):
        is_palindrome[i][i] = True

    for i in range(n - 1):
        is_palindrome[i][i + 1] = (s[i] == s[i + 1])

    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_palindrome[i][j] = (s[i] == s[j]) and is_palindrome[i + 1][j - 1]

    # DP for minimum cuts
    dp = [float('inf')] * n

    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n - 1]

# Test cases
assert min_cut("aab") == 1
assert min_cut("aba") == 0
```

---

## 🎯 Advanced Problem Categories

### **State Compression DP**

```python
# Problem: Traveling Salesman
def traveling_salesman(graph):
    n = len(graph)
    dp = {}

    def tsp(mask, pos):
        if mask == (1 << n) - 1:
            return graph[pos][0]

        if (mask, pos) in dp:
            return dp[(mask, pos)]

        result = float('inf')
        for city in range(n):
            if mask & (1 << city) == 0:
                new_mask = mask | (1 << city)
                cost = graph[pos][city] + tsp(new_mask, city)
                result = min(result, cost)

        dp[(mask, pos)] = result
        return result

    return tsp(1, 0)
```

### **DP on Trees**

```python
# Problem: Tree DP - Maximum sum with no adjacent nodes
def rob_tree(root):
    def helper(node):
        if not node:
            return [0, 0]  # [rob, skip]

        left = helper(node.left)
        right = helper(node.right)

        rob = node.val + left[1] + right[1]
        skip = max(left) + max(right)

        return [rob, skip]

    return max(helper(root))
```

---

## 📊 Problem Patterns Summary

### **Linear DP (1D)**

1. Fibonacci-like problems
2. House robber variants
3. Decode ways
4. Coin change problems

### **Grid DP (2D)**

1. Path counting problems
2. Edit distance variants
3. Longest common subsequence
4. String matching

### **Interval DP**

1. Burst balloons
2. Matrix chain multiplication
3. Palindrome partitioning

### **State Compression**

1. Traveling salesman
2. Assignment problems
3. Subset problems with constraints

---

_Master these 100+ problems and you'll be ready for any DP challenge in interviews! 🚀_
