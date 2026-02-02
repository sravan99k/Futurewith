---
title: "Dynamic Programming Interview Questions"
level: "Interview Preparation"
difficulty: "Easy to Hard"
time: "30-90 minutes per session"
tags: ["dsa", "dynamic-programming", "dp", "interview", "coding-interview"]
---

# ⚡ Dynamic Programming Interview Questions

_Top 50+ Interview Questions with Solutions_

---

## 📊 Question Categories

| Category         | Count        | Difficulty  | Companies                 |
| ---------------- | ------------ | ----------- | ------------------------- |
| **1D Linear DP** | 15 questions | Easy-Medium | Google, Meta, Amazon      |
| **2D Grid DP**   | 10 questions | Medium      | Microsoft, Apple, Netflix |
| **String DP**    | 12 questions | Medium-Hard | Google, Amazon, Uber      |
| **Advanced DP**  | 8 questions  | Hard        | Google, Meta, ByteDance   |
| **Optimization** | 5 questions  | Hard        | All FAANG                 |

---

## 🌱 EASY LEVEL - Foundation Questions

### **Q1: Climbing Stairs (Meta, Google)**

**Difficulty:** ⭐ Easy | **Frequency:** Very High | **Time:** 15 minutes

```python
"""
PROBLEM:
You're climbing stairs. You can climb 1 or 2 steps at a time.
In how many distinct ways can you climb to the top?

EXAMPLES:
Input: n = 2  → Output: 2 (1+1, 2)
Input: n = 3  → Output: 3 (1+1+1, 1+2, 2+1)

FOLLOW-UP QUESTIONS:
1. What if you can climb 1, 2, or 3 steps?
2. What if some steps are broken (can't step on them)?
3. Can you solve in O(1) space?
"""

def climb_stairs(n):
    """
    Time: O(n), Space: O(1)
    Pattern: Each step is sum of previous 2 ways
    """
    if n <= 2:
        return n

    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr

    return prev1

# INTERVIEWER QUESTIONS TO EXPECT:
# "Explain why this is DP problem"
# "Can you optimize space?"
# "What if we have k steps instead of 2?"

# EXTENSION: k steps allowed
def climb_stairs_k_steps(n, k):
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i] += dp[i - j]

    return dp[n]

# Test cases
assert climb_stairs(2) == 2
assert climb_stairs(3) == 3
assert climb_stairs(4) == 5
```

### **Q2: House Robber (Amazon, Microsoft)**

**Difficulty:** ⭐ Easy | **Frequency:** High | **Time:** 20 minutes

```python
"""
PROBLEM:
Rob houses in a row, but can't rob adjacent houses.
Return maximum amount you can rob.

EXAMPLES:
Input: [2,7,9,3,1]  → Output: 12 (rob 2+9+1)
Input: [5,1,3,9]    → Output: 14 (rob 5+9)

FOLLOW-UP QUESTIONS:
1. What if houses are in a circle?
2. What if it's a binary tree?
3. Can you return which houses to rob?
"""

def rob(nums):
    """
    Time: O(n), Space: O(1)
    Pattern: At each house, choose max(rob, skip)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2, prev1 = nums[0], max(nums[0], nums[1])

    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr

    return prev1

# With path reconstruction
def rob_with_path(nums):
    if not nums:
        return 0, []

    n = len(nums)
    dp = [0] * n
    parent = [False] * n

    dp[0] = nums[0]
    parent[0] = True

    if n > 1:
        if nums[1] > nums[0]:
            dp[1] = nums[1]
            parent[1] = True
        else:
            dp[1] = nums[0]
            parent[1] = False

    for i in range(2, n):
        if dp[i-2] + nums[i] > dp[i-1]:
            dp[i] = dp[i-2] + nums[i]
            parent[i] = True
        else:
            dp[i] = dp[i-1]
            parent[i] = False

    # Reconstruct path
    path = []
    i = n - 1
    while i >= 0:
        if parent[i]:
            path.append(i)
            i -= 2
        else:
            i -= 1

    return dp[n-1], path[::-1]

# CIRCULAR HOUSE ROBBER
def rob_circular(nums):
    if len(nums) <= 2:
        return max(nums) if nums else 0

    # Case 1: Rob first house (can't rob last)
    case1 = rob(nums[:-1])

    # Case 2: Don't rob first house (can rob last)
    case2 = rob(nums[1:])

    return max(case1, case2)

# Test cases
assert rob([2,7,9,3,1]) == 12
assert rob([5,1,3,9]) == 14
```

### **Q3: Coin Change (Google, Amazon)**

**Difficulty:** ⭐ Easy-Medium | **Frequency:** Very High | **Time:** 25 minutes

```python
"""
PROBLEM:
Find minimum number of coins to make given amount.
You have infinite supply of each coin type.

EXAMPLES:
Input: coins = [1,3,4], amount = 6  → Output: 2 (3+3)
Input: coins = [2], amount = 3      → Output: -1 (impossible)

FOLLOW-UP QUESTIONS:
1. Return the actual coins used?
2. Count number of ways to make amount?
3. What if each coin can be used only once?
"""

def coin_change(coins, amount):
    """
    Time: O(amount × coins), Space: O(amount)
    Pattern: For each amount, try all coins
    """
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

# COUNT WAYS variation
def coin_change_ways(coins, amount):
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]

    return dp[amount]

# Test cases
assert coin_change([1,3,4], 6) == 2
assert coin_change([2], 3) == -1
assert coin_change_ways([1,2,5], 5) == 4
```

---

## ⚡ MEDIUM LEVEL - Core Interview Questions

### **Q4: Unique Paths (Meta, Google)**

**Difficulty:** ⚡ Medium | **Frequency:** High | **Time:** 20 minutes

```python
"""
PROBLEM:
Robot moves from top-left to bottom-right in m×n grid.
Can only move right or down. Count unique paths.

EXAMPLES:
Input: m = 3, n = 7  → Output: 28
Input: m = 3, n = 2  → Output: 3

FOLLOW-UP QUESTIONS:
1. What if some cells have obstacles?
2. What if robot can move in 4 directions?
3. Can you solve in O(min(m,n)) space?
"""

def unique_paths(m, n):
    """
    Time: O(m×n), Space: O(n) - optimized
    Pattern: paths[i][j] = paths[i-1][j] + paths[i][j-1]
    """
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j-1]

    return dp[n-1]

# WITH OBSTACLES
def unique_paths_obstacles(obstacle_grid):
    if not obstacle_grid or obstacle_grid[0][0] == 1:
        return 0

    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [0] * n
    dp[0] = 1

    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]

    return dp[n-1]

# MATHEMATICAL SOLUTION (Combination)
def unique_paths_math(m, n):
    from math import factorial
    return factorial(m + n - 2) // (factorial(m - 1) * factorial(n - 1))

# Test cases
assert unique_paths(3, 7) == 28
assert unique_paths(3, 2) == 3
```

### **Q5: Longest Increasing Subsequence (Google, Amazon)**

**Difficulty:** ⚡ Medium | **Frequency:** Very High | **Time:** 30 minutes

```python
"""
PROBLEM:
Find length of longest strictly increasing subsequence.

EXAMPLES:
Input: [10,9,2,5,3,7,101,18]  → Output: 4 ([2,3,7,18])
Input: [0,1,0,3,2,3]          → Output: 4 ([0,1,2,3])

FOLLOW-UP QUESTIONS:
1. Return the actual subsequence?
2. What about non-decreasing (≤ instead of <)?
3. Can you solve in O(n log n)?
"""

# O(n²) Solution
def length_of_lis_basic(nums):
    """
    Time: O(n²), Space: O(n)
    Pattern: For each element, check all previous smaller elements
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

# O(n log n) Optimized Solution
def length_of_lis_optimized(nums):
    """
    Time: O(n log n), Space: O(n)
    Use binary search to maintain sorted sequence
    """
    if not nums:
        return 0

    tails = []  # tails[i] = smallest ending element of LIS of length i+1

    for num in nums:
        # Binary search for position
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        # Extend or replace
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num

    return len(tails)

# WITH SUBSEQUENCE RECONSTRUCTION
def lis_with_sequence(nums):
    if not nums:
        return 0, []

    n = len(nums)
    dp = [1] * n
    parent = [-1] * n

    max_length = 1
    max_index = 0

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j

                if dp[i] > max_length:
                    max_length = dp[i]
                    max_index = i

    # Reconstruct sequence
    sequence = []
    current = max_index
    while current != -1:
        sequence.append(nums[current])
        current = parent[current]

    sequence.reverse()
    return max_length, sequence

# Test cases
assert length_of_lis_basic([10,9,2,5,3,7,101,18]) == 4
assert length_of_lis_optimized([0,1,0,3,2,3]) == 4
```

### **Q6: Edit Distance (Google, Meta)**

**Difficulty:** ⚡ Medium-Hard | **Frequency:** High | **Time:** 35 minutes

```python
"""
PROBLEM:
Convert word1 to word2 using minimum operations:
- Insert a character
- Delete a character
- Replace a character

EXAMPLES:
Input: word1 = "horse", word2 = "ros"  → Output: 3
Input: word1 = "intention", word2 = "execution"  → Output: 5

FOLLOW-UP QUESTIONS:
1. Return the actual operations?
2. What if operations have different costs?
3. Can you optimize space to O(min(m,n))?
"""

def min_distance(word1, word2):
    """
    Time: O(m×n), Space: O(m×n)
    Pattern: If chars match, no cost; else min of 3 operations
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters from word1
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters to make word2

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # Delete from word1
                    dp[i][j-1],    # Insert into word1
                    dp[i-1][j-1]   # Replace in word1
                )

    return dp[m][n]

# SPACE OPTIMIZED O(min(m,n))
def min_distance_optimized(word1, word2):
    # Ensure word2 is shorter for space optimization
    if len(word1) < len(word2):
        word1, word2 = word2, word1

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

# WITH OPERATION TRACKING
def edit_distance_with_ops(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize base cases
    for i in range(1, m + 1):
        dp[i][0] = i
        ops[i][0] = [f"Delete '{word1[j]}'" for j in range(i)]

    for j in range(1, n + 1):
        dp[0][j] = j
        ops[0][j] = [f"Insert '{word2[j-1]}'" for j in range(j)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = ops[i-1][j-1][:]
            else:
                costs = [
                    (dp[i-1][j] + 1, ops[i-1][j] + [f"Delete '{word1[i-1]}'"]),
                    (dp[i][j-1] + 1, ops[i][j-1] + [f"Insert '{word2[j-1]}'"]),
                    (dp[i-1][j-1] + 1, ops[i-1][j-1] + [f"Replace '{word1[i-1]}' with '{word2[j-1]}'"]),
                ]

                min_cost, min_ops = min(costs, key=lambda x: x[0])
                dp[i][j] = min_cost
                ops[i][j] = min_ops

    return dp[m][n], ops[m][n]

# Test cases
assert min_distance("horse", "ros") == 3
assert min_distance("intention", "execution") == 5
cost, operations = edit_distance_with_ops("horse", "ros")
assert cost == 3
```

---

## 🔥 HARD LEVEL - Advanced Interview Questions

### **Q7: Regular Expression Matching (Google, Meta)**

**Difficulty:** 🔥 Hard | **Frequency:** Medium | **Time:** 45 minutes

```python
"""
PROBLEM:
Implement regex matching with '.' and '*':
- '.' matches any single character
- '*' matches zero or more of the preceding character

EXAMPLES:
Input: s = "aa", p = "a"      → Output: False
Input: s = "aa", p = "a*"     → Output: True
Input: s = "ab", p = ".*"     → Output: True

FOLLOW-UP QUESTIONS:
1. What if we add '+' (one or more)?
2. Can you optimize space?
3. How to handle edge cases?
"""

def is_match(s, p):
    """
    Time: O(m×n), Space: O(m×n)
    Pattern: Handle * specially, . matches any char
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # Empty pattern matches empty string
    dp[0][0] = True

    # Handle patterns like a*, a*b*, a*b*c*
    for j in range(2, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '*':
                # Zero occurrences of preceding character
                dp[i][j] = dp[i][j-2]

                # One or more occurrences
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] |= dp[i-1][j]
            else:
                # Regular character or '.'
                if p[j-1] == '.' or p[j-1] == s[i-1]:
                    dp[i][j] = dp[i-1][j-1]

    return dp[m][n]

# RECURSIVE WITH MEMOIZATION
def is_match_recursive(s, p):
    memo = {}

    def dp(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        if j == len(p):
            result = i == len(s)
        else:
            first_match = i < len(s) and (p[j] == '.' or p[j] == s[i])

            if j + 1 < len(p) and p[j + 1] == '*':
                result = dp(i, j + 2) or (first_match and dp(i + 1, j))
            else:
                result = first_match and dp(i + 1, j + 1)

        memo[(i, j)] = result
        return result

    return dp(0, 0)

# Test cases
assert is_match("aa", "a") == False
assert is_match("aa", "a*") == True
assert is_match("ab", ".*") == True
assert is_match("aab", "c*a*b") == True
```

### **Q8: Burst Balloons (Google, ByteDance)**

**Difficulty:** 🔥 Hard | **Frequency:** Medium | **Time:** 50 minutes

```python
"""
PROBLEM:
Burst balloons to maximize coins. When you burst balloon i:
- You get nums[left] × nums[i] × nums[right] coins
- Balloons left and right then become adjacent

EXAMPLES:
Input: [3,1,5,8]  → Output: 167

STRATEGY:
Think backwards - which balloon to burst LAST in each interval?

FOLLOW-UP QUESTIONS:
1. What if balloons have negative values?
2. Can you optimize the solution?
3. Explain the interval DP pattern?
"""

def max_coins(nums):
    """
    Time: O(n³), Space: O(n²)
    Pattern: Interval DP - decide last balloon to burst
    """
    # Add 1s at boundaries to handle edge cases
    nums = [1] + nums + [1]
    n = len(nums)

    # dp[i][j] = max coins from bursting all balloons between i and j
    dp = [[0] * n for _ in range(n)]

    # l is the length of interval
    for l in range(2, n):
        for i in range(n - l):
            j = i + l

            # Try bursting each balloon k last in interval (i, j)
            for k in range(i + 1, j):
                coins = nums[i] * nums[k] * nums[j]
                dp[i][j] = max(dp[i][j], dp[i][k] + coins + dp[k][j])

    return dp[0][n-1]

# WITH PATH RECONSTRUCTION
def max_coins_with_path(nums):
    original_nums = nums[:]
    nums = [1] + nums + [1]
    n = len(nums)

    dp = [[0] * n for _ in range(n)]
    choice = [[0] * n for _ in range(n)]

    for l in range(2, n):
        for i in range(n - l):
            j = i + l
            for k in range(i + 1, j):
                coins = nums[i] * nums[k] * nums[j]
                total = dp[i][k] + coins + dp[k][j]
                if total > dp[i][j]:
                    dp[i][j] = total
                    choice[i][j] = k

    def get_order(i, j):
        if i + 1 >= j:
            return []
        k = choice[i][j]
        return get_order(i, k) + get_order(k, j) + [k - 1]  # -1 for original indexing

    return dp[0][n-1], get_order(0, n-1)

# Test case
assert max_coins([3,1,5,8]) == 167
coins, order = max_coins_with_path([3,1,5,8])
assert coins == 167
```

---

## 💡 Interview Strategy & Tips

### **Common Interview Flow:**

**Phase 1: Problem Understanding (5 min)**

- Clarify input/output
- Ask about edge cases
- Confirm examples

**Phase 2: Approach Discussion (10 min)**

- Identify this is DP problem
- Define state and transitions
- Discuss time/space complexity

**Phase 3: Implementation (20-30 min)**

- Start with recursive solution
- Add memoization
- Convert to bottom-up if time permits

**Phase 4: Optimization & Testing (10 min)**

- Optimize space if possible
- Test with examples
- Discuss follow-up questions

### **Key Interview Phrases:**

```python
# WHEN EXPLAINING DP:
"This problem has optimal substructure because..."
"There are overlapping subproblems when..."
"The state represents..."
"The transition equation is..."

# WHEN OPTIMIZING:
"I can optimize space by using only the previous row..."
"Since we only need the last k values, we can use variables..."
"The rolling array technique reduces space from O(mn) to O(n)..."
```

### **Red Flags to Avoid:**

❌ Jumping straight to coding without explanation
❌ Not identifying base cases clearly
❌ Confusing index boundaries
❌ Not discussing time/space complexity
❌ Ignoring edge cases

### **Bonus Points:**

✅ Multiple solution approaches
✅ Space optimization techniques
✅ Handling follow-up questions
✅ Clean, readable code with comments
✅ Comprehensive testing

---

_Master these patterns and you'll ace any DP interview! 🚀_
