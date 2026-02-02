---
title: "Dynamic Programming Quick Reference Cheatsheet"
level: "All Levels"
time: "5 min quick reference"
tags: ["dsa", "dynamic-programming", "dp", "cheatsheet", "quick-reference"]
---

# ⚡ Dynamic Programming Cheatsheet

_Quick Reference for Interview & Practice_

---

## 📊 DP Pattern Recognition

| Problem Type    | Key Phrases                       | Pattern                         | Example         |
| --------------- | --------------------------------- | ------------------------------- | --------------- |
| **Count Ways**  | "how many ways", "number of ways" | `dp[i] = sum(dp[j])`            | Climbing Stairs |
| **Min/Max**     | "minimum", "maximum", "optimal"   | `dp[i] = min/max(dp[j] + cost)` | Coin Change     |
| **True/False**  | "possible", "can we", "exists"    | `dp[i] = dp[j] OR condition`    | Word Break      |
| **Subsequence** | "subsequence", "substring"        | 2D DP                           | LCS, LIS        |

---

## 🎯 Time & Space Complexity

| Problem       | Naive           | DP Solution       | Space     | Optimized Space |
| ------------- | --------------- | ----------------- | --------- | --------------- |
| Fibonacci     | O(2^n)          | O(n)              | O(n)      | **O(1)**        |
| Coin Change   | O(amount^coins) | O(amount × coins) | O(amount) | O(amount)       |
| LCS           | O(2^(m+n))      | O(m × n)          | O(m × n)  | **O(min(m,n))** |
| Edit Distance | O(3^max(m,n))   | O(m × n)          | O(m × n)  | **O(min(m,n))** |

---

## 🔧 Essential DP Templates

### **Template 1: 1D Linear DP**

```python
def linear_dp(arr):
    n = len(arr)
    dp = [0] * n
    dp[0] = base_case

    for i in range(1, n):
        # Recurrence relation
        dp[i] = max/min/count(dp[j] + transition) for j < i

    return dp[n-1]

# Space Optimized (if only depends on previous few states)
def linear_dp_optimized(arr):
    prev2 = prev1 = base_case
    for i in range(1, n):
        curr = transition_function(prev2, prev1, arr[i])
        prev2, prev1 = prev1, curr
    return prev1
```

### **Template 2: 2D Grid DP**

```python
def grid_dp(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # Initialize base cases
    dp[0][0] = grid[0][0]

    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue

            dp[i][j] = grid[i][j] + min/max/sum(
                dp[i-1][j] if i > 0 else inf,
                dp[i][j-1] if j > 0 else inf
            )

    return dp[m-1][n-1]
```

### **Template 3: Subsequence DP**

```python
def subsequence_dp(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases (empty strings)
    for i in range(m + 1):
        dp[i][0] = base_value
    for j in range(n + 1):
        dp[0][j] = base_value

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + match_cost
            else:
                dp[i][j] = optimal(
                    dp[i-1][j] + delete_cost,
                    dp[i][j-1] + insert_cost,
                    dp[i-1][j-1] + replace_cost
                )

    return dp[m][n]
```

### **Template 4: Interval DP**

```python
def interval_dp(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]

    # Base case: single elements
    for i in range(n):
        dp[i][i] = base_value

    # Fill by interval length
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            dp[i][j] = optimal(
                dp[i][k] + dp[k+1][j] + cost(i, k, j)
                for k in range(i, j)
            )

    return dp[0][n-1]
```

---

## 🎨 Common DP Patterns

### **Pattern 1: Fibonacci-like**

```python
# Recurrence: f(n) = f(n-1) + f(n-2)
# Examples: Fibonacci, Climbing Stairs, House Robber

def fibonacci_pattern(n):
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1
```

### **Pattern 2: Decision at Each Step**

```python
# At each step: take it or leave it
# Examples: Knapsack, Subset Sum, Partition

def decision_dp(items, target):
    dp = [False] * (target + 1)
    dp[0] = True

    for item in items:
        for j in range(target, item - 1, -1):
            dp[j] = dp[j] or dp[j - item]

    return dp[target]
```

### **Pattern 3: Path in Grid**

```python
# Move right/down in grid
# Examples: Unique Paths, Min Path Sum

def path_dp(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    # First row
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    # First column
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]
```

### **Pattern 4: String Matching**

```python
# Compare two strings character by character
# Examples: Edit Distance, LCS, Regex Matching

def string_match_dp(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

---

## 🚀 Space Optimization Techniques

### **1D Array to Variables**

```python
# When dp[i] only depends on dp[i-1], dp[i-2]
prev2, prev1 = base1, base2
for i in range(2, n):
    curr = function(prev2, prev1)
    prev2, prev1 = prev1, curr
```

### **2D Array to 1D Array**

```python
# When dp[i][j] only depends on current and previous row
dp = [0] * (n + 1)
for i in range(m):
    for j in range(n-1, -1, -1):  # Reverse order
        dp[j] = function(dp[j], dp[j-1])
```

### **Rolling Array**

```python
# Use only two rows instead of full 2D array
prev = [0] * (n + 1)
curr = [0] * (n + 1)
for i in range(m):
    for j in range(n):
        curr[j] = function(prev[j], curr[j-1])
    prev, curr = curr, prev
```

---

## 🧠 Problem-Solving Framework

### **Step 1: Identify DP**

- [ ] Optimal substructure exists?
- [ ] Overlapping subproblems?
- [ ] Can break into smaller problems?

### **Step 2: Define State**

- What information do we need to store?
- `dp[i]` = answer for first i elements
- `dp[i][j]` = answer for elements i to j

### **Step 3: Find Recurrence**

- How does `dp[i]` relate to previous states?
- Base cases?
- Transition equation?

### **Step 4: Implementation Order**

- Top-down (memoization) or bottom-up (tabulation)?
- What order to fill the table?

### **Step 5: Optimize**

- Can we reduce space complexity?
- Can we avoid redundant calculations?

---

## 🎯 Quick Problem Identification

### **When to Use DP:**

✅ **Use DP when you see:**

- "optimal", "minimum", "maximum"
- "count number of ways"
- "is it possible"
- "longest", "shortest"
- Choices at each step affect future

❌ **Don't use DP when:**

- Problem has no optimal substructure
- Greedy algorithm works
- Simple iteration suffices

---

## 📝 Common Mistakes & Tips

### **Mistakes to Avoid:**

1. **Wrong base cases** - Check edge cases
2. **Wrong iteration order** - Dependencies matter
3. **Off-by-one errors** - Index carefully
4. **Memory limits** - Optimize space when needed

### **Pro Tips:**

1. **Start with recursion** - Add memoization later
2. **Draw small examples** - Understand the pattern
3. **Check boundaries** - Handle empty inputs
4. **Verify with examples** - Test your logic

---

## 🔥 Interview Essentials

### **Must-Know Problems:**

1. **Fibonacci** - Basic DP concept
2. **Climbing Stairs** - Classic 1D DP
3. **Coin Change** - Optimization DP
4. **House Robber** - Decision DP
5. **Unique Paths** - Grid DP
6. **Longest Common Subsequence** - 2D DP
7. **Edit Distance** - String DP
8. **Maximum Subarray** - Kadane's algorithm

### **Time Complexity Analysis:**

- **1D DP**: Usually O(n) or O(n²)
- **2D DP**: Usually O(m × n)
- **Interval DP**: Usually O(n³)
- **Bitmask DP**: Usually O(2ⁿ × n)

### **Space Optimization:**

- **Rolling array**: 2D → 1D
- **State compression**: Variables instead of array
- **In-place**: Modify input array

---

## ⚡ Quick Code Snippets

### **Memoization Decorator**

```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def dp_function(n):
    if n <= 1:
        return n
    return dp_function(n-1) + dp_function(n-2)
```

### **Bottom-up Template**

```python
def bottom_up_dp(n):
    dp = [0] * (n + 1)
    dp[0] = base_case_0
    dp[1] = base_case_1

    for i in range(2, n + 1):
        dp[i] = recurrence_relation(dp, i)

    return dp[n]
```

---

_Master these patterns and you'll solve any DP problem! 🚀_
