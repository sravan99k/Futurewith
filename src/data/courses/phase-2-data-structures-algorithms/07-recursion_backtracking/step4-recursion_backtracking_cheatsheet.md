---
title: "Recursion & Backtracking Quick Reference Cheatsheet"
level: "All Levels"
time: "5 min quick reference"
tags: ["dsa", "recursion", "backtracking", "cheatsheet", "quick-reference"]
---

# 🔄 Recursion & Backtracking Cheatsheet

_Quick Reference for Interview & Practice_

---

## 📊 Complexity Guide

| Algorithm Type       | Time       | Space    | Example                |
| -------------------- | ---------- | -------- | ---------------------- |
| **Linear Recursion** | O(n)       | O(n)     | Factorial, Array sum   |
| **Tree Recursion**   | O(2^n)     | O(n)     | Fibonacci, Subsets     |
| **Divide & Conquer** | O(n log n) | O(log n) | Merge sort             |
| **Backtracking**     | O(b^d)     | O(d)     | N-Queens, Permutations |

_b = branching factor, d = depth_

---

## 🎯 Recursion Framework

### **Universal Recursion Template**

```python
def recursive_function(problem):
    # 1. BASE CASE - Stop condition
    if is_base_case(problem):
        return base_solution(problem)

    # 2. RECURSIVE CASE - Break down problem
    subproblem = make_smaller(problem)

    # 3. RECURSIVE CALL
    subresult = recursive_function(subproblem)

    # 4. COMBINE RESULTS
    return combine(subresult, current_level_work)
```

### **Common Recursion Patterns**

```python
# Pattern 1: Linear Recursion
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Pattern 2: Tree Recursion
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Pattern 3: Divide and Conquer
def binary_search(arr, target, left, right):
    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)
```

---

## 🔙 Backtracking Framework

### **Universal Backtracking Template**

```python
def backtrack(choices, path, result):
    # Base case: solution is complete
    if is_complete_solution(path):
        result.append(path.copy())
        return

    # Try all possible choices
    for choice in get_available_choices(choices, path):
        # Make choice
        path.append(choice)

        # Recurse with choice
        if is_valid_choice(choice, path):
            backtrack(choices, path, result)

        # Backtrack: undo choice
        path.pop()
```

### **Key Backtracking Patterns**

```python
# Pattern 1: Permutations
def permute(nums):
    result = []

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for num in nums:
            if num not in path:
                path.append(num)
                backtrack(path)
                path.pop()

    backtrack([])
    return result

# Pattern 2: Combinations
def combine(n, k):
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        for i in range(start, n + 1):
            path.append(i)
            backtrack(i + 1, path)
            path.pop()

    backtrack(1, [])
    return result

# Pattern 3: Subset Generation
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

---

## 🎨 Classic Problems & Solutions

### **N-Queens Problem**

```python
def solve_n_queens(n):
    result = []
    board = ['.' * n for _ in range(n)]

    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonals
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False

        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False

        return True

    def backtrack(row):
        if row == n:
            result.append(board[:])
            return

        for col in range(n):
            if is_safe(row, col):
                board[row] = board[row][:col] + 'Q' + board[row][col+1:]
                backtrack(row + 1)
                board[row] = board[row][:col] + '.' + board[row][col+1:]

    backtrack(0)
    return result
```

### **Generate Parentheses**

```python
def generate_parentheses(n):
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return

        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)

        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result
```

### **Word Search**

```python
def word_search(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(row, col, index):
        if index == len(word):
            return True

        if (row < 0 or row >= rows or col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False

        temp = board[row][col]
        board[row][col] = '#'  # Mark as visited

        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))

        board[row][col] = temp  # Restore
        return found

    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True

    return False
```

---

## ⚡ Optimization Techniques

### **1. Memoization**

```python
# Without memoization - O(2^n)
def fibonacci_slow(n):
    if n <= 1:
        return n
    return fibonacci_slow(n-1) + fibonacci_slow(n-2)

# With memoization - O(n)
def fibonacci_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]
```

### **2. Pruning**

```python
def combination_sum_pruned(candidates, target):
    result = []
    candidates.sort()  # Important for pruning

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            num = candidates[i]

            # Pruning: stop if number > remaining
            if num > remaining:
                break

            path.append(num)
            backtrack(i, path, remaining - num)
            path.pop()

    backtrack(0, [], target)
    return result
```

### **3. Early Termination**

```python
def n_queens_optimized(n):
    result = []

    # Use sets for O(1) conflict checking
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row, board):
        if row == n:
            result.append(board[:])
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board.append('.' * col + 'Q' + '.' * (n - col - 1))

            backtrack(row + 1, board)

            board.pop()
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0, [])
    return result
```

---

## 🚀 Problem Recognition

### **Use Recursion When:**

✅ Problem can be broken into similar subproblems
✅ Tree/graph traversal needed
✅ Mathematical recurrence exists
✅ Natural recursive structure

### **Use Backtracking When:**

✅ Need to find all solutions
✅ Constraint satisfaction problems
✅ Combinatorial optimization
✅ State space search

### **Avoid Recursion When:**

❌ Simple iteration suffices
❌ Deep recursion (stack overflow risk)
❌ No overlapping subproblems
❌ Tail recursion not optimized

---

## 🎯 Interview Tips

### **Recursion Checklist:**

1. **Base Case** - When to stop?
2. **Recursive Case** - How to break down?
3. **Progress** - Moving toward base case?
4. **Combine** - How to merge results?

### **Backtracking Checklist:**

1. **Choice** - What options are available?
2. **Constraint** - What makes choice valid?
3. **Goal** - When is solution complete?
4. **Backtrack** - How to undo choice?

### **Common Mistakes:**

❌ Missing base case
❌ Infinite recursion
❌ Not making progress
❌ Forgetting to backtrack
❌ Modifying shared state

### **Debugging Tips:**

```python
# Add trace to understand recursion
def trace_recursion(func):
    def wrapper(*args, **kwargs):
        wrapper.depth += 1
        indent = "  " * wrapper.depth
        print(f"{indent}→ {func.__name__}{args}")

        result = func(*args, **kwargs)

        print(f"{indent}← {result}")
        wrapper.depth -= 1
        return result

    wrapper.depth = -1
    return wrapper
```

---

## 📊 Time Complexity Cheatsheet

| Problem               | Time      | Space | Key Insight               |
| --------------------- | --------- | ----- | ------------------------- |
| **Factorial**         | O(n)      | O(n)  | Linear recursion          |
| **Fibonacci (naive)** | O(2^n)    | O(n)  | Tree recursion            |
| **Fibonacci (memo)**  | O(n)      | O(n)  | Memoization               |
| **Permutations**      | O(n!)     | O(n)  | n choices, then n-1, etc. |
| **Subsets**           | O(2^n)    | O(n)  | Each element: in or out   |
| **N-Queens**          | O(n!)     | O(n)  | Constraint satisfaction   |
| **Sudoku**            | O(9^(n²)) | O(n²) | 9 choices per empty cell  |

---

_Master recursion and backtracking patterns for coding interview success! 🚀_
