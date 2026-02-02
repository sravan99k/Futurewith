---
title: "Recursion & Backtracking Interview Questions"
level: "Interview Preparation"
difficulty: "Easy to Hard"
time: "30-90 minutes per session"
tags: ["dsa", "recursion", "backtracking", "interview", "coding-interview"]
---

# 🔄 Recursion & Backtracking Interview Questions

_Top 30+ Interview Questions with Solutions_

---

## 📊 Question Categories

| Category              | Count        | Difficulty  | Companies                 |
| --------------------- | ------------ | ----------- | ------------------------- |
| **Basic Recursion**   | 8 questions  | Easy        | All companies             |
| **Tree Recursion**    | 8 questions  | Medium      | Google, Meta, Amazon      |
| **Backtracking**      | 12 questions | Medium-Hard | Google, Amazon, Microsoft |
| **Advanced Patterns** | 8 questions  | Hard        | Google, Meta, ByteDance   |

---

## 🌱 EASY LEVEL - Foundation Questions

### **Q1: Factorial (All Companies)**

**Difficulty:** ⭐ Easy | **Frequency:** High | **Time:** 15 minutes

```python
"""
PROBLEM:
Calculate factorial of a non-negative integer n.
n! = n × (n-1) × (n-2) × ... × 1

EXAMPLES:
Input: n = 5  → Output: 120
Input: n = 0  → Output: 1

FOLLOW-UP QUESTIONS:
1. How to handle very large numbers?
2. Can you do it iteratively?
3. What about negative numbers?
"""

def factorial_recursive(n):
    """
    Time: O(n), Space: O(n)
    Pattern: Linear recursion with single recursive call
    """
    # Base case
    if n <= 1:
        return 1

    # Recursive case
    return n * factorial_recursive(n - 1)

def factorial_iterative(n):
    """
    Time: O(n), Space: O(1)
    More efficient for large n
    """
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def factorial_tail_recursive(n, accumulator=1):
    """
    Tail recursive version (optimized in some languages)
    """
    if n <= 1:
        return accumulator
    return factorial_tail_recursive(n - 1, n * accumulator)

# INTERVIEWER QUESTIONS TO EXPECT:
# "Why does recursion use more memory?"
# "What happens with very large n?"
# "Can you trace the execution?"

# Test cases
assert factorial_recursive(5) == 120
assert factorial_recursive(0) == 1
assert factorial_iterative(10) == 3628800
```

### **Q2: Fibonacci Number (Meta, Amazon)**

**Difficulty:** ⭐ Easy | **Frequency:** Very High | **Time:** 20 minutes

```python
"""
PROBLEM:
Find the nth Fibonacci number.
F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)

EXAMPLES:
Input: n = 6  → Output: 8
Input: n = 10 → Output: 55

FOLLOW-UP QUESTIONS:
1. Why is naive recursion slow?
2. How does memoization help?
3. Can you do it in O(1) space?
"""

def fibonacci_naive(n):
    """
    Time: O(2^n), Space: O(n)
    Classic example of inefficient recursion
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

def fibonacci_memoized(n, memo={}):
    """
    Time: O(n), Space: O(n)
    Demonstrates power of memoization
    """
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

def fibonacci_bottom_up(n):
    """
    Time: O(n), Space: O(1)
    Most efficient approach
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

# KEY INTERVIEW INSIGHTS:
# "Fibonacci demonstrates exponential recursion problem"
# "Memoization transforms O(2^n) to O(n)"
# "Bottom-up eliminates recursion overhead"

# Performance comparison demo
import time

# fibonacci_naive(35) takes ~3 seconds
# fibonacci_memoized(35) takes ~0.001 seconds
# fibonacci_bottom_up(35) takes ~0.0001 seconds
```

---

## ⚡ MEDIUM LEVEL - Core Interview Questions

### **Q3: Generate Parentheses (Google, Meta)**

**Difficulty:** ⚡ Medium | **Frequency:** Very High | **Time:** 30 minutes

```python
"""
PROBLEM:
Generate all valid combinations of n pairs of parentheses.

Input: n = 3
Output: ["((()))", "(()())", "(())()", "()(())", "()()()"]

FOLLOW-UP QUESTIONS:
1. How to ensure parentheses are valid?
2. Can you count valid combinations without generating?
3. What if we have different types of brackets?
"""

def generate_parentheses(n):
    """
    Time: O(4^n / √n), Space: O(n)
    Pattern: Constraint-based backtracking
    """
    result = []

    def backtrack(current, open_count, close_count):
        # Base case: used all parentheses
        if len(current) == 2 * n:
            result.append(current)
            return

        # Add opening parenthesis if we can
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)

        # Add closing parenthesis if it would be valid
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result

def count_valid_parentheses(n):
    """
    Count without generating - Catalan number
    """
    if n <= 1:
        return 1

    count = 0
    for i in range(n):
        count += count_valid_parentheses(i) * count_valid_parentheses(n - 1 - i)

    return count

def generate_with_multiple_types(n, types="()"):
    """
    Extension: multiple bracket types
    """
    result = []

    def backtrack(current, open_counts, close_counts):
        if len(current) == 2 * n * len(types):
            result.append(current)
            return

        # Try each bracket type
        for i, (open_br, close_br) in enumerate(types):
            # Add opening bracket
            if open_counts[i] < n:
                new_open = open_counts[:]
                new_open[i] += 1
                backtrack(current + open_br, new_open, close_counts)

            # Add closing bracket
            if close_counts[i] < open_counts[i]:
                new_close = close_counts[:]
                new_close[i] += 1
                backtrack(current + close_br, open_counts, new_close)

    backtrack("", [0] * len(types), [0] * len(types))
    return result

# ALGORITHM INSIGHTS:
# "Key insight: open_count ≤ n, close_count ≤ open_count"
# "This ensures valid parentheses at every step"
# "Classic example of constraint satisfaction"

# Test cases
result = generate_parentheses(3)
assert len(result) == 5
assert count_valid_parentheses(3) == 5
```

### **Q4: N-Queens (Google, Amazon)**

**Difficulty:** ⚡ Medium-Hard | **Frequency:** High | **Time:** 45 minutes

```python
"""
PROBLEM:
Place n queens on n×n chessboard so no two queens attack each other.

Input: n = 4
Output: 2 distinct solutions

FOLLOW-UP QUESTIONS:
1. How to optimize conflict detection?
2. Can you count solutions without generating?
3. What about N-Kings or N-Rooks?
"""

def solve_n_queens(n):
    """
    Time: O(n!), Space: O(n)
    Pattern: Constraint satisfaction with pruning
    """
    result = []
    board = ['.' * n for _ in range(n)]

    def is_safe(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

        # Check diagonal (top-left to bottom-right)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1

        # Check diagonal (top-right to bottom-left)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1

        return True

    def backtrack(row):
        # Base case: all queens placed
        if row == n:
            result.append([row[:] for row in board])
            return

        # Try placing queen in each column
        for col in range(n):
            if is_safe(row, col):
                # Place queen
                board[row] = board[row][:col] + 'Q' + board[row][col+1:]

                # Recurse to next row
                backtrack(row + 1)

                # Backtrack: remove queen
                board[row] = board[row][:col] + '.' + board[row][col+1:]

    backtrack(0)
    return result

def solve_n_queens_optimized(n):
    """
    Optimized version with O(1) conflict detection
    """
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
            # Quick conflict check using sets
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            board.append('.' * col + 'Q' + '.' * (n - col - 1))

            backtrack(row + 1, board)

            # Backtrack
            board.pop()
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0, [])
    return result

def count_n_queens(n):
    """Count solutions without storing them"""
    count = [0]  # Use list for mutable reference

    cols = set()
    diag1 = set()
    diag2 = set()

    def backtrack(row):
        if row == n:
            count[0] += 1
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            backtrack(row + 1)

            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count[0]

# OPTIMIZATION INSIGHTS:
# "Use sets instead of board scanning for O(1) conflict check"
# "Diagonal conflicts: row-col and row+col are constant"
# "Early pruning dramatically reduces search space"

# Test cases
solutions = solve_n_queens(4)
assert len(solutions) == 2
assert count_n_queens(8) == 92  # Famous result
```

### **Q5: Word Search (Amazon, Microsoft)**

**Difficulty:** ⚡ Medium | **Frequency:** High | **Time:** 35 minutes

```python
"""
PROBLEM:
Find if word exists in 2D character board.
Word must be formed by adjacent cells (horizontal/vertical).

Input: board = [["A","B","C","E"],
                ["S","F","C","S"],
                ["A","D","E","E"]]
       word = "ABCCED"
Output: True

FOLLOW-UP QUESTIONS:
1. What if we need to find all words?
2. Can we use the same cell multiple times?
3. How to optimize for multiple word searches?
"""

def word_search(board, word):
    """
    Time: O(N × 4^L), Space: O(L)
    where N = board size, L = word length
    Pattern: 2D backtracking with state modification
    """
    if not board or not board[0]:
        return False

    rows, cols = len(board), len(board[0])

    def backtrack(row, col, index):
        # Base case: found the word
        if index == len(word):
            return True

        # Check boundaries and character match
        if (row < 0 or row >= rows or col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False

        # Mark current cell as visited
        temp = board[row][col]
        board[row][col] = '#'

        # Explore all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))

        # Restore original value (backtrack)
        board[row][col] = temp

        return found

    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True

    return False

def word_search_with_visited_set(board, word):
    """
    Alternative using visited set (doesn't modify board)
    """
    if not board or not board[0]:
        return False

    rows, cols = len(board), len(board[0])

    def backtrack(row, col, index, visited):
        if index == len(word):
            return True

        if (row < 0 or row >= rows or col < 0 or col >= cols or
            (row, col) in visited or board[row][col] != word[index]):
            return False

        visited.add((row, col))

        found = (backtrack(row + 1, col, index + 1, visited) or
                backtrack(row - 1, col, index + 1, visited) or
                backtrack(row, col + 1, index + 1, visited) or
                backtrack(row, col - 1, index + 1, visited))

        visited.remove((row, col))  # Backtrack

        return found

    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0, set()):
                return True

    return False

# KEY INSIGHTS:
# "Critical to mark cells as visited to avoid cycles"
# "Must restore state after recursive call (backtrack)"
# "4^L worst case when exploring all directions"

# Test cases
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
assert word_search(board, "ABCCED") == True
assert word_search(board, "SEE") == True
assert word_search(board, "ABCB") == False
```

---

## 🔥 HARD LEVEL - Advanced Interview Questions

### **Q6: Sudoku Solver (Google, Meta)**

**Difficulty:** 🔥 Hard | **Frequency:** Medium | **Time:** 60 minutes

```python
"""
PROBLEM:
Solve a 9×9 Sudoku puzzle using backtracking.

CONSTRAINTS:
- Each row contains digits 1-9
- Each column contains digits 1-9
- Each 3×3 sub-box contains digits 1-9

FOLLOW-UP QUESTIONS:
1. How to optimize empty cell selection?
2. Can you validate if puzzle has unique solution?
3. How to generate valid Sudoku puzzles?
"""

def solve_sudoku(board):
    """
    Time: O(9^(empty_cells)), Space: O(1)
    Pattern: Constraint satisfaction with validation
    """

    def is_valid(board, row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False

        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False

        # Check 3×3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    # Try numbers 1-9
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num

                            if backtrack():
                                return True

                            board[i][j] = '.'  # Backtrack

                    return False  # No valid number found

        return True  # All cells filled

    backtrack()

def solve_sudoku_optimized(board):
    """
    Optimized with better cell selection strategy
    """

    def get_candidates(board, row, col):
        """Get possible candidates for empty cell"""
        used = set()

        # Check row, column, and box
        for i in range(9):
            if board[row][i] != '.':
                used.add(board[row][i])
            if board[i][col] != '.':
                used.add(board[i][col])

        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] != '.':
                    used.add(board[i][j])

        return [str(i) for i in range(1, 10) if str(i) not in used]

    def find_best_empty_cell():
        """Find empty cell with minimum candidates (MRV heuristic)"""
        best_cell = None
        min_candidates = 10

        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    candidates = get_candidates(board, i, j)
                    if len(candidates) < min_candidates:
                        min_candidates = len(candidates)
                        best_cell = (i, j, candidates)

                        if min_candidates == 0:
                            return best_cell  # Dead end

        return best_cell

    def backtrack():
        cell_info = find_best_empty_cell()

        if cell_info is None:
            return True  # No empty cells left

        row, col, candidates = cell_info

        if not candidates:
            return False  # No valid candidates

        for num in candidates:
            board[row][col] = num

            if backtrack():
                return True

            board[row][col] = '.'

        return False

    backtrack()

# OPTIMIZATION STRATEGIES:
# "Most Constraining Variable (MCV): choose cell with fewest candidates"
# "Constraint propagation: eliminate impossible values early"
# "Arc consistency: maintain constraints during search"

# Example board
sudoku_board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]

solve_sudoku(sudoku_board)
```

---

## 💡 Interview Strategy & Tips

### **Common Interview Flow:**

**Phase 1: Problem Understanding (5 min)**

```python
# Key questions to ask:
# "Should I use recursion or iteration?"
# "Are there any constraints on input size?"
# "Do you want all solutions or just one?"
# "Can I modify the input?"
```

**Phase 2: Approach Discussion (10 min)**

```python
# Framework explanation:
# "I'll use [recursion/backtracking] because..."
# "Base case will be..."
# "Recursive case will be..."
# "Time complexity will be..."
```

**Phase 3: Implementation (20-30 min)**

```python
# Implementation strategy:
# 1. Start with base case
# 2. Implement recursive case
# 3. Add constraint checking
# 4. Test with examples
```

### **Key Interview Phrases:**

```python
# When explaining recursion:
"The base case handles the simplest scenario..."
"Each recursive call works on a smaller subproblem..."
"We combine results from subproblems to solve original..."

# When explaining backtracking:
"We try each possibility systematically..."
"If a choice leads to dead end, we backtrack..."
"The key is to undo changes when backtracking..."
```

### **Common Pitfalls:**

❌ **Don't:**

- Forget base cases
- Create infinite recursion
- Forget to backtrack in backtracking problems
- Modify global state incorrectly

✅ **Do:**

- Trace execution with small examples
- Consider time/space complexity
- Handle edge cases
- Use clear variable names

### **Optimization Techniques:**

1. **Memoization** for overlapping subproblems
2. **Pruning** to eliminate invalid branches early
3. **Constraint propagation** in CSP problems
4. **Heuristics** for better search order

---

_Master these recursion and backtracking patterns for interview success! 🚀_
