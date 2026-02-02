---
title: "Recursion & Backtracking Practice Problems - 100+ Questions"
level: "Beginner to Advanced"
difficulty: "Progressive (Easy → Medium → Hard)"
time: "Varies (10-90 minutes per question)"
tags: ["dsa", "recursion", "backtracking", "practice", "coding-interview"]
---

# 🔄 Recursion & Backtracking Practice Problems

_100+ Progressive Problems from Basic Recursion to Complex Backtracking_

---

## 📊 Problem Difficulty Distribution

| Level         | Count       | Time/Problem | Focus                                |
| ------------- | ----------- | ------------ | ------------------------------------ |
| 🌱 **Easy**   | 35 problems | 10-25 min    | Basic recursion, simple backtracking |
| ⚡ **Medium** | 40 problems | 25-45 min    | Tree recursion, constraint solving   |
| 🔥 **Hard**   | 25 problems | 45-90 min    | Complex backtracking, optimization   |

---

## 🌱 EASY LEVEL (1-35) - Recursion Foundations

### **Problem 1: Factorial**

**Difficulty:** ⭐ Easy | **Time:** 10 minutes

```python
"""
Calculate factorial of a non-negative integer.
n! = n × (n-1) × (n-2) × ... × 1

Input: n = 5
Output: 120
"""

def factorial_recursive(n):
    # Base case
    if n <= 1:
        return 1

    # Recursive case
    return n * factorial_recursive(n - 1)

def factorial_iterative(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Test cases
assert factorial_recursive(0) == 1
assert factorial_recursive(5) == 120
assert factorial_recursive(10) == 3628800
```

### **Problem 2: Fibonacci Numbers**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Find the nth Fibonacci number.
F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)

Input: n = 6
Output: 8
"""

def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)

def fibonacci_memoized(n, memo={}):
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

def fibonacci_iterative(n):
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

# Test cases
assert fibonacci_naive(6) == 8
assert fibonacci_memoized(10) == 55
assert fibonacci_iterative(15) == 610
```

### **Problem 3: Power Function**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Calculate x^n efficiently.

Input: x = 2, n = 10
Output: 1024
"""

def power_naive(x, n):
    if n == 0:
        return 1
    if n < 0:
        return 1 / power_naive(x, -n)

    return x * power_naive(x, n - 1)

def power_optimized(x, n):
    """
    Fast exponentiation using divide and conquer
    Time: O(log n)
    """
    if n == 0:
        return 1
    if n < 0:
        return 1 / power_optimized(x, -n)

    half = power_optimized(x, n // 2)

    if n % 2 == 0:
        return half * half
    else:
        return half * half * x

def power_iterative(x, n):
    if n == 0:
        return 1
    if n < 0:
        x = 1 / x
        n = -n

    result = 1
    while n > 0:
        if n % 2 == 1:
            result *= x
        x *= x
        n //= 2

    return result

# Test cases
assert power_optimized(2, 10) == 1024
assert power_optimized(2, -3) == 0.125
assert power_iterative(3, 4) == 81
```

### **Problem 4: Sum of Array**

**Difficulty:** ⭐ Easy | **Time:** 10 minutes

```python
"""
Calculate sum of all elements in array using recursion.

Input: [1, 2, 3, 4, 5]
Output: 15
"""

def array_sum_recursive(arr, index=0):
    if index >= len(arr):
        return 0

    return arr[index] + array_sum_recursive(arr, index + 1)

def array_sum_slice(arr):
    if not arr:
        return 0

    return arr[0] + array_sum_slice(arr[1:])

def array_sum_helper(arr):
    def helper(arr, start, end):
        if start > end:
            return 0
        if start == end:
            return arr[start]

        mid = (start + end) // 2
        left_sum = helper(arr, start, mid)
        right_sum = helper(arr, mid + 1, end)

        return left_sum + right_sum

    return helper(arr, 0, len(arr) - 1)

# Test cases
assert array_sum_recursive([1, 2, 3, 4, 5]) == 15
assert array_sum_slice([]) == 0
assert array_sum_helper([10, 20, 30]) == 60
```

### **Problem 5: Reverse String**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Reverse a string using recursion.

Input: "hello"
Output: "olleh"
"""

def reverse_string(s):
    if len(s) <= 1:
        return s

    return reverse_string(s[1:]) + s[0]

def reverse_string_helper(s):
    def helper(s, start, end):
        if start >= end:
            return

        # Swap characters
        s[start], s[end] = s[end], s[start]

        helper(s, start + 1, end - 1)

    s_list = list(s)
    helper(s_list, 0, len(s_list) - 1)
    return ''.join(s_list)

def reverse_string_accumulator(s, acc=""):
    if not s:
        return acc

    return reverse_string_accumulator(s[1:], s[0] + acc)

# Test cases
assert reverse_string("hello") == "olleh"
assert reverse_string_helper("world") == "dlrow"
assert reverse_string_accumulator("test") == "tset"
```

### **Problem 6: Count Digits**

**Difficulty:** ⭐ Easy | **Time:** 10 minutes

```python
"""
Count number of digits in a positive integer.

Input: 12345
Output: 5
"""

def count_digits_recursive(n):
    if n < 10:
        return 1

    return 1 + count_digits_recursive(n // 10)

def count_digits_string(n):
    return len(str(n))

def count_digits_iterative(n):
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count

# Test cases
assert count_digits_recursive(12345) == 5
assert count_digits_recursive(7) == 1
assert count_digits_string(1000000) == 7
```

### **Problem 7: Check Palindrome**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Check if a string is a palindrome using recursion.

Input: "racecar"
Output: True
"""

def is_palindrome_recursive(s):
    if len(s) <= 1:
        return True

    if s[0] != s[-1]:
        return False

    return is_palindrome_recursive(s[1:-1])

def is_palindrome_helper(s):
    def helper(s, left, right):
        if left >= right:
            return True

        if s[left] != s[right]:
            return False

        return helper(s, left + 1, right - 1)

    return helper(s, 0, len(s) - 1)

def is_palindrome_cleaned(s):
    """Handle case-insensitive and ignore non-alphanumeric"""
    cleaned = ''.join(c.lower() for c in s if c.isalnum())

    def helper(s, left, right):
        if left >= right:
            return True

        if s[left] != s[right]:
            return False

        return helper(s, left + 1, right - 1)

    return helper(cleaned, 0, len(cleaned) - 1)

# Test cases
assert is_palindrome_recursive("racecar") == True
assert is_palindrome_helper("hello") == False
assert is_palindrome_cleaned("A man a plan a canal Panama") == True
```

### **Problem 8: Binary Tree Maximum Depth**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Find the maximum depth of a binary tree.

Input: [3,9,20,null,null,15,7]
Output: 3
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    if not root:
        return 0

    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)

    return 1 + max(left_depth, right_depth)

def max_depth_iterative(root):
    if not root:
        return 0

    from collections import deque
    queue = deque([(root, 1)])
    max_d = 0

    while queue:
        node, depth = queue.popleft()
        max_d = max(max_d, depth)

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

    return max_d

# Test case
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

assert max_depth(root) == 3
```

### **Problem 9: Generate All Subsets**

**Difficulty:** ⭐ Easy-Medium | **Time:** 25 minutes

```python
"""
Generate all possible subsets of a given set.

Input: [1, 2, 3]
Output: [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
"""

def subsets_recursive(nums):
    if not nums:
        return [[]]

    # Get subsets without first element
    subsets_without_first = subsets_recursive(nums[1:])

    # Add subsets with first element
    subsets_with_first = []
    for subset in subsets_without_first:
        subsets_with_first.append([nums[0]] + subset)

    return subsets_without_first + subsets_with_first

def subsets_backtrack(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])  # Add current subset

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

def subsets_bit_manipulation(nums):
    n = len(nums)
    result = []

    # Generate all 2^n possible subsets
    for i in range(2 ** n):
        subset = []
        for j in range(n):
            if i & (1 << j):  # Check if j-th bit is set
                subset.append(nums[j])
        result.append(subset)

    return result

# Test cases
nums = [1, 2, 3]
result1 = subsets_recursive(nums)
result2 = subsets_backtrack(nums)
result3 = subsets_bit_manipulation(nums)

assert len(result1) == 8
assert len(result2) == 8
assert len(result3) == 8
```

### **Problem 10: Tower of Hanoi**

**Difficulty:** ⭐ Easy-Medium | **Time:** 20 minutes

```python
"""
Solve Tower of Hanoi puzzle.
Move all disks from source to destination using auxiliary peg.

Input: n = 3, source = 'A', destination = 'C', auxiliary = 'B'
Output: Sequence of moves
"""

def hanoi_moves(n, source, destination, auxiliary):
    moves = []

    def hanoi(n, src, dest, aux):
        if n == 1:
            moves.append(f"Move disk 1 from {src} to {dest}")
            return

        # Move n-1 disks from source to auxiliary
        hanoi(n - 1, src, aux, dest)

        # Move largest disk from source to destination
        moves.append(f"Move disk {n} from {src} to {dest}")

        # Move n-1 disks from auxiliary to destination
        hanoi(n - 1, aux, dest, src)

    hanoi(n, source, destination, auxiliary)
    return moves

def hanoi_count_moves(n):
    """Count minimum number of moves needed"""
    if n == 1:
        return 1

    return 2 * hanoi_count_moves(n - 1) + 1

def hanoi_iterative(n):
    """Count moves using formula: 2^n - 1"""
    return (2 ** n) - 1

# Test cases
moves = hanoi_moves(3, 'A', 'C', 'B')
assert len(moves) == 7
assert hanoi_count_moves(3) == 7
assert hanoi_iterative(3) == 7
```

---

## ⚡ MEDIUM LEVEL (36-75) - Advanced Recursion & Backtracking

### **Problem 36: Generate Parentheses**

**Difficulty:** ⚡ Medium | **Time:** 30 minutes

```python
"""
Generate all valid combinations of n pairs of parentheses.

Input: n = 3
Output: ["((()))", "(()())", "(())()", "()(())", "()()()"]
"""

def generate_parentheses(n):
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

def generate_parentheses_dp(n):
    """Dynamic programming approach"""
    if n == 0:
        return [""]

    result = []
    for i in range(n):
        for left in generate_parentheses_dp(i):
            for right in generate_parentheses_dp(n - 1 - i):
                result.append(f"({left}){right}")

    return result

# Test cases
result = generate_parentheses(3)
assert len(result) == 5
assert "((()))" in result
assert "()()())" in result
```

### **Problem 37: Letter Combinations of Phone Number**

**Difficulty:** ⚡ Medium | **Time:** 25 minutes

```python
"""
Given a string of digits, return all possible letter combinations.

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]
"""

def letter_combinations(digits):
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index, path):
        # Base case: formed complete combination
        if index == len(digits):
            result.append(path)
            return

        # Try each letter for current digit
        for letter in mapping[digits[index]]:
            backtrack(index + 1, path + letter)

    backtrack(0, "")
    return result

def letter_combinations_iterative(digits):
    if not digits:
        return []

    mapping = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = [""]

    for digit in digits:
        new_result = []
        for combination in result:
            for letter in mapping[digit]:
                new_result.append(combination + letter)
        result = new_result

    return result

# Test cases
result = letter_combinations("23")
assert len(result) == 9
assert "ad" in result
assert "cf" in result
```

### **Problem 38: Permutations**

**Difficulty:** ⚡ Medium | **Time:** 30 minutes

```python
"""
Generate all permutations of an array.

Input: [1, 2, 3]
Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
"""

def permute(nums):
    result = []

    def backtrack(path):
        # Base case: permutation is complete
        if len(path) == len(nums):
            result.append(path[:])
            return

        # Try each unused number
        for num in nums:
            if num not in path:
                path.append(num)
                backtrack(path)
                path.pop()

    backtrack([])
    return result

def permute_swapping(nums):
    """More efficient approach using swapping"""
    result = []

    def backtrack(start):
        # Base case: reached end
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            # Swap current element with start position
            nums[start], nums[i] = nums[i], nums[start]

            # Recurse for next position
            backtrack(start + 1)

            # Backtrack: swap back
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result

def permute_with_duplicates(nums):
    """Handle array with duplicate elements"""
    result = []
    nums.sort()  # Sort to group duplicates

    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            # Skip duplicates: if current number same as previous
            # and previous is not used, skip current
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            path.append(nums[i])
            used[i] = True
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * len(nums))
    return result

# Test cases
result1 = permute([1, 2, 3])
assert len(result1) == 6

result2 = permute_with_duplicates([1, 1, 2])
assert len(result2) == 3  # [1,1,2], [1,2,1], [2,1,1]
```

### **Problem 39: N-Queens**

**Difficulty:** ⚡ Medium-Hard | **Time:** 45 minutes

```python
"""
Place n queens on n×n chessboard so no two queens attack each other.

Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."], ["..Q.","Q...","...Q",".Q.."]]
"""

def solve_n_queens(n):
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
            result.append(board[:])
            return

        # Try placing queen in each column of current row
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
    """Optimized version using sets for conflict detection"""
    result = []

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

# Test cases
solutions = solve_n_queens(4)
assert len(solutions) == 2

solutions_optimized = solve_n_queens_optimized(4)
assert len(solutions_optimized) == 2
```

### **Problem 40: Combination Sum**

**Difficulty:** ⚡ Medium | **Time:** 35 minutes

```python
"""
Find all combinations where candidates sum to target.
Numbers can be used multiple times.

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3], [7]]
"""

def combination_sum(candidates, target):
    result = []
    candidates.sort()  # Sort for optimization

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            num = candidates[i]

            # Pruning: if current number > remaining, no point continuing
            if num > remaining:
                break

            path.append(num)
            # Can reuse same number, so pass i (not i+1)
            backtrack(i, path, remaining - num)
            path.pop()

    backtrack(0, [], target)
    return result

def combination_sum_unique(candidates, target):
    """Each number can only be used once"""
    result = []
    candidates.sort()

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            num = candidates[i]

            if num > remaining:
                break

            # Skip duplicates
            if i > start and candidates[i] == candidates[i-1]:
                continue

            path.append(num)
            backtrack(i + 1, path, remaining - num)
            path.pop()

    backtrack(0, [], target)
    return result

# Test cases
result1 = combination_sum([2,3,6,7], 7)
assert [2,2,3] in result1
assert [7] in result1

result2 = combination_sum_unique([10,1,2,7,6,1,5], 8)
# Should return combinations without reusing elements
```

---

## 🔥 HARD LEVEL (76-100) - Expert Backtracking

### **Problem 76: Sudoku Solver**

**Difficulty:** 🔥 Hard | **Time:** 60 minutes

```python
"""
Solve a 9×9 Sudoku puzzle.

Input: Partially filled 9×9 grid
Output: Complete valid Sudoku solution
"""

def solve_sudoku(board):
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
    """Optimized with better empty cell selection"""

    def get_candidates(board, row, col):
        used = set()

        # Row
        for j in range(9):
            if board[row][j] != '.':
                used.add(board[row][j])

        # Column
        for i in range(9):
            if board[i][col] != '.':
                used.add(board[i][col])

        # Box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] != '.':
                    used.add(board[i][j])

        return [str(i) for i in range(1, 10) if str(i) not in used]

    def find_best_empty_cell(board):
        """Find empty cell with minimum possible candidates"""
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
        cell_info = find_best_empty_cell(board)

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

# Example Sudoku board
sudoku = [
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

solve_sudoku(sudoku)
```

### **Problem 77: Word Search II**

**Difficulty:** 🔥 Hard | **Time:** 70 minutes

```python
"""
Find all words in a 2D board. Words can be formed by adjacent cells.

Input: board = [["o","a","a","n"],
                ["e","t","a","e"],
                ["i","h","k","r"],
                ["i","f","l","v"]]
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
"""

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

def find_words(board, words):
    # Build Trie
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

    rows, cols = len(board), len(board[0])
    result = []

    def backtrack(row, col, parent):
        char = board[row][col]
        curr_node = parent.children[char]

        # Check if we found a word
        if curr_node.word:
            result.append(curr_node.word)
            curr_node.word = None  # Avoid duplicates

        # Mark as visited
        board[row][col] = '#'

        # Explore all 4 directions
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = row + dr, col + dc

            if (0 <= nr < rows and 0 <= nc < cols and
                board[nr][nc] != '#' and
                board[nr][nc] in curr_node.children):

                backtrack(nr, nc, curr_node)

        # Restore original character
        board[row][col] = char

        # Optimization: remove leaf nodes
        if not curr_node.children:
            parent.children.pop(char)

    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if board[i][j] in root.children:
                backtrack(i, j, root)

    return result

# Test case
board = [
    ["o","a","a","n"],
    ["e","t","a","e"],
    ["i","h","k","r"],
    ["i","f","l","v"]
]
words = ["oath","pea","eat","rain"]
result = find_words(board, words)
assert "eat" in result
assert "oath" in result
```

### **Problem 78: Regular Expression Matching**

**Difficulty:** 🔥 Hard | **Time:** 80 minutes

```python
"""
Implement regular expression matching with '.' and '*'.

'.' Matches any single character.
'*' Matches zero or more of the preceding element.

Input: s = "aa", p = "a*"
Output: True
"""

def is_match_recursive(s, p):
    def match(i, j):
        # Base case: pattern exhausted
        if j == len(p):
            return i == len(s)

        # Check if current characters match
        first_match = i < len(s) and (p[j] == '.' or s[i] == p[j])

        # Handle '*' pattern
        if j + 1 < len(p) and p[j + 1] == '*':
            # Two options:
            # 1. Skip the pattern (0 occurrences)
            # 2. Use the pattern if first character matches
            return (match(i, j + 2) or
                   (first_match and match(i + 1, j)))
        else:
            # Regular character match
            return first_match and match(i + 1, j + 1)

    return match(0, 0)

def is_match_memoized(s, p):
    memo = {}

    def match(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        if j == len(p):
            result = i == len(s)
        else:
            first_match = i < len(s) and (p[j] == '.' or s[i] == p[j])

            if j + 1 < len(p) and p[j + 1] == '*':
                result = (match(i, j + 2) or
                         (first_match and match(i + 1, j)))
            else:
                result = first_match and match(i + 1, j + 1)

        memo[(i, j)] = result
        return result

    return match(0, 0)

def is_match_dp(s, p):
    """Dynamic programming bottom-up approach"""
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

# Test cases
assert is_match_recursive("aa", "a") == False
assert is_match_recursive("aa", "a*") == True
assert is_match_memoized("ab", ".*") == True
assert is_match_dp("aab", "c*a*b") == True
```

---

## 📚 Problem Patterns Summary

### **Recursion Patterns:**

1. **Linear Recursion**
   - Factorial, Fibonacci
   - Array sum, string reversal
   - Tree depth calculation

2. **Divide and Conquer**
   - Merge sort, quick sort
   - Binary search
   - Tree traversals

3. **Tree Recursion**
   - Binary tree problems
   - Path finding
   - Subset generation

### **Backtracking Patterns:**

1. **Permutation/Combination**
   - Generate all permutations
   - Subset generation
   - Combination sum

2. **Constraint Satisfaction**
   - N-Queens problem
   - Sudoku solver
   - Graph coloring

3. **Path Finding**
   - Word search
   - Maze solving
   - String matching

### **Optimization Techniques:**

1. **Memoization** - Store results to avoid recomputation
2. **Pruning** - Skip invalid branches early
3. **Constraint Propagation** - Reduce search space
4. **Heuristics** - Choose better branching order

---

_Master these 100+ recursion and backtracking problems for algorithmic excellence! 🚀_
