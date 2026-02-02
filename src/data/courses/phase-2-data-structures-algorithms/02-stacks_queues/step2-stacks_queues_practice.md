# 🏋️ Stacks & Queues - Practice Problems

> **Total Problems:** 100+ | **Difficulty Levels:** Easy → Medium → Hard
> **Estimated Time:** 20-30 hours to complete all problems

---

## 📋 Table of Contents

1. [Easy Problems (1-35)](#easy-problems)
2. [Medium Problems (36-75)](#medium-problems)
3. [Hard Problems (76-100)](#hard-problems)
4. [Solutions with Explanations](#solutions)

---

## 🟢 EASY PROBLEMS (1-35)

### **Basic Stack Operations**

#### Problem 1: Implement Stack Using Array

**Difficulty:** Easy | **Time:** 10 min | **LeetCode:** -

Implement a stack with push, pop, peek, and isEmpty operations.

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        """Add item to top of stack"""
        self.items.append(item)

    def pop(self):
        """Remove and return top item"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items.pop()

    def peek(self):
        """Return top item without removing"""
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.items[-1]

    def is_empty(self):
        """Check if stack is empty"""
        return len(self.items) == 0

    def size(self):
        """Return number of items"""
        return len(self.items)

# Time: O(1) for all operations
# Space: O(n) for n elements
```

---

#### Problem 2: Implement Queue Using Array

**Difficulty:** Easy | **Time:** 10 min

```python
from collections import deque

class Queue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        """Add item to rear"""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items.popleft()

    def front(self):
        """Return front item without removing"""
        if self.is_empty():
            raise IndexError("Queue is empty")
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

# Time: O(1) for all operations (using deque)
# Space: O(n)
```

---

#### Problem 3: Valid Parentheses ⭐

**Difficulty:** Easy | **Time:** 15 min | **LeetCode:** 20

Given a string containing characters '(', ')', '{', '}', '[', ']', determine if valid.

```python
def isValid(s: str) -> bool:
    """
    Valid means:
    - Open brackets closed in correct order
    - Same type brackets

    Example: "()" -> True, "()[]{}" -> True, "(]" -> False
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:  # Opening bracket
            stack.append(char)

    return len(stack) == 0

# Time: O(n), Space: O(n)
```

**Test Cases:**

```python
assert isValid("()") == True
assert isValid("()[]{}") == True
assert isValid("(]") == False
assert isValid("([)]") == False
assert isValid("{[]}") == True
```

---

#### Problem 4: Implement Stack Using Queues

**Difficulty:** Easy | **Time:** 15 min | **LeetCode:** 225

```python
from collections import deque

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        """Push element to stack (O(n))"""
        self.q.append(x)
        # Rotate queue to make new element at front
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        """Pop top element (O(1))"""
        return self.q.popleft()

    def top(self) -> int:
        """Get top element (O(1))"""
        return self.q[0]

    def empty(self) -> bool:
        """Check if empty (O(1))"""
        return len(self.q) == 0

# Time: push O(n), others O(1)
# Space: O(n)
```

---

#### Problem 5: Implement Queue Using Stacks

**Difficulty:** Easy | **Time:** 15 min | **LeetCode:** 232

```python
class MyQueue:
    def __init__(self):
        self.input_stack = []
        self.output_stack = []

    def push(self, x: int) -> None:
        """Push to input stack (O(1))"""
        self.input_stack.append(x)

    def pop(self) -> int:
        """Pop from output stack (amortized O(1))"""
        self._transfer()
        return self.output_stack.pop()

    def peek(self) -> int:
        """Peek from output stack (amortized O(1))"""
        self._transfer()
        return self.output_stack[-1]

    def empty(self) -> bool:
        """Check if both stacks empty (O(1))"""
        return not self.input_stack and not self.output_stack

    def _transfer(self):
        """Transfer from input to output if needed"""
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())

# Time: amortized O(1) for all operations
# Space: O(n)
```

---

#### Problem 6: Min Stack ⭐⭐

**Difficulty:** Easy | **Time:** 15 min | **LeetCode:** 155

Design stack supporting push, pop, top, and retrieving minimum in O(1).

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # Parallel stack tracking minimums

    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push current min to min_stack
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

# Time: O(1) for all operations
# Space: O(n) for min_stack
```

---

#### Problem 7: Remove All Adjacent Duplicates

**Difficulty:** Easy | **Time:** 10 min | **LeetCode:** 1047

```python
def removeDuplicates(s: str) -> str:
    """
    Remove adjacent duplicates repeatedly.
    Example: "abbaca" -> "ca"
    """
    stack = []

    for char in s:
        if stack and stack[-1] == char:
            stack.pop()  # Remove duplicate
        else:
            stack.append(char)

    return ''.join(stack)

# Time: O(n), Space: O(n)
```

---

#### Problem 8: Baseball Game

**Difficulty:** Easy | **Time:** 10 min | **LeetCode:** 682

```python
def calPoints(operations: list[str]) -> int:
    """
    Calculate points based on operations:
    - Integer: add to score
    - '+': sum of last two
    - 'D': double last score
    - 'C': remove last score
    """
    stack = []

    for op in operations:
        if op == '+':
            stack.append(stack[-1] + stack[-2])
        elif op == 'D':
            stack.append(stack[-1] * 2)
        elif op == 'C':
            stack.pop()
        else:
            stack.append(int(op))

    return sum(stack)

# Time: O(n), Space: O(n)
```

---

#### Problem 9: Backspace String Compare

**Difficulty:** Easy | **Time:** 15 min | **LeetCode:** 844

```python
def backspaceCompare(s: str, t: str) -> bool:
    """
    Compare strings after processing backspaces (#).
    Example: "ab#c", "ad#c" -> True
    """
    def build(s):
        stack = []
        for char in s:
            if char != '#':
                stack.append(char)
            elif stack:
                stack.pop()
        return ''.join(stack)

    return build(s) == build(t)

# Time: O(n + m), Space: O(n + m)
```

---

#### Problem 10: Next Greater Element I

**Difficulty:** Easy | **Time:** 15 min | **LeetCode:** 496

```python
def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    Find next greater element for each element in nums1 within nums2.
    """
    # Build next greater map for nums2
    next_greater = {}
    stack = []

    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)

    # Fill remaining with -1
    for num in stack:
        next_greater[num] = -1

    # Build result for nums1
    return [next_greater[num] for num in nums1]

# Time: O(n + m), Space: O(n)
```

---

### **Problems 11-20: Basic Applications**

#### Problem 11: Reverse String Using Stack

```python
def reverseString(s: str) -> str:
    stack = list(s)
    return ''.join(reversed(stack))
```

#### Problem 12: Reverse First K Elements of Queue

```python
from collections import deque

def reverseK(queue, k):
    if k <= 0 or k > len(queue):
        return queue

    stack = []
    # Push first k to stack
    for _ in range(k):
        stack.append(queue.popleft())

    # Push back from stack
    while stack:
        queue.append(stack.pop())

    # Move remaining to back
    for _ in range(len(queue) - k):
        queue.append(queue.popleft())

    return queue
```

#### Problem 13: Sort Stack

```python
def sortStack(stack):
    """Sort stack using another stack"""
    temp_stack = []

    while stack:
        temp = stack.pop()

        # Move larger elements back
        while temp_stack and temp_stack[-1] > temp:
            stack.append(temp_stack.pop())

        temp_stack.append(temp)

    return temp_stack
```

#### Problem 14: Check Balanced Parentheses

```python
def isBalanced(s: str) -> bool:
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0
```

#### Problem 15: Evaluate Postfix Expression

```python
def evalRPN(tokens: list[str]) -> int:
    """
    Evaluate Reverse Polish Notation.
    Example: ["2","1","+","3","*"] -> ((2 + 1) * 3) = 9
    """
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]
```

#### Problem 16: Recent Calls Counter

```python
from collections import deque

class RecentCounter:
    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        self.queue.append(t)
        # Remove calls older than 3000ms
        while self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)
```

#### Problem 17: Remove Outermost Parentheses

```python
def removeOuterParentheses(s: str) -> str:
    result = []
    count = 0

    for char in s:
        if char == '(':
            if count > 0:
                result.append(char)
            count += 1
        else:
            count -= 1
            if count > 0:
                result.append(char)

    return ''.join(result)
```

#### Problem 18: Make String Great

```python
def makeGood(s: str) -> str:
    """Remove adjacent chars with same letter but different case"""
    stack = []

    for char in s:
        if stack and stack[-1].swapcase() == char:
            stack.pop()
        else:
            stack.append(char)

    return ''.join(stack)
```

#### Problem 19: Build Array with Stack Operations

```python
def buildArray(target: list[int], n: int) -> list[str]:
    result = []
    current = 1

    for num in target:
        while current < num:
            result.extend(["Push", "Pop"])
            current += 1
        result.append("Push")
        current += 1

    return result
```

#### Problem 20: Maximum Nesting Depth

```python
def maxDepth(s: str) -> int:
    """Find maximum nesting depth of parentheses"""
    max_depth = 0
    current_depth = 0

    for char in s:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1

    return max_depth
```

---

## 🟡 MEDIUM PROBLEMS (36-75)

### **Problem 21: Daily Temperatures ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 739

```python
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    """
    Find days until warmer temperature.
    Example: [73,74,75,71,69,72,76,73] -> [1,1,4,2,1,1,0,0]

    Pattern: Monotonic stack (decreasing)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stores indices

    for i in range(n):
        # While current temp is warmer than stack top
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)

    return result

# Time: O(n), Space: O(n)
```

---

### **Problem 22: Decode String ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 394

```python
def decodeString(s: str) -> str:
    """
    Decode encoded string.
    Example: "3[a]2[bc]" -> "aaabcbc"
             "3[a2[c]]" -> "accaccacc"
    """
    stack = []
    current_num = 0
    current_str = ""

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Save current state
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            # Pop and decode
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char

    return current_str

# Time: O(n), Space: O(n)
```

---

### **Problem 23: Asteroid Collision ⭐⭐**

**Difficulty:** Medium | **Time:** 25 min | **LeetCode:** 735

```python
def asteroidCollision(asteroids: list[int]) -> list[int]:
    """
    Simulate asteroid collisions.
    Positive = moving right, Negative = moving left

    Example: [5, 10, -5] -> [5, 10]
             [8, -8] -> []
             [10, 2, -5] -> [10]
    """
    stack = []

    for asteroid in asteroids:
        while stack and asteroid < 0 < stack[-1]:
            # Collision occurs
            if stack[-1] < -asteroid:
                stack.pop()  # Stack asteroid destroyed
                continue
            elif stack[-1] == -asteroid:
                stack.pop()  # Both destroyed
            break  # Current asteroid destroyed
        else:
            stack.append(asteroid)

    return stack

# Time: O(n), Space: O(n)
```

---

### **Problem 24: Simplify Path ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 71

```python
def simplifyPath(path: str) -> str:
    """
    Simplify Unix file path.
    Example: "/home/" -> "/home"
             "/a/./b/../../c/" -> "/c"
    """
    stack = []

    for part in path.split('/'):
        if part == '..' and stack:
            stack.pop()
        elif part and part != '.' and part != '..':
            stack.append(part)

    return '/' + '/'.join(stack)

# Time: O(n), Space: O(n)
```

---

### **Problem 25: Remove K Digits ⭐⭐**

**Difficulty:** Medium | **Time:** 25 min | **LeetCode:** 402

```python
def removeKdigits(num: str, k: int) -> str:
    """
    Remove k digits to make smallest number.
    Example: "1432219", k=3 -> "1219"

    Pattern: Monotonic stack (increasing)
    """
    stack = []

    for digit in num:
        # Remove larger digits
        while stack and k > 0 and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)

    # Remove remaining k digits from end
    stack = stack[:len(stack) - k] if k > 0 else stack

    # Remove leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'

# Time: O(n), Space: O(n)
```

---

### **Problem 26: Online Stock Span ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 901

```python
class StockSpanner:
    """
    Calculate stock price span (consecutive days with price <= today).
    """
    def __init__(self):
        self.stack = []  # (price, span)

    def next(self, price: int) -> int:
        span = 1

        # Accumulate spans of smaller prices
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]

        self.stack.append((price, span))
        return span

# Time: O(1) amortized, Space: O(n)
```

---

### **Problem 27: Validate Stack Sequences ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 946

```python
def validateStackSequences(pushed: list[int], popped: list[int]) -> bool:
    """
    Check if popped sequence is valid for pushed sequence.
    """
    stack = []
    j = 0  # Pointer for popped

    for num in pushed:
        stack.append(num)
        # Try to match popped sequence
        while stack and stack[-1] == popped[j]:
            stack.pop()
            j += 1

    return len(stack) == 0

# Time: O(n), Space: O(n)
```

---

### **Problem 28: Design Circular Queue ⭐⭐**

**Difficulty:** Medium | **Time:** 25 min | **LeetCode:** 622

```python
class MyCircularQueue:
    def __init__(self, k: int):
        self.queue = [0] * k
        self.size = 0
        self.capacity = k
        self.front = 0
        self.rear = -1

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = value
        self.size += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True

    def Front(self) -> int:
        return -1 if self.isEmpty() else self.queue[self.front]

    def Rear(self) -> int:
        return -1 if self.isEmpty() else self.queue[self.rear]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.capacity
```

---

### **Problem 29: Score of Parentheses ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 856

```python
def scoreOfParentheses(s: str) -> int:
    """
    Calculate score: () = 1, AB = A + B, (A) = 2 * A
    Example: "(())" -> 2, "()()" -> 2, "(()(()))" -> 6
    """
    stack = [0]  # Start with base score

    for char in s:
        if char == '(':
            stack.append(0)  # New level
        else:
            curr = stack.pop()
            # () gives 1, (A) gives 2*A
            score = 1 if curr == 0 else 2 * curr
            stack[-1] += score

    return stack[0]
```

---

### **Problem 30: Next Greater Element II ⭐⭐**

**Difficulty:** Medium | **Time:** 20 min | **LeetCode:** 503

```python
def nextGreaterElements(nums: list[int]) -> list[int]:
    """
    Find next greater element in circular array.
    """
    n = len(nums)
    result = [-1] * n
    stack = []

    # Traverse array twice for circular
    for i in range(2 * n):
        idx = i % n
        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]
        if i < n:
            stack.append(idx)

    return result
```

---

### **Problems 31-50: Advanced Applications**

#### Problem 31: Largest Rectangle in Histogram ⭐⭐⭐

```python
def largestRectangleArea(heights: list[int]) -> int:
    """
    Find largest rectangle in histogram.
    Pattern: Monotonic stack
    """
    stack = []
    max_area = 0
    heights.append(0)  # Sentinel

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
```

#### Problem 32: Maximal Rectangle ⭐⭐⭐

```python
def maximalRectangle(matrix: list[list[str]]) -> int:
    """Use histogram approach for each row"""
    if not matrix:
        return 0

    max_area = 0
    heights = [0] * len(matrix[0])

    for row in matrix:
        for i, val in enumerate(row):
            heights[i] = heights[i] + 1 if val == '1' else 0
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area
```

#### Problem 33: Sliding Window Maximum ⭐⭐⭐

```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """
    Find maximum in each sliding window.
    Use monotonic deque (decreasing).
    """
    dq = deque()
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside window
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Add to result after first window
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## 🔴 HARD PROBLEMS (76-100)

### **Problem 76: Basic Calculator ⭐⭐⭐**

**Difficulty:** Hard | **Time:** 30 min | **LeetCode:** 224

```python
def calculate(s: str) -> int:
    """
    Evaluate expression with +, -, (, ).
    Example: "(1+(4+5+2)-3)+(6+8)" -> 23
    """
    stack = []
    num = 0
    sign = 1
    result = 0

    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '+':
            result += sign * num
            num = 0
            sign = 1
        elif char == '-':
            result += sign * num
            num = 0
            sign = -1
        elif char == '(':
            # Save current result and sign
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif char == ')':
            result += sign * num
            num = 0
            # Apply sign and add previous result
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result

    result += sign * num
    return result
```

---

### **Problem 77: Trapping Rain Water ⭐⭐⭐**

**Difficulty:** Hard | **Time:** 30 min | **LeetCode:** 42

```python
def trap(height: list[int]) -> int:
    """
    Calculate trapped rainwater.

    Approach 1: Using stacks
    """
    stack = []
    water = 0

    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = height[stack.pop()]

            if not stack:
                break

            distance = i - stack[-1] - 1
            bounded_height = min(height[stack[-1]], h) - bottom
            water += distance * bounded_height

        stack.append(i)

    return water

# Alternative: Two pointers (O(1) space)
def trap_two_pointers(height: list[int]) -> int:
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

---

### **Problem 78: Longest Valid Parentheses ⭐⭐⭐**

**Difficulty:** Hard | **Time:** 30 min | **LeetCode:** 32

```python
def longestValidParentheses(s: str) -> int:
    """
    Find length of longest valid parentheses substring.
    Example: "(()" -> 2, ")()())" -> 4
    """
    stack = [-1]  # Base for length calculation
    max_length = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)  # New base
            else:
                max_length = max(max_length, i - stack[-1])

    return max_length
```

---

### **Problem 79: Max Stack ⭐⭐⭐**

**Difficulty:** Hard | **Time:** 30 min

```python
import heapq

class MaxStack:
    """
    Stack supporting push, pop, top, peekMax, popMax in O(log n).
    """
    def __init__(self):
        self.stack = []
        self.max_heap = []
        self.removed = set()
        self.count = 0

    def push(self, x: int) -> None:
        self.stack.append((self.count, x))
        heapq.heappush(self.max_heap, (-x, -self.count))
        self.count += 1

    def pop(self) -> int:
        self._cleanup_stack()
        count, val = self.stack.pop()
        self.removed.add(count)
        return val

    def top(self) -> int:
        self._cleanup_stack()
        return self.stack[-1][1]

    def peekMax(self) -> int:
        self._cleanup_heap()
        return -self.max_heap[0][0]

    def popMax(self) -> int:
        self._cleanup_heap()
        val, count = heapq.heappop(self.max_heap)
        self.removed.add(-count)
        return -val

    def _cleanup_stack(self):
        while self.stack and self.stack[-1][0] in self.removed:
            self.stack.pop()

    def _cleanup_heap(self):
        while self.max_heap and -self.max_heap[0][1] in self.removed:
            heapq.heappop(self.max_heap)
```

---

### **Problem 80: Basic Calculator III ⭐⭐⭐**

**Difficulty:** Hard | **Time:** 40 min

```python
def calculate(s: str) -> int:
    """
    Evaluate expression with +, -, *, /, (, ).
    """
    def helper(s, i):
        stack = []
        num = 0
        sign = '+'

        while i < len(s):
            char = s[i]

            if char.isdigit():
                num = num * 10 + int(char)
            elif char == '(':
                num, i = helper(s, i + 1)

            if char in '+-*/)' or i == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    stack.append(int(stack.pop() / num))

                if char == ')':
                    return sum(stack), i

                sign = char
                num = 0

            i += 1

        return sum(stack), i

    return helper(s, 0)[0]
```

---

## 📚 Additional Practice Problems (81-100)

**Problem 81:** Car Fleet
**Problem 82:** Sum of Subarray Minimums
**Problem 83:** Remove Duplicate Letters
**Problem 84:** Shortest Unsorted Continuous Subarray
**Problem 85:** Design Hit Counter
**Problem 86:** Ternary Expression Parser
**Problem 87:** Exclusive Time of Functions
**Problem 88:** Flatten Nested List Iterator
**Problem 89:** Mini Parser
**Problem 90:** Design Snake Game
**Problem 91:** Moving Average from Data Stream
**Problem 92:** Design File System
**Problem 93:** Design In-Memory File System
**Problem 94:** Binary Tree Zigzag Level Order
**Problem 95:** Reverse Substrings Between Parentheses
**Problem 96:** Find the Most Competitive Subsequence
**Problem 97:** Constrained Subsequence Sum
**Problem 98:** Number of Visible People in Queue
**Problem 99:** Maximum Frequency Stack
**Problem 100:** Maximum Width Ramp

---

## 🎯 Problem-Solving Patterns

### **Pattern 1: Monotonic Stack**

Used when you need to find next/previous greater/smaller element.

**Template:**

```python
def monotonic_stack(arr):
    stack = []
    result = []

    for i, val in enumerate(arr):
        # Maintain monotonic property
        while stack and arr[stack[-1]] < val:  # or > for decreasing
            stack.pop()

        # Process current element
        result.append(stack[-1] if stack else -1)
        stack.append(i)

    return result
```

**Use Cases:**

- Next Greater Element
- Daily Temperatures
- Largest Rectangle
- Trapping Rain Water

---

### **Pattern 2: Expression Evaluation**

Used for calculator problems.

**Template:**

```python
def evaluate_expression(s):
    stack = []
    num = 0
    sign = '+'

    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)

        if char in '+-*/' or i == len(s) - 1:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            # ... handle *, /

            sign = char
            num = 0

    return sum(stack)
```

---

### **Pattern 3: Valid Parentheses Variants**

Check balanced brackets, calculate scores, etc.

**Template:**

```python
def valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return not stack
```

---

## 💡 Time Complexity Guide

| Operation    | Stack | Queue | Deque |
| ------------ | ----- | ----- | ----- |
| Push/Enqueue | O(1)  | O(1)  | O(1)  |
| Pop/Dequeue  | O(1)  | O(1)  | O(1)  |
| Peek         | O(1)  | O(1)  | O(1)  |
| Search       | O(n)  | O(n)  | O(n)  |

**Space Complexity:** Generally O(n) for n elements

---

## 🚀 Practice Schedule

**Week 1:** Easy problems (1-35)

- Day 1-2: Basic implementations
- Day 3-4: Valid parentheses variants
- Day 5-6: Simple applications
- Day 7: Review

**Week 2:** Medium problems (36-75)

- Day 1-2: Monotonic stack
- Day 3-4: Design problems
- Day 5-6: Expression evaluation
- Day 7: Mock interview

**Week 3:** Hard problems (76-100)

- Day 1-2: Calculator problems
- Day 3-4: Advanced monotonic stack
- Day 5-6: Complex design
- Day 7: Final review

---

**Total Practice Time:** 30-40 hours
**Problems to Master:** 100+
**Interview Readiness:** 3-4 weeks

Good luck with your practice! 🎯
