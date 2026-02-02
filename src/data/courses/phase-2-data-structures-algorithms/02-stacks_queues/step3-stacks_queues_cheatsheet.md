# 📝 Stacks & Queues - Quick Reference Cheatsheet

> **Interview Prep** | **Last-Minute Revision** | **Pattern Recognition**

---

## 🎯 Core Concepts (30 seconds)

### **Stack - LIFO (Last In, First Out)**

```
Think: Stack of plates 🍽️
Operations: push() O(1), pop() O(1), peek() O(1)
```

### **Queue - FIFO (First In, First Out)**

```
Think: Line at coffee shop ☕
Operations: enqueue() O(1), dequeue() O(1), front() O(1)
```

---

## ⚡ Quick Implementations

### **Stack (Python)**

```python
# List-based (fastest)
stack = []
stack.append(x)      # push
stack.pop()          # pop
stack[-1]            # peek
len(stack) == 0      # is_empty

# Using collections.deque
from collections import deque
stack = deque()
stack.append(x)      # push
stack.pop()          # pop
stack[-1]            # peek
```

### **Queue (Python)**

```python
# ALWAYS use deque for O(1) operations
from collections import deque

queue = deque()
queue.append(x)      # enqueue
queue.popleft()      # dequeue
queue[0]             # front
queue[-1]            # rear
len(queue) == 0      # is_empty
```

### **Priority Queue**

```python
import heapq

heap = []
heapq.heappush(heap, x)      # O(log n)
heapq.heappop(heap)          # O(log n)
heap[0]                       # peek min O(1)
```

---

## 🎨 Common Patterns (3 minutes)

### **Pattern 1: Monotonic Stack** ⭐⭐⭐

**Use When:** Finding next/previous greater/smaller element

```python
def next_greater_element(arr):
    """Find next greater element for each element"""
    result = [-1] * len(arr)
    stack = []  # Store indices

    for i in range(len(arr)):
        # While current is greater than stack top
        while stack and arr[i] > arr[stack[-1]]:
            result[stack.pop()] = arr[i]
        stack.append(i)

    return result

# Time: O(n), Space: O(n)
```

**Variations:**

```python
# Next Greater: while stack and arr[i] > arr[stack[-1]]
# Next Smaller: while stack and arr[i] < arr[stack[-1]]
# Previous Greater: iterate right to left
# Previous Smaller: iterate right to left
```

**LeetCode Problems:**

- Daily Temperatures (739)
- Next Greater Element I (496)
- Next Greater Element II (503)
- Largest Rectangle in Histogram (84)
- Trapping Rain Water (42)

---

### **Pattern 2: Valid Parentheses** ⭐⭐

**Use When:** Matching pairs, balanced brackets

```python
def is_valid(s: str) -> bool:
    """Check if parentheses are balanced"""
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
```

**LeetCode Problems:**

- Valid Parentheses (20)
- Remove Outermost Parentheses (1021)
- Score of Parentheses (856)
- Longest Valid Parentheses (32)

---

### **Pattern 3: Expression Evaluation** ⭐⭐⭐

**Use When:** Calculator problems, postfix evaluation

```python
def eval_rpn(tokens: list[str]) -> int:
    """Evaluate Reverse Polish Notation"""
    stack = []

    for token in tokens:
        if token in '+-*/':
            b = stack.pop()
            a = stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            else: stack.append(int(a / b))
        else:
            stack.append(int(token))

    return stack[0]
```

**Infix to Postfix:**

```python
def infix_to_postfix(expr):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    stack = []
    output = []

    for char in expr:
        if char.isalnum():
            output.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        else:  # Operator
            while stack and stack[-1] != '(' and \
                  precedence.get(stack[-1], 0) >= precedence[char]:
                output.append(stack.pop())
            stack.append(char)

    while stack:
        output.append(stack.pop())

    return ''.join(output)
```

**LeetCode Problems:**

- Evaluate RPN (150)
- Basic Calculator (224)
- Basic Calculator II (227)
- Basic Calculator III (772)

---

### **Pattern 4: Sliding Window Maximum** ⭐⭐

**Use When:** Finding max/min in sliding window

```python
from collections import deque

def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """Find maximum in each window of size k"""
    dq = deque()  # Stores indices
    result = []

    for i in range(len(nums)):
        # Remove elements outside window
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain decreasing order
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result after first window
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

### **Pattern 5: Stack Using Queue / Queue Using Stack** ⭐

**Use When:** Design problems

```python
# Queue using two stacks
class MyQueue:
    def __init__(self):
        self.input = []
        self.output = []

    def push(self, x):
        self.input.append(x)

    def pop(self):
        self._transfer()
        return self.output.pop()

    def _transfer(self):
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())

# Stack using one queue
class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
```

---

## 🔥 Interview Favorites (5 minutes)

### **1. Min Stack** ⭐⭐⭐

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1]) if self.min_stack else val
        self.min_stack.append(min_val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
```

---

### **2. LRU Cache** ⭐⭐⭐

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

### **3. Decode String** ⭐⭐

```python
def decodeString(s: str) -> str:
    stack = []
    curr_num = 0
    curr_str = ""

    for char in s:
        if char.isdigit():
            curr_num = curr_num * 10 + int(char)
        elif char == '[':
            stack.append((curr_str, curr_num))
            curr_str, curr_num = "", 0
        elif char == ']':
            prev_str, num = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += char

    return curr_str
```

---

### **4. Daily Temperatures** ⭐⭐

```python
def dailyTemperatures(temps):
    result = [0] * len(temps)
    stack = []

    for i, temp in enumerate(temps):
        while stack and temp > temps[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)

    return result
```

---

### **5. Trapping Rain Water** ⭐⭐⭐

```python
def trap(height):
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

## 📊 Complexity Cheatsheet

| Data Structure          | Push/Enqueue   | Pop/Dequeue | Peek | Search | Space |
| ----------------------- | -------------- | ----------- | ---- | ------ | ----- |
| **Stack (List)**        | O(1) amortized | O(1)        | O(1) | O(n)   | O(n)  |
| **Stack (Linked List)** | O(1)           | O(1)        | O(1) | O(n)   | O(n)  |
| **Queue (Deque)**       | O(1)           | O(1)        | O(1) | O(n)   | O(n)  |
| **Queue (List)**        | O(1)           | ❌ O(n)     | O(1) | O(n)   | O(n)  |
| **Priority Queue**      | O(log n)       | O(log n)    | O(1) | O(n)   | O(n)  |
| **Circular Queue**      | O(1)           | O(1)        | O(1) | O(n)   | O(k)  |

---

## 🎯 Problem Recognition Guide

**Use Stack When:**

- ✅ Need to reverse something
- ✅ Matching pairs (parentheses)
- ✅ Backtracking (undo operations)
- ✅ DFS traversal
- ✅ Expression evaluation
- ✅ Next greater/smaller element

**Use Queue When:**

- ✅ Order matters (FIFO)
- ✅ BFS traversal
- ✅ Scheduling tasks
- ✅ Streaming data
- ✅ Level-order traversal
- ✅ Recent calls/requests

**Use Deque When:**

- ✅ Need both ends access
- ✅ Sliding window max/min
- ✅ Palindrome checking
- ✅ Queue + Stack hybrid

**Use Priority Queue When:**

- ✅ Need min/max repeatedly
- ✅ K largest/smallest
- ✅ Merge K sorted
- ✅ Dijkstra's algorithm

---

## 🔧 Common Edge Cases

```python
# Always check:
- Empty stack/queue
- Single element
- All elements same
- Negative numbers
- Large numbers (overflow)
- Duplicate elements
- Invalid input

# Template:
if not stack:
    return default_value

if len(stack) == 1:
    return special_case
```

---

## 💡 Pro Tips

### **Stack Tips:**

1. Use dummy node for cleaner code
2. Track min/max with parallel stack
3. For reversal: push all, then pop all
4. Check bounds before pop/peek

### **Queue Tips:**

1. ALWAYS use `collections.deque`
2. For circular: use modulo `%`
3. Two stacks = queue (O(1) amortized)
4. Priority queue = heap

### **Optimization:**

1. In-place when possible
2. Monotonic stack for O(n)
3. Avoid list.pop(0) - use deque!
4. PreCompute when beneficial

---

## 🚀 Interview Speedrun (2 minutes)

### **Most Common Questions:**

1. Valid Parentheses - Stack with hashmap
2. Min Stack - Two stacks
3. Daily Temperatures - Monotonic stack
4. Implement Queue using Stacks - Two stacks
5. Decode String - Stack with numbers

### **Quick Decision Tree:**

```
Need last element? → Stack
Need first element? → Queue
Need max/min repeatedly? → Priority Queue
Finding next greater? → Monotonic Stack
Matching pairs? → Stack
BFS? → Queue
DFS? → Stack or Recursion
Sliding window max? → Monotonic Deque
```

---

## 📝 Before Interview Checklist

**Memorize:**

- [ ] Valid parentheses template
- [ ] Monotonic stack template
- [ ] Stack using queues
- [ ] Queue using stacks
- [ ] Min stack implementation

**Practice:**

- [ ] 5 easy problems
- [ ] 5 medium problems
- [ ] 2 hard problems
- [ ] 1 design problem

**Review:**

- [ ] Time complexities
- [ ] Edge cases
- [ ] When to use each structure
- [ ] Common patterns

---

## 🎓 Top 20 Must-Know Problems

| #   | Problem                      | Pattern            | Difficulty |
| --- | ---------------------------- | ------------------ | ---------- |
| 1   | Valid Parentheses            | Stack              | Easy       |
| 2   | Min Stack                    | Design             | Easy       |
| 3   | Implement Queue using Stacks | Design             | Easy       |
| 4   | Implement Stack using Queues | Design             | Easy       |
| 5   | Daily Temperatures           | Monotonic Stack    | Medium     |
| 6   | Next Greater Element I       | Monotonic Stack    | Easy       |
| 7   | Decode String                | Stack              | Medium     |
| 8   | Asteroid Collision           | Stack              | Medium     |
| 9   | Remove K Digits              | Monotonic Stack    | Medium     |
| 10  | Evaluate RPN                 | Stack              | Medium     |
| 11  | Simplify Path                | Stack              | Medium     |
| 12  | Basic Calculator             | Stack              | Hard       |
| 13  | Largest Rectangle            | Monotonic Stack    | Hard       |
| 14  | Trapping Rain Water          | Stack/Two Pointers | Hard       |
| 15  | Sliding Window Maximum       | Monotonic Deque    | Hard       |
| 16  | Score of Parentheses         | Stack              | Medium     |
| 17  | Validate Stack Sequences     | Stack              | Medium     |
| 18  | Online Stock Span            | Monotonic Stack    | Medium     |
| 19  | Design Circular Queue        | Design             | Medium     |
| 20  | LRU Cache                    | Design             | Medium     |

---

## 🔥 Last-Minute Cramming (5 minutes)

```python
# 1. Stack basics
stack = []
stack.append(x)  # push
x = stack.pop()  # pop
x = stack[-1]    # peek

# 2. Queue basics
from collections import deque
q = deque()
q.append(x)      # enqueue
x = q.popleft()  # dequeue

# 3. Valid parentheses
def isValid(s):
    stack = []
    pairs = {')':'(', '}':'{', ']':'['}
    for c in s:
        if c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
        else:
            stack.append(c)
    return not stack

# 4. Monotonic stack (next greater)
def nextGreater(arr):
    res = [-1] * len(arr)
    stack = []
    for i in range(len(arr)):
        while stack and arr[i] > arr[stack[-1]]:
            res[stack.pop()] = arr[i]
        stack.append(i)
    return res

# 5. Min stack
class MinStack:
    def __init__(self):
        self.s = []
        self.mins = []
    def push(self, x):
        self.s.append(x)
        self.mins.append(x if not self.mins else min(x, self.mins[-1]))
    def getMin(self):
        return self.mins[-1]
```

---

## 🎯 Final Tips

**During Interview:**

1. Clarify if LIFO or FIFO needed
2. Ask about duplicates, negatives
3. Discuss time/space tradeoffs
4. Start with brute force
5. Optimize using patterns

**Common Mistakes:**
❌ Using list.pop(0) for queue (O(n))
❌ Not checking empty before pop
❌ Forgetting to return from base case
❌ Off-by-one errors in circular queue
❌ Not handling edge cases

**Remember:**
✅ Stack = LIFO = Plate stack
✅ Queue = FIFO = Coffee line
✅ Monotonic stack = Next greater/smaller
✅ Two stacks = Queue
✅ Always use deque for queue!

---

**Good luck! You got this! 🚀**

_Print this page and keep it handy during interview prep!_
