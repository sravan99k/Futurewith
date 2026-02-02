# 🎯 Stacks & Queues - FAANG Interview Questions

> **Target Companies:** Google, Meta, Amazon, Apple, Microsoft, Netflix
> **Difficulty Level:** Easy to Hard
> **Time to Master:** 2-3 weeks

---

## 📋 Table of Contents

1. [Company-Specific Questions](#company-specific-questions)
2. [Top 25 FAANG Questions](#top-25-faang-questions)
3. [Design Pattern Questions](#design-pattern-questions)
4. [Advanced Scenarios](#advanced-scenarios)
5. [Interview Strategy](#interview-strategy)

---

## 🏢 Company-Specific Questions

### **Google Questions**

#### Q1: Decode String ⭐⭐⭐

**Asked:** 150+ times | **LeetCode:** 394

```python
def decodeString(s: str) -> str:
    """
    Decode: "3[a]2[bc]" -> "aaabcbc"
            "3[a2[c]]" -> "accaccacc"

    Google loves this - tests recursion + stack understanding
    """
    stack = []
    curr_num = 0
    curr_str = ""

    for char in s:
        if char.isdigit():
            curr_num = curr_num * 10 + int(char)
        elif char == '[':
            # Save current state
            stack.append((curr_str, curr_num))
            curr_str = ""
            curr_num = 0
        elif char == ']':
            # Restore and decode
            prev_str, num = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += char

    return curr_str

# Time: O(n), Space: O(n)
# Why Google asks: Tests nested structures, recursion thinking
```

**Follow-up Questions:**

- What if numbers can be very large?
- How to handle invalid input?
- Can you do it without stack? (Recursion)

---

#### Q2: Largest Rectangle in Histogram ⭐⭐⭐

**Asked:** 200+ times | **LeetCode:** 84

```python
def largestRectangleArea(heights: list[int]) -> int:
    """
    Find largest rectangle in histogram.

    Key insight: For each bar, find left and right boundaries
    Use monotonic increasing stack
    """
    stack = []
    max_area = 0
    heights.append(0)  # Sentinel to clear stack

    for i, h in enumerate(heights):
        # Pop all bars taller than current
        while stack and heights[stack[-1]] > h:
            height_idx = stack.pop()
            height = heights[height_idx]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area

# Time: O(n), Space: O(n)
# Google's favorite monotonic stack problem
```

**Visualization:**

```
Heights: [2, 1, 5, 6, 2, 3]

    6 █
  5 █ █
      █ █     3 █
  2 █ █ █ 2 █ █
  1 █ █ █ █ █ █

Largest: 10 (height=5, width=2 -> bars at index 2,3)
```

---

#### Q3: Trapping Rain Water ⭐⭐⭐

**Asked:** 300+ times | **LeetCode:** 42

```python
def trap(height: list[int]) -> int:
    """
    Calculate water trapped after rain.

    Approach 1: Two pointers (optimal)
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0

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

# Approach 2: Using Stack
def trap_stack(height: list[int]) -> int:
    """Stack approach - easier to explain in interview"""
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

# Time: O(n), Space: O(n) for stack, O(1) for two pointers
```

---

### **Meta (Facebook) Questions**

#### Q4: Simplify Path ⭐⭐

**Asked:** 100+ times | **LeetCode:** 71

```python
def simplifyPath(path: str) -> str:
    """
    Simplify Unix file path.
    "/a/./b/../../c/" -> "/c"

    Meta loves testing understanding of file systems
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

**Edge Cases:**

```python
assert simplifyPath("/../") == "/"
assert simplifyPath("/home//foo/") == "/home/foo"
assert simplifyPath("/a/./b/../../c/") == "/c"
```

---

#### Q5: Remove All Adjacent Duplicates II ⭐⭐

**Asked:** 80+ times | **LeetCode:** 1209

```python
def removeDuplicates(s: str, k: int) -> str:
    """
    Remove k consecutive duplicate characters.
    "deeedbbcccbdaa", k=3 -> "aa"

    Stack stores (char, count)
    """
    stack = []  # [(char, count)]

    for char in s:
        if stack and stack[-1][0] == char:
            stack[-1] = (char, stack[-1][1] + 1)
            if stack[-1][1] == k:
                stack.pop()
        else:
            stack.append((char, 1))

    return ''.join(char * count for char, count in stack)

# Time: O(n), Space: O(n)
```

---

#### Q6: Basic Calculator II ⭐⭐

**Asked:** 200+ times | **LeetCode:** 227

```python
def calculate(s: str) -> int:
    """
    Evaluate expression with +, -, *, /.
    "3+2*2" -> 7
    " 3/2 " -> 1

    Meta tests operator precedence understanding
    """
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
            elif sign == '*':
                stack.append(stack.pop() * num)
            else:
                # Python division: truncate toward zero
                stack.append(int(stack.pop() / num))

            if i < len(s) - 1:
                sign = char
            num = 0

    return sum(stack)

# Time: O(n), Space: O(n)
```

---

### **Amazon Questions**

#### Q7: Daily Temperatures ⭐⭐⭐

**Asked:** 250+ times | **LeetCode:** 739

```python
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    """
    Find days until warmer temperature.
    [73,74,75,71,69,72,76,73] -> [1,1,4,2,1,1,0,0]

    Classic monotonic stack - Amazon's favorite
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Indices of days

    for i, temp in enumerate(temperatures):
        # While current temp is warmer
        while stack and temp > temperatures[stack[-1]]:
            prev_day = stack.pop()
            result[prev_day] = i - prev_day
        stack.append(i)

    return result

# Time: O(n), Space: O(n)
```

**Why Amazon loves this:**

- Real-world application (temperature tracking)
- Tests monotonic stack pattern
- Common in system design (stock prices, metrics)

---

#### Q8: Sliding Window Maximum ⭐⭐⭐

**Asked:** 150+ times | **LeetCode:** 239

```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """
    Find maximum in each sliding window.
    nums = [1,3,-1,-3,5,3,6,7], k = 3
    Output: [3,3,5,5,6,7]

    Use monotonic deque (decreasing order)
    """
    dq = deque()  # Store indices
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside window
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (maintain decreasing)
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Add to result after first complete window
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Time: O(n), Space: O(k)
```

---

#### Q9: Online Stock Span ⭐⭐

**Asked:** 100+ times | **LeetCode:** 901

```python
class StockSpanner:
    """
    Calculate stock price span.
    Span = max consecutive days (ending today) with price <= today

    Amazon uses this in trading systems
    """
    def __init__(self):
        self.stack = []  # (price, span)

    def next(self, price: int) -> int:
        span = 1

        # Accumulate spans of smaller/equal prices
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]

        self.stack.append((price, span))
        return span

# Time: O(1) amortized, Space: O(n)

# Example:
# Input: [100, 80, 60, 70, 60, 75, 85]
# Output: [1, 1, 1, 2, 1, 4, 6]
```

---

### **Microsoft Questions**

#### Q10: Valid Parentheses ⭐⭐⭐

**Asked:** 500+ times | **LeetCode:** 20

```python
def isValid(s: str) -> bool:
    """
    Check if parentheses are balanced.

    Microsoft's most asked question - tests fundamentals
    """
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in pairs:  # Closing bracket
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:  # Opening bracket
            stack.append(char)

    return len(stack) == 0

# Time: O(n), Space: O(n)
```

**Variations Microsoft asks:**

```python
# 1. Minimum removals to make valid
def minRemoveToMakeValid(s: str) -> str:
    stack = []
    s = list(s)

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                s[i] = ''

    # Remove unmatched '('
    for i in stack:
        s[i] = ''

    return ''.join(s)

# 2. Longest valid parentheses
def longestValidParentheses(s: str) -> int:
    stack = [-1]
    max_len = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])

    return max_len
```

---

#### Q11: Implement Min Stack ⭐⭐⭐

**Asked:** 200+ times | **LeetCode:** 155

```python
class MinStack:
    """
    Stack supporting getMin in O(1).

    Microsoft loves design questions
    """
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1]) if self.min_stack else val
        self.min_stack.append(min_val)

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

# All operations O(1) time, O(n) space

# Space-optimized version:
class MinStackOptimized:
    def __init__(self):
        self.stack = []
        self.min_val = float('inf')

    def push(self, val: int) -> None:
        if val <= self.min_val:
            self.stack.append(self.min_val)
            self.min_val = val
        self.stack.append(val)

    def pop(self) -> None:
        if self.stack.pop() == self.min_val:
            self.min_val = self.stack.pop()

    def getMin(self) -> int:
        return self.min_val
```

---

## 🎯 Top 25 FAANG Questions

| #   | Problem                      | Company   | Difficulty | Pattern            |
| --- | ---------------------------- | --------- | ---------- | ------------------ |
| 1   | Valid Parentheses            | All       | Easy       | Stack              |
| 2   | Min Stack                    | Microsoft | Easy       | Design             |
| 3   | Daily Temperatures           | Amazon    | Medium     | Monotonic Stack    |
| 4   | Decode String                | Google    | Medium     | Stack              |
| 5   | Largest Rectangle            | Google    | Hard       | Monotonic Stack    |
| 6   | Trapping Rain Water          | Google    | Hard       | Stack/Two Pointers |
| 7   | Simplify Path                | Meta      | Medium     | Stack              |
| 8   | Basic Calculator II          | Meta      | Medium     | Stack              |
| 9   | Sliding Window Max           | Amazon    | Hard       | Monotonic Deque    |
| 10  | Online Stock Span            | Amazon    | Medium     | Monotonic Stack    |
| 11  | Implement Queue using Stacks | All       | Easy       | Design             |
| 12  | Asteroid Collision           | Meta      | Medium     | Stack              |
| 13  | Remove K Digits              | Google    | Medium     | Monotonic Stack    |
| 14  | Score of Parentheses         | Google    | Medium     | Stack              |
| 15  | Next Greater Element I       | Amazon    | Easy       | Monotonic Stack    |
| 16  | Validate Stack Sequences     | Microsoft | Medium     | Stack              |
| 17  | Design Circular Queue        | All       | Medium     | Design             |
| 18  | Evaluate RPN                 | All       | Medium     | Stack              |
| 19  | Maximal Rectangle            | Amazon    | Hard       | Monotonic Stack    |
| 20  | Basic Calculator             | Google    | Hard       | Stack              |
| 21  | Longest Valid Parentheses    | Google    | Hard       | Stack              |
| 22  | Remove Duplicates II         | Meta      | Medium     | Stack              |
| 23  | Car Fleet                    | Amazon    | Medium     | Monotonic Stack    |
| 24  | Sum of Subarray Minimums     | Google    | Medium     | Monotonic Stack    |
| 25  | Exclusive Time               | Meta      | Medium     | Stack              |

---

## 🏗️ Design Pattern Questions

### **Pattern 1: LRU Cache** ⭐⭐⭐

**Asked by:** Google (300+ times), Amazon (200+ times)

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used cache - O(1) get and put

    Why asked: Tests design skills + data structure knowledge
    """
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove least recently used

# Follow-up: Implement without OrderedDict (use Doubly Linked List + HashMap)
```

---

### **Pattern 2: Design Browser History** ⭐⭐

**Asked by:** Google, Amazon

```python
class BrowserHistory:
    """
    Simulate browser back/forward navigation.

    Use two stacks: back_stack and forward_stack
    """
    def __init__(self, homepage: str):
        self.back_stack = []
        self.current = homepage
        self.forward_stack = []

    def visit(self, url: str) -> None:
        self.back_stack.append(self.current)
        self.current = url
        self.forward_stack.clear()  # Clear forward history

    def back(self, steps: int) -> str:
        while steps > 0 and self.back_stack:
            self.forward_stack.append(self.current)
            self.current = self.back_stack.pop()
            steps -= 1
        return self.current

    def forward(self, steps: int) -> str:
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.current)
            self.current = self.forward_stack.pop()
            steps -= 1
        return self.current
```

---

### **Pattern 3: Design Hit Counter** ⭐⭐

**Asked by:** Google, Meta

```python
from collections import deque

class HitCounter:
    """
    Count hits in last 300 seconds.

    Use queue to track timestamps
    """
    def __init__(self):
        self.hits = deque()

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)

    def getHits(self, timestamp: int) -> int:
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        return len(self.hits)

# Time: O(1) amortized, Space: O(n)
```

---

## 🔥 Advanced Scenarios

### **Scenario 1: System Design Integration**

**Q: Design a task scheduler with priorities**

```python
import heapq
from collections import deque

class TaskScheduler:
    """
    Schedule tasks with cooldown period.

    Amazon asks this for distributed systems
    """
    def __init__(self, n: int):  # n = cooldown
        self.cooldown = n
        self.heap = []  # Min heap of (-frequency, task)
        self.queue = deque()  # (task, available_time)
        self.time = 0

    def leastInterval(self, tasks: list[str], n: int) -> int:
        # Count frequencies
        freq = {}
        for task in tasks:
            freq[task] = freq.get(task, 0) + 1

        # Build max heap
        heap = [-count for count in freq.values()]
        heapq.heapify(heap)

        time = 0
        while heap or queue:
            time += 1

            if heap:
                count = heapq.heappop(heap) + 1
                if count < 0:
                    queue.append((count, time + n))

            if queue and queue[0][1] == time:
                heapq.heappush(heap, queue.popleft()[0])

        return time
```

---

### **Scenario 2: Real-Time Data Processing**

**Q: Design streaming median calculator**

```python
import heapq

class MedianFinder:
    """
    Find median from data stream.

    Use two heaps: max_heap (left half), min_heap (right half)
    Google/Amazon ask for streaming data
    """
    def __init__(self):
        self.small = []  # Max heap (negative values)
        self.large = []  # Min heap

    def addNum(self, num: int) -> None:
        # Add to max heap
        heapq.heappush(self.small, -num)

        # Balance: ensure small <= large
        if self.small and self.large and (-self.small[0] > self.large[0]):
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)

        # Balance sizes
        if len(self.small) > len(self.large) + 1:
            val = -heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small):
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -val)

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0

# Time: O(log n) add, O(1) find
# Space: O(n)
```

---

## 💡 Interview Strategy

### **Before Coding:**

1. **Clarify requirements**
   - Is input always valid?
   - What's the size constraint?
   - Any duplicate elements?

2. **Choose right data structure**
   - LIFO behavior? → Stack
   - FIFO behavior? → Queue
   - Min/Max queries? → Priority Queue
   - Both ends? → Deque

3. **Identify pattern**
   - Matching pairs? → Stack with hashmap
   - Next greater? → Monotonic stack
   - Expression? → Stack with operators
   - Design? → Multiple data structures

### **During Coding:**

1. Handle edge cases first
2. Use meaningful variable names
3. Explain your approach
4. Test with examples

### **After Coding:**

1. Trace through example
2. Check edge cases
3. Analyze complexity
4. Discuss optimization

---

## 🎓 Common Mistakes to Avoid

❌ **Mistake 1:** Using list.pop(0) for queue

```python
# WRONG - O(n) time
queue = []
queue.pop(0)  # Slow!

# CORRECT - O(1) time
from collections import deque
queue = deque()
queue.popleft()  # Fast!
```

❌ **Mistake 2:** Not checking if empty

```python
# WRONG
def pop(self):
    return self.stack.pop()  # Crashes if empty!

# CORRECT
def pop(self):
    if not self.stack:
        raise IndexError("Stack is empty")
    return self.stack.pop()
```

❌ **Mistake 3:** Wrong monotonic stack condition

```python
# For NEXT GREATER
while stack and arr[i] > arr[stack[-1]]:  # Correct

# For NEXT SMALLER
while stack and arr[i] < arr[stack[-1]]:  # Correct
```

---

## 🚀 Final Preparation Checklist

**Week 1: Fundamentals**

- [ ] Implement stack (array & linked list)
- [ ] Implement queue (deque & linked list)
- [ ] Solve 10 easy problems
- [ ] Master valid parentheses pattern

**Week 2: Patterns**

- [ ] Master monotonic stack
- [ ] Solve 10 medium problems
- [ ] Practice design questions
- [ ] Learn expression evaluation

**Week 3: Advanced**

- [ ] Solve 5 hard problems
- [ ] Mock interviews
- [ ] Review company-specific questions
- [ ] Practice explaining solutions

---

## 📚 Key Takeaways

**What Interviewers Look For:**

1. ✅ Understanding of when to use stack vs queue
2. ✅ Ability to recognize patterns
3. ✅ Clean, bug-free implementation
4. ✅ Good communication
5. ✅ Complexity analysis

**Most Important Skills:**

1. Valid parentheses (100% must know)
2. Monotonic stack (80% of medium problems)
3. Design skills (common in senior roles)
4. Expression evaluation (calculator problems)
5. Two stacks = Queue (fundamental understanding)

---

**Good luck with your FAANG interviews! 🎯**

_Remember: Master the patterns, not individual problems!_
