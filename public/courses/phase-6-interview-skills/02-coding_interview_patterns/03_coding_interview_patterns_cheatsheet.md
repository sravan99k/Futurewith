# Coding Interview Patterns - Quick Reference Cheatsheet

## Pattern Recognition Quick Guide

### 🔍 Problem → Pattern Mapping

```markdown
**Array/String Keywords → Pattern:**
"sorted array" + "target sum" → Two Pointers
"subarray" + "maximum/minimum" → Sliding Window
"permutations" | "combinations" → Backtracking/Subsets
"rotated sorted array" → Modified Binary Search
"find missing" + "range [1,n]" → Cyclic Sort

**LinkedList Keywords → Pattern:**
"cycle" | "loop" → Fast & Slow Pointers
"reverse" | "k-group" → In-place Reversal
"merge sorted lists" → K-way Merge
"middle element" → Fast & Slow Pointers

**Tree Keywords → Pattern:**
"path sum" | "root to leaf" → Tree DFS
"level order" | "level by level" → Tree BFS
"diameter" | "height" → Tree DFS
"serialize/deserialize" → DFS + String Processing

**Optimization Keywords → Pattern:**
"top K" | "kth largest/smallest" → Top K Elements/Heap
"median" | "middle value" → Two Heaps
"overlapping subproblems" → Dynamic Programming
"dependencies" | "prerequisites" → Topological Sort

**Graph Keywords → Pattern:**
"connected components" → DFS/Union-Find
"shortest path" → BFS/Dijkstra
"course schedule" → Topological Sort
"islands" | "regions" → DFS/BFS
```

## Core Pattern Templates

### 🔄 Two Pointers

```python
# Opposite Direction
def two_pointers_opposite(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        curr_sum = arr[left] + arr[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return [-1, -1]

# Same Direction
def remove_duplicates(arr):
    slow = 0
    for fast in range(len(arr)):
        if arr[slow] != arr[fast]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1
```

### 🪟 Sliding Window

```python
# Fixed Window
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable Window
def longest_substring_no_repeats(s):
    left = 0
    char_map = {}
    max_length = 0

    for right in range(len(s)):
        if s[right] in char_map:
            left = max(left, char_map[s[right]] + 1)
        char_map[s[right]] = right
        max_length = max(max_length, right - left + 1)

    return max_length
```

### 🐰🐢 Fast & Slow Pointers

```python
# Cycle Detection
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Find Middle
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### ⏰ Merge Intervals

```python
def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1]:  # Overlap
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged
```

### 🔄 Cyclic Sort

```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        correct_pos = nums[i] - 1  # For range [1,n]
        if nums[i] != nums[correct_pos]:
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            i += 1
    return nums

# Find Missing Number
def find_missing(nums):
    i = 0
    while i < len(nums):
        if nums[i] < len(nums) and nums[i] != i:
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        else:
            i += 1

    for i in range(len(nums)):
        if nums[i] != i:
            return i
    return len(nums)
```

### 🔗 LinkedList Reversal

```python
# Basic Reversal
def reverse_list(head):
    prev = None
    current = head

    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    return prev

# Reverse Sublist
def reverse_between(head, left, right):
    # Skip first left-1 nodes
    current = head
    prev = None
    for _ in range(left - 1):
        prev = current
        current = current.next

    # Reverse sublist
    last_of_first = prev
    last_of_sublist = current

    for _ in range(right - left + 1):
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    # Connect parts
    if last_of_first:
        last_of_first.next = prev
    else:
        head = prev
    last_of_sublist.next = current

    return head
```

### 🌳 Tree DFS

```python
# Path Sum
def has_path_sum(root, target):
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == target

    return (has_path_sum(root.left, target - root.val) or
            has_path_sum(root.right, target - root.val))

# All Paths
def all_paths(root):
    def dfs(node, path, result):
        if not node:
            return

        path.append(node.val)

        if not node.left and not node.right:
            result.append(path[:])
        else:
            dfs(node.left, path, result)
            dfs(node.right, path, result)

        path.pop()  # Backtrack

    result = []
    dfs(root, [], result)
    return result
```

### 📊 Tree BFS

```python
def level_order(root):
    if not root:
        return []

    result = []
    queue = [root]

    while queue:
        level_size = len(queue)
        level_nodes = []

        for _ in range(level_size):
            node = queue.pop(0)
            level_nodes.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level_nodes)

    return result
```

### 🔢 Two Heaps

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.max_heap = []  # For smaller half
        self.min_heap = []  # For larger half

    def add_num(self, num):
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        # Rebalance
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        return -self.max_heap[0]
```

### 🔍 Modified Binary Search

```python
# Search Rotated Array
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1

# Find Peak Element
def find_peak(nums):
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid

    return left
```

### ⚡ XOR Pattern

```python
# Single Number
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Two Single Numbers
def two_single_numbers(nums):
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Find rightmost set bit
    rightmost_bit = xor_all & -xor_all

    num1 = num2 = 0
    for num in nums:
        if num & rightmost_bit:
            num1 ^= num
        else:
            num2 ^= num

    return [num1, num2]
```

### 🎯 Top K Elements

```python
# K Largest Elements
def k_largest(nums, k):
    import heapq
    return heapq.nlargest(k, nums)

# Using Min Heap
def k_largest_heap(nums, k):
    import heapq
    heap = []

    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)

    return heap
```

### 🔀 K-way Merge

```python
def merge_k_lists(lists):
    import heapq
    heap = []

    # Initialize heap
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

### 📋 Topological Sort

```python
# Kahn's Algorithm
def topological_sort(vertices, edges):
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = {i: 0 for i in range(vertices)}

    for parent, child in edges:
        graph[parent].append(child)
        in_degree[child] += 1

    queue = deque([v for v in range(vertices) if in_degree[v] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == vertices else []
```

## Time & Space Complexity Quick Reference

### ⏱️ Common Complexities

```markdown
**Time Complexity Hierarchy:**
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n) < O(n!)

**Pattern Complexities:**
Two Pointers: O(n) time, O(1) space
Sliding Window: O(n) time, O(k) space
Fast & Slow Pointers: O(n) time, O(1) space
Merge Intervals: O(n log n) time, O(1) space
Cyclic Sort: O(n) time, O(1) space
Tree DFS: O(n) time, O(h) space (h = height)
Tree BFS: O(n) time, O(w) space (w = width)
Binary Search: O(log n) time, O(1) space
Top K Elements: O(n log k) time, O(k) space
```

### 📊 Data Structure Operations

```markdown
**Array/List:**
Access: O(1) Search: O(n) Insert: O(n) Delete: O(n)

**Hash Table:**
Access: N/A Search: O(1)_ Insert: O(1)_ Delete: O(1)\*

**Binary Tree:**
Access: O(log n) Search: O(log n) Insert: O(log n) Delete: O(log n)

**Heap:**
Access: O(1) Search: O(n) Insert: O(log n) Delete: O(log n)

\*Average case
```

## Decision Trees

### 🌳 Array Problem Decision Tree

```
Looking at array problem:
├── Is array sorted?
│   ├── YES: Two Pointers or Binary Search
│   └── NO: Consider other patterns
├── Need subarray/substring?
│   ├── Fixed size: Sliding Window (fixed)
│   └── Variable size: Sliding Window (variable)
├── Finding pairs/triplets?
│   └── Two Pointers (after sorting if needed)
├── All combinations/permutations?
│   └── Backtracking/Subsets
└── Missing/duplicate in [1,n]?
    └── Cyclic Sort
```

### 🔗 LinkedList Problem Decision Tree

```
Looking at LinkedList problem:
├── Cycle-related?
│   └── Fast & Slow Pointers
├── Reversing nodes?
│   └── In-place Reversal
├── Finding position (middle, nth)?
│   └── Fast & Slow Pointers
├── Merging multiple lists?
│   └── K-way Merge
└── Palindrome checking?
    └── Fast & Slow + Reversal
```

### 🌲 Tree Problem Decision Tree

```
Looking at tree problem:
├── Level-by-level processing?
│   └── Tree BFS
├── Path-related (sum, count)?
│   └── Tree DFS
├── Tree properties (height, diameter)?
│   └── Tree DFS (often post-order)
├── Connecting same-level nodes?
│   └── Tree BFS
└── Serialization/Deserialization?
    └── Tree DFS + String processing
```

## Common Pitfalls & Solutions

### ⚠️ Two Pointers Pitfalls

```python
# ❌ Forgetting to check bounds
while left < right:  # Should be left < right, not left <= right

# ❌ Moving wrong pointer
if sum < target:
    right -= 1  # Should be left += 1

# ✅ Correct bounds and movement
while left < right:
    if sum < target:
        left += 1
    else:
        right -= 1
```

### ⚠️ Sliding Window Pitfalls

```python
# ❌ Not updating window state correctly
window_chars[s[left]] -= 1
left += 1  # Should check if count becomes 0

# ✅ Proper window state management
window_chars[s[left]] -= 1
if window_chars[s[left]] == 0:
    del window_chars[s[left]]
left += 1
```

### ⚠️ Tree Traversal Pitfalls

```python
# ❌ Not checking for null nodes
if root.left:  # Should check root first
    dfs(root.left)

# ✅ Proper null checking
if not root:
    return
if root.left:
    dfs(root.left)
```

### ⚠️ Binary Search Pitfalls

```python
# ❌ Integer overflow
mid = (left + right) / 2

# ✅ Avoid overflow
mid = left + (right - left) // 2

# ❌ Infinite loop conditions
while left <= right:
    if condition:
        left = mid  # Should be mid + 1

# ✅ Proper loop termination
while left < right:
    if condition:
        right = mid
    else:
        left = mid + 1
```

## Interview Strategy

### 🎯 Problem-Solving Framework

```markdown
**1. Understand (2 minutes):**

- Read problem carefully
- Identify inputs, outputs, constraints
- Ask clarifying questions
- Work through examples

**2. Plan (3-5 minutes):**

- Recognize the pattern(s)
- Discuss approach with interviewer
- Analyze time/space complexity
- Consider edge cases

**3. Code (15-20 minutes):**

- Start with brute force if complex
- Implement optimal solution
- Test with examples
- Handle edge cases

**4. Optimize (2-5 minutes):**

- Review for improvements
- Discuss alternative approaches
- Analyze final complexity
- Consider follow-up questions
```

### 🗣️ Communication Tips

```markdown
**While Coding:**

- Verbalize your thought process
- Explain why you choose certain patterns
- Mention trade-offs and alternatives
- Ask for feedback on approach

**Common Phrases:**

- "This looks like a [pattern] problem because..."
- "I'll use [pattern] since we need to..."
- "The time complexity is O(n) because..."
- "An edge case to consider is..."
- "Alternative approaches include..."
```

### 🚀 Last-Minute Checklist

```markdown
**Before Interview:**
□ Review pattern templates
□ Practice pattern recognition on 5 random problems
□ Refresh on complexity analysis
□ Prepare clarifying questions list

**During Problem Solving:**
□ Identify pattern within first 2 minutes
□ Explain approach before coding
□ Test solution with given examples
□ Consider edge cases (empty, single element, duplicates)
□ Analyze and state time/space complexity

**Code Quality:**
□ Use descriptive variable names
□ Add comments for complex logic
□ Handle edge cases explicitly
□ Validate inputs when appropriate
```

---

## Pattern Priority for Interviews

### 🥇 Must-Know Patterns (Practice Daily)

1. **Two Pointers** - Most versatile, many variations
2. **Sliding Window** - Common in string/array problems
3. **Tree DFS** - Essential for tree problems
4. **Binary Search** - Efficient search in sorted data
5. **Dynamic Programming** - Optimization problems

### 🥈 Important Patterns (Practice 2-3x/week)

6. **Tree BFS** - Level-order tree problems
7. **Fast & Slow Pointers** - Cycle detection, positioning
8. **Top K Elements** - Heap-based problems
9. **Merge Intervals** - Scheduling, range problems
10. **Backtracking** - Combinatorial problems

### 🥉 Good-to-Know Patterns (Practice 1x/week)

11. **K-way Merge** - Multi-source merging
12. **Topological Sort** - Dependency resolution
13. **Cyclic Sort** - Specific to [1,n] range problems
14. **XOR** - Mathematical bit manipulation
15. **LinkedList Reversal** - In-place modifications

**Remember:** Focus on understanding WHY patterns work, not just memorizing implementations. This enables adaptation to new problem variations during interviews.---

## 🔄 Common Confusions

### Confusion 1: Keyword Mapping vs. Problem Understanding

**The Confusion:** Using keyword matching to choose patterns without understanding the underlying problem structure and requirements.
**The Clarity:** Keywords are hints, not rules. You need to understand the problem's core requirements and data structure to choose the right approach.
**Why It Matters:** Different problems can have similar keywords but require different solutions. Surface-level pattern matching leads to wrong approaches and poor solutions.

### Confusion 2: Pattern Priority vs. Personal Comfort

**The Confusion:** Following suggested pattern priority without considering your own strengths, weaknesses, and the specific roles you're targeting.
**The Clarity:** Your personal pattern priority should be based on your current skill level, the companies you're interviewing with, and your comfort with different problem types.
**Why It Matters:** A personalized approach is more effective than generic advice. What matters most is improving your specific weaknesses while leveraging your strengths.

### Confusion 3: Memorization vs. Understanding Application

**The Confusion:** Memorizing the exact implementation of each pattern rather than understanding the underlying logic and being able to adapt it.
**The Clarity:** Cheatsheets should be memory aids, not replacement for understanding. You need to understand why each pattern works to apply it effectively.
**Why It Matters:** Interviewers will present variations that don't match memorized templates exactly. Without understanding, you can't adapt patterns to new situations.

### Confusion 4: Implementation Details vs. Core Logic

**The Confusion:** Focusing too much on specific implementation details (variable names, loop structures) rather than understanding the core algorithmic approach.
**The Clarity:** Different languages and situations require different implementations. The core logic and approach are more important than specific syntax.
**Why It Matters:** Real interviews allow language flexibility and expect you to focus on algorithmic thinking rather than specific implementation details.

### Confusion 5: Pattern Combinations vs. Single Patterns

**The Confusion:** Learning patterns in isolation without practicing how they combine to solve complex problems with multiple requirements.
**The Clarity:** Many interview problems require combining multiple patterns. Understanding how patterns work together is crucial for complex problem solving.
**Why It Matters:** Real systems and complex problems require multiple algorithmic approaches. Pattern combination skills demonstrate advanced problem-solving ability.

### Confusion 6: Time Complexity Focus vs. Practical Implementation

**The Confusion:** Focusing only on achieving optimal time complexity without considering implementation complexity and practical constraints.
**The Clarity:** In interviews, a simpler O(n²) solution that you can implement correctly is better than a complex O(n log n) solution with bugs.
**Why It Matters:** Working, correct solutions are better than optimized but broken solutions. Balance theoretical efficiency with practical implementation feasibility.

### Confusion 7: Edge Case Handling vs. Core Algorithm

**The Confusion:** Getting caught up in edge cases without ensuring the core algorithm is solid and the main problem is solved correctly.
**The Clarity:** Start with the core algorithm for the main case, then handle edge cases. Trying to handle everything at once often leads to confusion and bugs.
**Why It Matters:** Demonstrating systematic problem-solving and clear thinking is valued. Edge cases are important, but not at the expense of solving the main problem.

### Confusion 8: Interview vs. Real-World Application

**The Confusion:** Treating interview patterns as only relevant for interviews without understanding their practical applications in real software development.
**The Clarity:** Interview patterns represent fundamental algorithmic concepts that have real-world applications. Understanding both the interview value and practical use cases builds deeper expertise.
**Why It Matters:** This dual understanding helps you think about problems more holistically and demonstrates genuine technical depth, not just interview preparation.

## 📝 Micro-Quiz

### Question 1: When you see "subarray" with a condition like "sum equals target," the most likely pattern is:

A) Two Pointers
B) Sliding Window
C) Dynamic Programming
D) Hash Map
**Answer:** B
**Explanation:** Sliding Window is ideal for subarray problems with conditions. It efficiently maintains a running sum while sliding the window to find the desired condition.

### Question 2: For problems involving "finding the middle element," the recommended approach is:

A) Count total elements and divide by 2
B) Fast & Slow Pointers
C) Use an index variable
D) Convert to array and use random access
**Answer:** B
**Explanation:** Fast & Slow Pointers efficiently finds the middle element in O(n) time and O(1) space, without needing to count elements first or convert to an array.

### Question 3: When you need to "find all combinations/permutations" of a set, you should use:

A) Dynamic Programming
B) Backtracking
C) Iterative loops
D) Recursion without backtracking
**Answer:** B
**Explanation:** Backtracking is specifically designed for exploring all possible combinations and arrangements, making it the ideal choice for permutation and combination problems.

### Question 4: The key difference between Two Pointers and Sliding Window is:

A) Two Pointers works with sorted arrays, Sliding Window doesn't
B) Two Pointers moves pointers in same direction, Sliding Window in opposite
C) Two Pointers for pairs, Sliding Window for subarrays
D) There is no significant difference
**Answer:** C
**Explanation:** While both use pointers, Two Pointers typically finds pairs or works with opposite ends, while Sliding Window maintains a contiguous subarray with conditions.

### Question 5: For problems asking "find the kth smallest/largest element," you should consider:

A) Sorting the entire array
B) Top K Elements pattern (Heap/QuickSelect)
C) Linear search through array
D) Binary search in sorted array
**Answer:** B
**Explanation:** Top K Elements pattern (using heaps or QuickSelect) is more efficient than full sorting for finding kth elements, with O(n log k) or O(n) average time complexity.

### Question 6: The most important thing to focus on when using this cheatsheet is:

A) Memorizing all the keyword mappings exactly
B) Understanding the underlying logic of each pattern
C) Practicing implementation of each pattern multiple times
D) Knowing the exact time complexity of each pattern
**Answer:** B
**Explanation:** Understanding the underlying logic enables you to adapt patterns to new problems and variations. Memorization without understanding is fragile and doesn't transfer well to novel situations.

**Mastery Threshold:** 80% (5/6 correct)

## 💭 Reflection Prompts

1. **Personal Pattern Profile:** Analyze your own pattern strengths and weaknesses. Which patterns come naturally to you? Which ones challenge you? How does this profile influence your preparation strategy and interview readiness?

2. **Transfer Learning:** Think about a technical skill you've developed outside of coding patterns (like design patterns, frameworks, or tools). How did you develop this expertise? What strategies can you apply to pattern learning?

3. **Application Awareness:** Consider a recent project or problem you solved. Which coding interview patterns could have helped, even if you didn't use them explicitly? How do interview patterns relate to real-world problem-solving?

## 🏃 Mini Sprint Project (1-3 hours)

**Project: "Personalized Pattern Quick Reference"**

Create a customized pattern reference system tailored to your specific needs and preparation goals:

**Requirements:**

1. Adapt the general pattern priorities to your personal skill level and interview targets
2. Create a personal pattern application guide with your own examples and variations
3. Build a quick recognition checklist based on your common problem types
4. Design a performance tracking system for your pattern practice
5. Develop a personal improvement plan focused on your weakest patterns

**Deliverables:**

- Personalized pattern priority list
- Customized application guide with personal examples
- Quick recognition checklist for your common problems
- Performance tracking template
- Focused improvement plan for weak areas

## 🚀 Full Project Extension (10-25 hours)

**Project: "AI-Powered Pattern Recognition Assistant"**

Build an intelligent system that helps you quickly identify and apply the right patterns during interview preparation:

**Core System Components:**

1. **Problem Analysis Engine**: AI-powered analysis of problem statements to suggest likely patterns and approaches
2. **Pattern Recommendation System**: Personalized pattern suggestions based on problem characteristics and your skill level
3. **Real-Time Implementation Helper**: Step-by-step guidance for implementing chosen patterns with code examples
4. **Performance Optimization Advisor**: Suggestions for improving time/space complexity of your solutions
5. **Interview Simulation Assistant**: Practice problems with real-time pattern guidance and feedback

**Advanced Features:**

- Natural language processing for problem statement analysis
- Machine learning models trained on successful pattern applications
- Integration with coding practice platforms
- Real-time code analysis and pattern detection
- Personalized difficulty adjustment based on performance
- Collaborative pattern learning with peer matching
- Visual pattern representation and animation
- Mobile app for quick pattern reference during study

**Implementation Architecture:**

- Modern web application with AI/ML capabilities
- Natural language processing for problem analysis
- Real-time pattern recognition and recommendation
- Interactive coding environment with syntax highlighting
- Performance analytics and learning progress tracking
- Integration APIs for popular coding platforms
- Mobile-responsive design for cross-device access
- Cloud-based learning model that improves with usage

**Pattern Intelligence Features:**

- Automatic pattern identification from problem descriptions
- Difficulty level assessment for each pattern
- Common variation examples and solutions
- Time/space complexity analysis and optimization suggestions
- Common pitfalls and how to avoid them
- Real-world applications and use cases for each pattern
- Peer success patterns and solution approaches
- Interview-specific tips and communication strategies

**Expected Outcome:** An intelligent pattern recognition and application assistant that accelerates your learning, provides personalized guidance, and builds confidence in pattern selection and implementation during technical interviews.
