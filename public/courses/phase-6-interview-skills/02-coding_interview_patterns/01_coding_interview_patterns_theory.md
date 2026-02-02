# Coding Interview Patterns - Theory & Fundamentals

## Table of Contents

1. [Introduction to Coding Patterns](#introduction-to-coding-patterns)
2. [Pattern Recognition Framework](#pattern-recognition-framework)
3. [Two Pointers Pattern](#two-pointers-pattern)
4. [Sliding Window Pattern](#sliding-window-pattern)
5. [Fast & Slow Pointers Pattern](#fast--slow-pointers-pattern)
6. [Merge Intervals Pattern](#merge-intervals-pattern)
7. [Cyclic Sort Pattern](#cyclic-sort-pattern)
8. [In-place Reversal of LinkedList](#in-place-reversal-of-linkedlist)
9. [Tree Depth First Search](#tree-depth-first-search)
10. [Tree Breadth First Search](#tree-breadth-first-search)
11. [Two Heaps Pattern](#two-heaps-pattern)
12. [Subsets Pattern](#subsets-pattern)
13. [Modified Binary Search](#modified-binary-search)
14. [Bitwise XOR Pattern](#bitwise-xor-pattern)
15. [Top K Elements Pattern](#top-k-elements-pattern)
16. [K-way Merge Pattern](#k-way-merge-pattern)
17. [Topological Sort Pattern](#topological-sort-pattern)
18. [Dynamic Programming Patterns](#dynamic-programming-patterns)
19. [Graph Algorithms Patterns](#graph-algorithms-patterns)
20. [Advanced Patterns & Combinations](#advanced-patterns--combinations)

## Introduction to Coding Patterns

### What Are Coding Patterns?

Coding patterns are reusable problem-solving templates that help identify the underlying structure of algorithmic problems. Instead of memorizing hundreds of individual problems, understanding patterns allows you to recognize similar problem types and apply proven solution approaches.

### Why Patterns Matter in Interviews

```markdown
**Benefits of Pattern-Based Thinking:**
• Faster problem recognition and solution development
• Consistent approach to unknown problems
• Better communication with interviewers
• Reduced anxiety through familiar frameworks
• More efficient preparation and practice

**Pattern vs. Memorization:**
❌ Memorizing 500+ individual solutions
✅ Learning 15-20 patterns applicable to thousands of problems

❌ Panic when seeing new problem
✅ Systematic pattern matching and adaptation

❌ Explaining specific code implementations
✅ Discussing algorithmic approaches and reasoning
```

### How to Use This Guide

1. **Understand the Pattern**: Learn when and why to use each pattern
2. **Master the Template**: Internalize the basic implementation structure
3. **Practice Recognition**: Develop ability to identify patterns in new problems
4. **Apply Variations**: Adapt patterns to solve related problem types
5. **Combine Patterns**: Use multiple patterns together for complex problems

### Pattern Classification Framework

```markdown
**By Data Structure:**
• Array/String Patterns: Two Pointers, Sliding Window
• LinkedList Patterns: Fast & Slow Pointers, Reversal
• Tree Patterns: DFS, BFS, Tree DP
• Graph Patterns: Traversal, Topological Sort
• Heap Patterns: Two Heaps, Top K Elements

**By Problem Type:**
• Search Problems: Binary Search, BFS/DFS
• Optimization Problems: Dynamic Programming, Greedy
• Sorting Problems: Merge Sort variants, Cyclic Sort
• Interval Problems: Merge Intervals, Sweep Line
• Combinatorial Problems: Subsets, Permutations

**By Time Complexity:**
• O(1): Hash Table lookups, Array access
• O(log n): Binary Search, Heap operations
• O(n): Linear traversal, Two Pointers
• O(n log n): Sorting, Divide and Conquer
• O(2^n): Backtracking, Exponential generation
```

## Pattern Recognition Framework

### Problem Analysis Checklist

```markdown
**Step 1: Input Analysis**
□ What data structure is given? (Array, LinkedList, Tree, Graph)
□ Is the input sorted or unsorted?
□ Are there duplicate elements?
□ What are the constraints (size, value range)?

**Step 2: Output Requirements**
□ What exactly needs to be returned? (Value, index, boolean, new structure)
□ Are there multiple valid answers?
□ Is in-place modification allowed?
□ What should happen with edge cases?

**Step 3: Pattern Indicators**
□ Do we need to find pairs/triplets? → Two Pointers
□ Are we looking at subarrays/substrings? → Sliding Window  
□ Is there a cycle to detect? → Fast & Slow Pointers
□ Are we dealing with intervals? → Merge Intervals
□ Do we need top/bottom K elements? → Heap patterns
□ Are we generating combinations? → Backtracking/Subsets
```

### Pattern Matching Decision Tree

```markdown
**Array/String Problems:**
├── Two elements with target sum? → Two Pointers
├── Subarray with condition? → Sliding Window
├── Rotate/reverse in place? → Two Pointers
└── All possible combinations? → Backtracking

**LinkedList Problems:**
├── Cycle detection? → Fast & Slow Pointers
├── Find middle/kth element? → Fast & Slow Pointers
├── Reverse list/sublists? → In-place Reversal
└── Merge sorted lists? → K-way Merge

**Tree Problems:**
├── Path sum/counting? → Tree DFS
├── Level-order traversal? → Tree BFS
├── Validate properties? → Tree DFS
└── Find diameter/height? → Tree DFS

**Optimization Problems:**
├── Overlapping subproblems? → Dynamic Programming
├── Locally optimal choices? → Greedy
├── Search in sorted space? → Binary Search
└── Multiple decision points? → DFS/Backtracking
```

## Two Pointers Pattern

### When to Use Two Pointers

```markdown
**Problem Characteristics:**
• Dealing with sorted arrays or strings
• Looking for pairs/triplets that meet certain criteria
• Need to compare elements at different positions
• Reversing or rearranging elements in-place

**Common Problem Types:**
• Two Sum in sorted array
• Remove duplicates from sorted array
• Reverse a string or array
• Find triplets with given sum
• Container with most water
• Valid palindrome checking
```

### Two Pointers Templates

#### Template 1: Opposite Direction Pointers

```python
def two_pointers_opposite(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # Process current pair
        current_sum = arr[left] + arr[right]

        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1  # Need larger sum
        else:
            right -= 1  # Need smaller sum

    return [-1, -1]  # Not found
```

**Use Cases:**

- Two Sum in sorted array
- Valid Palindrome
- Container With Most Water
- 3Sum problem

#### Template 2: Same Direction Pointers

```python
def two_pointers_same_direction(arr):
    slow = fast = 0

    while fast < len(arr):
        # Process with fast pointer
        process(arr[fast])

        # Move fast pointer
        fast += 1

        # Conditionally move slow pointer
        if should_move_slow():
            slow += 1

    return slow  # Often returns the new length
```

**Use Cases:**

- Remove duplicates from sorted array
- Remove element from array
- Move zeros to end
- Partition array

### Time and Space Complexity

- **Time Complexity:** O(n) - each element visited at most once
- **Space Complexity:** O(1) - only using two pointer variables
- **Key Advantage:** Replaces nested loops that would be O(n²)

### Problem Solving Framework

```markdown
**Step 1: Identify Pointers**
• Determine starting positions (usually 0 and n-1, or both at 0)
• Understand what each pointer represents

**Step 2: Define Movement Rules**
• When to move left pointer
• When to move right pointer
• When to move both pointers

**Step 3: Termination Condition**
• Usually when pointers meet or cross
• Sometimes when one pointer reaches end

**Step 4: Handle Edge Cases**
• Empty array or string
• Single element
• All elements same
• No valid solution exists
```

## Sliding Window Pattern

### When to Use Sliding Window

```markdown
**Problem Characteristics:**
• Finding something in subarrays/substrings of specific size
• Looking for optimal subarray (max/min length, sum, etc.)
• Counting subarrays with certain properties
• String matching and pattern finding

**Common Problem Types:**
• Maximum sum subarray of size K
• Longest substring without repeating characters
• Minimum window substring
• Find all anagrams in string
• Longest substring with K distinct characters
```

### Sliding Window Templates

#### Template 1: Fixed Window Size

```python
def sliding_window_fixed(arr, k):
    if len(arr) < k:
        return -1

    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

**Use Cases:**

- Maximum sum of subarray size K
- Average of subarrays size K
- Maximum of all subarrays size K

#### Template 2: Variable Window Size

```python
def sliding_window_variable(s):
    left = 0
    result = 0
    window_data = {}

    for right in range(len(s)):
        # Expand window - add right element
        char = s[right]
        window_data[char] = window_data.get(char, 0) + 1

        # Contract window while condition violated
        while window_violates_condition(window_data):
            left_char = s[left]
            window_data[left_char] -= 1
            if window_data[left_char] == 0:
                del window_data[left_char]
            left += 1

        # Update result with current valid window
        result = max(result, right - left + 1)

    return result
```

**Use Cases:**

- Longest substring without repeating characters
- Minimum window substring
- Longest substring with at most K distinct characters

### Key Components of Sliding Window

```markdown
**Window State Tracking:**
• Hash map for character/element frequencies
• Variables for sum, count, or other metrics
• Conditions for valid vs invalid windows

**Window Operations:**
• Expand: Add right element to window
• Contract: Remove left element from window  
• Update: Modify result based on current window

**Optimization Principles:**
• Each element added and removed exactly once
• Avoids recalculating window properties from scratch
• Maintains window invariants efficiently
```

## Fast & Slow Pointers Pattern

### When to Use Fast & Slow Pointers

```markdown
**Problem Characteristics:**
• Detecting cycles in sequences (arrays, linked lists)
• Finding middle elements or specific positions
• Determining if sequence has certain properties
• Problems involving linked list manipulation

**Common Problem Types:**
• Linked List Cycle Detection (Floyd's Algorithm)
• Find middle of linked list
• Happy number problem
• Palindromic linked list
• Find duplicate number in array
```

### Fast & Slow Pointers Template

```python
def fast_slow_pointers(head):
    if not head or not head.next:
        return None

    slow = fast = head

    # Phase 1: Detect if cycle exists
    while fast and fast.next:
        slow = slow.next        # Move 1 step
        fast = fast.next.next   # Move 2 steps

        if slow == fast:  # Cycle detected
            break

    # Phase 2: Find cycle start (if needed)
    if fast and fast.next:  # Cycle exists
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow  # Start of cycle

    return None  # No cycle
```

### Algorithm Analysis

```markdown
**Why It Works:**
• Fast pointer gains one position per iteration on slow pointer
• In a cycle, fast pointer will eventually lap slow pointer
• Mathematical proof: If cycle length is C, fast catches slow in at most C iterations

**Time Complexity:** O(n)
• At most 2n iterations (n for detection + n for finding start)

**Space Complexity:** O(1)
• Only uses two pointer variables

**Key Insight:**
• Distance from start to cycle beginning = Distance from meeting point to cycle beginning
```

### Variations and Applications

```markdown
**Find Middle Element:**
• When fast reaches end, slow is at middle
• Handles both odd and even length sequences

**Happy Number:**
• Use digit sum transformation as "next" function
• Cycle detection determines if number is happy

**Remove Nth from End:**
• Fast pointer gets n-step head start
• When fast reaches end, slow is at target position

**Palindromic Linked List:**
• Find middle, reverse second half, compare both halves
```

## Merge Intervals Pattern

### When to Use Merge Intervals

```markdown
**Problem Characteristics:**
• Dealing with overlapping time intervals, ranges, or segments
• Need to merge, insert, or remove intervals
• Finding intersection or union of interval sets
• Scheduling and resource allocation problems

**Common Problem Types:**
• Merge overlapping intervals
• Insert interval into sorted intervals
• Meeting rooms scheduling
• Minimum platforms needed
• Non-overlapping intervals
```

### Merge Intervals Template

```python
def merge_intervals(intervals):
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last_merged = merged[-1]

        # Check if current overlaps with last merged interval
        if current[0] <= last_merged[1]:  # Overlapping
            # Merge by updating end time
            last_merged[1] = max(last_merged[1], current[1])
        else:  # Non-overlapping
            # Add as new interval
            merged.append(current)

    return merged
```

### Key Concepts

```markdown
**Interval Overlap Conditions:**
• Two intervals [a,b] and [c,d] overlap if: max(a,c) <= min(b,d)
• Equivalent: !(b < c or d < a)

**Sorting Strategy:**
• Usually sort by start time
• Sometimes sort by end time (for greedy scheduling)
• Custom sorting based on problem requirements

**Merging Logic:**
• If overlap: new_end = max(interval1_end, interval2_end)
• If no overlap: add as separate interval
```

### Problem Categories

```markdown
**Merging Problems:**
• Basic interval merging
• Insert new interval
• Remove overlapping intervals

**Scheduling Problems:**
• Meeting rooms (can all meetings happen?)
• Meeting rooms II (minimum rooms needed)
• Non-overlapping intervals (minimum removals)

**Resource Allocation:**
• Minimum platforms for trains
• Employee free time
• Range sum queries
```

## Cyclic Sort Pattern

### When to Use Cyclic Sort

```markdown
**Problem Characteristics:**
• Array contains numbers in range [1, n] or [0, n-1]
• Finding missing, duplicate, or misplaced numbers
• Array elements can be used as indices
• Need to sort or organize elements by their value

**Common Problem Types:**
• Find missing number
• Find duplicate number
• Find all missing numbers
• Find all duplicates
• First missing positive
```

### Cyclic Sort Template

```python
def cyclic_sort(nums):
    i = 0
    while i < len(nums):
        # Calculate correct position for current number
        correct_pos = nums[i] - 1  # For range [1, n]
        # correct_pos = nums[i] for range [0, n-1]

        # If number is not in correct position
        if nums[i] != nums[correct_pos]:
            # Swap to place number in correct position
            nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
        else:
            # Number is in correct position, move to next
            i += 1

    return nums
```

### Algorithm Mechanics

```markdown
**Core Idea:**
• Each number knows its correct index position
• Swap elements until each is in correct place
• Missing or duplicate numbers will be obvious after sorting

**Time Complexity:** O(n)
• Each number is moved at most once to its correct position

**Space Complexity:** O(1)
• Only swaps elements in place

**Key Insight:**
• After cyclic sort, nums[i] should equal i+1 (for 1-based)
• Any deviation indicates missing/duplicate numbers
```

### Problem Solving Steps

```markdown
**Step 1: Cyclic Sort**
• Place each number in its correct position
• Handle out-of-range numbers appropriately

**Step 2: Find Anomalies**
• Scan array to find positions where nums[i] != i+1
• These positions indicate missing/duplicate numbers

**Step 3: Return Result**
• Based on problem requirements (missing, duplicates, etc.)
```

## In-place Reversal of LinkedList

### When to Use LinkedList Reversal

```markdown
**Problem Characteristics:**
• Reversing entire linked list or sublists
• Modifying linked list structure in-place
• K-group reversal patterns
• Palindromic linked list checking

**Common Problem Types:**
• Reverse linked list
• Reverse nodes in k-group
• Reverse sublist between positions
• Swap nodes in pairs
• Palindromic linked list
```

### LinkedList Reversal Template

```python
def reverse_linkedlist(head):
    prev = None
    current = head

    while current:
        # Store next node before breaking link
        next_temp = current.next

        # Reverse the link
        current.next = prev

        # Move pointers one step forward
        prev = current
        current = next_temp

    return prev  # New head of reversed list
```

### Sublist Reversal Template

```python
def reverse_sublist(head, m, n):
    if m == n:
        return head

    # Skip first m-1 nodes
    prev = None
    current = head
    for _ in range(m - 1):
        prev = current
        current = current.next

    # Store connection points
    last_of_first_part = prev
    last_of_sublist = current

    # Reverse sublist
    prev = None
    for _ in range(n - m + 1):
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp

    # Connect with remaining parts
    if last_of_first_part:
        last_of_first_part.next = prev
    else:
        head = prev

    last_of_sublist.next = current
    return head
```

### Key Concepts

```markdown
**Pointer Management:**
• prev: Points to previous node in new order
• current: Currently processing node
• next_temp: Stores next node before modifying links

**Edge Cases:**
• Empty list (head is None)
• Single node list
• Reversing entire list vs sublist
• Invalid positions for sublist reversal

**Connection Points:**
• Track where reversed section connects to rest of list
• Handle cases where reversal starts at head
```

## Tree Depth First Search

### When to Use Tree DFS

```markdown
**Problem Characteristics:**
• Need to traverse all nodes in a tree
• Looking for paths from root to leaves
• Calculating tree properties (height, diameter, sum)
• Validating tree structure or properties

**Common Problem Types:**
• Binary tree path sum
• Maximum depth of tree
• Validate binary search tree
• Tree diameter
• Lowest common ancestor
```

### Tree DFS Templates

#### Template 1: Basic DFS Traversal

```python
def tree_dfs(root):
    if not root:
        return

    # Pre-order: Process root first
    process(root)
    tree_dfs(root.left)
    tree_dfs(root.right)

    # In-order: Process root between children
    tree_dfs(root.left)
    process(root)
    tree_dfs(root.right)

    # Post-order: Process root after children
    tree_dfs(root.left)
    tree_dfs(root.right)
    process(root)
```

#### Template 2: Path Sum DFS

```python
def path_sum_dfs(root, target_sum):
    def dfs(node, current_path, current_sum):
        if not node:
            return

        # Add current node to path
        current_path.append(node.val)
        current_sum += node.val

        # Check if leaf node with target sum
        if not node.left and not node.right and current_sum == target_sum:
            result.append(current_path[:])  # Copy current path

        # Continue DFS
        dfs(node.left, current_path, current_sum)
        dfs(node.right, current_path, current_sum)

        # Backtrack
        current_path.pop()

    result = []
    dfs(root, [], 0)
    return result
```

### DFS Problem Categories

```markdown
**Path Problems:**
• Find all root-to-leaf paths
• Path sum calculations
• Maximum path sum
• Path with given sequence

**Tree Property Problems:**
• Tree height/depth
• Tree diameter
• Balanced tree validation
• Symmetric tree checking

**Tree Construction/Modification:**
• Build tree from traversals
• Mirror/invert tree
• Flatten tree to linked list
• Clone tree with random pointers
```

## Tree Breadth First Search

### When to Use Tree BFS

```markdown
**Problem Characteristics:**
• Level-by-level tree processing
• Finding minimum depth or shortest path
• Level-order traversal requirements
• Connecting nodes at same level

**Common Problem Types:**
• Level order traversal
• Zigzag level order traversal
• Minimum depth of binary tree
• Connect level order siblings
• Right view of binary tree
```

### Tree BFS Template

```python
from collections import deque

def tree_bfs(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)

            # Add children to queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result
```

### BFS Variations

```markdown
**Level-by-Level Processing:**
• Process all nodes at current level before moving to next
• Track level size to separate levels

**Zigzag Traversal:**
• Alternate between left-to-right and right-to-left
• Use deque with appendleft for reverse levels

**Single Value per Level:**
• Average of level, minimum/maximum in level
• Rightmost/leftmost node in level

**Level Connection:**
• Connect all nodes at same level
• Build next pointers for level-order navigation
```

## Two Heaps Pattern

### When to Use Two Heaps

```markdown
**Problem Characteristics:**
• Need to track both smallest and largest elements
• Finding median in dynamic data
• Balancing two sets of data
• Sliding window median problems

**Common Problem Types:**
• Find median from data stream
• Sliding window median
• IPO (maximize capital with constraints)
• Meeting scheduler
```

### Two Heaps Template

```python
import heapq

class MedianFinder:
    def __init__(self):
        # Max heap for smaller half (use negative values)
        self.max_heap = []
        # Min heap for larger half
        self.min_heap = []

    def add_number(self, num):
        # Add to appropriate heap
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        # Rebalance heaps
        self.rebalance()

    def rebalance(self):
        # Ensure max_heap has at most 1 more element than min_heap
        if len(self.max_heap) > len(self.min_heap) + 1:
            element = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, element)
        elif len(self.min_heap) > len(self.max_heap):
            element = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -element)

    def find_median(self):
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2
        else:
            return -self.max_heap[0]
```

### Key Concepts

```markdown
**Heap Balancing:**
• Max heap size ∈ [min_heap_size, min_heap_size + 1]
• Always insert into appropriate heap first, then rebalance

**Median Access:**
• If equal sizes: median = (max_heap_top + min_heap_top) / 2
• If max_heap larger: median = max_heap_top

**Time Complexity:**
• Insert: O(log n) for heap operations
• Find median: O(1) for accessing heap tops
```

## Subsets Pattern

### When to Use Subsets Pattern

```markdown
**Problem Characteristics:**
• Generate all possible combinations or subsets
• Explore all possible choices at each step
• Combinatorial optimization problems
• Constraint satisfaction with backtracking

**Common Problem Types:**
• Generate all subsets
• Generate permutations
• Combination sum
• Palindromic partitioning
• N-Queens problem
```

### Subsets Templates

#### Template 1: Iterative Subsets

```python
def generate_subsets(nums):
    subsets = [[]]

    for num in nums:
        # Create new subsets by adding current number to existing subsets
        new_subsets = []
        for subset in subsets:
            new_subsets.append(subset + [num])
        subsets.extend(new_subsets)

    return subsets
```

#### Template 2: Recursive Backtracking

```python
def generate_subsets_backtrack(nums):
    def backtrack(start, current_subset):
        # Add current subset to result
        result.append(current_subset[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            # Choose
            current_subset.append(nums[i])

            # Explore
            backtrack(i + 1, current_subset)

            # Unchoose (backtrack)
            current_subset.pop()

    result = []
    backtrack(0, [])
    return result
```

### Backtracking Framework

```markdown
**Three Steps:**

1. **Choose:** Add element to current solution
2. **Explore:** Recursively build rest of solution
3. **Unchoose:** Remove element (backtrack)

**Key Decisions:**
• What constitutes a valid solution?
• How to avoid duplicate solutions?
• When to prune invalid branches?
• How to optimize with constraints?

**Optimization Techniques:**
• Sort input to enable early termination
• Use visited arrays to avoid revisiting elements
• Prune branches that can't lead to valid solutions
```

## Modified Binary Search

### When to Use Modified Binary Search

```markdown
**Problem Characteristics:**
• Searching in sorted or rotated arrays
• Finding target in infinite or unknown size arrays
• Peak finding and local extrema
• First/last occurrence in sorted array

**Common Problem Types:**
• Search in rotated sorted array
• Find peak element
• Search in sorted matrix
• Find first/last occurrence
• Minimum in rotated sorted array
```

### Modified Binary Search Templates

#### Template 1: Basic Modified Binary Search

```python
def modified_binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        # Determine which half is sorted
        if arr[left] <= arr[mid]:  # Left half is sorted
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

#### Template 2: Find Boundary Binary Search

```python
def find_boundary(arr, condition):
    """Find rightmost position where condition is True"""
    left, right = 0, len(arr) - 1
    boundary_index = -1

    while left <= right:
        mid = left + (right - left) // 2

        if condition(arr[mid]):
            boundary_index = mid
            left = mid + 1  # Continue searching right
        else:
            right = mid - 1  # Search left

    return boundary_index
```

### Key Modifications

```markdown
**Rotation Handling:**
• Identify which half is sorted
• Check if target lies in sorted half
• Adjust search accordingly

**Duplicate Elements:**
• Handle cases where arr[left] == arr[mid] == arr[right]
• May need to increment left or decrement right

**Boundary Finding:**
• Find first/last occurrence of element
• Find insertion point for element
• Find peak elements in mountain arrays
```

## Bitwise XOR Pattern

### When to Use XOR Pattern

```markdown
**Problem Characteristics:**
• Finding unique elements when others appear multiple times
• Space-efficient solutions using bit manipulation
• Problems involving pairs and cancellation
• Detecting differences in bit patterns

**XOR Properties:**
• a ⊕ a = 0 (number XOR with itself is 0)
• a ⊕ 0 = a (number XOR with 0 is itself)
• XOR is commutative and associative
• XOR of even count of same numbers is 0

**Common Problem Types:**
• Single number (others appear twice)
• Two single numbers (others appear twice)
• Missing number in array
• Find duplicate number
• Complement of base 10 integer
```

### XOR Pattern Templates

#### Template 1: Find Single Number

```python
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result  # All pairs cancel out, leaving single number
```

#### Template 2: Find Two Single Numbers

```python
def two_single_numbers(nums):
    # XOR all numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Find rightmost set bit
    rightmost_set_bit = 1
    while (xor_all & rightmost_set_bit) == 0:
        rightmost_set_bit <<= 1

    # Partition numbers based on rightmost set bit
    num1 = num2 = 0
    for num in nums:
        if num & rightmost_set_bit:
            num1 ^= num
        else:
            num2 ^= num

    return [num1, num2]
```

### Bit Manipulation Techniques

```markdown
**Common Bit Operations:**
• Check if bit is set: (n & (1 << i)) != 0
• Set bit: n | (1 << i)
• Clear bit: n & ~(1 << i)
• Toggle bit: n ^ (1 << i)
• Get rightmost set bit: n & -n

**XOR Problem Solving Steps:**

1. Understand what cancels out (pairs, duplicates)
2. Apply XOR to eliminate known patterns
3. Use bit manipulation to extract information
4. Handle edge cases and constraints
```

## Top K Elements Pattern

### When to Use Top K Pattern

```markdown
**Problem Characteristics:**
• Finding K largest/smallest elements
• Maintaining top K elements in dynamic data
• K closest elements to target
• Frequency-based K problems

**Common Problem Types:**
• Kth largest element in array
• Top K frequent elements
• K closest points to origin
• Kth smallest element in sorted matrix
• K largest elements in stream
```

### Top K Pattern Templates

#### Template 1: Heap-based Top K

```python
import heapq

def top_k_elements(nums, k):
    # Use min heap for top K largest elements
    min_heap = []

    for num in nums:
        heapq.heappush(min_heap, num)

        # Keep only K elements in heap
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return list(min_heap)
```

#### Template 2: QuickSelect for Kth Element

```python
def kth_largest_quickselect(nums, k):
    def partition(left, right, pivot_index):
        pivot_val = nums[pivot_index]
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]

        store_index = left
        for i in range(left, right):
            if nums[i] > pivot_val:  # For kth largest
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1

        # Move pivot to final position
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index

    def quickselect(left, right, k_smallest):
        if left == right:
            return nums[left]

        pivot_index = left + (right - left) // 2
        pivot_index = partition(left, right, pivot_index)

        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return quickselect(left, pivot_index - 1, k_smallest)
        else:
            return quickselect(pivot_index + 1, right, k_smallest)

    return quickselect(0, len(nums) - 1, k - 1)
```

### Algorithm Choices

```markdown
**Heap vs QuickSelect:**
• Heap: O(N log K) time, O(K) space, stable
• QuickSelect: O(N) average time, O(1) space, unstable

**Min Heap vs Max Heap:**
• Top K largest → Use Min Heap of size K
• Top K smallest → Use Max Heap of size K

**Frequency Problems:**
• Count frequencies first
• Use heap with custom comparator
• Consider bucket sort for limited frequency ranges
```

## K-way Merge Pattern

### When to Use K-way Merge

```markdown
**Problem Characteristics:**
• Merging K sorted arrays or lists
• Finding smallest/largest among K sources
• Combining multiple sorted data streams
• Matrix traversal with sorted properties

**Common Problem Types:**
• Merge K sorted lists
• Kth smallest element in sorted matrix
• Smallest range covering elements from K lists
• Find pairs with smallest sum from two arrays
```

### K-way Merge Template

```python
import heapq

def merge_k_sorted_lists(lists):
    min_heap = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))  # (value, list_index, element_index)

    result = []

    while min_heap:
        value, list_idx, element_idx = heapq.heappop(min_heap)
        result.append(value)

        # Add next element from same list
        if element_idx + 1 < len(lists[list_idx]):
            next_element = lists[list_idx][element_idx + 1]
            heapq.heappush(min_heap, (next_element, list_idx, element_idx + 1))

    return result
```

### Key Concepts

```markdown
**Heap Management:**
• Maintain one element from each source in heap
• Always process smallest element next
• Add next element from same source after processing

**Time Complexity:**
• O(N log K) where N is total elements, K is number of sources
• Each element added/removed from heap once

**Space Complexity:**
• O(K) for heap storage
• Additional space for result if needed

**Variations:**
• Merge K sorted linked lists
• K-way merge sort implementation
• Streaming data merge applications
```

## Topological Sort Pattern

### When to Use Topological Sort

```markdown
**Problem Characteristics:**
• Dependencies between tasks or elements
• Directed Acyclic Graph (DAG) problems
• Ordering with precedence constraints
• Course scheduling and prerequisite problems

**Common Problem Types:**
• Course schedule (can finish all courses?)
• Course schedule II (order to take courses)
• Alien dictionary (character ordering)
• Minimum height trees
• Task scheduling with dependencies
```

### Topological Sort Templates

#### Template 1: Kahn's Algorithm (BFS-based)

```python
from collections import deque, defaultdict

def topological_sort_kahns(vertices, edges):
    # Build adjacency list and in-degree count
    adj_list = defaultdict(list)
    in_degree = {i: 0 for i in range(vertices)}

    for parent, child in edges:
        adj_list[parent].append(child)
        in_degree[child] += 1

    # Find all vertices with no incoming edges
    queue = deque([v for v in range(vertices) if in_degree[v] == 0])

    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        # Remove edges from this vertex
        for neighbor in adj_list[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycle
    if len(result) != vertices:
        return []  # Cycle detected

    return result
```

#### Template 2: DFS-based Topological Sort

```python
def topological_sort_dfs(vertices, edges):
    adj_list = defaultdict(list)
    for parent, child in edges:
        adj_list[parent].append(child)

    visited = set()
    visiting = set()  # For cycle detection
    result = []

    def dfs(vertex):
        if vertex in visiting:  # Back edge found, cycle exists
            return False
        if vertex in visited:
            return True

        visiting.add(vertex)
        for neighbor in adj_list[vertex]:
            if not dfs(neighbor):
                return False

        visiting.remove(vertex)
        visited.add(vertex)
        result.append(vertex)
        return True

    # Process all vertices
    for v in range(vertices):
        if v not in visited:
            if not dfs(v):
                return []  # Cycle detected

    return result[::-1]  # Reverse for correct order
```

### Algorithm Comparison

```markdown
**Kahn's vs DFS:**
• Kahn's: BFS-based, intuitive, good for finding any valid order
• DFS: Recursive, finds lexicographically last valid order
• Both: O(V + E) time complexity, O(V) space

**Cycle Detection:**
• Kahn's: If result length != vertex count
• DFS: If back edge detected (visiting set)

**Applications:**
• Build systems (dependency compilation)
• Course scheduling
• Task prioritization
• Dependency resolution
```

## Dynamic Programming Patterns

### DP Pattern Categories

```markdown
**Linear DP:**
• 1D array, each state depends on previous states
• Examples: Fibonacci, house robber, climbing stairs

**Grid DP:**
• 2D array, state depends on adjacent cells
• Examples: unique paths, minimum path sum, edit distance

**Interval DP:**
• Process intervals of increasing size
• Examples: palindromic substrings, matrix chain multiplication

**Tree DP:**
• DP on tree structures, combine child results
• Examples: tree diameter, house robber III, binary tree cameras

**State Machine DP:**
• Multiple states with transitions
• Examples: stock trading, state-dependent problems
```

### DP Problem-Solving Framework

```markdown
**Step 1: Identify DP Nature**
□ Optimal substructure (optimal solution contains optimal subsolutions)
□ Overlapping subproblems (same subproblems solved multiple times)

**Step 2: Define State**
□ What parameters uniquely identify a subproblem?
□ What's the meaning of dp[i][j]?

**Step 3: State Transition**
□ How to compute current state from previous states?
□ What are the base cases?

**Step 4: Implementation**
□ Top-down (memoization) vs bottom-up (tabulation)
□ Space optimization opportunities

**Step 5: Optimization**
□ Can we reduce space complexity?
□ Are there redundant computations?
```

## Graph Algorithms Patterns

### Graph Representation

```python
# Adjacency List
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Adjacency Matrix
graph = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]
```

### Graph Traversal Patterns

#### BFS Template

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        vertex = queue.popleft()
        process(vertex)

        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

#### DFS Template

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    process(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

### Advanced Graph Algorithms

```markdown
**Shortest Path:**
• Dijkstra's Algorithm: Single source, non-negative weights
• Bellman-Ford: Single source, handles negative weights
• Floyd-Warshall: All pairs shortest paths

**Minimum Spanning Tree:**
• Kruskal's Algorithm: Edge-based, uses Union-Find
• Prim's Algorithm: Vertex-based, uses priority queue

**Strongly Connected Components:**
• Tarjan's Algorithm: One pass DFS
• Kosaraju's Algorithm: Two pass DFS

**Network Flow:**
• Ford-Fulkerson: Maximum flow
• Edmonds-Karp: BFS-based Ford-Fulkerson
```

## Advanced Patterns & Combinations

### Pattern Combinations

```markdown
**Two Pointers + Sliding Window:**
• Minimum window substring
• Longest substring with at most K distinct characters
• Subarray product less than K

**DFS + Memoization:**
• Word break II
• Unique paths with obstacles
• Palindromic partitioning

**Binary Search + Graph:**
• Find minimum in rotated sorted array
• Search 2D matrix
• Median of two sorted arrays

**Heap + Hash Map:**
• Top K frequent elements
• Task scheduler
• Sliding window maximum
```

### Advanced Problem-Solving Strategies

```markdown
**Pattern Recognition Steps:**

1. **Input Analysis:** What data structures and constraints?
2. **Output Requirements:** What exactly needs to be computed?
3. **Constraint Analysis:** Time/space limitations, input size
4. **Pattern Mapping:** Which patterns fit the problem characteristics?
5. **Adaptation:** How to modify standard patterns for this problem?

**Optimization Techniques:**
• **Space Optimization:** 2D DP to 1D, rolling arrays
• **Time Optimization:** Memoization, early termination
• **Algorithm Selection:** Choose optimal algorithm for constraints
• **Data Structure Choice:** Balance between time and space complexity

**Testing Strategy:**
• **Edge Cases:** Empty input, single element, maximum constraints
• **Boundary Conditions:** First/last elements, min/max values
• **Performance Testing:** Large inputs, worst-case scenarios
• **Correctness Verification:** Manual trace through examples
```

---

## Pattern Selection Guide

### Quick Pattern Identification

```markdown
**Array/String Keywords → Pattern:**
• "sorted array" + "target sum" → Two Pointers
• "subarray" + "condition" → Sliding Window
• "permutations/combinations" → Backtracking
• "rotated sorted" → Modified Binary Search

**LinkedList Keywords → Pattern:**
• "cycle" → Fast & Slow Pointers
• "reverse" → In-place Reversal
• "merge sorted lists" → K-way Merge

**Tree Keywords → Pattern:**
• "path" → Tree DFS
• "level order" → Tree BFS
• "depth/height" → Tree DFS

**Graph Keywords → Pattern:**
• "dependencies" → Topological Sort
• "shortest path" → BFS/Dijkstra
• "connected components" → DFS/Union-Find

**Optimization Keywords → Pattern:**
• "overlapping subproblems" → Dynamic Programming
• "top K" → Heap/QuickSelect
• "median" → Two Heaps
```

This comprehensive guide provides the theoretical foundation for recognizing and applying coding patterns in technical interviews. The key is to practice identifying patterns quickly and adapting them to solve variations of known problems.---

## 🔄 Common Confusions

### Confusion 1: Pattern Recognition vs. Problem Solving

**The Confusion:** Some candidates think learning patterns means you just memorize solutions and apply them directly, rather than understanding the underlying logic.
**The Clarity:** Patterns are frameworks for thinking, not templates to copy. You need to understand why the pattern works and how to adapt it to specific problems.
**Why It Matters:** Interviewers will present variations that require adaptation. Without understanding the principles, you'll fail when the problem doesn't match the exact pattern you memorized.

### Confusion 2: Over-Engineering Pattern Applications

**The Confusion:** Some candidates try to use advanced patterns for simple problems, showing off complexity rather than appropriateness.
**The Clarity:** Choose the simplest pattern that solves the problem effectively. Over-engineering demonstrates poor judgment about complexity trade-offs.
**Why It Matters:** In real work, you need to choose appropriate tools for the job. Using a sledgehammer to crack a nut suggests lack of engineering judgment.

### Confusion 3: Focusing Only on Common Patterns

**The Confusion:** Spending all practice time on the most common patterns (two pointers, sliding window) while ignoring less common but important ones.
**The Clarity:** Interviewers use patterns to test your breadth of knowledge. Ignoring less common patterns leaves gaps that can be easily exploited.
**Why It Matters:** You need to be prepared for any pattern the interviewer might test. A narrow focus shows limited preparation and understanding.

### Confusion 4: Not Practicing Pattern Variation

**The Confusion:** Learning patterns by solving the same example problems repeatedly without exploring variations and edge cases.
**The Clarity:** Mastery requires practice with variations, not just the original problem. Real interviews present novel problem statements.
**Why It Matters:** If you can only solve the exact problems you've seen, you won't recognize the pattern in different contexts. Flexibility is key.

### Confusion 5: Mixing Up Pattern Applications

**The Confusion:** Applying the wrong pattern to a problem because you focus on keywords rather than understanding the core problem structure.
**The Clarity:** Keywords are hints, not rules. You need to understand the problem's structure and requirements to choose the right approach.
**Why It Matters:** Surface-level pattern matching can lead to completely wrong solutions. You need deeper understanding of problem characteristics.

### Confusion 6: Ignoring Pattern Combinations

**The Confusion:** Learning patterns in isolation and not practicing how they combine to solve complex problems.
**The Clarity:** Many interview problems require combining multiple patterns. Real systems are complex and require multiple approaches.
**Why It Matters:** Complex problems often require sophisticated solutions using multiple patterns. Isolated pattern knowledge is insufficient for challenging problems.

### Confusion 7: Time vs. Space Trade-offs

**The Confusion:** Not understanding when patterns optimize for time vs. space, leading to inappropriate solution choices.
**The Clarity:** Different patterns have different trade-offs. Understanding these helps you choose the right approach for specific constraints.
**Why It Matters:** Real systems have resource constraints. Interviewers expect you to understand and communicate trade-offs when proposing solutions.

### Confusion 8: Pattern Recognition Speed Pressure

**The Confusion:** Feeling overwhelmed when you can't immediately identify the pattern, especially under interview time pressure.
**The Clarity:** Even experienced developers need time to analyze problems. The process of elimination and systematic approach is as important as pattern recognition speed.
**Why It Matters:** Under pressure, systematic thinking and problem breakdown is more valuable than quick pattern identification. The journey matters.

## 📝 Micro-Quiz

### Question 1: When you see a problem about "finding the longest subarray that meets a condition," the most likely pattern is:

A) Two Pointers
B) Sliding Window
C) Fast & Slow Pointers
D) Modified Binary Search
**Answer:** B
**Explanation:** Sliding window is the go-to pattern for problems involving subarrays/substrings with conditions. The window slides along the array, maintaining the condition while tracking the best result.

### Question 2: You notice a problem involves finding if a linked list has a cycle. The best approach is:

A) Use a Set to track visited nodes
B) Use Fast & Slow Pointers
C) Use a HashMap to store node references
D) Use recursion to traverse
**Answer:** B
**Explanation:** Fast & Slow Pointers (Floyd's cycle detection algorithm) is the optimal solution with O(1) space complexity. The faster pointer will eventually meet the slower one if a cycle exists.

### Question 3: For problems requiring "finding the kth smallest/largest element," the recommended pattern is:

A) Merge Sort
B) QuickSort
C) Top K Elements (Heap/QuickSelect)
D) Binary Search
**Answer:** C
**Explanation:** Top K Elements pattern uses heaps or QuickSelect to find kth elements efficiently. This approach has better average performance than sorting the entire array.

### Question 4: When you need to "merge multiple sorted lists," the best pattern is:

A) K-way Merge using a Min Heap
B) Merge all pairs first
C) Use divide and conquer
D) Concatenate and sort
**Answer:** A
**Explanation:** K-way Merge with a Min Heap efficiently merges multiple sorted lists by always taking the smallest current element from the heap, maintaining O(n log k) complexity.

### Question 5: Problems involving "finding all possible permutations/combinations" typically use:

A) Two Pointers
B) Backtracking (DFS)
C) Dynamic Programming
D) Greedy Algorithm
**Answer:** B
**Explanation:** Backtracking pattern is designed for problems requiring exploring all possible arrangements or selections. It systematically explores all valid combinations.

### Question 6: The most important skill in pattern recognition is:

A) Memorizing the most patterns
B) Understanding the underlying logic of each pattern
C) Recognizing keywords in problem statements
D) Solving problems as quickly as possible
**Answer:** B
**Explanation:** Understanding the logic and principles behind each pattern is more important than memorization. This allows you to adapt patterns to new problems and variations.

**Mastery Threshold:** 80% (5/6 correct)

## 💭 Reflection Prompts

1. **Pattern Philosophy:** Think about how patterns work in your field. How do you recognize recurring structures or solutions in your work? How can you apply this same pattern recognition skill to coding interview problems?

2. **Adaptation Challenges:** Recall a time when you tried to apply a familiar solution to a slightly different problem and it didn't work. What did you learn about when and how to adapt solutions? How can you apply this learning to pattern application?

3. **Complexity Judgment:** Consider a recent technical decision where you had to choose between a simple approach and a complex one. How did you decide which was appropriate? How can you develop this judgment for choosing the right pattern for interview problems?

## 🏃 Mini Sprint Project (1-3 hours)

**Project: "Pattern Recognition Speed Training"**

Build a system to improve your pattern recognition speed and accuracy:

**Requirements:**

1. Create a collection of 20 coding problems across 8-10 different patterns
2. For each problem, identify the pattern, explain why it fits, and outline the solution approach
3. Practice identifying patterns by reading problem descriptions only (no solving)
4. Time yourself and track improvement in recognition speed
5. Create a personal "pattern vocabulary" with key indicators for each pattern

**Deliverables:**

- Pattern recognition training problems with analysis
- Personal pattern identification guide
- Performance tracking system showing improvement
- Quick reference guide for pattern indicators

## 🚀 Full Project Extension (10-25 hours)

**Project: "Comprehensive Pattern Mastery System"**

Build a complete system for learning, practicing, and mastering coding interview patterns:

**Core System Components:**

1. **Pattern Learning Engine**: Structured lessons for each pattern with theory, examples, variations, and common pitfalls
2. **Adaptive Practice System**: Problems that adapt to your skill level and help you practice weak patterns more frequently
3. **Pattern Combination Trainer**: Complex problems that require combining multiple patterns with guided solution paths
4. **Performance Analytics**: Track pattern recognition speed, solution accuracy, and improvement trends across all patterns
5. **Interview Simulation**: Mock interviews that test pattern knowledge and application under realistic time pressure

**Advanced Features:**

- Visual pattern recognition training with animated examples
- Peer matching for collaborative pattern learning
- Expert solution review and feedback
- Pattern application in real-world contexts
- Integration with coding practice platforms
- Mobile app for pattern learning on-the-go
- AI-powered pattern suggestions for new problems
- Gamified learning with achievements and progress tracking

**Technical Implementation:**

- Modern web application with responsive design
- Interactive coding environment for practice
- Real-time collaboration features
- Performance tracking and analytics
- Integration with popular coding platforms
- Offline capability for pattern reference
- Export capabilities for study materials
- Search and filter system for pattern-specific content

**Pattern Coverage:**

- Two Pointers, Sliding Window, Fast & Slow Pointers
- Merge Intervals, Cyclic Sort, In-place LinkedList Reversal
- Tree DFS, Tree BFS, Two Heaps, Subsets
- Modified Binary Search, Bitwise XOR, Top K Elements
- K-way Merge, Topological Sort, Dynamic Programming
- Backtracking, Greedy Algorithms, and advanced variations

**Expected Outcome:** A comprehensive pattern mastery system that accelerates your learning, provides structured practice, and builds confidence in applying patterns effectively during technical interviews.
