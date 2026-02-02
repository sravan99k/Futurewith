---
title: "Greedy Algorithms Complete Guide"
level: "Intermediate to Advanced"
estimated_time: "80 minutes"
prerequisites:
  [Basic algorithms, Sorting, Basic graph theory, Mathematical optimization]
skills_gained:
  [
    Greedy choice properties,
    Activity selection,
    Interval scheduling,
    Graph algorithms,
    Optimization strategies,
    Problem verification,
  ]
success_criteria:
  [
    "Identify when greedy algorithms apply",
    "Implement classic greedy algorithms correctly",
    "Prove greedy choice properties mathematically",
    "Detect when greedy approaches fail",
    "Optimize greedy solutions for different constraints",
    "Apply greedy thinking to new problem domains",
  ]
version: 1.0
last_updated: 2025-11-11
---

# 🏃‍♂️ Greedy Algorithms: Master the Art of Local Optimization

## Learning Goals

By the end of this comprehensive guide, you will be able to:

- Understand and apply the greedy choice property to identify suitable problems
- Implement classic greedy algorithms including activity selection and Huffman coding
- Prove mathematically why greedy algorithms work for specific problem types
- Recognize when greedy approaches are optimal vs when they fail
- Design custom greedy strategies for novel optimization problems
- Analyze time and space complexity of greedy solutions
- Apply greedy algorithms to real-world scheduling and resource allocation problems
- Debug and verify greedy algorithm correctness

## TL;DR

Greedy algorithms make locally optimal choices at each step, hoping to achieve global optimality. They work when problems have the greedy choice property and optimal substructure. Key examples include activity selection, Dijkstra's algorithm, and Huffman coding - providing efficient solutions for many optimization problems.

## Common Confusions & Mistakes

- **Confusion: "Greedy vs Dynamic Programming"** — Greedy makes irreversible local choices, DP explores all possibilities; greedy is faster but may miss optimal solutions.

- **Confusion: "Local vs Global Optimum"** — Greedy focuses on best immediate choice, which may not lead to overall best solution; need to verify this property.

- **Confusion: "Greedy Choice Property"** — The property that local optimal choices lead to global optimal solutions; must be proven for each application.

- **Confusion: "Sorting vs Greedy"** — Sorting is often a preprocessing step in greedy algorithms, but not all greedy algorithms require sorting.

- **Quick Debug Tip:** For greedy issues, verify the greedy choice property holds, check for integer vs real number domains, and test edge cases carefully.

- **Optimality Verification:** Always prove or test whether your greedy approach actually produces optimal solutions; many problems require counter-examples.

- **Constraint Violations:** Greedy algorithms must check validity of each choice against problem constraints; invalid choices break the solution.

## Micro-Quiz (80% mastery required)

1. **Q:** What are the two key properties needed for a greedy algorithm to work? **A:** Greedy choice property (local optimal leads to global optimal) and optimal substructure (optimal solution contains optimal sub-solutions).

2. **Q:** Why does activity selection sort by finish time? **A:** To maximize the number of activities, always choose the one that finishes earliest, leaving more time for subsequent activities.

3. **Q:** How do you prove a greedy algorithm is correct? **A:** Use exchange argument or proof by induction to show that any optimal solution can be transformed to match the greedy solution without losing optimality.

4. **Q:** When should you NOT use a greedy approach? **A:** When the problem doesn't have the greedy choice property, when the solution space is disconnected, or when small local choices can lead to poor global outcomes.

5. **Q:** What's the difference between greedy and hill-climbing? **A:** Greedy makes one irreversible choice per step, hill-climbing can make temporary suboptimal moves to reach better solutions.

## Reflection Prompts

- **Problem Recognition:** How would you identify if a new optimization problem might be solvable with a greedy approach?

- **Proof Strategy:** What mathematical techniques would you use to prove that a greedy algorithm works for a specific problem?

- **Failure Analysis:** How would you determine if a greedy approach fails for a given problem, and what alternatives would you consider?

_Make the best choice at each step and trust it leads to global optimum_

## Why This Matters

In the real world, perfect solutions aren't always feasible. Greedy algorithms provide **efficient shortcuts** that solve approximately 70% of optimization problems encountered in software engineering. They power:

- **Financial Trading Systems** - Real-time portfolio optimization decisions
- **Resource Management** - Cloud computing resource allocation and scheduling
- **Network Routing** - Internet packet routing and CDN optimization
- **Machine Learning** - Feature selection and model optimization
- **Database Query Planning** - Query execution order optimization
- **Logistics & Supply Chain** - Delivery route optimization and warehouse management

Understanding greedy algorithms means you'll recognize when to apply fast, efficient solutions versus when to invest time in more complex algorithms like dynamic programming or backtracking. This strategic thinking separates good programmers from great engineers.

---

## 🎬 Story Hook: The Coin Change Problem

**Imagine a cashier giving change:**

- **Greedy approach:** Always give the largest coin possible
- **US coins:** $1.00, $0.25, $0.10, $0.05, $0.01 → Works perfectly!
- **Other systems:** May need different strategies

**Real-world uses:**

- 💰 **Financial Trading** - Optimal buy/sell strategies
- 📦 **Resource Scheduling** - CPU, memory allocation
- 🚚 **Route Optimization** - Shortest delivery routes
- 📊 **Data Compression** - Huffman coding
- 🔋 **Energy Management** - Power distribution

---

## 📋 Table of Contents

1. [What Are Greedy Algorithms?](#what-are-greedy-algorithms)
2. [Greedy Algorithm Properties](#greedy-properties)
3. [Classic Greedy Problems](#classic-greedy-problems)
4. [Interval Scheduling Problems](#interval-scheduling)
5. [Graph Greedy Algorithms](#graph-greedy-algorithms)
6. [Advanced Greedy Techniques](#advanced-greedy)
7. [When Greedy Fails](#when-greedy-fails)
8. [Interview Essentials](#interview-essentials)

---

## 🎯 What Are Greedy Algorithms?

### **Definition & Core Concept:**

```python
"""
Greedy Algorithm: Make locally optimal choices at each step,
hoping to find a global optimum.

Key Characteristics:
1. GREEDY CHOICE - Make best immediate decision
2. OPTIMAL SUBSTRUCTURE - Optimal solution contains optimal sub-solutions
3. NEVER RECONSIDER - Once choice is made, stick with it
4. BUILD UP - Construct solution step by step
"""

def greedy_template(items, selection_criteria):
    """
    Universal greedy algorithm template
    """
    result = []

    # Sort items according to greedy criteria
    sorted_items = sorted(items, key=selection_criteria, reverse=True)

    for item in sorted_items:
        # Make greedy choice if it's valid
        if is_valid_choice(item, result):
            result.append(item)

            # Check if we have complete solution
            if is_complete_solution(result):
                break

    return result

# Example: Activity Selection Problem
def activity_selection(activities):
    """
    Select maximum number of non-overlapping activities
    Greedy choice: Always pick activity that finishes earliest
    """
    # Sort by finish time
    activities.sort(key=lambda x: x[1])  # (start, finish)

    selected = [activities[0]]
    last_finish = activities[0][1]

    for start, finish in activities[1:]:
        # If activity starts after last one finishes
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected

# Test
activities = [(1, 3), (2, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)]
print(activity_selection(activities))
# Output: [(1, 3), (3, 5), (5, 7), (8, 9)]
```

### **Greedy vs Other Approaches:**

```python
# GREEDY: Always choose best immediate option
def coin_change_greedy(coins, amount):
    """
    Works for standard coin systems (US coins)
    Time: O(n), Space: O(1)
    """
    coins.sort(reverse=True)
    result = []

    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin

    return result if amount == 0 else None

# DYNAMIC PROGRAMMING: Consider all possibilities
def coin_change_dp(coins, amount):
    """
    Works for any coin system, guarantees optimal solution
    Time: O(amount × coins), Space: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# US coins: Greedy works perfectly
us_coins = [25, 10, 5, 1]
print(coin_change_greedy(us_coins, 67))  # [25, 25, 10, 5, 1, 1]
print(coin_change_dp(us_coins, 67))     # 5 (same result)

# Non-standard coins: Greedy may fail
weird_coins = [4, 3, 1]
print(coin_change_greedy(weird_coins, 6))  # [4, 1, 1] = 3 coins
print(coin_change_dp(weird_coins, 6))      # 2 coins (3, 3)
```

---

## 🔑 Greedy Algorithm Properties

### **When Greedy Works:**

```python
def greedy_choice_property():
    """
    GREEDY CHOICE PROPERTY:
    A global optimum can be reached by making locally optimal choices.

    Examples where it works:
    - Activity Selection (earliest finish time)
    - Fractional Knapsack (highest value/weight ratio)
    - Huffman Coding (lowest frequency first)
    - Dijkstra's Algorithm (shortest distance first)
    """
    pass

def optimal_substructure_property():
    """
    OPTIMAL SUBSTRUCTURE:
    Optimal solution to problem contains optimal solutions to subproblems.

    If we remove the greedy choice from optimal solution,
    what remains must be optimal solution to remaining subproblem.
    """
    pass

# Example: Proving Activity Selection is Optimal
def prove_activity_selection_optimal():
    """
    Proof by Exchange Argument:

    1. Let A be solution by greedy algorithm (earliest finish time)
    2. Let O be any optimal solution
    3. If A ≠ O, find first difference
    4. Exchange first activity in O with first activity in A
    5. This gives us solution no worse than O
    6. Repeat until we get A, proving A is optimal
    """

    # Greedy choice: activity with earliest finish time
    # Why? It leaves most room for remaining activities
    activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]

    # Sort by finish time (greedy choice)
    activities.sort(key=lambda x: x[1])

    selected = []
    last_finish = -1

    for start, finish in activities:
        if start >= last_finish:
            selected.append((start, finish))
            last_finish = finish

    return selected
```

---

## 🎯 Classic Greedy Problems

### **Problem 1: Activity Selection**

```python
def activity_selection_detailed(activities):
    """
    Select maximum number of non-overlapping activities.

    Greedy Strategy: Always pick activity that ends earliest
    Why it works: Leaves maximum time for remaining activities

    Time: O(n log n), Space: O(1)
    """

    # Sort by end time
    activities.sort(key=lambda x: x[1])

    selected = []
    last_end = -1

    for start, end in activities:
        if start >= last_end:  # No overlap
            selected.append((start, end))
            last_end = end

    return selected

def activity_selection_with_weights(activities):
    """
    Select activities to maximize total weight (not count)
    This is NOT a greedy problem - needs DP!
    """
    # This shows when greedy fails
    # Need dynamic programming for weighted version
    pass

# Test
activities = [(1, 2, 50), (3, 5, 20), (6, 19, 100), (2, 100, 200)]
# Greedy by earliest end: (1,2), (3,5), (6,19) = value 170
# Optimal: (2,100) = value 200
```

### **Problem 2: Fractional Knapsack**

```python
def fractional_knapsack(capacity, items):
    """
    Fill knapsack to maximize value (can take fractions of items).

    Greedy Strategy: Sort by value/weight ratio, take highest first
    Why it works: Getting most value per unit weight is always optimal

    Time: O(n log n), Space: O(1)
    """

    # Calculate value/weight ratio for each item
    items_with_ratio = []
    for i, (weight, value) in enumerate(items):
        ratio = value / weight
        items_with_ratio.append((ratio, weight, value, i))

    # Sort by ratio in descending order
    items_with_ratio.sort(reverse=True)

    total_value = 0
    remaining_capacity = capacity
    selected = []

    for ratio, weight, value, original_index in items_with_ratio:
        if remaining_capacity >= weight:
            # Take entire item
            selected.append((original_index, 1.0, value))
            total_value += value
            remaining_capacity -= weight
        elif remaining_capacity > 0:
            # Take fraction of item
            fraction = remaining_capacity / weight
            selected.append((original_index, fraction, value * fraction))
            total_value += value * fraction
            remaining_capacity = 0
            break

    return total_value, selected

# Example
items = [(10, 60), (20, 100), (30, 120)]  # (weight, value)
capacity = 50

max_value, selection = fractional_knapsack(capacity, items)
print(f"Max value: {max_value}")
print(f"Selection: {selection}")
# Output: Max value: 240.0
# Take all of item 1 (ratio=6), all of item 2 (ratio=5), 2/3 of item 3 (ratio=4)
```

### **Problem 3: Huffman Coding**

```python
import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(text):
    """
    Build optimal prefix-free encoding using Huffman algorithm.

    Greedy Strategy: Always merge two nodes with lowest frequencies
    Why it works: Minimizes expected code length

    Time: O(n log n), Space: O(n)
    """

    # Count character frequencies
    frequencies = Counter(text)

    # Special case: single character
    if len(frequencies) == 1:
        char = list(frequencies.keys())[0]
        return {char: '0'}, frequencies[char]

    # Create min heap with nodes
    heap = []
    for char, freq in frequencies.items():
        node = HuffmanNode(char, freq)
        heapq.heappush(heap, node)

    # Build Huffman tree
    while len(heap) > 1:
        # Get two nodes with minimum frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # Create new internal node
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    # Generate codes
    root = heap[0]
    codes = {}

    def generate_codes(node, code=""):
        if node:
            if node.char:  # Leaf node
                codes[node.char] = code or "0"  # Handle single char case
            else:
                generate_codes(node.left, code + "0")
                generate_codes(node.right, code + "1")

    generate_codes(root)

    # Calculate total bits needed
    total_bits = sum(frequencies[char] * len(codes[char]) for char in codes)

    return codes, total_bits

def encode_text(text, codes):
    """Encode text using Huffman codes"""
    return ''.join(codes[char] for char in text)

def decode_text(encoded, root):
    """Decode text using Huffman tree"""
    decoded = []
    current = root

    for bit in encoded:
        if bit == '0':
            current = current.left
        else:
            current = current.right

        if current.char:  # Leaf node
            decoded.append(current.char)
            current = root

    return ''.join(decoded)

# Example
text = "this is an example of a huffman tree"
codes, total_bits = huffman_coding(text)

print("Character codes:")
for char, code in sorted(codes.items()):
    print(f"'{char}': {code}")

encoded = encode_text(text, codes)
print(f"\nOriginal: {len(text) * 8} bits")
print(f"Huffman:  {total_bits} bits")
print(f"Compression ratio: {len(text) * 8 / total_bits:.2f}x")
```

---

## ⏰ Interval Scheduling Problems

### **Problem 1: Meeting Rooms**

```python
def can_attend_all_meetings(intervals):
    """
    Check if person can attend all meetings (no overlaps).

    Time: O(n log n), Space: O(1)
    """
    intervals.sort(key=lambda x: x[0])  # Sort by start time

    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:  # Overlap detected
            return False

    return True

def min_meeting_rooms(intervals):
    """
    Find minimum number of meeting rooms needed.

    Greedy Strategy: Track room usage with start/end events
    Time: O(n log n), Space: O(n)
    """

    events = []

    # Create start and end events
    for start, end in intervals:
        events.append((start, 1))   # Meeting starts (+1 room)
        events.append((end, -1))    # Meeting ends (-1 room)

    # Sort events (end events come before start events at same time)
    events.sort(key=lambda x: (x[0], x[1]))

    rooms_needed = 0
    max_rooms = 0

    for time, delta in events:
        rooms_needed += delta
        max_rooms = max(max_rooms, rooms_needed)

    return max_rooms

# Alternative using heap
def min_meeting_rooms_heap(intervals):
    """
    Using heap to track earliest ending meeting
    """
    if not intervals:
        return 0

    import heapq

    intervals.sort(key=lambda x: x[0])  # Sort by start time
    heap = []  # Min heap of end times

    for start, end in intervals:
        # If earliest ending meeting has ended, reuse room
        if heap and heap[0] <= start:
            heapq.heappop(heap)

        # Assign room (or create new one)
        heapq.heappush(heap, end)

    return len(heap)

# Test
meetings = [(0, 30), (5, 10), (15, 20)]
print(can_attend_all_meetings(meetings))  # False
print(min_meeting_rooms(meetings))        # 2
```

### **Problem 2: Non-overlapping Intervals**

```python
def erase_overlap_intervals(intervals):
    """
    Remove minimum number of intervals to make rest non-overlapping.

    Greedy Strategy: Keep intervals that end earliest
    Time: O(n log n), Space: O(1)
    """

    if not intervals:
        return 0

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    end = intervals[0][1]
    remove_count = 0

    for i in range(1, len(intervals)):
        if intervals[i][0] < end:  # Overlap
            remove_count += 1
        else:
            end = intervals[i][1]

    return remove_count

def merge_intervals(intervals):
    """
    Merge all overlapping intervals.

    Time: O(n log n), Space: O(n)
    """

    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        if current[0] <= last[1]:  # Overlap
            # Merge intervals
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged

# Test
intervals = [[1,2],[2,3],[3,4],[1,3]]
print(erase_overlap_intervals(intervals))  # 1
print(merge_intervals(intervals))          # [(1, 4)]
```

---

## 🌐 Graph Greedy Algorithms

### **Minimum Spanning Tree: Kruskal's Algorithm**

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        return True

def kruskal_mst(n, edges):
    """
    Find Minimum Spanning Tree using Kruskal's algorithm.

    Greedy Strategy: Always add cheapest edge that doesn't create cycle
    Why it works: Cut property of MSTs

    Time: O(E log E), Space: O(V)
    """

    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    mst = []
    total_cost = 0

    for u, v, weight in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_cost += weight

            if len(mst) == n - 1:  # MST complete
                break

    return mst, total_cost

# Test
edges = [(0, 1, 4), (0, 2, 4), (1, 2, 2), (1, 3, 6), (2, 3, 3)]
mst, cost = kruskal_mst(4, edges)
print(f"MST edges: {mst}")
print(f"Total cost: {cost}")
```

### **Shortest Path: Dijkstra's Algorithm**

```python
import heapq

def dijkstra(graph, start):
    """
    Find shortest paths from start to all vertices.

    Greedy Strategy: Always visit unvisited vertex with shortest distance
    Why it works: Optimal substructure + greedy choice property

    Time: O((V + E) log V), Space: O(V)
    """

    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    pq = [(0, start)]
    visited = set()

    while pq:
        curr_dist, u = heapq.heappop(pq)

        if u in visited:
            continue

        visited.add(u)

        for v, weight in graph[u]:
            if v not in visited:
                new_dist = curr_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))

    return distances

# Test
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 1), ('D', 5)],
    'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
    'D': [('B', 5), ('C', 8), ('E', 2)],
    'E': [('C', 10), ('D', 2)]
}

distances = dijkstra(graph, 'A')
print(distances)
# Output: {'A': 0, 'B': 3, 'C': 2, 'D': 8, 'E': 10}
```

---

## 🚀 Advanced Greedy Techniques

### **Gas Station Problem**

```python
def can_complete_circuit(gas, cost):
    """
    Find starting gas station to complete circular trip.

    Greedy Strategy: Start from station where we first get positive balance
    Time: O(n), Space: O(1)
    """

    total_tank = 0
    curr_tank = 0
    start_station = 0

    for i in range(len(gas)):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]

        # If we can't reach next station, start from next station
        if curr_tank < 0:
            start_station = i + 1
            curr_tank = 0

    # If total gas >= total cost, solution exists
    return start_station if total_tank >= 0 else -1

def can_complete_circuit_detailed(gas, cost):
    """
    More detailed version with explanation
    """
    n = len(gas)

    # Check if solution exists
    if sum(gas) < sum(cost):
        return -1

    # Find starting point using greedy strategy
    tank = 0
    start = 0

    for i in range(n):
        tank += gas[i] - cost[i]

        # If tank becomes negative, we can't reach here from start
        # So try starting from next station
        if tank < 0:
            start = i + 1
            tank = 0

    return start

# Test
gas = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
print(can_complete_circuit(gas, cost))  # 3
```

### **Jump Game**

```python
def can_jump(nums):
    """
    Determine if you can reach the last index.

    Greedy Strategy: Keep track of farthest reachable position
    Time: O(n), Space: O(1)
    """

    farthest = 0

    for i in range(len(nums)):
        if i > farthest:
            return False

        farthest = max(farthest, i + nums[i])

        if farthest >= len(nums) - 1:
            return True

    return False

def jump_game_min_jumps(nums):
    """
    Find minimum number of jumps to reach the end.

    Greedy Strategy: For each jump, go as far as possible within current range
    Time: O(n), Space: O(1)
    """

    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        # If we've reached the end of current jump range
        if i == current_end:
            jumps += 1
            current_end = farthest

            # If we can reach the end
            if current_end >= len(nums) - 1:
                break

    return jumps

# Test
nums1 = [2, 3, 1, 1, 4]
print(can_jump(nums1))           # True
print(jump_game_min_jumps(nums1)) # 2

nums2 = [3, 2, 1, 0, 4]
print(can_jump(nums2))           # False
```

---

## ❌ When Greedy Fails

### **Problems Where Greedy Doesn't Work:**

```python
def coin_change_counterexample():
    """
    Coin change with non-standard denominations

    Coins: [4, 3, 1]
    Amount: 6

    Greedy: 4 + 1 + 1 = 3 coins
    Optimal: 3 + 3 = 2 coins
    """

    coins = [4, 3, 1]
    amount = 6

    # Greedy approach (WRONG for this case)
    greedy_result = []
    remaining = amount

    for coin in sorted(coins, reverse=True):
        while remaining >= coin:
            greedy_result.append(coin)
            remaining -= coin

    print(f"Greedy: {greedy_result} = {len(greedy_result)} coins")

    # Correct DP approach
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    print(f"Optimal: {dp[amount]} coins")

def knapsack_01_counterexample():
    """
    0/1 Knapsack (can't take fractions)

    Items: [(w=10, v=60), (w=20, v=100), (w=30, v=120)]
    Capacity: 50

    Greedy by value/weight: Take item 1 (ratio=6) and 2 (ratio=5) = value 160
    Optimal: Take item 2 and 3 = value 220
    """

    items = [(10, 60), (20, 100), (30, 120)]  # (weight, value)
    capacity = 50

    # Greedy by value/weight ratio (WRONG)
    items_with_ratio = [(v/w, w, v) for w, v in items]
    items_with_ratio.sort(reverse=True)

    greedy_value = 0
    greedy_weight = 0

    for ratio, weight, value in items_with_ratio:
        if greedy_weight + weight <= capacity:
            greedy_value += value
            greedy_weight += weight

    print(f"Greedy: value = {greedy_value}")

    # Correct DP approach
    def knapsack_dp(items, capacity):
        n = len(items)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            weight, value = items[i-1]
            for w in range(capacity + 1):
                if weight <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w-weight] + value)
                else:
                    dp[i][w] = dp[i-1][w]

        return dp[n][capacity]

    optimal_value = knapsack_dp(items, capacity)
    print(f"Optimal: value = {optimal_value}")

coin_change_counterexample()
knapsack_01_counterexample()
```

### **How to Recognize When Greedy Won't Work:**

```python
def greedy_checklist():
    """
    Red flags that indicate greedy might not work:

    ✅ GREEDY WORKS when:
    - Making locally optimal choice doesn't preclude globally optimal solution
    - Problem has optimal substructure
    - Problem has greedy choice property

    ❌ GREEDY FAILS when:
    - Local optimum != global optimum
    - Need to consider multiple future states
    - Choices are interdependent
    - Problem requires "looking ahead"

    Examples:
    - Activity Selection ✅ (earliest end time is always optimal)
    - Fractional Knapsack ✅ (highest value/weight ratio is optimal)
    - 0/1 Knapsack ❌ (can't take fractions, need to consider combinations)
    - Coin Change ❌ (depends on coin denominations)
    - Longest Path ❌ (need to explore all paths)
    """

    problems = {
        "Greedy Works": [
            "Activity Selection",
            "Fractional Knapsack",
            "Huffman Coding",
            "Minimum Spanning Tree",
            "Single-source Shortest Path (non-negative weights)"
        ],

        "Greedy Fails": [
            "0/1 Knapsack",
            "Coin Change (arbitrary denominations)",
            "Longest Path Problem",
            "Traveling Salesman Problem",
            "All-pairs Shortest Path"
        ]
    }

    return problems

# Example: Testing if greedy works
def test_greedy_optimality(problem_instance, greedy_solution, optimal_solution):
    """
    General framework to test if greedy gives optimal solution
    """
    greedy_result = greedy_solution(problem_instance)
    optimal_result = optimal_solution(problem_instance)

    return greedy_result == optimal_result
```

---

## 🎯 Interview Essentials

### **Greedy Algorithm Recognition:**

```python
def recognize_greedy_problem():
    """
    Key phrases that suggest greedy approach:

    🔍 LOOK FOR:
    - "Maximum/Minimum"
    - "Optimal"
    - "Scheduling"
    - "Earliest/Latest"
    - "Interval problems"
    - "Resource allocation"

    🤔 ASK YOURSELF:
    - Can I make the best local choice at each step?
    - Will the local optimum lead to global optimum?
    - Is there a natural ordering/sorting strategy?
    - Do I need to reconsider previous choices?
    """

    indicators = {
        "Strong Greedy Indicators": [
            "Schedule maximum activities",
            "Minimize waiting time",
            "Find shortest path",
            "Minimize completion time",
            "Resource allocation problems"
        ],

        "Consider Alternatives": [
            "Maximize/minimize combinations",
            "Count number of ways",
            "Find all possible solutions",
            "Complex interdependencies"
        ]
    }

    return indicators

# Template for greedy problems
def greedy_problem_template():
    """
    Standard approach to greedy problems:

    1. IDENTIFY the greedy choice property
    2. PROVE that greedy choice leads to optimal solution
    3. SORT data according to greedy criteria
    4. ITERATE and make greedy choices
    5. BUILD solution incrementally
    """

    def solve_greedy_problem(input_data):
        # Step 1: Define greedy choice criteria
        def greedy_criteria(item):
            # Return value to sort by (e.g., end_time, ratio, etc.)
            pass

        # Step 2: Sort input according to criteria
        sorted_data = sorted(input_data, key=greedy_criteria)

        # Step 3: Make greedy choices
        solution = []
        for item in sorted_data:
            if is_valid_choice(item, solution):
                solution.append(item)

        return solution
```

### **Common Greedy Patterns:**

```python
# Pattern 1: Interval Scheduling
def interval_pattern(intervals):
    # Sort by end time, pick non-overlapping
    intervals.sort(key=lambda x: x[1])
    # ... rest of logic

# Pattern 2: Resource Allocation
def resource_pattern(items, capacity):
    # Sort by efficiency/ratio
    items.sort(key=lambda x: x.value/x.cost, reverse=True)
    # ... rest of logic

# Pattern 3: Shortest Path
def shortest_path_pattern(graph, start):
    # Always pick unvisited node with minimum distance
    # ... Dijkstra's algorithm logic

# Pattern 4: Minimum/Maximum Selection
def min_max_pattern(items):
    # Sort by key criterion, pick greedily
    items.sort(key=lambda x: x.priority)
    # ... rest of logic
```

---

## 🏃‍♂️ Mini Sprint Project: Event Scheduler Optimizer

**Time Required:** 25-40 minutes  
**Difficulty:** Intermediate  
**Skills Practiced:** Activity selection, interval scheduling, greedy verification

### Project Overview

Build an intelligent event scheduling system that maximizes attendance using greedy algorithms.

### Core Requirements

1. **Event Management System**
   - Parse event data with start/end times, priority scores
   - Implement activity selection algorithm
   - Handle overlapping events intelligently

2. **Greedy Algorithm Implementation**
   - Sort events by finish time
   - Select non-overlapping events
   - Track total value/priority achieved

3. **Performance Analysis**
   - Compare greedy vs brute force results
   - Generate scheduling reports
   - Handle edge cases (single events, no overlaps)

### Starter Code

```python
class EventScheduler:
    def __init__(self, events):
        self.events = events  # Each event: (start, end, priority, name)

    def greedy_schedule(self):
        """
        Implement activity selection to maximize total priority
        Returns: (selected_events, total_priority)
        """
        # Step 1: Sort by finish time
        # Step 2: Greedy selection
        # Step 3: Calculate total priority
        pass

    def verify_greedy_optimal(self):
        """
        Compare greedy solution with brute force for validation
        """
        # Generate all possible valid schedules
        # Compare with greedy result
        pass

# Test with sample data
events = [
    (9, 11, 5, "Team Meeting"),
    (8, 10, 3, "Code Review"),
    (10, 12, 4, "Design Review"),
    (11, 13, 6, "Client Call"),
    (8, 9, 2, "Quick Sync")
]

scheduler = EventScheduler(events)
selected, total = scheduler.greedy_schedule()
print(f"Selected events: {selected}")
print(f"Total priority: {total}")
```

### Success Criteria

- [ ] Correctly implements activity selection algorithm
- [ ] Handles overlapping event detection
- [ ] Generates optimal scheduling results
- [ ] Includes performance comparison with brute force
- [ ] Provides clear output and documentation

### Extension Challenges

1. **Multi-track Scheduling** - Schedule events across multiple rooms/tracks
2. **Weighted Events** - Different event types have different scheduling constraints
3. **Real-time Updates** - Handle dynamic event additions/cancellations

---

## 🚀 Full Project Extension: Smart Resource Allocation Platform

**Time Required:** 6-10 hours  
**Difficulty:** Advanced  
**Skills Practiced:** Multi-objective optimization, system design, real-world application

### Project Overview

Design and implement a comprehensive resource allocation platform used by companies to optimize workforce scheduling, server resource distribution, and project task assignment using multiple greedy algorithms.

### Core Architecture

#### 1. Multi-Algorithm Engine

```python
class ResourceAllocator:
    def __init__(self):
        self.schedulers = {
            'activity_selection': ActivitySelector(),
            'fractional_knapsack': KnapsackOptimizer(),
            'interval_scheduling': IntervalScheduler(),
            'minimum_spanning_tree': GraphOptimizer()
        }

    def optimize_workforce(self, employees, projects, skills_matrix):
        """
        Assign employees to projects using greedy skill-based matching
        """
        # Greedy assignment based on skill fit and availability
        pass

    def optimize_server_resources(self, workloads, servers):
        """
        Distribute computing workloads using resource efficiency greedy
        """
        # Sort by resource efficiency ratio
        pass

    def optimize_project_timeline(self, tasks, dependencies, resources):
        """
        Schedule project tasks using activity selection with constraints
        """
        # Greedy task ordering by duration and resource availability
        pass
```

#### 2. Real-World Data Integration

- **Calendar Integration** - Import Google Calendar, Outlook schedules
- **Skills Database** - Employee skill levels and certifications
- **Resource Constraints** - Budget limits, equipment availability
- **Performance Metrics** - Historical productivity data

#### 3. Advanced Greedy Features

- **Dynamic Re-optimization** - Handle real-time constraint changes
- **Multi-objective Optimization** - Balance cost, time, quality simultaneously
- **Conflict Resolution** - Automatic conflict detection and resolution
- **Scenario Planning** - "What-if" analysis for different resource allocations

#### 4. User Interface & Reporting

- **Interactive Dashboard** - Visual resource allocation displays
- **Optimization Reports** - Efficiency metrics and improvements
- **Export Capabilities** - Integration with project management tools
- **Notification System** - Alert users to optimization opportunities

### Implementation Phases

#### Phase 1: Core Algorithm Implementation (2-3 hours)

- Implement all four greedy algorithms
- Create comprehensive test suite
- Build performance benchmarking system

#### Phase 2: Data Model & Integration (2-3 hours)

- Design database schema for resources, constraints, outcomes
- Implement data import/export functionality
- Create mock real-world datasets for testing

#### Phase 3: User Interface & Visualization (2-3 hours)

- Build interactive dashboard with D3.js or similar
- Create algorithm selection and configuration interface
- Implement result visualization and comparison tools

#### Phase 4: Advanced Features & Polish (1-2 hours)

- Add real-time optimization capabilities
- Implement performance monitoring and alerts
- Create comprehensive documentation and user guides

### Success Criteria

- [ ] All four greedy algorithms implemented and tested
- [ ] Real-world data integration working
- [ ] Interactive dashboard with visualization
- [ ] Performance metrics and reporting
- [ ] Handles edge cases and constraint violations
- [ ] Professional documentation and examples

### Technical Stack Recommendations

- **Backend:** Python with FastAPI or Django
- **Frontend:** React with D3.js for visualization
- **Database:** PostgreSQL for complex queries
- **Testing:** pytest with comprehensive test coverage
- **Deployment:** Docker containerization

### Learning Outcomes

This project demonstrates mastery of greedy algorithms in production contexts, showcasing ability to:

- Design scalable optimization systems
- Handle real-world constraints and edge cases
- Build user interfaces for complex algorithms
- Integrate multiple algorithmic approaches
- Measure and optimize system performance

---

## 📊 Complexity Analysis

### **Time Complexity Patterns:**

| Problem Type            | Sorting    | Algorithm      | Total              | Notes                   |
| ----------------------- | ---------- | -------------- | ------------------ | ----------------------- |
| **Activity Selection**  | O(n log n) | O(n)           | **O(n log n)**     | Dominated by sorting    |
| **Fractional Knapsack** | O(n log n) | O(n)           | **O(n log n)**     | Sort by ratio           |
| **Huffman Coding**      | -          | O(n log n)     | **O(n log n)**     | Heap operations         |
| **Kruskal's MST**       | O(E log E) | O(E α(V))      | **O(E log E)**     | Sort edges + Union-Find |
| **Dijkstra's**          | -          | O((V+E) log V) | **O((V+E) log V)** | Priority queue          |

### **When to Choose Greedy:**

- ✅ Problem has greedy choice property
- ✅ Optimal substructure exists
- ✅ Need efficient solution
- ✅ Local choices don't affect global optimum

---

_Master greedy algorithms for efficient optimization solutions! 🚀_
