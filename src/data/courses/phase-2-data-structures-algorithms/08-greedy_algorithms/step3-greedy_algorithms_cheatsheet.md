# Greedy Algorithms Quick Reference

## Core Greedy Strategy

### Greedy Choice Property

- **Local Optimal Choice**: Make choice that seems best at the moment
- **Optimal Substructure**: Problem can be broken into subproblems
- **No Regret**: Early choices don't need to be reconsidered

## Common Problem Patterns

### 1. Activity Selection

```python
# Sort by earliest finish time
activities.sort(key=lambda x: x[1])  # x[1] = end time
selected = [activities[0]]
last_end = activities[0][1]

for start, end in activities[1:]:
    if start >= last_end:
        selected.append((start, end))
        last_end = end
```

### 2. Job Sequencing with Deadlines

```python
# Sort by profit (descending)
jobs.sort(key=lambda x: x[2], reverse=True)  # x[2] = profit

max_deadline = max(job[1] for job in jobs)  # x[1] = deadline
time_slots = [False] * max_deadline

for job_id, deadline, profit in jobs:
    for slot in range(deadline - 1, -1, -1):
        if not time_slots[slot]:
            time_slots[slot] = True
            break
```

### 3. Fractional Knapsack

```python
# Sort by value/weight ratio (descending)
items.sort(key=lambda x: x[0]/x[1], reverse=True)  # x[0]=value, x[1]=weight

total_value = 0
remaining_capacity = capacity

for value, weight in items:
    if remaining_capacity == 0:
        break

    if weight <= remaining_capacity:
        total_value += value
        remaining_capacity -= weight
    else:
        total_value += value * (remaining_capacity / weight)
        remaining_capacity = 0
```

### 4. Minimum Number of Coins

```python
# Sort coins in descending order
coins.sort(reverse=True)

coin_count = 0
remaining = amount

for coin in coins:
    if remaining == 0:
        break

    if coin <= remaining:
        num_coins = remaining // coin
        coin_count += num_coins
        remaining -= num_coins * coin
```

## Template Solutions

### Template 1: Interval Scheduling

```python
def interval_scheduling(intervals):
    """
    intervals: list of [start, end]
    """
    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    selected = []
    last_end = float('-inf')

    for start, end in intervals:
        if start >= last_end:
            selected.append([start, end])
            last_end = end

    return selected
```

### Template 2: Resource Allocation

```python
def resource_allocation(tasks, resources):
    """
    tasks: list of [deadline, profit]
    resources: available resources
    """
    # Sort by profit
    tasks.sort(key=lambda x: x[1], reverse=True)

    resource_used = [False] * resources
    total_profit = 0

    for deadline, profit in tasks:
        # Find available resource slot
        for slot in range(min(deadline, resources) - 1, -1, -1):
            if not resource_used[slot]:
                resource_used[slot] = True
                total_profit += profit
                break

    return total_profit
```

### Template 3: Fuel Station

```python
def min_refuel_stops(target, start_fuel, stations):
    """
    Find minimum number of refueling stops
    """
    import heapq

    max_distance = start_fuel
    stops = 0
    i = 0
    max_heap = []  # max-heap of fuel

    while max_distance < target:
        # Add all reachable stations
        while i < len(stations) and stations[i][0] <= max_distance:
            heapq.heappush(max_heap, -stations[i][1])
            i += 1

        if not max_heap:
            return -1  # Cannot reach target

        # Refuel at station with most fuel
        max_distance += -heapq.heappop(max_heap)
        stops += 1

    return stops
```

## Problem-Specific Strategies

### Interval Problems

```python
# Minimum number of platforms
def min_platforms(arrival, departure):
    arrival.sort()
    departure.sort()

    platforms = 0
    max_platforms = 0
    i = j = 0

    while i < len(arrival):
        if arrival[i] <= departure[j]:
            platforms += 1
            max_platforms = max(max_platforms, platforms)
            i += 1
        else:
            platforms -= 1
            j += 1

    return max_platforms
```

### Meeting Room Problems

```python
# Maximum meetings in one room
def max_meetings(start, end):
    meetings = list(zip(start, end))
    meetings.sort(key=lambda x: x[1])

    count = 1
    last_end = meetings[0][1]

    for s, e in meetings[1:]:
        if s >= last_end:
            count += 1
            last_end = e

    return count

# Multiple rooms
def max_meetings_multi_room(start, end, k):
    meetings = list(zip(start, end))
    meetings.sort()

    import heapq
    room_heap = []  # min-heap of end times

    for s, e in meetings:
        if room_heap and room_heap[0] <= s:
            heapq.heappop(room_heap)  # Free room

        if len(room_heap) < k:
            heapq.heappush(room_heap, e)

    return len(room_heap)
```

### Coin Change Variants

```python
# Maximum number of coins
def max_coins(coins, amount):
    if amount == 0:
        return 0

    coins.sort()
    count = 0
    remaining = amount

    for coin in reversed(coins):
        if remaining == 0:
            break

        num = remaining // coin
        if num > 0:
            count += num
            remaining -= num * coin

    return count if remaining == 0 else -1
```

## Advanced Patterns

### Huffman Coding

```python
import heapq
from collections import Counter

def huffman_coding(freq):
    # Create min-heap of nodes
    heap = []
    for char, f in freq.items():
        heapq.heappush(heap, (f, [char]))

    while len(heap) > 1:
        f1, left = heapq.heappop(heap)
        f2, right = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, [left, right]))

    return heap[0][1]  # Return tree structure
```

### Prim's MST

```python
def prim_mst(graph):
    import heapq

    vertices = set(graph.keys())
    visited = set()
    heap = [(0, list(vertices)[0])]  # (cost, vertex)
    mst_cost = 0
    mst = []

    while heap and len(visited) < len(vertices):
        cost, vertex = heapq.heappop(heap)

        if vertex in visited:
            continue

        visited.add(vertex)
        mst_cost += cost

        for neighbor, weight in graph[vertex].items():
            if neighbor not in visited:
                heapq.heappush(heap, (weight, neighbor))

    return mst_cost
```

## Common Mistakes to Avoid

### 1. Wrong Sorting Order

```python
# ❌ Wrong: Sort by start time for activity selection
activities.sort(key=lambda x: x[0])  # Start time

# ✅ Correct: Sort by end time
activities.sort(key=lambda x: x[1])  # End time
```

### 2. Ignoring Edge Cases

```python
# ❌ Not handling empty input
def greedy_algorithm(items):
    # Missing empty check
    result = []
    # ...

# ✅ Handle edge cases
def greedy_algorithm(items):
    if not items:
        return []
    # ...
```

### 3. Forgetting to Prove Greedy Choice

```python
# Always consider:
# 1. Is there optimal substructure?
# 2. Does greedy choice lead to optimal solution?
# 3. Can you find a counterexample?
```

## Time Complexity Quick Reference

| Problem                 | Time       | Space | Greedy Strategy  |
| ----------------------- | ---------- | ----- | ---------------- |
| Activity Selection      | O(n log n) | O(1)  | Sort by end time |
| Job Sequencing          | O(n log n) | O(n)  | Sort by profit   |
| Fractional Knapsack     | O(n log n) | O(1)  | Sort by ratio    |
| Gas Station             | O(n)       | O(1)  | Linear scan      |
| Coin Change (Canonical) | O(n)       | O(1)  | Descending sort  |
| Minimum Platforms       | O(n log n) | O(1)  | Two pointers     |
| Huffman Coding          | O(n log n) | O(n)  | Priority queue   |
| Prim's MST              | O(E log V) | O(V)  | Priority queue   |

## When Greedy Works

### ✅ Greedy Optimal Cases:

1. **Activity Selection**: Non-overlapping intervals
2. **Fractional Knapsack**: Can take fractions
3. **Huffman Coding**: Optimal prefix codes
4. **Kruskal's/Prim's MST**: Greedy edge selection
5. **Dijkstra's Shortest Path**: Non-negative weights
6. **Canonical Coin Systems**: Powers of coin denominations

### ❌ Greedy Fails:

1. **0-1 Knapsack**: Cannot take fractions
2. **Coin Change (Non-canonical)**: [1, 3, 4] for amount 6
3. **Longest Common Subsequence**: Need dynamic programming
4. **Matrix Chain Multiplication**: Subproblem overlapping

## Decision Tree

```
Problem Type?
├── Scheduling
│   ├── Single resource → Sort by end time
│   └── Multiple resources → Priority queue
├── Selection
│   ├── Maximize value/weight → Sort by ratio
│   └── Maximize count → Sort by efficiency
├── Allocation
│   ├── One-time allocation → Sort by profit/deadline
│   └── Continuous allocation → Binary search
└── Optimization
    ├── Tree-based → Prim's/Kruskal's
    └── Path-based → Dijkstra's
```

## Interview Success Tips

1. **Identify the pattern** quickly
2. **Prove greedy choice** or provide intuition
3. **Handle edge cases** explicitly
4. **Optimize space** when possible
5. **Test with examples** including edge cases
6. **Consider alternatives** if greedy fails
7. **Explain complexity** clearly
