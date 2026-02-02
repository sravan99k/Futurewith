# Greedy Algorithms Practice Exercises

## Basic Greedy Problems

### 1. Activity Selection Problem

```python
def select_activities(activities):
    """
    Select maximum number of non-overlapping activities
    activities: list of (start_time, end_time)
    """
    # TODO: Sort by end time and select greedily
    pass

# Test cases:
activities1 = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 8), (5, 9), (6, 10), (8, 11), (8, 12), (2, 13), (12, 14)]
# Expected: [(1, 4), (5, 7), (8, 11), (12, 14)]

activities2 = [(10, 20), (12, 25), (20, 30)]
# Expected: [(10, 20), (20, 30)]
```

**Solution**:

```python
def select_activities(activities):
    if not activities:
        return []

    # Sort activities by end time
    activities.sort(key=lambda x: x[1])

    selected = [activities[0]]
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected
```

### 2. Job Sequencing with Deadlines

```python
def job_sequencing(jobs):
    """
    Maximize profit by scheduling jobs before deadlines
    jobs: list of (job_id, deadline, profit)
    """
    # TODO: Sort by profit and schedule greedily
    pass

# Test:
jobs = [('a', 2, 100), ('b', 1, 19), ('c', 2, 27), ('d', 1, 25), ('e', 3, 15)]
# Expected: c, a, e (profit: 27 + 100 + 15 = 142)
```

**Solution**:

```python
def job_sequencing(jobs):
    if not jobs:
        return []

    # Sort jobs by profit (descending)
    jobs.sort(key=lambda x: x[2], reverse=True)

    max_deadline = max(job[1] for job in jobs)
    time_slots = [False] * max_deadline
    result = []

    for job_id, deadline, profit in jobs:
        # Find available slot from deadline to 1
        for slot in range(deadline - 1, -1, -1):
            if not time_slots[slot]:
                time_slots[slot] = True
                result.append(job_id)
                break

    return result
```

### 3. Minimum Number of Coins

```python
def min_coins(coins, amount):
    """
    Find minimum number of coins to make amount
    coins: list of available coin denominations
    amount: target amount
    """
    # TODO: Use greedy (works only for canonical coin systems)
    pass

# Test:
coins1 = [1, 2, 5], amount1 = 11  # Expected: 3 (5+5+1)
coins2 = [1, 2, 5], amount2 = 3   # Expected: 2 (2+1)
coins3 = [1, 3, 4], amount3 = 6   # Expected: 2 (3+3) but greedy might fail
```

**Solution**:

```python
def min_coins(coins, amount):
    if amount == 0:
        return 0

    if not coins:
        return -1

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

    return coin_count if remaining == 0 else -1
```

### 4. Fractional Knapsack

```python
def fractional_knapsack(items, capacity):
    """
    Maximize value with fractional items
    items: list of (value, weight)
    capacity: maximum weight capacity
    """
    # TODO: Sort by value/weight ratio and take fractions
    pass

# Test:
items1 = [(60, 10), (100, 20), (120, 30)], capacity1 = 50
# Expected: Maximum value = 240 (10kg*6 + 20kg*5 + 20kg*6)

items2 = [(70, 10), (100, 20), (120, 30)], capacity2 = 50
# Expected: Maximum value = 260
```

**Solution**:

```python
def fractional_knapsack(items, capacity):
    if not items or capacity <= 0:
        return 0

    # Sort by value/weight ratio (descending)
    items.sort(key=lambda x: x[0]/x[1], reverse=True)

    total_value = 0
    remaining_capacity = capacity

    for value, weight in items:
        if remaining_capacity == 0:
            break

        if weight <= remaining_capacity:
            # Take whole item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction
            fraction = remaining_capacity / weight
            total_value += value * fraction
            remaining_capacity = 0

    return total_value
```

### 5. Gas Station Problem

```python
def gas_station(gas, cost):
    """
    Find starting gas station for complete circuit
    gas: gas available at each station
    cost: cost to reach next station
    """
    # TODO: Find start index using greedy approach
    pass

# Test:
gas1 = [1, 2, 3, 4, 5], cost1 = [3, 4, 5, 1, 2]  # Expected: 3
gas2 = [2, 3, 4], cost2 = [3, 4, 3]              # Expected: 2
```

**Solution**:

```python
def gas_station(gas, cost):
    if not gas or len(gas) != len(cost):
        return -1

    total_gas = 0
    total_cost = 0
    start = 0
    tank = 0

    for i in range(len(gas)):
        total_gas += gas[i]
        total_cost += cost[i]
        tank += gas[i] - cost[i]

        if tank < 0:
            start = i + 1
            tank = 0

    return start if total_gas >= total_cost else -1
```

## Intermediate Greedy Problems

### 6. Maximum Meetings in a Room

```python
def max_meetings(start, end, room_count):
    """
    Schedule maximum meetings given multiple rooms
    start: list of start times
    end: list of end times
    room_count: number of available rooms
    """
    # TODO: Use min-heap to track room availability
    pass

# Test:
start = [1, 3, 0, 5, 8, 5]
end = [2, 4, 6, 7, 9, 9]
room_count = 2
# Expected: Maximum number of meetings
```

**Solution**:

```python
import heapq

def max_meetings(start, end, room_count):
    if not start or not end or len(start) != len(end):
        return 0

    meetings = list(zip(start, end))
    meetings.sort()  # Sort by start time

    # Min-heap to track end times of occupied rooms
    occupied_rooms = []
    scheduled = 0

    for s, e in meetings:
        # Free up rooms that have ended
        while occupied_rooms and occupied_rooms[0] <= s:
            heapq.heappop(occupied_rooms)

        # Assign room if available
        if len(occupied_rooms) < room_count:
            heapq.heappush(occupied_rooms, e)
            scheduled += 1

    return scheduled
```

### 7. Minimum Platforms Required

```python
def min_platforms(arrival, departure):
    """
    Find minimum platforms needed at railway station
    arrival: list of arrival times
    departure: list of departure times
    """
    # TODO: Sort and count overlapping intervals
    pass

# Test:
arrival = [900, 940, 950, 1100, 1500, 1800]
departure = [910, 1200, 1120, 1130, 1900, 2000]
# Expected: 3
```

**Solution**:

```python
def min_platforms(arrival, departure):
    if not arrival or not departure or len(arrival) != len(departure):
        return 0

    # Sort both arrays
    arrival.sort()
    departure.sort()

    platforms_needed = 0
    max_platforms = 0
    i = j = 0

    while i < len(arrival):
        if arrival[i] <= departure[j]:
            platforms_needed += 1
            i += 1
            max_platforms = max(max_platforms, platforms_needed)
        else:
            platforms_needed -= 1
            j += 1

    return max_platforms
```

### 8. Largest Number from Array

```python
def largest_number(nums):
    """
    Form the largest possible number from array of integers
    nums: list of integers
    """
    # TODO: Sort using custom comparator
    pass

# Test:
nums1 = [10, 2]              # Expected: "210"
nums2 = [3, 30, 34, 5, 9]    # Expected: "9534330"
```

**Solution**:

```python
def largest_number(nums):
    if not nums:
        return "0"

    # Convert to strings for comparison
    str_nums = list(map(str, nums))

    # Custom sort: x+y > y+x means x should come before y
    from functools import cmp_to_key

    def compare(x, y):
        if x + y > y + x:
            return -1
        elif x + y < y + x:
            return 1
        else:
            return 0

    str_nums.sort(key=cmp_to_key(compare))

    result = ''.join(str_nums)

    # Handle case where result starts with '0'
    return "0" if result[0] == '0' else result
```

## Advanced Greedy Problems

### 9. N Meetings in One Room

```python
def n_meetings(start, finish):
    """
    Find maximum number of meetings that can be held in one room
    """
    # TODO: Optimize for single room scheduling
    pass

# Test:
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
# Expected: Maximum meetings = 4
```

**Solution**:

```python
def n_meetings(start, finish):
    if not start or not finish or len(start) != len(finish):
        return 0

    # Create list of meetings and sort by finish time
    meetings = list(zip(start, finish))
    meetings.sort(key=lambda x: x[1])

    count = 1  # First meeting is always possible
    last_finish = meetings[0][1]

    for s, f in meetings[1:]:
        if s >= last_finish:
            count += 1
            last_finish = f

    return count
```

### 10. Min Cost to Connect All Points

```python
def min_cost_connect_points(points):
    """
    Minimum cost to connect all points (like minimum spanning tree)
    points: list of [x, y] coordinates
    """
    # TODO: Use Prim's algorithm
    pass

# Test:
points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
# Expected: 20
```

**Solution**:

```python
import heapq
from math import sqrt

def min_cost_connect_points(points):
    if not points or len(points) < 2:
        return 0

    n = len(points)
    visited = [False] * n
    min_cost = 0
    heap = [(0, 0)]  # (cost, node)

    while len(visited) != n or heap:
        cost, node = heapq.heappop(heap)

        if visited[node]:
            continue

        visited[node] = True
        min_cost += cost

        # Add all unvisited neighbors
        for i in range(n):
            if not visited[i]:
                # Calculate distance
                dist = abs(points[node][0] - points[i][0]) + abs(points[node][1] - points[i][1])
                heapq.heappush(heap, (dist, i))

    return min_cost
```

### 11. Minimum Number of Refueling Stops

```python
def min_refuel_stops(target, start_fuel, stations):
    """
    Find minimum number of refueling stops to reach target
    target: target distance
    start_fuel: initial fuel
    stations: list of [distance, fuel] stations
    """
    # TODO: Use max-heap to select best stations
    pass

# Test:
target, start_fuel, stations = 100, 10, [[10, 60], [20, 30], [30, 30], [60, 40]]
# Expected: 2 stops
```

**Solution**:

```python
import heapq

def min_refuel_stops(target, start_fuel, stations):
    max_distance = start_fuel
    stops = 0
    i = 0
    max_heap = []  # max-heap of fuel amounts

    while max_distance < target:
        # Add all stations we can reach
        while i < len(stations) and stations[i][0] <= max_distance:
            heapq.heappush(max_heap, -stations[i][1])
            i += 1

        # If we can't reach target and no more stations, impossible
        if not max_heap:
            return -1

        # Fuel up at the station with most fuel
        max_distance += -heapq.heappop(max_heap)
        stops += 1

    return stops
```

### 12. Lemonade Change

```python
def lemonade_change(bills):
    """
    Check if you can give correct change to all customers
    bills: list of bills received (5, 10, 20)
    """
    # TODO: Simulate with greedy change giving
    pass

# Test:
bills1 = [5, 5, 5, 10, 20]  # Expected: True
bills2 = [5, 5, 10]         # Expected: True
bills3 = [10, 10]           # Expected: False
```

**Solution**:

```python
def lemonade_change(bills):
    five, ten, twenty = 0, 0, 0

    for bill in bills:
        if bill == 5:
            five += 1
        elif bill == 10:
            if five == 0:
                return False
            five -= 1
            ten += 1
        elif bill == 20:
            # Prefer giving one 10 and one 5 over three 5s
            if ten > 0 and five > 0:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False

    return True
```

## Pattern Recognition Exercises

### Exercise Set A: Interval Scheduling

1. **Room Allocation**: Given intervals [start, end], allocate minimum rooms
2. **Course Schedule**: Given prerequisites as [course, prerequisite], find order
3. **Task Scheduler**: Given tasks with cooldown, find minimum time

### Exercise Set B: Resource Allocation

1. **Job Scheduling**: Maximize profit with deadlines
2. **Resource Loading**: Distribute resources optimally
3. **Inventory Management**: Stock items optimally

### Exercise Set C: Greedy Selection

1. **Huffman Coding**: Build optimal prefix codes
2. **Kruskal's MST**: Build minimum spanning tree
3. **Dijkstra's Shortest Path**: Find shortest paths

## Time Complexity Analysis

| Problem             | Time Complexity | Space Complexity | Greedy Strategy          |
| ------------------- | --------------- | ---------------- | ------------------------ |
| Activity Selection  | O(n log n)      | O(1)             | Sort by end time         |
| Job Sequencing      | O(n log n)      | O(n)             | Sort by profit           |
| Fractional Knapsack | O(n log n)      | O(1)             | Sort by value/weight     |
| Gas Station         | O(n)            | O(1)             | Greedy start selection   |
| Coin Change         | O(n log n)      | O(1)             | Greedy coin selection    |
| Minimum Platforms   | O(n log n)      | O(1)             | Two-pointer technique    |
| Largest Number      | O(n log n)      | O(n)             | Custom string comparison |

## When Greedy Works

### Sufficient Conditions:

1. **Optimal Substructure**: Problem can be broken into subproblems
2. **Greedy Choice Property**: Local optimal choice leads to global optimum

### When Greedy Fails:

1. **Non-canonical coin systems**: [1, 3, 4] for amount 6
2. **0-1 Knapsack**: Cannot take fractions
3. **Longest increasing subsequence**: Need dynamic programming

## Practice Tips

1. **Identify the problem type**: Sorting, selection, or resource allocation
2. **Prove greedy choice property** when possible
3. **Consider edge cases**: Empty inputs, single elements
4. **Test with small examples** to verify logic
5. **Look for counterexamples** when greedy might fail
