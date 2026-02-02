# Greedy Algorithms Interview Questions

## Basic Level Questions

### 1. Activity Selection Problem

**Question**: Given a set of activities with start and end times, select the maximum number of non-overlapping activities.

**Answer**:

```python
def select_activities(activities):
    """
    Time: O(n log n) for sorting
    Space: O(1)
    """
    if not activities:
        return []

    # Sort by end time
    activities.sort(key=lambda x: x[1])

    selected = [activities[0]]
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected

# Alternative: Return count
def max_activities(activities):
    activities.sort(key=lambda x: x[1])
    count = 1
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            count += 1
            last_end = end

    return count
```

**Follow-up Questions**:

- How would you handle weighted activities?
- What if activities can be preempted?

### 2. Job Sequencing with Deadlines

**Question**: Given jobs with deadlines and profits, schedule jobs to maximize total profit.

**Answer**:

```python
def job_sequencing(jobs):
    """
    Time: O(n log n) for sorting
    Space: O(n) for time slots
    """
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

# Alternative: Return total profit
def max_profit_job_sequencing(jobs):
    jobs.sort(key=lambda x: x[2], reverse=True)

    max_deadline = max(job[1] for job in jobs)
    time_slots = [None] * max_deadline
    total_profit = 0

    for job_id, deadline, profit in jobs:
        for slot in range(min(deadline, max_deadline) - 1, -1, -1):
            if time_slots[slot] is None:
                time_slots[slot] = (job_id, profit)
                total_profit += profit
                break

    return total_profit
```

**Follow-up Questions**:

- How to handle multiple workers?
- What if jobs have different durations?

### 3. Minimum Number of Coins

**Question**: Find minimum number of coins needed to make a given amount using given coin denominations.

**Answer**:

```python
def min_coins(coins, amount):
    """
    Works for canonical coin systems (1, 2, 5, 10, 20, etc.)
    Time: O(n log n) for sorting
    """
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

# Alternative: Using dynamic programming for non-canonical systems
def min_coins_dp(coins, amount):
    """
    Works for all coin systems
    Time: O(amount * len(coins))
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

**Follow-up Questions**:

- When does greedy work? When does it fail?
- How to handle large amounts efficiently?

### 4. Gas Station Problem

**Question**: Find starting gas station to complete circuit given gas costs and gas available.

**Answer**:

```python
def gas_station(gas, cost):
    """
    Time: O(n)
    Space: O(1)
    """
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

# Alternative: Brute force for verification
def gas_station_brute(gas, cost):
    n = len(gas)

    for start in range(n):
        tank = 0
        for i in range(n):
            tank += gas[(start + i) % n] - cost[(start + i) % n]
            if tank < 0:
                break
        else:
            return start

    return -1
```

## Intermediate Level Questions

### 5. Fractional Knapsack

**Question**: Given items with value and weight, find maximum value with weight limit allowing fractions.

**Answer**:

```python
def fractional_knapsack(items, capacity):
    """
    Time: O(n log n) for sorting
    Space: O(1)
    """
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
            total_value += value
            remaining_capacity -= weight
        else:
            fraction = remaining_capacity / weight
            total_value += value * fraction
            remaining_capacity = 0

    return total_value

# Alternative: Return selected items
def fractional_knapsack_items(items, capacity):
    items.sort(key=lambda x: x[0]/x[1], reverse=True)

    selected = []
    remaining_capacity = capacity

    for value, weight in items:
        if remaining_capacity == 0:
            break

        if weight <= remaining_capacity:
            selected.append((value, weight, 1.0))
            remaining_capacity -= weight
        else:
            fraction = remaining_capacity / weight
            selected.append((value, weight * fraction, fraction))
            remaining_capacity = 0

    return selected
```

**Follow-up Questions**:

- How is this different from 0-1 knapsack?
- When can you use greedy vs dynamic programming?

### 6. Minimum Platforms Required

**Question**: Find minimum platforms needed at railway station for given arrival and departure times.

**Answer**:

```python
def min_platforms(arrival, departure):
    """
    Time: O(n log n) for sorting
    Space: O(1)
    """
    if not arrival or not departure or len(arrival) != len(departure):
        return 0

    arrival.sort()
    departure.sort()

    platforms_needed = 0
    max_platforms = 0
    i = j = 0

    while i < len(arrival):
        if arrival[i] <= departure[j]:
            platforms_needed += 1
            max_platforms = max(max_platforms, platforms_needed)
            i += 1
        else:
            platforms_needed -= 1
            j += 1

    return max_platforms

# Alternative: Using events
def min_platforms_events(arrival, departure):
    events = []
    for time in arrival:
        events.append((time, 1))  # Arrival event
    for time in departure:
        events.append((time, -1))  # Departure event

    events.sort(key=lambda x: (x[0], -x[1]))  # Arrivals before departures

    platforms = 0
    max_platforms = 0

    for time, event in events:
        platforms += event
        max_platforms = max(max_platforms, platforms)

    return max_platforms
```

**Follow-up Questions**:

- How to handle overlapping intervals?
- What if times are in different formats?

### 7. Largest Number from Array

**Question**: Given array of integers, form the largest possible number.

**Answer**:

```python
def largest_number(nums):
    """
    Time: O(n log n) for sorting
    Space: O(n) for string conversion
    """
    if not nums:
        return "0"

    # Convert to strings
    str_nums = list(map(str, nums))

    # Custom sort: compare concatenations
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

    # Handle leading zeros
    return "0" if result[0] == '0' else result

# Alternative: Using key function
def largest_number_key(nums):
    def compare(x, y):
        return int(y + x) - int(x + y)

    str_nums = list(map(str, nums))
    str_nums.sort(key=cmp_to_key(compare))
    result = ''.join(str_nums)
    return "0" if result[0] == '0' else result
```

## Advanced Level Questions

### 8. N Meetings in One Room

**Question**: Schedule maximum meetings in one room given start and end times.

**Answer**:

```python
def n_meetings(start, end):
    """
    Time: O(n log n) for sorting
    Space: O(1)
    """
    if not start or not end or len(start) != len(end):
        return 0

    # Create meetings and sort by end time
    meetings = list(zip(start, end))
    meetings.sort(key=lambda x: x[1])

    count = 1  # First meeting
    last_end = meetings[0][1]

    for s, e in meetings[1:]:
        if s >= last_end:
            count += 1
            last_end = e

    return count

# Return actual meeting sequence
def n_meetings_sequence(start, end):
    meetings = list(zip(start, end, range(len(start))))
    meetings.sort(key=lambda x: x[1])

    selected = [meetings[0]]
    last_end = meetings[0][1]

    for s, e, idx in meetings[1:]:
        if s >= last_end:
            selected.append((s, e, idx))
            last_end = e

    return selected
```

### 9. Minimum Refueling Stops

**Question**: Find minimum number of refueling stops to reach destination.

**Answer**:

```python
import heapq

def min_refuel_stops(target, start_fuel, stations):
    """
    Time: O(n log n)
    Space: O(n)
    """
    max_distance = start_fuel
    stops = 0
    i = 0
    max_heap = []  # Max-heap of fuel amounts

    while max_distance < target:
        # Add all reachable stations
        while i < len(stations) and stations[i][0] <= max_distance:
            heapq.heappush(max_heap, -stations[i][1])
            i += 1

        if not max_heap:
            return -1  # Cannot reach target

        # Fuel up at station with most fuel
        max_distance += -heapq.heappop(max_heap)
        stops += 1

    return stops

# Alternative: Return sequence of stops
def min_refuel_stops_sequence(target, start_fuel, stations):
    max_distance = start_fuel
    stops = []
    i = 0
    max_heap = []

    while max_distance < target:
        while i < len(stations) and stations[i][0] <= max_distance:
            heapq.heappush(max_heap, (-stations[i][1], i))
            i += 1

        if not max_heap:
            return -1

        fuel, station_idx = heapq.heappop(max_heap)
        max_distance += -fuel
        stops.append(station_idx)

    return stops
```

**Follow-up Questions**:

- How to handle stations with different fuel prices?
- What if fuel consumption varies by vehicle type?

### 10. Lemonade Change

**Question**: Determine if you can give correct change to all customers.

**Answer**:

```python
def lemonade_change(bills):
    """
    Time: O(n)
    Space: O(1)
    """
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

# Alternative: Track all possible change combinations
def lemonade_change_advanced(bills):
    from collections import defaultdict

    change = defaultdict(int)  # {denomination: count}
    change[5] = 0
    change[10] = 0
    change[20] = 0

    for bill in bills:
        if bill == 5:
            change[5] += 1
        elif bill == 10:
            if change[5] == 0:
                return False
            change[5] -= 1
            change[10] += 1
        elif bill == 20:
            # Prefer 10+5 over 5+5+5
            if change[10] > 0 and change[5] > 0:
                change[10] -= 1
                change[5] -= 1
            elif change[5] >= 3:
                change[5] -= 3
            else:
                return False

    return True
```

## System Design Questions

### 11. Design Task Scheduler

**Question**: Design a task scheduler that prioritizes tasks by deadline and profit.

**Answer**:

```python
import heapq
from datetime import datetime, timedelta

class TaskScheduler:
    def __init__(self, max_concurrent_tasks=1):
        self.max_concurrent = max_concurrent_tasks
        self.task_queue = []  # Min-heap by deadline
        self.running_tasks = []  # (end_time, task_id, task)
        self.task_id = 0

    def add_task(self, task, deadline, profit):
        self.task_id += 1
        heapq.heappush(self.task_queue, (deadline, profit, self.task_id, task))

    def get_next_task(self):
        if not self.task_queue or len(self.running_tasks) >= self.max_concurrent:
            return None

        deadline, profit, task_id, task = heapq.heappop(self.task_queue)
        end_time = datetime.now() + timedelta(hours=1)  # Assume 1-hour tasks
        self.running_tasks.append((end_time, task_id, task))
        return task

    def complete_finished_tasks(self):
        now = datetime.now()
        self.running_tasks = [task for task in self.running_tasks if task[0] > now]
```

### 12. Design Resource Allocator

**Question**: Design a resource allocation system that optimally distributes resources.

**Answer**:

```python
import heapq

class ResourceAllocator:
    def __init__(self, resources):
        self.resources = resources
        self.available = set(range(len(resources)))
        self.allocated = {}

    def allocate(self, requirement):
        """
        requirement: (demand, profit, deadline)
        """
        demand, profit, deadline = requirement
        candidates = []

        for resource_id in self.available:
            capacity = self.resources[resource_id]
            if capacity >= demand:
                efficiency = profit / demand
                candidates.append((-efficiency, resource_id, capacity))

        if not candidates:
            return None

        # Greedy: select most efficient resource
        candidates.sort()
        efficiency, resource_id, capacity = candidates[0]

        self.allocated[resource_id] = requirement
        self.available.remove(resource_id)

        return resource_id

    def deallocate(self, resource_id):
        if resource_id in self.allocated:
            del self.allocated[resource_id]
            self.available.add(resource_id)

    def get_utilization(self):
        allocated_capacity = sum(self.resources[i] for i in self.allocated)
        total_capacity = sum(self.resources)
        return allocated_capacity / total_capacity
```

## Common Follow-up Questions

### Q1: When does greedy fail?

**Answer**:

- Non-canonical coin systems (e.g., [1, 3, 4] for amount 6)
- 0-1 Knapsack (cannot take fractions)
- Problems requiring global optimization

### Q2: How to prove greedy is optimal?

**Answer**:

1. **Optimal Substructure**: Problem can be broken into subproblems
2. **Greedy Choice Property**: Local optimal choice leads to global optimum
3. **Exchange Argument**: Show any optimal solution can be transformed to greedy solution

### Q3: What if multiple greedy strategies work?

**Answer**:

- Consider tie-breaking strategies
- Test edge cases
- Analyze time complexity differences

### Q4: How to handle real-time constraints?

**Answer**:

- Use data structures for efficient updates (heaps, BSTs)
- Consider incremental recomputation
- Cache intermediate results

### Q5: How to optimize space complexity?

**Answer**:

- Use in-place sorting when possible
- Stream processing for large datasets
- Lazy evaluation for heavy computations

## Interview Success Strategies

1. **Identify the pattern** quickly
2. **State assumptions** about input constraints
3. **Prove greedy choice** or provide strong intuition
4. **Handle edge cases** explicitly
5. **Analyze complexity** thoroughly
6. **Test with examples** including counterexamples
7. **Consider alternatives** when greedy fails
8. **Optimize for interview** - clean, readable code
