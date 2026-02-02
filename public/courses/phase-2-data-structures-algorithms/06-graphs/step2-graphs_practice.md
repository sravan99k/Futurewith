---
title: "Graphs Practice Problems - 100+ Questions"
level: "Beginner to Advanced"
difficulty: "Progressive (Easy → Medium → Hard)"
time: "Varies (15-90 minutes per question)"
tags:
  [
    "dsa",
    "graphs",
    "bfs",
    "dfs",
    "shortest-path",
    "practice",
    "coding-interview",
  ]
---

# 🌐 Graphs Practice Problems

_100+ Progressive Problems from Basic Traversal to Advanced Algorithms_

---

## 📊 Problem Difficulty Distribution

| Level         | Count       | Time/Problem | Focus                             |
| ------------- | ----------- | ------------ | --------------------------------- |
| 🌱 **Easy**   | 35 problems | 15-30 min    | BFS/DFS, basic graph operations   |
| ⚡ **Medium** | 40 problems | 30-60 min    | Shortest paths, topological sort  |
| 🔥 **Hard**   | 25 problems | 60-90 min    | Advanced algorithms, optimization |

---

## 🌱 EASY LEVEL (1-35) - Graph Fundamentals

### **Problem 1: Build Graph from Edge List**

**Difficulty:** ⭐ Easy | **Time:** 15 minutes

```python
"""
Build an adjacency list from a list of edges.

Input: edges = [[0,1],[1,2],[2,0]], directed = False
Output: {0: [1,2], 1: [0,2], 2: [1,0]}
"""

def build_graph(edges, directed=False):
    graph = {}

    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []

        graph[u].append(v)
        if not directed:
            graph[v].append(u)

    return graph

# Test cases
edges1 = [[0,1],[1,2],[2,0]]
assert build_graph(edges1) == {0: [1,2], 1: [0,2], 2: [1,0]}

edges2 = [[0,1],[1,2]]
assert build_graph(edges2, directed=True) == {0: [1], 1: [2], 2: []}
```

### **Problem 2: Graph Traversal - BFS**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Perform BFS traversal starting from a given node.

Input: graph = {0: [1,2], 1: [0,3,4], 2: [0], 3: [1], 4: [1]}, start = 0
Output: [0, 1, 2, 3, 4]
"""

from collections import deque

def bfs_traversal(graph, start):
    if start not in graph:
        return []

    visited = set()
    queue = deque([start])
    result = []

    visited.add(start)

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result

# Test case
graph = {0: [1,2], 1: [0,3,4], 2: [0], 3: [1], 4: [1]}
result = bfs_traversal(graph, 0)
assert set(result) == {0,1,2,3,4}
assert result[0] == 0  # Start node comes first
```

### **Problem 3: Graph Traversal - DFS**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Perform DFS traversal starting from a given node.

Input: graph = {0: [1,2], 1: [0,3,4], 2: [0], 3: [1], 4: [1]}, start = 0
Output: [0, 1, 3, 4, 2] (order may vary)
"""

def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()

    if start not in graph:
        return []

    visited.add(start)
    result = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result

def dfs_iterative(graph, start):
    if start not in graph:
        return []

    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)

            # Add neighbors in reverse order for consistent results
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result

# Test cases
graph = {0: [1,2], 1: [0,3,4], 2: [0], 3: [1], 4: [1]}
result_rec = dfs_recursive(graph, 0)
result_iter = dfs_iterative(graph, 0)

assert set(result_rec) == {0,1,2,3,4}
assert set(result_iter) == {0,1,2,3,4}
```

### **Problem 4: Check if Path Exists**

**Difficulty:** ⭐ Easy | **Time:** 20 minutes

```python
"""
Check if there's a path between two nodes.

Input: graph = {0: [1,2], 1: [3], 2: [], 3: []}, source = 0, target = 3
Output: True
"""

def has_path_bfs(graph, source, target):
    if source == target:
        return True

    if source not in graph:
        return False

    visited = set()
    queue = deque([source])
    visited.add(source)

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == target:
                return True

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False

def has_path_dfs(graph, source, target, visited=None):
    if visited is None:
        visited = set()

    if source == target:
        return True

    if source in visited or source not in graph:
        return False

    visited.add(source)

    for neighbor in graph[source]:
        if has_path_dfs(graph, neighbor, target, visited):
            return True

    return False

# Test cases
graph = {0: [1,2], 1: [3], 2: [], 3: []}
assert has_path_bfs(graph, 0, 3) == True
assert has_path_bfs(graph, 0, 4) == False
assert has_path_dfs(graph, 0, 3) == True
assert has_path_dfs(graph, 2, 3) == False
```

### **Problem 5: Count Connected Components**

**Difficulty:** ⭐ Easy | **Time:** 25 minutes

```python
"""
Count the number of connected components in an undirected graph.

Input: n = 5, edges = [[0,1],[1,2],[3,4]]
Output: 2 (components: {0,1,2} and {3,4})
"""

def count_components(n, edges):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    components = 0

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for i in range(n):
        if i not in visited:
            dfs(i)
            components += 1

    return components

def count_components_union_find(n, edges):
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for u, v in edges:
        union(u, v)

    return len(set(find(i) for i in range(n)))

# Test cases
assert count_components(5, [[0,1],[1,2],[3,4]]) == 2
assert count_components(5, [[0,1],[1,2],[2,3],[3,4]]) == 1
assert count_components_union_find(5, [[0,1],[1,2],[3,4]]) == 2
```

### **Problem 6: Shortest Path in Unweighted Graph**

**Difficulty:** ⭐ Easy | **Time:** 25 minutes

```python
"""
Find shortest path between two nodes in unweighted graph.

Input: edges = [[0,1],[1,2],[2,3]], start = 0, end = 3
Output: [0,1,2,3] (path), 3 (distance)
"""

def shortest_path_unweighted(edges, start, end):
    # Build graph
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    if start == end:
        return [start], 0

    if start not in graph or end not in graph:
        return None, float('inf')

    # BFS to find shortest path
    queue = deque([(start, [start], 0)])
    visited = {start}

    while queue:
        node, path, dist = queue.popleft()

        for neighbor in graph[node]:
            if neighbor == end:
                return path + [neighbor], dist + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], dist + 1))

    return None, float('inf')

# Test cases
edges = [[0,1],[1,2],[2,3]]
path, dist = shortest_path_unweighted(edges, 0, 3)
assert path == [0,1,2,3]
assert dist == 3

path, dist = shortest_path_unweighted(edges, 0, 4)
assert path is None
assert dist == float('inf')
```

### **Problem 7: Detect Cycle in Undirected Graph**

**Difficulty:** ⭐ Easy | **Time:** 30 minutes

```python
"""
Detect if there's a cycle in an undirected graph.

Input: n = 5, edges = [[0,1],[1,2],[2,3],[3,4],[4,1]]
Output: True (cycle: 1-2-3-4-1)
"""

def has_cycle_undirected_dfs(n, edges):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node, parent):
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:  # Back edge found
                return True

        return False

    for i in range(n):
        if i not in visited:
            if dfs(i, -1):
                return True

    return False

def has_cycle_undirected_union_find(n, edges):
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return True  # Cycle found
        parent[px] = py
        return False

    for u, v in edges:
        if union(u, v):
            return True

    return False

# Test cases
assert has_cycle_undirected_dfs(5, [[0,1],[1,2],[2,3],[3,4],[4,1]]) == True
assert has_cycle_undirected_dfs(4, [[0,1],[1,2],[2,3]]) == False
assert has_cycle_undirected_union_find(5, [[0,1],[1,2],[2,3],[3,4],[4,1]]) == True
```

### **Problem 8: Number of Islands**

**Difficulty:** ⭐ Easy | **Time:** 30 minutes

```python
"""
Count number of islands in a 2D grid.
'1' represents land, '0' represents water.

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
"""

def num_islands_dfs(grid):
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    islands = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or grid[r][c] == '0'):
            return

        visited.add((r, c))

        # Visit all 4 directions
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                islands += 1

    return islands

def num_islands_bfs(grid):
    if not grid or not grid[0]:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    islands = 0

    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))

        while queue:
            r, c = queue.popleft()

            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + dr, c + dc

                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in visited and grid[nr][nc] == '1'):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                bfs(r, c)
                islands += 1

    return islands

# Test cases
grid1 = [
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
]
assert num_islands_dfs(grid1) == 1
assert num_islands_bfs(grid1) == 1

grid2 = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]
assert num_islands_dfs(grid2) == 3
```

### **Problem 9: Clone Graph**

**Difficulty:** ⭐ Easy-Medium | **Time:** 35 minutes

```python
"""
Clone an undirected graph. Each node has a value and list of neighbors.

Node definition:
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph_dfs(node):
    if not node:
        return None

    clones = {}

    def dfs(original):
        if original in clones:
            return clones[original]

        clone = Node(original.val)
        clones[original] = clone

        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))

        return clone

    return dfs(node)

def clone_graph_bfs(node):
    if not node:
        return None

    clones = {node: Node(node.val)}
    queue = deque([node])

    while queue:
        original = queue.popleft()

        for neighbor in original.neighbors:
            if neighbor not in clones:
                clones[neighbor] = Node(neighbor.val)
                queue.append(neighbor)

            clones[original].neighbors.append(clones[neighbor])

    return clones[node]

# Test case creation
def create_test_graph():
    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node4 = Node(4)

    node1.neighbors = [node2, node4]
    node2.neighbors = [node1, node3]
    node3.neighbors = [node2, node4]
    node4.neighbors = [node1, node3]

    return node1

# Test
original = create_test_graph()
cloned_dfs = clone_graph_dfs(original)
cloned_bfs = clone_graph_bfs(original)

assert cloned_dfs.val == 1
assert len(cloned_dfs.neighbors) == 2
assert cloned_dfs is not original  # Different object
```

### **Problem 10: Valid Tree**

**Difficulty:** ⭐ Easy-Medium | **Time:** 30 minutes

```python
"""
Check if n nodes and given edges form a valid tree.
A valid tree has exactly n-1 edges and is connected.

Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: True
"""

def valid_tree(n, edges):
    # Tree must have exactly n-1 edges
    if len(edges) != n - 1:
        return False

    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Check if all nodes are connected using BFS
    visited = set()
    queue = deque([0])
    visited.add(0)

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n

def valid_tree_union_find(n, edges):
    if len(edges) != n - 1:
        return False

    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False  # Cycle found
        parent[px] = py
        return True

    for u, v in edges:
        if not union(u, v):
            return False

    return True

# Test cases
assert valid_tree(5, [[0,1],[0,2],[0,3],[1,4]]) == True
assert valid_tree(5, [[0,1],[1,2],[2,3],[1,3],[1,4]]) == False  # Has cycle
assert valid_tree_union_find(5, [[0,1],[0,2],[0,3],[1,4]]) == True
```

---

## ⚡ MEDIUM LEVEL (36-75) - Intermediate Graph Algorithms

### **Problem 36: Course Schedule**

**Difficulty:** ⚡ Medium | **Time:** 35 minutes

```python
"""
Determine if you can finish all courses given prerequisites.
This is cycle detection in directed graph.

Input: numCourses = 2, prerequisites = [[1,0]]
Output: True (take course 0 first, then course 1)

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: False (circular dependency)
"""

def can_finish(num_courses, prerequisites):
    # Build adjacency list and in-degree array
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Use Kahn's algorithm (topological sort)
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses

def can_finish_dfs(num_courses, prerequisites):
    # Build adjacency list
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # DFS with coloring: WHITE=0, GRAY=1, BLACK=2
    colors = [0] * num_courses

    def has_cycle(course):
        if colors[course] == 1:  # GRAY - back edge, cycle found
            return True
        if colors[course] == 2:  # BLACK - already processed
            return False

        colors[course] = 1  # Mark as processing

        for next_course in graph[course]:
            if has_cycle(next_course):
                return True

        colors[course] = 2  # Mark as processed
        return False

    for i in range(num_courses):
        if colors[i] == 0:
            if has_cycle(i):
                return False

    return True

# Test cases
assert can_finish(2, [[1,0]]) == True
assert can_finish(2, [[1,0],[0,1]]) == False
assert can_finish_dfs(2, [[1,0]]) == True
assert can_finish_dfs(2, [[1,0],[0,1]]) == False
```

### **Problem 37: Course Schedule II**

**Difficulty:** ⚡ Medium | **Time:** 40 minutes

```python
"""
Return the order in which courses should be taken.

Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3] (any valid topological order)
"""

def find_order(num_courses, prerequisites):
    # Build graph and in-degree
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return order if len(order) == num_courses else []

def find_order_dfs(num_courses, prerequisites):
    # Build adjacency list
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    WHITE, GRAY, BLACK = 0, 1, 2
    colors = [WHITE] * num_courses
    order = []

    def dfs(course):
        if colors[course] == GRAY:
            return False  # Cycle detected
        if colors[course] == BLACK:
            return True

        colors[course] = GRAY

        for next_course in graph[course]:
            if not dfs(next_course):
                return False

        colors[course] = BLACK
        order.append(course)
        return True

    for i in range(num_courses):
        if colors[i] == WHITE:
            if not dfs(i):
                return []

    return order[::-1]  # Reverse for correct topological order

# Test cases
result1 = find_order(4, [[1,0],[2,0],[3,1],[3,2]])
assert len(result1) == 4 and result1[0] == 0

result2 = find_order(2, [[1,0],[0,1]])
assert result2 == []

result3 = find_order_dfs(4, [[1,0],[2,0],[3,1],[3,2]])
assert len(result3) == 4
```

### **Problem 38: Network Delay Time (Dijkstra's Algorithm)**

**Difficulty:** ⚡ Medium | **Time:** 45 minutes

```python
"""
Find minimum time for signal to reach all nodes from source.

Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2 (time for signal to reach all nodes from node 2)
"""

import heapq

def network_delay_time(times, n, k):
    # Build adjacency list with weights
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra's algorithm
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0

    heap = [(0, k)]
    visited = set()

    while heap:
        curr_dist, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)

        for v, weight in graph[u]:
            if v not in visited:
                new_dist = curr_dist + weight
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))

    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1

# Optimized version without visited set
def network_delay_time_optimized(times, n, k):
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((v, w))

    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0

    heap = [(0, k)]

    while heap:
        curr_dist, u = heapq.heappop(heap)

        if curr_dist > distances[u]:
            continue

        for v, weight in graph[u]:
            new_dist = curr_dist + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1

# Test cases
times1 = [[2,1,1],[2,3,1],[3,4,1]]
assert network_delay_time(times1, 4, 2) == 2

times2 = [[1,2,1]]
assert network_delay_time(times2, 2, 1) == 1
assert network_delay_time(times2, 2, 2) == -1
```

### **Problem 39: Cheapest Flights with K Stops**

**Difficulty:** ⚡ Medium | **Time:** 50 minutes

```python
"""
Find cheapest flight from src to dst with at most k stops.

Input: flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200 (0->1->2 costs 200, 0->2 costs 500)
"""

def find_cheapest_price_bellman_ford(n, flights, src, dst, k):
    # Initialize distances
    distances = [float('inf')] * n
    distances[src] = 0

    # Relax edges at most k+1 times (k stops means k+1 flights)
    for i in range(k + 1):
        temp_distances = distances[:]

        for u, v, price in flights:
            if distances[u] != float('inf'):
                temp_distances[v] = min(temp_distances[v], distances[u] + price)

        distances = temp_distances

    return distances[dst] if distances[dst] != float('inf') else -1

def find_cheapest_price_dijkstra(n, flights, src, dst, k):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v, price in flights:
        graph[u].append((v, price))

    # Modified Dijkstra: (cost, node, stops_used)
    heap = [(0, src, 0)]
    visited = {}  # (node, stops) -> min_cost

    while heap:
        cost, node, stops = heapq.heappop(heap)

        if node == dst:
            return cost

        if stops > k:
            continue

        if (node, stops) in visited and visited[(node, stops)] <= cost:
            continue

        visited[(node, stops)] = cost

        for neighbor, price in graph[node]:
            new_cost = cost + price
            if stops < k + 1:  # Can still make more stops
                heapq.heappush(heap, (new_cost, neighbor, stops + 1))

    return -1

def find_cheapest_price_bfs(n, flights, src, dst, k):
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v, price in flights:
        graph[u].append((v, price))

    # BFS with level tracking
    queue = deque([(src, 0)])  # (node, cost)
    min_cost = float('inf')

    for stops in range(k + 2):  # 0 to k+1 stops
        if not queue:
            break

        for _ in range(len(queue)):
            node, cost = queue.popleft()

            if node == dst:
                min_cost = min(min_cost, cost)
                continue

            if stops <= k:  # Can make more flights
                for neighbor, price in graph[node]:
                    if cost + price < min_cost:  # Pruning
                        queue.append((neighbor, cost + price))

    return min_cost if min_cost != float('inf') else -1

# Test cases
flights = [[0,1,100],[1,2,100],[0,2,500]]
assert find_cheapest_price_bellman_ford(3, flights, 0, 2, 1) == 200
assert find_cheapest_price_dijkstra(3, flights, 0, 2, 1) == 200
assert find_cheapest_price_bfs(3, flights, 0, 2, 1) == 200

flights2 = [[0,1,100],[1,2,100],[0,2,500]]
assert find_cheapest_price_bellman_ford(3, flights2, 0, 2, 0) == 500
```

### **Problem 40: Word Ladder**

**Difficulty:** ⚡ Medium | **Time:** 45 minutes

```python
"""
Transform beginWord to endWord changing one letter at a time.
Each intermediate word must be in wordList.

Input: beginWord = "hit", endWord = "cog",
       wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5 ("hit" -> "hot" -> "dot" -> "dog" -> "cog")
"""

def ladder_length(begin_word, end_word, word_list):
    if end_word not in word_list:
        return 0

    word_set = set(word_list)
    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, length = queue.popleft()

        if word == end_word:
            return length

        # Try changing each character
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]

                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        queue.append((new_word, length + 1))

    return 0

def ladder_length_bidirectional(begin_word, end_word, word_list):
    if end_word not in word_list:
        return 0

    word_set = set(word_list)

    # Bidirectional BFS
    begin_set = {begin_word}
    end_set = {end_word}
    visited = set()
    length = 1

    while begin_set and end_set:
        # Always expand the smaller set
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set

        next_begin_set = set()

        for word in begin_set:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]

                        if new_word in end_set:
                            return length + 1

                        if new_word in word_set and new_word not in visited:
                            visited.add(new_word)
                            next_begin_set.add(new_word)

        begin_set = next_begin_set
        length += 1

    return 0

# Test cases
word_list = ["hot","dot","dog","lot","log","cog"]
assert ladder_length("hit", "cog", word_list) == 5
assert ladder_length_bidirectional("hit", "cog", word_list) == 5

assert ladder_length("hit", "cog", ["hot","dot","dog","lot","log"]) == 0
```

---

## 🔥 HARD LEVEL (76-100) - Advanced Graph Mastery

### **Problem 76: Shortest Path to Get All Keys**

**Difficulty:** 🔥 Hard | **Time:** 70 minutes

```python
"""
Find shortest path to collect all keys in a grid maze.
'@' = start, lowercase = key, uppercase = lock, '#' = wall

Input: grid = ["@.a.#","###.#","b.A.B"]
Output: 8 (collect key 'a', then 'b', then pass through lock 'A' and 'B')
"""

def shortest_path_all_keys(grid):
    if not grid or not grid[0]:
        return -1

    rows, cols = len(grid), len(grid[0])
    start = None
    key_count = 0

    # Find start position and count keys
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '@':
                start = (r, c)
            elif grid[r][c].islower():
                key_count += 1

    if key_count == 0:
        return 0

    # BFS with state: (row, col, keys_bitmask)
    queue = deque([(start[0], start[1], 0, 0)])  # r, c, keys, steps
    visited = set([(start[0], start[1], 0)])
    target_keys = (1 << key_count) - 1  # All keys collected

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        r, c, keys, steps = queue.popleft()

        if keys == target_keys:
            return steps

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
                cell = grid[nr][nc]
                new_keys = keys

                # Check if we can pass
                if cell.isupper():  # Lock
                    key_needed = cell.lower()
                    key_bit = ord(key_needed) - ord('a')
                    if not (keys & (1 << key_bit)):  # Don't have key
                        continue
                elif cell.islower():  # Key
                    key_bit = ord(cell) - ord('a')
                    new_keys |= (1 << key_bit)

                state = (nr, nc, new_keys)
                if state not in visited:
                    visited.add(state)
                    queue.append((nr, nc, new_keys, steps + 1))

    return -1

# Test case
grid1 = ["@.a.#","###.#","b.A.B"]
assert shortest_path_all_keys(grid1) == 8

grid2 = ["@..aA","..B#.","....b"]
assert shortest_path_all_keys(grid2) == 6
```

### **Problem 77: Alien Dictionary**

**Difficulty:** 🔥 Hard | **Time:** 60 minutes

```python
"""
Given sorted dictionary of alien language, find the order of characters.

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
"""

def alien_order(words):
    # Step 1: Initialize graph and in_degree
    graph = {}
    in_degree = {}

    # Initialize all characters
    for word in words:
        for char in word:
            if char not in graph:
                graph[char] = set()
                in_degree[char] = 0

    # Step 2: Build graph by comparing adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))

        # Check for invalid case: word1 is prefix of word2 but longer
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""

        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break

    # Step 3: Topological sort using Kahn's algorithm
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if valid (no cycle)
    if len(result) != len(in_degree):
        return ""

    return ''.join(result)

# Test cases
words1 = ["wrt","wrf","er","ett","rftt"]
result1 = alien_order(words1)
assert result1 == "wertf"

words2 = ["z","x"]
assert alien_order(words2) == "zx"

words3 = ["z","x","z"]
assert alien_order(words3) == ""  # Invalid
```

### **Problem 78: Critical Connections (Bridges)**

**Difficulty:** 🔥 Hard | **Time:** 75 minutes

```python
"""
Find all critical connections (bridges) in a network.
A bridge is an edge that, if removed, increases the number of components.

Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
Output: [[1,3]] (removing edge 1-3 disconnects node 3)
"""

def critical_connections(n, connections):
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    discovery = [0] * n  # Discovery time
    low = [0] * n       # Lowest reachable node
    parent = [-1] * n
    bridges = []
    time = [0]  # Use list to make it mutable in nested function

    def bridge_dfs(u):
        visited[u] = True
        discovery[u] = low[u] = time[0]
        time[0] += 1

        for v in graph[u]:
            if not visited[v]:
                parent[v] = u
                bridge_dfs(v)

                # Update low value
                low[u] = min(low[u], low[v])

                # Check if edge u-v is a bridge
                if low[v] > discovery[u]:
                    bridges.append([u, v])

            elif v != parent[u]:  # Back edge
                low[u] = min(low[u], discovery[v])

    # Run DFS from all unvisited nodes
    for i in range(n):
        if not visited[i]:
            bridge_dfs(i)

    return bridges

def critical_connections_tarjan(n, connections):
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    discovery = {}
    low = {}
    parent = {}
    bridges = []
    time = 0

    def tarjan(u):
        nonlocal time
        visited.add(u)
        discovery[u] = low[u] = time
        time += 1

        for v in graph[u]:
            if v not in visited:
                parent[v] = u
                tarjan(v)

                low[u] = min(low[u], low[v])

                if low[v] > discovery[u]:
                    bridges.append([min(u, v), max(u, v)])
            elif v != parent.get(u, -1):
                low[u] = min(low[u], discovery[v])

    for i in range(n):
        if i not in visited:
            tarjan(i)

    return sorted(bridges)

# Test cases
connections1 = [[0,1],[1,2],[2,0],[1,3]]
result1 = critical_connections(4, connections1)
assert [1, 3] in result1 or [3, 1] in result1

connections2 = [[0,1]]
assert critical_connections(2, connections2) == [[0,1]]
```

### **Problem 79: Minimum Cost to Make Graph Connected**

**Difficulty:** 🔥 Hard | **Time:** 60 minutes

```python
"""
Find minimum cost to make all nodes connected.
You can remove existing edges and add new ones.

Input: n = 4, connections = [[0,1],[0,2],[1,2]]
Output: 1 (need 1 more edge to connect node 3)
"""

def make_connected(n, connections):
    # Need at least n-1 edges to connect n nodes
    if len(connections) < n - 1:
        return -1

    # Find number of connected components using Union-Find
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    redundant_edges = 0

    for u, v in connections:
        if not union(u, v):
            redundant_edges += 1

    # Count connected components
    components = len(set(find(i) for i in range(n)))

    # Need (components - 1) edges to connect all components
    edges_needed = components - 1

    return edges_needed if redundant_edges >= edges_needed else -1

def make_connected_dfs(n, connections):
    if len(connections) < n - 1:
        return -1

    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    components = 0

    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)

    for i in range(n):
        if not visited[i]:
            dfs(i)
            components += 1

    return components - 1

# Test cases
assert make_connected(4, [[0,1],[0,2],[1,2]]) == 1
assert make_connected(6, [[0,1],[0,2],[0,3],[1,2],[1,3]]) == 2
assert make_connected(6, [[0,1],[0,2],[0,3],[1,2]]) == -1
assert make_connected_dfs(4, [[0,1],[0,2],[1,2]]) == 1
```

### **Problem 80: Swim in Rising Water**

**Difficulty:** 🔥 Hard | **Time:** 70 minutes

```python
"""
Find minimum time to swim from top-left to bottom-right.
At time t, you can only swim in cells with elevation ≤ t.

Input: grid = [[0,2],[1,3]]
Output: 3 (wait until time 3 to swim through cell (1,1))
"""

def swim_in_water_binary_search(grid):
    n = len(grid)

    def can_swim(time_limit):
        if grid[0][0] > time_limit:
            return False

        visited = set()
        queue = deque([(0, 0)])
        visited.add((0, 0))

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            r, c = queue.popleft()

            if r == n - 1 and c == n - 1:
                return True

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if (0 <= nr < n and 0 <= nc < n and
                    (nr, nc) not in visited and
                    grid[nr][nc] <= time_limit):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return False

    # Binary search on time
    left, right = 0, max(max(row) for row in grid)

    while left < right:
        mid = (left + right) // 2
        if can_swim(mid):
            right = mid
        else:
            left = mid + 1

    return left

def swim_in_water_dijkstra(grid):
    n = len(grid)

    # Modified Dijkstra: minimize maximum elevation on path
    heap = [(grid[0][0], 0, 0)]  # (max_elevation, row, col)
    visited = set()

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while heap:
        max_elevation, r, c = heapq.heappop(heap)

        if (r, c) in visited:
            continue

        visited.add((r, c))

        if r == n - 1 and c == n - 1:
            return max_elevation

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if (0 <= nr < n and 0 <= nc < n and
                (nr, nc) not in visited):
                new_elevation = max(max_elevation, grid[nr][nc])
                heapq.heappush(heap, (new_elevation, nr, nc))

    return -1

def swim_in_water_union_find(grid):
    n = len(grid)

    # Create list of cells sorted by elevation
    cells = []
    for r in range(n):
        for c in range(n):
            cells.append((grid[r][c], r, c))

    cells.sort()

    parent = {}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for elevation, r, c in cells:
        parent[(r, c)] = (r, c)

        # Connect with neighbors that have been processed
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < n and 0 <= nc < n and
                (nr, nc) in parent):
                union((r, c), (nr, nc))

        # Check if start and end are connected
        if ((0, 0) in parent and (n-1, n-1) in parent and
            find((0, 0)) == find((n-1, n-1))):
            return elevation

    return -1

# Test cases
grid1 = [[0,2],[1,3]]
assert swim_in_water_binary_search(grid1) == 3
assert swim_in_water_dijkstra(grid1) == 3
assert swim_in_water_union_find(grid1) == 3

grid2 = [[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]]
assert swim_in_water_binary_search(grid2) == 16
```

---

## 📚 Graph Problem Patterns Summary

### **Pattern Categories:**

1. **Traversal Problems**
   - Basic BFS/DFS
   - Connected components
   - Path finding

2. **Shortest Path Problems**
   - Unweighted: BFS
   - Weighted: Dijkstra, Bellman-Ford
   - All-pairs: Floyd-Warshall

3. **Topological Sort Problems**
   - Course scheduling
   - Dependency resolution
   - Alien dictionary

4. **Cycle Detection**
   - Undirected: DFS with parent
   - Directed: DFS with colors

5. **Minimum Spanning Tree**
   - Kruskal's algorithm
   - Prim's algorithm

6. **Advanced Algorithms**
   - Bridges and articulation points
   - Strongly connected components
   - Network flow

### **Time Complexity Summary:**

| Algorithm      | Time         | Space | Best For                    |
| -------------- | ------------ | ----- | --------------------------- |
| BFS/DFS        | O(V+E)       | O(V)  | Traversal, connectivity     |
| Dijkstra       | O((V+E)logV) | O(V)  | Single-source shortest path |
| Bellman-Ford   | O(VE)        | O(V)  | Negative weights            |
| Floyd-Warshall | O(V³)        | O(V²) | All-pairs shortest path     |
| Tarjan's       | O(V+E)       | O(V)  | Bridges, SCC                |

---

_Master these 100+ graph problems and you'll be ready for any network challenge! 🚀_
