---
title: "Graphs Quick Reference Cheatsheet"
level: "All Levels"
time: "5 min quick reference"
tags:
  [
    "dsa",
    "graphs",
    "bfs",
    "dfs",
    "shortest-path",
    "cheatsheet",
    "quick-reference",
  ]
---

# 🌐 Graphs Cheatsheet

_Quick Reference for Interview & Practice_

---

## 📊 Graph Algorithm Complexity

| Algorithm            | Time         | Space | Use Case                                |
| -------------------- | ------------ | ----- | --------------------------------------- |
| **BFS**              | O(V+E)       | O(V)  | Shortest path (unweighted), level-order |
| **DFS**              | O(V+E)       | O(V)  | Cycle detection, topological sort       |
| **Dijkstra**         | O((V+E)logV) | O(V)  | Shortest path (non-negative weights)    |
| **Bellman-Ford**     | O(VE)        | O(V)  | Shortest path (negative weights)        |
| **Floyd-Warshall**   | O(V³)        | O(V²) | All-pairs shortest paths                |
| **Kruskal MST**      | O(ElogE)     | O(V)  | Minimum spanning tree                   |
| **Prim MST**         | O(ElogV)     | O(V)  | Minimum spanning tree                   |
| **Topological Sort** | O(V+E)       | O(V)  | Dependency ordering                     |

---

## 🎯 When to Use Which Algorithm

| Problem Type                               | Algorithm      | Key Indicator                    |
| ------------------------------------------ | -------------- | -------------------------------- |
| **Shortest path (unweighted)**             | BFS            | Equal edge weights               |
| **Shortest path (weighted, non-negative)** | Dijkstra       | Non-negative weights             |
| **Shortest path (negative weights)**       | Bellman-Ford   | Negative edges exist             |
| **All-pairs shortest paths**               | Floyd-Warshall | Need distances between all pairs |
| **Cycle detection (undirected)**           | DFS            | Check connectivity               |
| **Cycle detection (directed)**             | DFS (3-color)  | Topological sort fails           |
| **Connected components**                   | DFS/BFS        | Group related nodes              |
| **Topological ordering**                   | Kahn's/DFS     | Dependencies, course scheduling  |
| **Minimum spanning tree**                  | Kruskal/Prim   | Connect all nodes minimally      |

---

## 🔧 Essential Templates

### **Template 1: BFS Traversal**

```python
from collections import deque

def bfs(graph, start):
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
```

### **Template 2: DFS Traversal**

```python
# Recursive DFS
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()

    visited.add(start)
    result = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))

    return result

# Iterative DFS
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)

            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result
```

### **Template 3: Shortest Path (BFS)**

```python
def shortest_path_bfs(graph, start, end):
    if start == end:
        return [start], 0

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
```

### **Template 4: Dijkstra's Algorithm**

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    pq = [(0, start)]
    visited = set()

    while pq:
        curr_dist, curr_node = heapq.heappop(pq)

        if curr_node in visited:
            continue

        visited.add(curr_node)

        for neighbor, weight in graph[curr_node]:
            if neighbor not in visited:
                new_dist = curr_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    return distances
```

### **Template 5: Cycle Detection**

```python
# Undirected Graph
def has_cycle_undirected(graph):
    visited = set()

    def dfs(node, parent):
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True

        return False

    for node in graph:
        if node not in visited:
            if dfs(node, None):
                return True

    return False

# Directed Graph (3-color DFS)
def has_cycle_directed(graph):
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {node: WHITE for node in graph}

    def dfs(node):
        if colors[node] == GRAY:
            return True
        if colors[node] == BLACK:
            return False

        colors[node] = GRAY

        for neighbor in graph[node]:
            if dfs(neighbor):
                return True

        colors[node] = BLACK
        return False

    for node in graph:
        if colors[node] == WHITE:
            if dfs(node):
                return True

    return False
```

### **Template 6: Topological Sort**

```python
# Kahn's Algorithm (BFS)
def topological_sort_bfs(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(graph) else []

# DFS-based
def topological_sort_dfs(graph):
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return stack[::-1]
```

### **Template 7: Union-Find**

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

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

        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

## 🎨 Common Graph Patterns

### **Pattern 1: Island Problems**

```python
# Count islands in 2D grid
def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    islands = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or grid[r][c] == '0'):
            return

        visited.add((r, c))

        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                islands += 1

    return islands
```

### **Pattern 2: Bipartite Check**

```python
def is_bipartite(graph):
    colors = {}

    def bfs(start):
        queue = deque([start])
        colors[start] = 0

        while queue:
            node = queue.popleft()

            for neighbor in graph[node]:
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[node]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[node]:
                    return False

        return True

    for node in graph:
        if node not in colors:
            if not bfs(node):
                return False

    return True
```

### **Pattern 3: Course Schedule**

```python
def can_finish(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

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
```

---

## 🚀 Graph Representations

### **Adjacency List (Most Common)**

```python
# Unweighted
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A'],
    'D': ['B']
}

# Weighted
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('D', 3)],
    'C': [('A', 2), ('D', 1)],
    'D': [('B', 3), ('C', 1)]
}
```

### **Adjacency Matrix**

```python
# For n vertices
matrix = [[0] * n for _ in range(n)]

# Add edge (i, j) with weight w
matrix[i][j] = w
matrix[j][i] = w  # For undirected graph
```

### **Edge List**

```python
# List of tuples: (u, v, weight)
edges = [(0, 1, 4), (1, 2, 3), (0, 2, 7)]
```

---

## 🔥 Quick Problem Recognition

### **BFS Problems:**

- Shortest path in unweighted graph
- Level-order traversal
- Minimum steps/moves
- "Shortest" in grid problems

### **DFS Problems:**

- Connected components
- Cycle detection
- Path existence
- Topological sort
- Backtracking

### **Dijkstra Problems:**

- Shortest path with weights
- "Minimum cost/time"
- Network delay time

### **Union-Find Problems:**

- Connected components
- Cycle detection in undirected graphs
- Dynamic connectivity

---

## 🎯 Interview Tips

### **Graph Construction:**

```python
# From edge list
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
```

### **Grid as Graph:**

```python
def get_neighbors(r, c, rows, cols):
    neighbors = []
    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors
```

### **Common Mistakes:**

- ❌ Forgetting to mark nodes as visited
- ❌ Not handling disconnected components
- ❌ Wrong base case in recursion
- ❌ Infinite loops in cyclic graphs
- ❌ Not considering edge cases (empty graph, single node)

### **Optimization Tips:**

- Use sets for O(1) visited checks
- Early termination when target found
- Bidirectional BFS for shortest path
- Choose right data structure for graph representation

---

## 📊 Space-Time Tradeoffs

| Representation | Space  | Add Edge | Remove Edge | Check Edge | Get Neighbors |
| -------------- | ------ | -------- | ----------- | ---------- | ------------- |
| **Adj List**   | O(V+E) | O(1)     | O(degree)   | O(degree)  | O(degree)     |
| **Adj Matrix** | O(V²)  | O(1)     | O(1)        | O(1)       | O(V)          |
| **Edge List**  | O(E)   | O(1)     | O(E)        | O(E)       | O(E)          |

**Choose based on:**

- **Sparse graphs** → Adjacency List
- **Dense graphs** → Adjacency Matrix
- **Simple storage** → Edge List

---

_Master these patterns and templates for graph interview success! 🚀_
