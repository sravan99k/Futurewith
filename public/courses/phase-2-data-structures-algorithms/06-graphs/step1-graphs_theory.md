# 🌐 Graphs: Master Network Algorithms

_Navigate the connected world with BFS, DFS, and Shortest Path algorithms_

---

# Comprehensive Learning System

title: "Graphs Complete Guide - BFS, DFS & Shortest Paths"
level: "Intermediate to Advanced"
time_to_complete: "20-25 hours"
prerequisites: ["Basic data structures", "Recursion fundamentals", "Queue and stack operations", "Time complexity analysis"]
skills_gained: ["Graph representation and implementation", "Breadth-first and depth-first search", "Shortest path algorithms", "Graph connectivity analysis", "Network flow and matching", "Advanced graph algorithms"]
success_criteria: ["Implement graphs using adjacency lists and matrices", "Master BFS and DFS traversal algorithms", "Apply shortest path algorithms (Dijkstra, Floyd-Warshall)", "Solve graph connectivity and cycle detection problems", "Implement advanced algorithms (MST, topological sort)", "Apply graph concepts to real-world network problems"]
tags: ["data structures", "graphs", "algorithms", "bfs", "dfs", "shortest path", "networks", "interview prep"]
description: "Master graph data structures and algorithms from basic traversals to advanced network problems. Learn BFS, DFS, shortest path algorithms, and their applications to real-world network scenarios."

---

---

## 🎬 Story Hook: The Social Network

**Imagine Facebook's friend network:**

- **BFS:** Find friends within 2-3 degrees of separation
- **DFS:** Discover connected components (friend groups)
- **Shortest Path:** Find shortest connection between any two people

**Real-world uses:**

- 🗺️ **GPS Navigation** - Finding shortest routes
- 🌐 **Internet Routing** - Packet delivery paths
- 🎯 **Social Networks** - Friend recommendations
- 🎮 **Game AI** - Pathfinding for characters
- 📊 **Dependency Analysis** - Build systems, prerequisites

---

## Learning Goals

By the end of this module, you will be able to:

1. **Understand Graph Fundamentals** - Grasp graph theory concepts, terminology, and different graph types
2. **Represent Graphs Efficiently** - Implement graphs using adjacency lists, matrices, and edge lists
3. **Master Graph Traversals** - Implement BFS and DFS with proper visited tracking and applications
4. **Solve Path-Finding Problems** - Apply Dijkstra's, Bellman-Ford, and Floyd-Warshall algorithms
5. **Analyze Graph Connectivity** - Find connected components, bridges, and articulation points
6. **Implement Advanced Algorithms** - Use minimum spanning trees, topological sorting, and network flow
7. **Apply to Real-World Problems** - Solve network routing, social networks, and dependency management
8. **Optimize Graph Performance** - Understand time/space complexity and choose appropriate representations

---

## TL;DR

Graphs model relationships and networks in the real world. **Start with graph representation**, **learn BFS/DFS traversals**, and **master shortest path algorithms**. Focus on understanding when to use BFS (shortest path in unweighted) vs DFS (connectivity, cycles), and apply these concepts to real-world network problems.

---

## 🎬 Story Hook: The Social Network

## 📋 Table of Contents

1. [Graph Fundamentals](#graph-fundamentals)
2. [Graph Representations](#graph-representations)
3. [Breadth-First Search (BFS)](#breadth-first-search)
4. [Depth-First Search (DFS)](#depth-first-search)
5. [Shortest Path Algorithms](#shortest-path-algorithms)
6. [Advanced Graph Algorithms](#advanced-algorithms)
7. [Graph Problems & Patterns](#graph-patterns)
8. [Interview Essentials](#interview-essentials)

---

## 🎯 Graph Fundamentals

### **What are Graphs?**

```python
"""
Graph: Collection of vertices (nodes) connected by edges

Types:
1. DIRECTED vs UNDIRECTED
2. WEIGHTED vs UNWEIGHTED
3. CYCLIC vs ACYCLIC
4. CONNECTED vs DISCONNECTED
"""

class GraphBasics:
    def __init__(self):
        """
        UNDIRECTED GRAPH:
        A --- B
        |     |
        C --- D

        DIRECTED GRAPH (Digraph):
        A --> B
        ^     |
        |     v
        C <-- D

        WEIGHTED GRAPH:
        A --5-- B
        |       |
        3       2
        |       |
        C --7-- D
        """
        pass

# Real-world examples
examples = {
    "Social Network": "Undirected, Unweighted (friendship is mutual)",
    "Web Pages": "Directed, Unweighted (links are one-way)",
    "Road Network": "Undirected, Weighted (roads have distances)",
    "Flight Routes": "Directed, Weighted (routes have costs)",
    "Family Tree": "Directed, Acyclic (DAG)",
    "Dependencies": "Directed, Acyclic (build order)"
}
```

### **Graph Properties:**

```python
def analyze_graph_properties(graph):
    """
    DENSITY: How many edges vs possible edges
    - Dense: Many edges (adjacency matrix)
    - Sparse: Few edges (adjacency list)

    CONNECTIVITY:
    - Connected: Path exists between any two nodes
    - Strongly Connected: (Directed) Path in both directions

    CYCLES:
    - Acyclic: No cycles (trees, DAGs)
    - Cyclic: Contains cycles
    """

    vertices = len(graph)
    edges = sum(len(neighbors) for neighbors in graph.values()) // 2

    max_edges = vertices * (vertices - 1) // 2
    density = edges / max_edges if max_edges > 0 else 0

    return {
        'vertices': vertices,
        'edges': edges,
        'density': density,
        'is_sparse': density < 0.1
    }
```

---

## 🗂️ Graph Representations

### **1. Adjacency List (Most Common)**

```python
class AdjacencyList:
    """
    Best for: Sparse graphs, most graph algorithms
    Space: O(V + E)
    Edge lookup: O(degree of vertex)
    """

    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def add_edge(self, u, v, directed=False):
        self.add_vertex(u)
        self.add_vertex(v)

        self.graph[u].append(v)
        if not directed:
            self.graph[v].append(u)

    def get_neighbors(self, vertex):
        return self.graph.get(vertex, [])

    def remove_edge(self, u, v, directed=False):
        if u in self.graph and v in self.graph[u]:
            self.graph[u].remove(v)
        if not directed and v in self.graph and u in self.graph[v]:
            self.graph[v].remove(u)

    def display(self):
        for vertex in self.graph:
            print(f"{vertex}: {self.graph[vertex]}")

# Example usage
graph = AdjacencyList()
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')]
for u, v in edges:
    graph.add_edge(u, v)

graph.display()
# Output:
# A: ['B', 'C']
# B: ['A', 'D']
# C: ['A', 'D']
# D: ['B', 'C']
```

### **2. Adjacency Matrix**

```python
class AdjacencyMatrix:
    """
    Best for: Dense graphs, quick edge lookup
    Space: O(V²)
    Edge lookup: O(1)
    """

    def __init__(self, vertices):
        self.V = vertices
        self.matrix = [[0] * vertices for _ in range(vertices)]
        self.vertex_map = {}  # Map names to indices
        self.index_map = {}   # Map indices to names

    def add_vertex(self, vertex):
        if vertex not in self.vertex_map:
            idx = len(self.vertex_map)
            self.vertex_map[vertex] = idx
            self.index_map[idx] = vertex

    def add_edge(self, u, v, weight=1, directed=False):
        u_idx = self.vertex_map[u]
        v_idx = self.vertex_map[v]

        self.matrix[u_idx][v_idx] = weight
        if not directed:
            self.matrix[v_idx][u_idx] = weight

    def has_edge(self, u, v):
        u_idx = self.vertex_map[u]
        v_idx = self.vertex_map[v]
        return self.matrix[u_idx][v_idx] != 0

    def get_neighbors(self, vertex):
        vertex_idx = self.vertex_map[vertex]
        neighbors = []

        for i in range(self.V):
            if self.matrix[vertex_idx][i] != 0:
                neighbors.append(self.index_map[i])

        return neighbors

    def display(self):
        print("   ", end="")
        for vertex in self.vertex_map:
            print(f"{vertex:3}", end="")
        print()

        for vertex in self.vertex_map:
            print(f"{vertex}: ", end="")
            vertex_idx = self.vertex_map[vertex]
            for j in range(self.V):
                print(f"{self.matrix[vertex_idx][j]:3}", end="")
            print()
```

### **3. Edge List**

```python
class EdgeList:
    """
    Best for: Simple storage, some algorithms (Kruskal's MST)
    Space: O(E)
    Edge operations: O(E) for searches
    """

    def __init__(self):
        self.edges = []
        self.vertices = set()

    def add_edge(self, u, v, weight=1, directed=False):
        self.vertices.add(u)
        self.vertices.add(v)
        self.edges.append((u, v, weight))

        if not directed:
            self.edges.append((v, u, weight))

    def get_vertices(self):
        return list(self.vertices)

    def get_edges(self):
        return self.edges

    def display(self):
        for u, v, weight in self.edges:
            print(f"{u} -> {v} (weight: {weight})")

# Comparison table
comparison = {
    'Operation': ['Space', 'Add Vertex', 'Add Edge', 'Remove Edge', 'Check Edge', 'Get Neighbors'],
    'Adj List': ['O(V+E)', 'O(1)', 'O(1)', 'O(degree)', 'O(degree)', 'O(degree)'],
    'Adj Matrix': ['O(V²)', 'O(V²)', 'O(1)', 'O(1)', 'O(1)', 'O(V)'],
    'Edge List': ['O(E)', 'O(1)', 'O(1)', 'O(E)', 'O(E)', 'O(E)']
}
```

---

## 🔍 Breadth-First Search (BFS)

### **BFS Fundamentals:**

```python
from collections import deque

def bfs_traversal(graph, start):
    """
    BFS explores level by level (shortest path in unweighted graphs)

    Time: O(V + E)
    Space: O(V) for queue and visited set

    Use Cases:
    - Shortest path in unweighted graphs
    - Level-order traversal
    - Check connectivity
    - Find connected components
    """

    visited = set()
    queue = deque([start])
    result = []

    visited.add(start)

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        # Visit all unvisited neighbors
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result

# Example
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

print(bfs_traversal(graph, 'A'))
# Output: ['A', 'B', 'C', 'D', 'E', 'F']
```

### **BFS with Distance Tracking:**

```python
def bfs_shortest_path(graph, start, target):
    """
    Find shortest path in unweighted graph
    Returns path and distance
    """

    if start == target:
        return [start], 0

    visited = set()
    queue = deque([(start, [start], 0)])  # (vertex, path, distance)
    visited.add(start)

    while queue:
        vertex, path, distance = queue.popleft()

        for neighbor in graph.get(vertex, []):
            if neighbor == target:
                return path + [neighbor], distance + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], distance + 1))

    return None, float('inf')  # No path exists

# Find shortest path
path, distance = bfs_shortest_path(graph, 'A', 'F')
print(f"Shortest path from A to F: {path}")
print(f"Distance: {distance}")
# Output: ['A', 'C', 'F'], Distance: 2
```

### **BFS for Connected Components:**

```python
def find_connected_components_bfs(graph):
    """
    Find all connected components using BFS
    """

    visited = set()
    components = []

    def bfs_component(start):
        component = []
        queue = deque([start])
        visited.add(start)

        while queue:
            vertex = queue.popleft()
            component.append(vertex)

            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return component

    # Check each vertex
    for vertex in graph:
        if vertex not in visited:
            component = bfs_component(vertex)
            components.append(component)

    return components

# Example with disconnected graph
disconnected_graph = {
    'A': ['B', 'C'],
    'B': ['A'],
    'C': ['A'],
    'D': ['E'],
    'E': ['D'],
    'F': []  # Isolated vertex
}

components = find_connected_components_bfs(disconnected_graph)
print("Connected Components:", components)
# Output: [['A', 'B', 'C'], ['D', 'E'], ['F']]
```

---

## 🕳️ Depth-First Search (DFS)

### **DFS Fundamentals:**

```python
def dfs_recursive(graph, start, visited=None, result=None):
    """
    DFS explores as far as possible before backtracking

    Time: O(V + E)
    Space: O(V) for recursion stack and visited set

    Use Cases:
    - Topological sorting
    - Cycle detection
    - Pathfinding with constraints
    - Tree/graph traversal
    """

    if visited is None:
        visited = set()
    if result is None:
        result = []

    visited.add(start)
    result.append(start)

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, result)

    return result

def dfs_iterative(graph, start):
    """
    Iterative DFS using stack
    (Order might differ from recursive version)
    """

    visited = set()
    stack = [start]
    result = []

    while stack:
        vertex = stack.pop()

        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)

            # Add neighbors to stack (reverse order for same result as recursive)
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)

    return result

# Example
print("DFS Recursive:", dfs_recursive(graph, 'A'))
print("DFS Iterative:", dfs_iterative(graph, 'A'))
# Output may vary but both are valid DFS traversals
```

### **DFS with Path Tracking:**

```python
def dfs_all_paths(graph, start, target, path=None, all_paths=None):
    """
    Find all possible paths from start to target using DFS
    """

    if path is None:
        path = []
    if all_paths is None:
        all_paths = []

    path = path + [start]

    if start == target:
        all_paths.append(path)
        return all_paths

    for neighbor in graph.get(start, []):
        if neighbor not in path:  # Avoid cycles
            dfs_all_paths(graph, neighbor, target, path, all_paths)

    return all_paths

# Find all paths from A to F
all_paths = dfs_all_paths(graph, 'A', 'F')
for i, path in enumerate(all_paths, 1):
    print(f"Path {i}: {path}")
```

### **Cycle Detection:**

```python
def has_cycle_undirected(graph):
    """
    Detect cycle in undirected graph using DFS
    """

    visited = set()

    def dfs_cycle_check(vertex, parent):
        visited.add(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                if dfs_cycle_check(neighbor, vertex):
                    return True
            elif neighbor != parent:  # Back edge found (not to parent)
                return True

        return False

    # Check each component
    for vertex in graph:
        if vertex not in visited:
            if dfs_cycle_check(vertex, None):
                return True

    return False

def has_cycle_directed(graph):
    """
    Detect cycle in directed graph using DFS with colors
    """

    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {vertex: WHITE for vertex in graph}

    def dfs_cycle_check(vertex):
        if colors[vertex] == GRAY:  # Back edge (cycle found)
            return True
        if colors[vertex] == BLACK:  # Already processed
            return False

        colors[vertex] = GRAY  # Mark as processing

        for neighbor in graph.get(vertex, []):
            if dfs_cycle_check(neighbor):
                return True

        colors[vertex] = BLACK  # Mark as completely processed
        return False

    # Check each vertex
    for vertex in graph:
        if colors[vertex] == WHITE:
            if dfs_cycle_check(vertex):
                return True

    return False

# Test cycle detection
cyclic_graph = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']  # Creates cycle
}

print("Has cycle (undirected):", has_cycle_undirected(cyclic_graph))
print("Has cycle (directed):", has_cycle_directed(cyclic_graph))
```

---

## 🗺️ Shortest Path Algorithms

### **1. Dijkstra's Algorithm (Non-negative weights)**

```python
import heapq

def dijkstra(graph, start):
    """
    Find shortest paths from start to all other vertices

    Time: O((V + E) log V) with binary heap
    Space: O(V)

    Requirements: Non-negative edge weights
    """

    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0

    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    previous = {vertex: None for vertex in graph}

    while pq:
        current_distance, current_vertex = heapq.heappop(pq)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        # Check all neighbors
        for neighbor, weight in graph.get(current_vertex, []):
            if neighbor not in visited:
                new_distance = current_distance + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (new_distance, neighbor))

    return distances, previous

def reconstruct_path(previous, start, target):
    """Reconstruct shortest path from start to target"""
    path = []
    current = target

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    if path[0] == start:
        return path
    else:
        return None  # No path exists

# Example with weighted graph
weighted_graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('A', 4), ('C', 1), ('D', 5)],
    'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
    'D': [('B', 5), ('C', 8), ('E', 2)],
    'E': [('C', 10), ('D', 2)]
}

distances, previous = dijkstra(weighted_graph, 'A')
print("Shortest distances from A:")
for vertex, distance in distances.items():
    path = reconstruct_path(previous, 'A', vertex)
    print(f"  {vertex}: {distance} via {' -> '.join(path) if path else 'No path'}")
```

### **2. Bellman-Ford Algorithm (Handles negative weights)**

```python
def bellman_ford(graph, start):
    """
    Find shortest paths, can handle negative weights
    Detects negative cycles

    Time: O(VE)
    Space: O(V)
    """

    # Get all vertices and edges
    vertices = set()
    edges = []

    for vertex in graph:
        vertices.add(vertex)
        for neighbor, weight in graph.get(vertex, []):
            vertices.add(neighbor)
            edges.append((vertex, neighbor, weight))

    # Initialize distances
    distances = {vertex: float('inf') for vertex in vertices}
    distances[start] = 0
    previous = {vertex: None for vertex in vertices}

    # Relax edges V-1 times
    for _ in range(len(vertices) - 1):
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                previous[v] = u

    # Check for negative cycles
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            return None, None  # Negative cycle detected

    return distances, previous

# Example with negative weights
negative_weight_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('C', -3), ('D', 2)],
    'C': [],
    'D': [('C', 5), ('B', -2)]
}

distances, previous = bellman_ford(negative_weight_graph, 'A')
if distances:
    print("Bellman-Ford distances from A:")
    for vertex, distance in distances.items():
        print(f"  {vertex}: {distance}")
else:
    print("Negative cycle detected!")
```

### **3. Floyd-Warshall Algorithm (All pairs shortest paths)**

```python
def floyd_warshall(graph):
    """
    Find shortest paths between all pairs of vertices

    Time: O(V³)
    Space: O(V²)
    """

    vertices = list(graph.keys())
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}

    # Initialize distance matrix
    distances = [[float('inf')] * n for _ in range(n)]
    next_vertex = [[None] * n for _ in range(n)]

    # Set distances to self as 0
    for i in range(n):
        distances[i][i] = 0

    # Set direct edge distances
    for u in graph:
        for v, weight in graph[u]:
            u_idx, v_idx = vertex_index[u], vertex_index[v]
            distances[u_idx][v_idx] = weight
            next_vertex[u_idx][v_idx] = v_idx

    # Floyd-Warshall main algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]
                    next_vertex[i][j] = next_vertex[i][k]

    return distances, next_vertex, vertices

def reconstruct_fw_path(next_vertex, vertices, start, end):
    """Reconstruct path from Floyd-Warshall results"""
    vertex_index = {v: i for i, v in enumerate(vertices)}
    start_idx, end_idx = vertex_index[start], vertex_index[end]

    if next_vertex[start_idx][end_idx] is None:
        return None

    path = [start]
    current = start_idx

    while current != end_idx:
        current = next_vertex[current][end_idx]
        path.append(vertices[current])

    return path

# Test Floyd-Warshall
distances, next_vertex, vertices = floyd_warshall(weighted_graph)

print("All pairs shortest distances:")
for i, u in enumerate(vertices):
    for j, v in enumerate(vertices):
        if distances[i][j] != float('inf'):
            path = reconstruct_fw_path(next_vertex, vertices, u, v)
            print(f"{u} to {v}: {distances[i][j]} via {' -> '.join(path)}")
```

---

## 🚀 Advanced Graph Algorithms

### **Topological Sort (DAG only)**

```python
def topological_sort_dfs(graph):
    """
    Topological sort using DFS
    Only works for Directed Acyclic Graphs (DAGs)
    """

    visited = set()
    stack = []

    def dfs(vertex):
        visited.add(vertex)

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor)

        stack.append(vertex)  # Add to stack after visiting all descendants

    # Visit all vertices
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)

    return stack[::-1]  # Reverse stack for topological order

def topological_sort_bfs(graph):
    """
    Topological sort using Kahn's algorithm (BFS approach)
    """

    # Calculate in-degrees
    in_degree = {vertex: 0 for vertex in graph}
    for vertex in graph:
        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

    # Start with vertices having no incoming edges
    queue = deque([vertex for vertex in in_degree if in_degree[vertex] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        # Remove edges and update in-degrees
        for neighbor in graph.get(vertex, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check if all vertices are included (no cycle)
    if len(result) == len(graph):
        return result
    else:
        return None  # Cycle detected

# Example: Course prerequisites
course_graph = {
    'Math101': [],
    'CS101': ['Math101'],
    'CS102': ['CS101'],
    'Stats': ['Math101'],
    'AI': ['CS102', 'Stats'],
    'ML': ['AI']
}

print("Topological order (DFS):", topological_sort_dfs(course_graph))
print("Topological order (BFS):", topological_sort_bfs(course_graph))
```

### **Minimum Spanning Tree (MST)**

```python
class UnionFind:
    """Union-Find data structure for Kruskal's algorithm"""

    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}

    def find(self, vertex):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])  # Path compression
        return self.parent[vertex]

    def union(self, u, v):
        root_u, root_v = self.find(u), self.find(v)

        if root_u != root_v:
            # Union by rank
            if self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            elif self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True
        return False

def kruskal_mst(graph):
    """
    Find Minimum Spanning Tree using Kruskal's algorithm

    Time: O(E log E)
    Space: O(V)
    """

    # Get all edges
    edges = []
    vertices = set()

    for u in graph:
        vertices.add(u)
        for v, weight in graph[u]:
            vertices.add(v)
            edges.append((weight, u, v))

    # Sort edges by weight
    edges.sort()

    # Initialize Union-Find
    uf = UnionFind(vertices)
    mst = []
    total_weight = 0

    for weight, u, v in edges:
        if uf.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight

            if len(mst) == len(vertices) - 1:
                break

    return mst, total_weight

def prim_mst(graph, start):
    """
    Find Minimum Spanning Tree using Prim's algorithm

    Time: O(E log V)
    Space: O(V)
    """

    visited = set([start])
    edges = []
    mst = []
    total_weight = 0

    # Add all edges from start vertex
    for neighbor, weight in graph.get(start, []):
        heapq.heappush(edges, (weight, start, neighbor))

    while edges and len(visited) < len(graph):
        weight, u, v = heapq.heappop(edges)

        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            total_weight += weight

            # Add new edges from v
            for neighbor, edge_weight in graph.get(v, []):
                if neighbor not in visited:
                    heapq.heappush(edges, (edge_weight, v, neighbor))

    return mst, total_weight

# Test MST algorithms
mst_kruskal, weight_k = kruskal_mst(weighted_graph)
mst_prim, weight_p = prim_mst(weighted_graph, 'A')

print(f"Kruskal MST: {mst_kruskal}, Total weight: {weight_k}")
print(f"Prim MST: {mst_prim}, Total weight: {weight_p}")
```

---

## 🎯 Graph Problems & Patterns

### **Pattern 1: Island Problems (2D Grid as Graph)**

```python
def num_islands(grid):
    """
    Count number of islands in 2D grid
    Island = connected group of 1s
    """

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

    # Check each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                islands += 1

    return islands

# Test
grid = [
    ['1','1','0','0','0'],
    ['1','1','0','0','0'],
    ['0','0','1','0','0'],
    ['0','0','0','1','1']
]
print("Number of islands:", num_islands(grid))  # Output: 3
```

### **Pattern 2: Bipartite Graph Check**

```python
def is_bipartite(graph):
    """
    Check if graph is bipartite (can be colored with 2 colors)
    """

    colors = {}

    def bfs_color(start):
        queue = deque([start])
        colors[start] = 0

        while queue:
            vertex = queue.popleft()

            for neighbor in graph.get(vertex, []):
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[vertex]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[vertex]:
                    return False

        return True

    # Check each component
    for vertex in graph:
        if vertex not in colors:
            if not bfs_color(vertex):
                return False

    return True

# Test
bipartite_graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

print("Is bipartite:", is_bipartite(bipartite_graph))  # True
```

### **Pattern 3: Course Schedule (Topological Sort)**

```python
def can_finish_courses(num_courses, prerequisites):
    """
    Check if all courses can be finished given prerequisites
    This is cycle detection in directed graph
    """

    # Build graph
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # BFS with Kahn's algorithm
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

def find_order(num_courses, prerequisites):
    """
    Find a valid order to finish all courses
    """

    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

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

# Test
prerequisites = [[1,0],[2,0],[3,1],[3,2]]
print("Can finish:", can_finish_courses(4, prerequisites))
print("Course order:", find_order(4, prerequisites))
```

---

## 🎯 Interview Essentials

### **Graph Algorithm Complexity Summary:**

| Algorithm      | Time Complexity  | Space Complexity | Use Case                             |
| -------------- | ---------------- | ---------------- | ------------------------------------ |
| BFS            | O(V + E)         | O(V)             | Shortest path (unweighted)           |
| DFS            | O(V + E)         | O(V)             | Cycle detection, topological sort    |
| Dijkstra       | O((V + E) log V) | O(V)             | Shortest path (non-negative weights) |
| Bellman-Ford   | O(VE)            | O(V)             | Shortest path (negative weights)     |
| Floyd-Warshall | O(V³)            | O(V²)            | All pairs shortest paths             |
| Kruskal's MST  | O(E log E)       | O(V)             | Minimum spanning tree                |
| Prim's MST     | O(E log V)       | O(V)             | Minimum spanning tree                |

### **When to Use Which Algorithm:**

```python
def choose_algorithm(problem_type, graph_properties):
    """
    Algorithm selection guide
    """

    if problem_type == "shortest_path":
        if graph_properties["weights"] == "none":
            return "BFS"
        elif graph_properties["weights"] == "non_negative":
            return "Dijkstra"
        elif graph_properties["weights"] == "can_be_negative":
            return "Bellman-Ford"
        elif graph_properties["need_all_pairs"]:
            return "Floyd-Warshall"

    elif problem_type == "connectivity":
        return "DFS or BFS"

    elif problem_type == "cycle_detection":
        if graph_properties["directed"]:
            return "DFS with colors"
        else:
            return "DFS with parent tracking"

    elif problem_type == "minimum_spanning_tree":
        if graph_properties["sparse"]:
            return "Kruskal"
        else:
            return "Prim"

    elif problem_type == "topological_order":
        return "DFS or Kahn's algorithm"

    return "Choose based on specific requirements"
```

---

---

## Common Confusions & Mistakes

### **1. "Directed vs Undirected Graph Confusion"**

**Confusion:** Not understanding the difference between directed and undirected graphs
**Reality:** Directed graphs have one-way edges, undirected graphs have two-way connections
**Solution:** Always determine graph directionality before choosing algorithms, as they affect traversal

### **2. "Adjacency List vs Matrix Choice"**

**Confusion:** Not knowing when to use adjacency lists vs adjacency matrices
**Reality:** Lists are better for sparse graphs, matrices for dense graphs
**Solution:** Choose based on graph density, memory constraints, and operation requirements

### **3. "BFS vs DFS Selection"**

**Confusion:** Using BFS when DFS would be more appropriate or vice versa
**Reality:** BFS finds shortest paths, DFS explores deep connections and components
**Solution:** BFS for shortest path in unweighted graphs, DFS for connectivity and cycles

### **4. "Graph Traversal with Cycles"**

**Confusion:** Not properly handling cycles during graph traversal
**Reality:** Unchecked cycles lead to infinite loops and incorrect results
**Solution:** Always use visited tracking with colors or boolean arrays in DFS/BFS

### **5. "Negative Weight Edge Handling"**

**Confusion:** Using Dijkstra's algorithm with negative weight edges
**Reality:** Dijkstra fails with negative weights; use Bellman-Ford instead
**Solution:** Check for negative weights and choose appropriate shortest path algorithm

### **6. "Connected Components vs Connected Graph"**

**Confusion:** Mixing up graph connectivity concepts
**Reality:** Connected components are subgraphs where any two vertices are connected
**Solution:** Use DFS/BFS to find components, understand they're disjoint subgraphs

### **7. "Memory Management in Large Graphs"**

**Confusion:** Not considering memory usage when working with large graphs
**Reality:** Graph representations can consume significant memory
**Solution:** Choose memory-efficient representations, use streaming for very large graphs

### **8. "Graph Problem Classification"**

**Confusion:** Not identifying the specific type of graph problem
**Reality:** Different graph problems require different algorithms and approaches
**Solution:** Classify problems (shortest path, connectivity, cycle detection) before choosing algorithms

---

## Micro-Quiz (80% Required for Mastery)

**Question 1:** What is the time complexity of BFS on a graph with V vertices and E edges?
a) O(V)
b) O(E)
c) O(V + E)
d) O(V × E)

**Question 2:** Which algorithm should you use to find the shortest path in a graph with negative weights?
a) Dijkstra's algorithm
b) BFS
c) Bellman-Ford algorithm
d) DFS

**Question 3:** What is the primary use case for topological sorting?
a) Finding shortest paths
b) Ordering tasks with dependencies
c) Detecting cycles
d) Finding connected components

**Question 4:** In DFS, what does the "color" technique prevent?
a) Memory overflow
b) Stack overflow
c) Revisiting nodes
d) Graph disconnection

**Question 5:** When is Prim's algorithm preferred over Kruskal's for minimum spanning tree?
a) For dense graphs
b) For sparse graphs
c) For directed graphs
d) For disconnected graphs

**Answer Key:** 1-c, 2-c, 3-b, 4-c, 5-a

---

## Reflection Prompts

**1. Network Design Challenge:**
You're designing a social media platform's friend recommendation system. What graph algorithms would you use? How would you handle the scale of millions of users and relationships? What are the performance trade-offs?

**2. Route Optimization Problem:**
You need to find the most efficient delivery routes for a logistics company. How would you model this as a graph problem? Which algorithms would you use? How would you handle real-time traffic updates?

**3. Dependency Management:**
You're building a build system that needs to compile code files with complex dependencies. How would you use graphs to manage this? What algorithms would help you find the optimal build order?

**4. Network Security Analysis:**
You're analyzing a computer network for potential vulnerabilities. How would you use graph algorithms to identify critical nodes? What would be the algorithmic approach to find the most vulnerable points?

---

## Mini Sprint Project (25-40 minutes)

**Project:** Build a Social Network Connection Finder

**Scenario:** Create a system that finds connections between people in a social network and determines degrees of separation.

**Requirements:**

1. **Graph Representation:** Model people as nodes, friendships as edges
2. **Connection Finding:** Find shortest path between any two people
3. **Degree Calculation:** Calculate degrees of separation
4. **Component Analysis:** Find friend groups and isolated users

**Deliverables:**

1. **Graph Implementation** - Build graph with adjacency list representation
2. **BFS Implementation** - Use BFS to find shortest paths
3. **Connection Analysis** - Find degrees of separation between users
4. **Component Detection** - Identify connected friend groups
5. **Performance Analysis** - Compare efficiency of different approaches

**Success Criteria:**

- Working graph implementation with proper data structures
- Correct BFS implementation for shortest path finding
- Accurate degree of separation calculations
- Proper connected component detection
- Clear analysis of algorithm efficiency

---

## Full Project Extension (8-12 hours)

**Project:** Build a Network Routing and Analysis Platform

**Scenario:** Create a comprehensive network routing system that handles multiple types of networks and provides various analysis capabilities.

**Extended Requirements:**

**1. Graph Representation System (2-3 hours)**

- Build flexible graph class supporting directed/undirected graphs
- Implement adjacency list, matrix, and edge list representations
- Add weighted and unweighted graph support
- Create graph validation and error handling

**2. Traversal and Search Algorithms (2-3 hours)**

- Implement BFS and DFS with different node ordering
- Add bidirectional BFS for faster shortest path finding
- Create graph component detection and analysis
- Implement cycle detection for directed and undirected graphs

**3. Shortest Path Algorithms (2-3 hours)**

- Build Dijkstra's algorithm for non-negative weights
- Implement Bellman-Ford for graphs with negative weights
- Add Floyd-Warshall for all-pairs shortest paths
- Create A\* algorithm for weighted graphs with heuristics

**4. Advanced Graph Applications (1-2 hours)**

- Implement minimum spanning tree (Prim and Kruskal)
- Add topological sorting for directed acyclic graphs
- Build network flow algorithms (Ford-Fulkerson)
- Create graph clustering and community detection

**5. Network Analysis Platform (1-2 hours)**

- Build network topology analysis tools
- Add centrality measures (betweenness, closeness, degree)
- Implement network robustness analysis
- Create visualization tools for network analysis

**Deliverables:**

1. **Complete graph library** with multiple representations and operations
2. **Comprehensive traversal algorithms** with BFS, DFS, and advanced variants
3. **Shortest path suite** with Dijkstra, Bellman-Ford, Floyd-Warshall, and A\*
4. **Advanced algorithms** for MST, topological sort, and network flow
5. **Network analysis tools** with centrality measures and robustness analysis
6. **Performance benchmarking** with algorithm comparison
7. **Real-world applications** demonstrating practical usage
8. **Documentation** with usage examples and algorithm explanations

**Success Criteria:**

- Functional graph library with comprehensive algorithm implementations
- Performance-optimized algorithms with proper complexity analysis
- Real-world network analysis capabilities
- Professional documentation and usage examples
- Demonstrated understanding of graph theory applications
- Clear performance comparisons and optimization strategies
- Working network analysis tools and visualizations
- Professional presentation of results and applications

**Bonus Challenges:**

- Distributed graph processing for massive networks
- Real-time dynamic graph updates and analysis
- Integration with geographic information systems (GIS)
- Machine learning integration for graph-based predictions
- Graph neural network implementation and training
- Large-scale network simulation and analysis
- Integration with existing network monitoring tools
- Performance optimization for memory-constrained environments

---

_Master these graph algorithms and you'll solve any network problem! 🚀_
