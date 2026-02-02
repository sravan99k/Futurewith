---
title: "Graphs Interview Questions"
level: "Interview Preparation"
difficulty: "Easy to Hard"
time: "30-90 minutes per session"
tags:
  [
    "dsa",
    "graphs",
    "bfs",
    "dfs",
    "shortest-path",
    "interview",
    "coding-interview",
  ]
---

# 🌐 Graphs Interview Questions

_Top 40+ Interview Questions with Solutions_

---

## 📊 Question Categories

| Category                | Count        | Difficulty | Companies                 |
| ----------------------- | ------------ | ---------- | ------------------------- |
| **Basic Traversal**     | 8 questions  | Easy       | All companies             |
| **Shortest Path**       | 10 questions | Medium     | Google, Meta, Amazon      |
| **Topological Sort**    | 8 questions  | Medium     | Microsoft, Apple, Netflix |
| **Advanced Algorithms** | 12 questions | Hard       | Google, Meta, ByteDance   |
| **System Design**       | 5 questions  | Hard       | All FAANG                 |

---

## 🌱 EASY LEVEL - Foundation Questions

### **Q1: Number of Islands (Meta, Amazon)**

**Difficulty:** ⭐ Easy | **Frequency:** Very High | **Time:** 20 minutes

```python
"""
PROBLEM:
Given a 2D binary grid, count the number of islands.
'1' represents land, '0' represents water.
Islands are connected horizontally or vertically.

EXAMPLES:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

FOLLOW-UP QUESTIONS:
1. What if islands can be connected diagonally?
2. What if we need to find the largest island?
3. Can you solve it with Union-Find?
"""

def num_islands(grid):
    """
    Time: O(m*n), Space: O(m*n) worst case for recursion stack
    Pattern: DFS/BFS to mark connected components
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

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                dfs(r, c)
                islands += 1

    return islands

# BFS Alternative
def num_islands_bfs(grid):
    from collections import deque

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

# INTERVIEWER QUESTIONS TO EXPECT:
# "Explain the difference between DFS and BFS for this problem"
# "What's the space complexity and why?"
# "How would you handle a very large grid that doesn't fit in memory?"

# Test cases
grid1 = [["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]
assert num_islands(grid1) == 1

grid2 = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
assert num_islands(grid2) == 3
```

### **Q2: Clone Graph (Google, Meta)**

**Difficulty:** ⭐ Easy-Medium | **Frequency:** High | **Time:** 25 minutes

```python
"""
PROBLEM:
Clone an undirected connected graph. Each node contains a value and
a list of neighbors.

FOLLOW-UP QUESTIONS:
1. What if the graph is directed?
2. How to handle disconnected components?
3. Can you solve with BFS instead of DFS?
"""

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def clone_graph_dfs(node):
    """
    Time: O(V+E), Space: O(V)
    Pattern: DFS with hashmap to track cloned nodes
    """
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
    """
    BFS alternative approach
    """
    if not node:
        return None

    from collections import deque

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

# INTERVIEWER INSIGHTS:
# "Why use a hashmap instead of visited set?"
# "How do you ensure each node is cloned exactly once?"
# "What happens with cycles in the graph?"

# Test case creation and verification would go here
```

---

## ⚡ MEDIUM LEVEL - Core Interview Questions

### **Q3: Course Schedule (Google, Amazon)**

**Difficulty:** ⚡ Medium | **Frequency:** Very High | **Time:** 30 minutes

```python
"""
PROBLEM:
Determine if you can finish all courses given prerequisites.

Input: numCourses = 2, prerequisites = [[1,0]]
Output: True

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: False

FOLLOW-UP QUESTIONS:
1. Return a valid course order?
2. What if some courses have no prerequisites?
3. How to detect which courses form a cycle?
"""

def can_finish(num_courses, prerequisites):
    """
    Time: O(V+E), Space: O(V+E)
    Pattern: Topological sort using Kahn's algorithm
    """
    # Build adjacency list and in-degree count
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses having no prerequisites
    from collections import deque
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    completed = 0

    while queue:
        course = queue.popleft()
        completed += 1

        # Remove this course's prerequisites from dependent courses
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)

    return completed == num_courses

def can_finish_dfs(num_courses, prerequisites):
    """
    DFS approach with cycle detection using 3-coloring
    """
    graph = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    WHITE, GRAY, BLACK = 0, 1, 2
    colors = [WHITE] * num_courses

    def has_cycle(course):
        if colors[course] == GRAY:  # Back edge - cycle detected
            return True
        if colors[course] == BLACK:  # Already processed
            return False

        colors[course] = GRAY  # Mark as processing

        for next_course in graph[course]:
            if has_cycle(next_course):
                return True

        colors[course] = BLACK  # Mark as completed
        return False

    # Check each course
    for i in range(num_courses):
        if colors[i] == WHITE:
            if has_cycle(i):
                return False

    return True

# COURSE SCHEDULE II - Return the order
def find_order(num_courses, prerequisites):
    graph = {i: [] for i in range(num_courses)}
    in_degree = [0] * num_courses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    from collections import deque
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

# KEY INTERVIEW POINTS:
# "This is essentially cycle detection in a directed graph"
# "Kahn's algorithm vs DFS-based topological sort"
# "In-degree represents remaining prerequisites"

# Test cases
assert can_finish(2, [[1,0]]) == True
assert can_finish(2, [[1,0],[0,1]]) == False
```

### **Q4: Word Ladder (Amazon, Meta)**

**Difficulty:** ⚡ Medium | **Frequency:** High | **Time:** 35 minutes

```python
"""
PROBLEM:
Transform beginWord to endWord changing one letter at a time.
Each transformed word must exist in wordList.

Input: beginWord = "hit", endWord = "cog",
       wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5

FOLLOW-UP QUESTIONS:
1. Return all shortest transformation sequences?
2. What if multiple paths have same length?
3. How to optimize for very large word lists?
"""

def ladder_length(begin_word, end_word, word_list):
    """
    Time: O(M²×N) where M=word length, N=word list size
    Space: O(M×N)
    Pattern: BFS for shortest path in unweighted graph
    """
    if end_word not in word_list:
        return 0

    word_set = set(word_list)
    from collections import deque

    queue = deque([(begin_word, 1)])
    visited = {begin_word}

    while queue:
        word, length = queue.popleft()

        if word == end_word:
            return length

        # Try changing each character position
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]

                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        queue.append((new_word, length + 1))

    return 0

def ladder_length_bidirectional(begin_word, end_word, word_list):
    """
    Bidirectional BFS optimization
    Time: O(M²×N), but faster in practice
    """
    if end_word not in word_list:
        return 0

    word_set = set(word_list)

    # Start from both ends
    begin_set = {begin_word}
    end_set = {end_word}
    visited = set()
    length = 1

    while begin_set and end_set:
        # Always expand the smaller frontier
        if len(begin_set) > len(end_set):
            begin_set, end_set = end_set, begin_set

        next_begin_set = set()

        for word in begin_set:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    if c != word[i]:
                        new_word = word[:i] + c + word[i+1:]

                        # If we meet the other frontier
                        if new_word in end_set:
                            return length + 1

                        if new_word in word_set and new_word not in visited:
                            visited.add(new_word)
                            next_begin_set.add(new_word)

        begin_set = next_begin_set
        length += 1

    return 0

# OPTIMIZATION INSIGHTS:
# "Why BFS instead of DFS?"
# "How does bidirectional search improve performance?"
# "Character transformation creates implicit graph"

# Test cases
word_list = ["hot","dot","dog","lot","log","cog"]
assert ladder_length("hit", "cog", word_list) == 5
assert ladder_length_bidirectional("hit", "cog", word_list) == 5
```

### **Q5: Network Delay Time (Google, Amazon)**

**Difficulty:** ⚡ Medium | **Frequency:** High | **Time:** 40 minutes

```python
"""
PROBLEM:
Find the time it takes for a signal to reach all nodes from source node.

Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

FOLLOW-UP QUESTIONS:
1. What if some nodes are unreachable?
2. How to find the path that reaches all nodes fastest?
3. Can you handle negative delays?
"""

def network_delay_time(times, n, k):
    """
    Time: O((V+E)logV), Space: O(V+E)
    Pattern: Dijkstra's algorithm for shortest paths
    """
    import heapq

    # Build adjacency list
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra's algorithm
    distances = {i: float('inf') for i in range(1, n + 1)}
    distances[k] = 0

    heap = [(0, k)]
    visited = set()

    while heap:
        curr_time, node = heapq.heappop(heap)

        if node in visited:
            continue

        visited.add(node)

        for neighbor, delay in graph[node]:
            if neighbor not in visited:
                new_time = curr_time + delay
                if new_time < distances[neighbor]:
                    distances[neighbor] = new_time
                    heapq.heappush(heap, (new_time, neighbor))

    max_time = max(distances.values())
    return max_time if max_time != float('inf') else -1

def network_delay_time_bellman_ford(times, n, k):
    """
    Bellman-Ford approach - can handle negative weights
    Time: O(VE), Space: O(V)
    """
    distances = [float('inf')] * (n + 1)
    distances[k] = 0

    # Relax edges n-1 times
    for _ in range(n - 1):
        updated = False
        for u, v, w in times:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                distances[v] = distances[u] + w
                updated = True

        if not updated:  # Early termination
            break

    max_time = max(distances[1:])
    return max_time if max_time != float('inf') else -1

# ALGORITHM SELECTION GUIDE:
# "When to use Dijkstra vs Bellman-Ford?"
# "How to detect if all nodes are reachable?"
# "Why use a min-heap in Dijkstra's?"

# Test cases
times = [[2,1,1],[2,3,1],[3,4,1]]
assert network_delay_time(times, 4, 2) == 2
```

---

## 🔥 HARD LEVEL - Advanced Interview Questions

### **Q6: Alien Dictionary (Google, Meta)**

**Difficulty:** 🔥 Hard | **Frequency:** Medium | **Time:** 50 minutes

```python
"""
PROBLEM:
Given sorted dictionary of alien language, deduce character order.

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"

FOLLOW-UP QUESTIONS:
1. What if the input is invalid?
2. Multiple valid orders possible?
3. How to handle duplicate words?
"""

def alien_order(words):
    """
    Time: O(C + min(U²,N)) where C=total chars, U=unique chars, N=words
    Space: O(U²)
    Pattern: Topological sort with careful edge case handling
    """
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

        # Invalid case: word1 is longer prefix of word2
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""

        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break

    # Step 3: Topological sort
    from collections import deque
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycle
    return ''.join(result) if len(result) == len(in_degree) else ""

# EDGE CASES TO DISCUSS:
# "What makes input invalid?"
# "How to handle words that are prefixes?"
# "Why topological sort works here?"

# Test cases
assert alien_order(["wrt","wrf","er","ett","rftt"]) == "wertf"
assert alien_order(["z","x"]) == "zx"
assert alien_order(["z","x","z"]) == ""  # Invalid
```

### **Q7: Critical Connections (Google, ByteDance)**

**Difficulty:** 🔥 Hard | **Frequency:** Medium | **Time:** 60 minutes

```python
"""
PROBLEM:
Find all critical connections (bridges) in a network.
A bridge is an edge whose removal increases connected components.

Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
Output: [[1,3]]

FOLLOW-UP QUESTIONS:
1. Find critical vertices (articulation points)?
2. What if we need to add minimum edges to remove all bridges?
3. How to handle dynamic graphs?
"""

def critical_connections(n, connections):
    """
    Time: O(V+E), Space: O(V+E)
    Pattern: Tarjan's bridge-finding algorithm
    """
    # Build adjacency list
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    visited = [False] * n
    disc = [0] * n      # Discovery time
    low = [0] * n       # Low-link value
    parent = [-1] * n
    bridges = []
    time = [0]  # Mutable time counter

    def bridge_dfs(u):
        visited[u] = True
        disc[u] = low[u] = time[0]
        time[0] += 1

        for v in graph[u]:
            if not visited[v]:
                parent[v] = u
                bridge_dfs(v)

                # Update low-link value
                low[u] = min(low[u], low[v])

                # Check if edge u-v is a bridge
                if low[v] > disc[u]:
                    bridges.append([min(u, v), max(u, v)])

            elif v != parent[u]:  # Back edge
                low[u] = min(low[u], disc[v])

    # Run DFS from all unvisited nodes
    for i in range(n):
        if not visited[i]:
            bridge_dfs(i)

    return sorted(bridges)

def critical_connections_alternative(n, connections):
    """
    Alternative implementation with cleaner structure
    """
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    rank = {}
    graph_rank = 0
    result = []

    def dfs(node, parent_rank):
        nonlocal graph_rank, result

        if node in visited:
            return rank[node]

        visited.add(node)
        rank[node] = graph_rank
        graph_rank += 1
        min_back_edge = rank[node]

        for neighbor in graph[node]:
            if neighbor not in visited:
                back_edge = dfs(neighbor, rank[node])
                if back_edge > rank[node]:
                    result.append([min(node, neighbor), max(node, neighbor)])
                min_back_edge = min(min_back_edge, back_edge)
            elif rank[neighbor] < parent_rank:
                min_back_edge = min(min_back_edge, rank[neighbor])

        return min_back_edge

    for i in range(n):
        if i not in visited:
            dfs(i, -1)

    return sorted(result)

# ALGORITHM INSIGHTS:
# "What's the difference between discovery time and low-link value?"
# "Why do we check low[v] > disc[u] for bridges?"
# "How does this relate to strongly connected components?"

# Test cases
connections = [[0,1],[1,2],[2,0],[1,3]]
result = critical_connections(4, connections)
assert [1, 3] in result or [3, 1] in result
```

---

## 💡 Interview Strategy & Tips

### **Common Interview Flow:**

**Phase 1: Problem Analysis (5 min)**

```python
# Questions to ask:
# "Is the graph directed or undirected?"
# "Are there weights on edges?"
# "Can there be self-loops or multiple edges?"
# "What's the size constraint (V, E)?"
```

**Phase 2: Approach Selection (10 min)**

```python
# Decision tree:
if "shortest path" in problem:
    if "unweighted":
        return "BFS"
    elif "weighted, non-negative":
        return "Dijkstra"
    elif "weighted, can be negative":
        return "Bellman-Ford"

elif "connectivity" in problem:
    return "DFS or BFS"

elif "topological order" in problem:
    return "Kahn's algorithm or DFS"
```

**Phase 3: Implementation (25-30 min)**

```python
# Always start with:
# 1. Graph representation
# 2. Main algorithm
# 3. Edge cases
# 4. Optimization if time permits
```

### **Key Interview Phrases:**

```python
# When explaining approach:
"This is essentially a [graph problem type] problem"
"I'll use [algorithm] because [reasoning]"
"The key insight is [pattern recognition]"

# When implementing:
"Let me build the adjacency list first"
"I'll use BFS/DFS to traverse the graph"
"This requires keeping track of visited nodes"

# When optimizing:
"We can optimize space by using Union-Find"
"Bidirectional search can reduce time complexity"
```

### **Common Pitfalls:**

❌ **Don't:**

- Forget to mark nodes as visited
- Assume graph is connected
- Mix up directed vs undirected logic
- Ignore edge cases (empty graph, cycles)

✅ **Do:**

- Clarify problem constraints first
- Choose appropriate data structure
- Handle disconnected components
- Test with examples

### **Advanced Topics to Mention:**

- **Strongly Connected Components** (Kosaraju's, Tarjan's)
- **Minimum Spanning Tree** (Kruskal's, Prim's)
- **Network Flow** (Ford-Fulkerson, Edmonds-Karp)
- **A\* Search** for pathfinding optimization

---

_Master these patterns and you'll handle any graph interview! 🚀_
