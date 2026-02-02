# 📝 Trees & BST - Quick Reference Cheatsheet

> **Last-Minute Revision** | **Interview Prep** | **Pattern Recognition**

---

## 🎯 Core Concepts (30 seconds)

### **Tree Structure**

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### **Key Properties**

- **Height:** Longest path to leaf
- **Depth:** Distance from root
- **BST:** Left < Root < Right

---

## ⚡ Essential Traversals

### **DFS Traversals**

```python
# Preorder: Root → Left → Right
def preorder(root):
    if not root: return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

# Inorder: Left → Root → Right (BST gives sorted!)
def inorder(root):
    if not root: return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

# Postorder: Left → Right → Root
def postorder(root):
    if not root: return
    postorder(root.left)
    postorder(root.right)
    print(root.val)
```

### **BFS (Level Order)**

```python
from collections import deque

def levelOrder(root):
    if not root: return []
    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        result.append(current_level)

    return result
```

---

## 🔥 Top 10 Patterns

### **1. Max Depth**

```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

### **2. Validate BST**

```python
def isValidBST(root, min_val=float('-inf'), max_val=float('inf')):
    if not root: return True
    if not (min_val < root.val < max_val): return False
    return (isValidBST(root.left, min_val, root.val) and
            isValidBST(root.right, root.val, max_val))
```

### **3. Path Sum**

```python
def hasPathSum(root, target):
    if not root: return False
    if not root.left and not root.right:
        return root.val == target
    remaining = target - root.val
    return (hasPathSum(root.left, remaining) or
            hasPathSum(root.right, remaining))
```

### **4. LCA in BST**

```python
def lowestCommonAncestor(root, p, q):
    if p.val < root.val and q.val < root.val:
        return lowestCommonAncestor(root.left, p, q)
    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor(root.right, p, q)
    return root
```

### **5. Invert Tree**

```python
def invertTree(root):
    if not root: return None
    root.left, root.right = root.right, root.left
    invertTree(root.left)
    invertTree(root.right)
    return root
```

### **6. Diameter**

```python
def diameterOfBinaryTree(root):
    self.diameter = 0

    def height(node):
        if not node: return 0
        left = height(node.left)
        right = height(node.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)

    height(root)
    return self.diameter
```

### **7. Symmetric Tree**

```python
def isSymmetric(root):
    def mirror(left, right):
        if not left and not right: return True
        if not left or not right: return False
        return (left.val == right.val and
                mirror(left.left, right.right) and
                mirror(left.right, right.left))
    return mirror(root, root) if root else True
```

### **8. Right Side View**

```python
def rightSideView(root):
    if not root: return []
    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:  # Last in level
                result.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
    return result
```

### **9. Kth Smallest in BST**

```python
def kthSmallest(root, k):
    stack = []
    curr = root
    count = 0

    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        count += 1
        if count == k:
            return curr.val
        curr = curr.right
```

### **10. Serialize/Deserialize**

```python
def serialize(root):
    if not root: return "null"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    def helper(nodes):
        val = next(nodes)
        if val == "null": return None
        node = TreeNode(int(val))
        node.left = helper(nodes)
        node.right = helper(nodes)
        return node
    return helper(iter(data.split(',')))
```

---

## 📊 Complexity Cheatsheet

| Operation | BST Avg  | BST Worst | Balanced |
| --------- | -------- | --------- | -------- |
| Search    | O(log n) | O(n)      | O(log n) |
| Insert    | O(log n) | O(n)      | O(log n) |
| Delete    | O(log n) | O(n)      | O(log n) |
| Traversal | O(n)     | O(n)      | O(n)     |

---

## 🎯 Pattern Recognition

**Use DFS (Recursion) when:**

- Need to explore all paths
- Working with tree structure
- Calculating depth/height

**Use BFS (Queue) when:**

- Level-order processing
- Shortest path to node
- Right/left side view

**Use BST Property when:**

- Values are sorted
- Need O(log n) search
- Finding kth element

---

## 💡 Common Mistakes

❌ **Forget null check**

```python
# WRONG
def maxDepth(root):
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

# RIGHT
def maxDepth(root):
    if not root: return 0  # CHECK NULL!
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

❌ **Wrong BST validation**

```python
# WRONG - Only checks immediate children
if root.left.val < root.val and root.right.val > root.val

# RIGHT - Use range validation
validate(node, min_val, max_val)
```

---

## 🚀 Interview Speedrun (2 min)

**Must Know (Top 5):**

1. Traversals (Pre/In/Post/Level order)
2. Validate BST
3. Max Depth
4. Path Sum
5. LCA

**Decision Tree:**

```
Need to visit all nodes? → DFS or BFS
Level-by-level? → BFS (queue)
Path problems? → DFS (recursion)
BST specific? → Use BST property
Sorted output? → Inorder traversal
```

---

## 📚 Top 20 Must-Know Problems

| #   | Problem               | Pattern          |
| --- | --------------------- | ---------------- |
| 1   | Max Depth             | DFS              |
| 2   | Invert Tree           | DFS              |
| 3   | Same Tree             | DFS              |
| 4   | Symmetric Tree        | DFS              |
| 5   | Path Sum              | DFS              |
| 6   | Level Order           | BFS              |
| 7   | Validate BST          | BST              |
| 8   | Kth Smallest          | BST              |
| 9   | LCA BST               | BST              |
| 10  | Right Side View       | BFS              |
| 11  | Diameter              | DFS              |
| 12  | Balanced Tree         | DFS              |
| 13  | Serialize/Deserialize | DFS              |
| 14  | Max Path Sum          | DFS              |
| 15  | Construct Tree        | Divide & Conquer |
| 16  | Flatten Tree          | DFS              |
| 17  | Zigzag Level          | BFS              |
| 18  | Merge Trees           | DFS              |
| 19  | Subtree Check         | DFS              |
| 20  | Min Depth             | BFS              |

---

## 🔥 Last-Minute Cramming

```python
# 1. Max Depth
def maxDepth(root):
    return 0 if not root else 1 + max(maxDepth(root.left), maxDepth(root.right))

# 2. Inorder (BST → Sorted)
def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

# 3. Level Order
def levelOrder(root):
    result, queue = [], deque([root]) if root else []
    while queue:
        result.append([node.val for node in queue])
        queue = [child for node in queue for child in (node.left, node.right) if child]
    return result

# 4. Validate BST
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    return not root or (lo < root.val < hi and
                        isValidBST(root.left, lo, root.val) and
                        isValidBST(root.right, root.val, hi))

# 5. Path Sum
def hasPathSum(root, sum):
    return (not root and sum == 0) or (root and (hasPathSum(root.left, sum - root.val) or
                                                   hasPathSum(root.right, sum - root.val)))
```

---

**Good luck! 🌳**

_Print this for interview day!_
