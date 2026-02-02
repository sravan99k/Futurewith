# 🌳 Trees & BST - Practice Problems

> **Total Problems:** 100+ | **Difficulty:** Easy → Medium → Hard
> **Time:** 30-40 hours | **Pattern-Based Learning**

---

## 📋 Table of Contents

1. [Easy Problems (1-35)](#easy-problems)
2. [Medium Problems (36-75)](#medium-problems)
3. [Hard Problems (76-100)](#hard-problems)

---

## 🟢 EASY PROBLEMS (1-35)

### **Problem 1: Maximum Depth of Binary Tree** ⭐⭐⭐

**LeetCode:** 104 | **Time:** 10 min

```python
def maxDepth(root):
    """Find maximum depth (height) of tree"""
    if not root:
        return 0

    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    return 1 + max(left_depth, right_depth)

# Time: O(n), Space: O(h)
```

### **Problem 2: Invert Binary Tree** ⭐⭐⭐

**LeetCode:** 226 | **Asked by:** Google (200+ times)

```python
def invertTree(root):
    """
    Swap left and right children of all nodes

    Example:
         4              4
       /   \          /   \
      2     7   →    7     2
     / \   / \      / \   / \
    1   3 6   9    9   6 3   1
    """
    if not root:
        return None

    # Swap children
    root.left, root.right = root.right, root.left

    # Recursively invert subtrees
    invertTree(root.left)
    invertTree(root.right)

    return root

# Time: O(n), Space: O(h)
```

### **Problem 3: Same Tree** ⭐⭐

**LeetCode:** 100

```python
def isSameTree(p, q):
    """Check if two trees are identical"""
    # Both null
    if not p and not q:
        return True

    # One null, other not
    if not p or not q:
        return False

    # Check value and recurse
    return (p.val == q.val and
            isSameTree(p.left, q.left) and
            isSameTree(p.right, q.right))

# Time: O(n), Space: O(h)
```

### **Problem 4: Symmetric Tree** ⭐⭐

**LeetCode:** 101

```python
def isSymmetric(root):
    """Check if tree is mirror of itself"""
    def is_mirror(left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False

        return (left.val == right.val and
                is_mirror(left.left, right.right) and
                is_mirror(left.right, right.left))

    return is_mirror(root, root) if root else True

# Time: O(n), Space: O(h)
```

### **Problem 5: Path Sum** ⭐⭐

**LeetCode:** 112

```python
def hasPathSum(root, targetSum):
    """Check if root-to-leaf path with sum exists"""
    if not root:
        return False

    # Leaf node
    if not root.left and not root.right:
        return root.val == targetSum

    remaining = targetSum - root.val
    return (hasPathSum(root.left, remaining) or
            hasPathSum(root.right, remaining))

# Time: O(n), Space: O(h)
```

### **Problem 6: Minimum Depth** ⭐⭐

**LeetCode:** 111

```python
def minDepth(root):
    """Find minimum depth to nearest leaf"""
    if not root:
        return 0

    # If one subtree is null, return other
    if not root.left:
        return 1 + minDepth(root.right)
    if not root.right:
        return 1 + minDepth(root.left)

    return 1 + min(minDepth(root.left), minDepth(root.right))

# Time: O(n), Space: O(h)
```

### **Problem 7: Balanced Binary Tree** ⭐⭐

**LeetCode:** 110

```python
def isBalanced(root):
    """Check if tree is height-balanced"""
    def height(node):
        if not node:
            return 0

        left = height(node.left)
        right = height(node.right)

        # If unbalanced, return -1
        if left == -1 or right == -1 or abs(left - right) > 1:
            return -1

        return 1 + max(left, right)

    return height(root) != -1

# Time: O(n), Space: O(h)
```

### **Problem 8: Merge Two Binary Trees** ⭐⭐

**LeetCode:** 617

```python
def mergeTrees(t1, t2):
    """Merge by summing overlapping nodes"""
    if not t1:
        return t2
    if not t2:
        return t1

    t1.val += t2.val
    t1.left = mergeTrees(t1.left, t2.left)
    t1.right = mergeTrees(t1.right, t2.right)

    return t1

# Time: O(min(m,n)), Space: O(min(m,n))
```

### **Problem 9: Diameter of Binary Tree** ⭐⭐⭐

**LeetCode:** 543

```python
def diameterOfBinaryTree(root):
    """Find longest path between any two nodes"""
    self.diameter = 0

    def height(node):
        if not node:
            return 0

        left = height(node.left)
        right = height(node.right)

        # Update diameter
        self.diameter = max(self.diameter, left + right)

        return 1 + max(left, right)

    height(root)
    return self.diameter

# Time: O(n), Space: O(h)
```

### **Problem 10: Subtree of Another Tree** ⭐⭐

**LeetCode:** 572

```python
def isSubtree(s, t):
    """Check if t is subtree of s"""
    if not s:
        return False

    if isSameTree(s, t):
        return True

    return isSubtree(s.left, t) or isSubtree(s.right, t)

# Time: O(m * n), Space: O(h)
```

---

## 🟡 MEDIUM PROBLEMS (36-75)

### **Problem 36: Binary Tree Level Order Traversal** ⭐⭐⭐

**LeetCode:** 102

```python
from collections import deque

def levelOrder(root):
    """Return level-order traversal as list of lists"""
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

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result

# Time: O(n), Space: O(n)
```

### **Problem 37: Validate Binary Search Tree** ⭐⭐⭐

**LeetCode:** 98 | **FAANG Favorite**

```python
def isValidBST(root):
    """Check if valid BST using range validation"""
    def validate(node, min_val, max_val):
        if not node:
            return True

        if not (min_val < node.val < max_val):
            return False

        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))

# Time: O(n), Space: O(h)
```

### **Problem 38: Kth Smallest in BST** ⭐⭐⭐

**LeetCode:** 230

```python
def kthSmallest(root, k):
    """Find kth smallest element using inorder traversal"""
    self.count = 0
    self.result = None

    def inorder(node):
        if not node or self.result is not None:
            return

        inorder(node.left)

        self.count += 1
        if self.count == k:
            self.result = node.val
            return

        inorder(node.right)

    inorder(root)
    return self.result

# Time: O(n), Space: O(h)
```

### **Problem 39: Lowest Common Ancestor of BST** ⭐⭐⭐

**LeetCode:** 235

```python
def lowestCommonAncestor(root, p, q):
    """Find LCA in BST using BST property"""
    # Both in left subtree
    if p.val < root.val and q.val < root.val:
        return lowestCommonAncestor(root.left, p, q)

    # Both in right subtree
    if p.val > root.val and q.val > root.val:
        return lowestCommonAncestor(root.right, p, q)

    # Split point
    return root

# Time: O(log n), Space: O(h)
```

### **Problem 40: Binary Tree Right Side View** ⭐⭐

**LeetCode:** 199

```python
def rightSideView(root):
    """Return values visible from right side"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.popleft()

            # Last node in level
            if i == level_size - 1:
                result.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result

# Time: O(n), Space: O(n)
```

### **Problem 41: Construct Binary Tree from Preorder and Inorder** ⭐⭐⭐

**LeetCode:** 105

```python
def buildTree(preorder, inorder):
    """
    Construct tree from traversals
    Preorder: root first
    Inorder: root in middle
    """
    if not preorder or not inorder:
        return None

    # Root is first in preorder
    root = TreeNode(preorder[0])

    # Find root in inorder
    mid = inorder.index(root.val)

    # Recursively build subtrees
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])

    return root

# Time: O(n), Space: O(n)
```

### **Problem 42: Flatten Binary Tree to Linked List** ⭐⭐

**LeetCode:** 114

```python
def flatten(root):
    """Flatten tree to linked list (preorder)"""
    if not root:
        return

    # Flatten subtrees
    flatten(root.left)
    flatten(root.right)

    # Save right subtree
    right_subtree = root.right

    # Move left to right
    root.right = root.left
    root.left = None

    # Find end and attach saved right
    current = root
    while current.right:
        current = current.right
    current.right = right_subtree

# Time: O(n), Space: O(h)
```

### **Problem 43: Path Sum II** ⭐⭐

**LeetCode:** 113

```python
def pathSum(root, targetSum):
    """Find all root-to-leaf paths with sum"""
    result = []

    def dfs(node, remaining, path):
        if not node:
            return

        path.append(node.val)

        # Leaf node with target sum
        if not node.left and not node.right and node.val == remaining:
            result.append(path[:])

        dfs(node.left, remaining - node.val, path)
        dfs(node.right, remaining - node.val, path)

        path.pop()  # Backtrack

    dfs(root, targetSum, [])
    return result

# Time: O(n), Space: O(h)
```

### **Problem 44: Binary Tree Zigzag Level Order** ⭐⭐

**LeetCode:** 103

```python
def zigzagLevelOrder(root):
    """Level order but alternate direction each level"""
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        current_level = deque()

        for _ in range(level_size):
            node = queue.popleft()

            if left_to_right:
                current_level.append(node.val)
            else:
                current_level.appendleft(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(current_level))
        left_to_right = not left_to_right

    return result

# Time: O(n), Space: O(n)
```

### **Problem 45: Count Complete Tree Nodes** ⭐⭐

**LeetCode:** 222

```python
def countNodes(root):
    """
    Count nodes in complete binary tree
    Optimized using complete tree property
    """
    if not root:
        return 0

    # Get left and right heights
    left_height = get_height(root.left)
    right_height = get_height(root.right)

    if left_height == right_height:
        # Left is perfect, count + recurse right
        return (1 << left_height) + countNodes(root.right)
    else:
        # Right is perfect, count + recurse left
        return (1 << right_height) + countNodes(root.left)

def get_height(node):
    height = 0
    while node:
        height += 1
        node = node.left
    return height

# Time: O(log² n), Space: O(log n)
```

---

## 🔴 HARD PROBLEMS (76-100)

### **Problem 76: Serialize and Deserialize Binary Tree** ⭐⭐⭐

**LeetCode:** 297 | **FAANG Favorite**

```python
class Codec:
    def serialize(self, root):
        """Convert tree to string"""
        if not root:
            return "null"

        return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"

    def deserialize(self, data):
        """Convert string to tree"""
        def helper(nodes):
            val = next(nodes)
            if val == "null":
                return None

            node = TreeNode(int(val))
            node.left = helper(nodes)
            node.right = helper(nodes)
            return node

        return helper(iter(data.split(',')))

# Time: O(n), Space: O(n)
```

### **Problem 77: Binary Tree Maximum Path Sum** ⭐⭐⭐

**LeetCode:** 124 | **Hard but Common**

```python
def maxPathSum(root):
    """
    Find maximum path sum (any node to any node)
    """
    self.max_sum = float('-inf')

    def max_gain(node):
        if not node:
            return 0

        # Only take positive gains
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        # Path through current node
        path_sum = node.val + left_gain + right_gain
        self.max_sum = max(self.max_sum, path_sum)

        # Return max gain if continue path
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return self.max_sum

# Time: O(n), Space: O(h)
```

### **Problem 78: Recover Binary Search Tree** ⭐⭐⭐

**LeetCode:** 99

```python
def recoverTree(root):
    """Fix BST where two nodes are swapped"""
    self.first = self.second = self.prev = None

    def inorder(node):
        if not node:
            return

        inorder(node.left)

        # Find two swapped nodes
        if self.prev and self.prev.val > node.val:
            if not self.first:
                self.first = self.prev
            self.second = node

        self.prev = node
        inorder(node.right)

    inorder(root)
    self.first.val, self.second.val = self.second.val, self.first.val

# Time: O(n), Space: O(h)
```

### **Problem 79: Binary Tree Cameras** ⭐⭐⭐

**LeetCode:** 968

```python
def minCameraCover(root):
    """Minimum cameras to monitor all nodes"""
    self.cameras = 0

    def dfs(node):
        if not node:
            return 2  # Covered

        left = dfs(node.left)
        right = dfs(node.right)

        # If child not covered, need camera
        if left == 0 or right == 0:
            self.cameras += 1
            return 1  # Has camera

        # If child has camera, current is covered
        if left == 1 or right == 1:
            return 2  # Covered

        return 0  # Not covered

    return (self.cameras + 1) if dfs(root) == 0 else self.cameras

# Time: O(n), Space: O(h)
```

### **Problem 80: Vertical Order Traversal** ⭐⭐⭐

**LeetCode:** 987

```python
from collections import defaultdict, deque

def verticalTraversal(root):
    """Return vertical order traversal"""
    if not root:
        return []

    # column -> [(row, value)]
    columns = defaultdict(list)
    queue = deque([(root, 0, 0)])  # (node, row, col)

    while queue:
        node, row, col = queue.popleft()
        columns[col].append((row, node.val))

        if node.left:
            queue.append((node.left, row + 1, col - 1))
        if node.right:
            queue.append((node.right, row + 1, col + 1))

    result = []
    for col in sorted(columns.keys()):
        # Sort by row, then value
        column_vals = [val for row, val in sorted(columns[col])]
        result.append(column_vals)

    return result

# Time: O(n log n), Space: O(n)
```

---

## 📚 Additional Problems (81-100)

**Problem 81:** All Nodes Distance K (LeetCode 863)
**Problem 82:** Delete Nodes and Return Forest (LeetCode 1110)
**Problem 83:** Maximum Difference Between Node and Ancestor (LeetCode 1026)
**Problem 84:** Sum Root to Leaf Numbers (LeetCode 129)
**Problem 85:** House Robber III (LeetCode 337)
**Problem 86:** Unique BST II (LeetCode 95)
**Problem 87:** Populating Next Right Pointers (LeetCode 116)
**Problem 88:** Sum of Left Leaves (LeetCode 404)
**Problem 89:** Find Duplicate Subtrees (LeetCode 652)
**Problem 90:** Trim BST (LeetCode 669)
**Problem 91:** Binary Tree Tilt (LeetCode 563)
**Problem 92:** Second Minimum Node (LeetCode 671)
**Problem 93:** Most Frequent Subtree Sum (LeetCode 508)
**Problem 94:** Binary Tree Pruning (LeetCode 814)
**Problem 95:** Distribute Coins (LeetCode 979)
**Problem 96:** Longest Univalue Path (LeetCode 687)
**Problem 97:** Maximum Binary Tree (LeetCode 654)
**Problem 98:** Find Bottom Left Value (LeetCode 513)
**Problem 99:** Add One Row (LeetCode 623)
**Problem 100:** Convert BST to Greater Tree (LeetCode 538)

---

## 🎯 Problem Patterns

### **Pattern 1: DFS (Recursion)**

- Max Depth, Min Depth
- Path Sum, Diameter
- Validate BST

### **Pattern 2: BFS (Level Order)**

- Level Order Traversal
- Right Side View
- Zigzag Level Order

### **Pattern 3: Divide & Conquer**

- Build Tree from Traversals
- Serialize/Deserialize
- Maximum Path Sum

### **Pattern 4: BST Properties**

- Validate BST
- Kth Smallest
- LCA in BST

---

**Practice Strategy:**

- Week 1: Easy problems (1-35)
- Week 2: Medium problems (36-75)
- Week 3: Hard problems (76-100)

**Good luck! 🌳**
