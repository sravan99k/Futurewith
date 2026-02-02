# 🎯 Trees & BST - FAANG Interview Questions

> **Target:** Google, Meta, Amazon, Apple, Microsoft
> **Success Rate:** Master these for 90% of tree interviews

---

## 🏢 Company-Specific Questions

### **Google - Loves Complex Tree Problems**

#### Q1: Binary Tree Maximum Path Sum ⭐⭐⭐

**Asked:** 300+ times | **LeetCode:** 124

```python
def maxPathSum(root):
    """
    Find maximum sum path between any two nodes.

    Google tests: recursion, global state, edge cases
    """
    self.max_sum = float('-inf')

    def max_gain(node):
        if not node:
            return 0

        # Only take positive gains
        left = max(max_gain(node.left), 0)
        right = max(max_gain(node.right), 0)

        # Path through current node
        price_newpath = node.val + left + right
        self.max_sum = max(self.max_sum, price_newpath)

        # For recursion, return max single path
        return node.val + max(left, right)

    max_gain(root)
    return self.max_sum
```

**Follow-up:** What if we need the actual path?

---

#### Q2: Serialize and Deserialize Binary Tree ⭐⭐⭐

**Asked:** 250+ times | **LeetCode:** 297

```python
class Codec:
    """
    Google asks: multiple serialization strategies
    1. Preorder with null markers
    2. Level-order
    3. Compact encoding
    """
    def serialize(self, root):
        if not root:
            return "null"
        return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"

    def deserialize(self, data):
        def helper(nodes):
            val = next(nodes)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = helper(nodes)
            node.right = helper(nodes)
            return node
        return helper(iter(data.split(',')))
```

---

### **Meta - Design & System Integration**

#### Q3: Binary Tree Vertical Order Traversal ⭐⭐

**Asked:** 150+ times | **LeetCode:** 314

```python
from collections import defaultdict, deque

def verticalOrder(root):
    """
    Meta asks: handle ties, optimize space
    """
    if not root:
        return []

    column_table = defaultdict(list)
    queue = deque([(root, 0)])

    while queue:
        node, column = queue.popleft()
        column_table[column].append(node.val)

        if node.left:
            queue.append((node.left, column - 1))
        if node.right:
            queue.append((node.right, column + 1))

    return [column_table[x] for x in sorted(column_table.keys())]
```

---

### **Amazon - Practical Applications**

#### Q4: Lowest Common Ancestor ⭐⭐⭐

**Asked:** 200+ times | **LeetCode:** 236

```python
def lowestCommonAncestor(root, p, q):
    """
    Amazon uses in: org charts, file systems

    Cases:
    1. Both in left → recurse left
    2. Both in right → recurse right
    3. Split → current is LCA
    """
    if not root or root == p or root == q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root  # Split point

    return left if left else right
```

---

## 🎯 Top 25 FAANG Questions

| #   | Problem               | Company   | Difficulty |
| --- | --------------------- | --------- | ---------- |
| 1   | Validate BST          | All       | Medium     |
| 2   | Max Depth             | All       | Easy       |
| 3   | Invert Tree           | Google    | Easy       |
| 4   | LCA                   | Amazon    | Medium     |
| 5   | Serialize/Deserialize | Google    | Hard       |
| 6   | Max Path Sum          | Google    | Hard       |
| 7   | Level Order           | All       | Medium     |
| 8   | Kth Smallest BST      | Meta      | Medium     |
| 9   | Right Side View       | Amazon    | Medium     |
| 10  | Diameter              | Meta      | Easy       |
| 11  | Path Sum              | Microsoft | Easy       |
| 12  | Symmetric Tree        | Microsoft | Easy       |
| 13  | Flatten Tree          | Amazon    | Medium     |
| 14  | Construct Tree        | Google    | Medium     |
| 15  | Vertical Order        | Meta      | Medium     |
| 16  | Zigzag Level          | Amazon    | Medium     |
| 17  | Balanced Tree         | All       | Easy       |
| 18  | Same Tree             | All       | Easy       |
| 19  | Merge Trees           | Amazon    | Easy       |
| 20  | Subtree Check         | Google    | Easy       |
| 21  | Count Complete Nodes  | Google    | Medium     |
| 22  | Binary Tree Cameras   | Google    | Hard       |
| 23  | Recover BST           | Amazon    | Medium     |
| 24  | All Nodes Distance K  | Meta      | Medium     |
| 25  | Delete Nodes Forest   | Amazon    | Medium     |

---

## 💡 Interview Strategy

### **Step-by-Step Approach**

**1. Clarify (30 sec)**

- Is it BST or regular tree?
- Can nodes have duplicate values?
- Can I modify the tree?
- What to return if tree is empty?

**2. Examples (1 min)**

```
Draw example:
    3
   / \
  5   1
 / \   \
6   2   8

Edge cases:
- Empty tree
- Single node
- Skewed tree (like linked list)
```

**3. Approach (2 min)**

- DFS or BFS?
- Recursive or iterative?
- Need extra data structure?
- Time/space complexity?

**4. Code (10-15 min)**

- Start with base case
- Handle recursion
- Test with example

**5. Test (2 min)**

- Run through example
- Check edge cases
- Verify complexity

---

## 🔥 Common Patterns

### **Pattern 1: Validate BST**

```python
def validate(node, min_val, max_val):
    if not node:
        return True
    if not (min_val < node.val < max_val):
        return False
    return (validate(node.left, min_val, node.val) and
            validate(node.right, node.val, max_val))
```

### **Pattern 2: Path Problems**

```python
def hasPath(root, target, path=[]):
    if not root:
        return False

    path.append(root.val)

    if not root.left and not root.right and root.val == target:
        return True

    if (hasPath(root.left, target - root.val, path) or
        hasPath(root.right, target - root.val, path)):
        return True

    path.pop()  # Backtrack
    return False
```

### **Pattern 3: Level Processing**

```python
def processLevels(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        result.append(level)
    return result
```

---

## 🎓 What Interviewers Look For

**1. Understanding (30%)**

- Do you understand the problem?
- Can you explain the approach?
- Do you consider edge cases?

**2. Code Quality (40%)**

- Clean, readable code
- Proper variable names
- Handles edge cases
- No bugs

**3. Optimization (20%)**

- Know time/space complexity
- Can improve brute force
- Understand tradeoffs

**4. Communication (10%)**

- Think out loud
- Ask clarifying questions
- Explain while coding

---

## ⚠️ Common Mistakes

❌ **Not handling null**

```python
# WRONG
height = max(height(node.left), height(node.right))

# RIGHT
if not node: return 0
height = max(height(node.left), height(node.right))
```

❌ **Wrong BST validation**

```python
# WRONG - only checks immediate children
if root.left.val < root.val and root.right.val > root.val

# RIGHT - checks entire subtree
validate(node, min_val, max_val)
```

❌ **Forgetting to return**

```python
# WRONG
def search(root, val):
    if root.val == val:
        return root
    search(root.left, val)  # Missing return!

# RIGHT
return search(root.left, val)
```

---

## 🚀 Preparation Checklist

**Week 1: Fundamentals**

- [ ] Understand tree structure
- [ ] Master all traversals
- [ ] Solve 10 easy problems
- [ ] Practice drawing trees

**Week 2: Patterns**

- [ ] BST operations
- [ ] Path problems
- [ ] Level-order variations
- [ ] Solve 15 medium problems

**Week 3: Advanced**

- [ ] Serialize/deserialize
- [ ] Max path sum
- [ ] Design problems
- [ ] Solve 5 hard problems

**Final Week: Mock Interviews**

- [ ] Solve random problems timed
- [ ] Explain solutions out loud
- [ ] Practice on whiteboard
- [ ] Review mistakes

---

**Success Formula:**
Understanding + Practice + Communication = FAANG Offer! 🎯

Good luck with your interviews! 🌳
