---
title: "Tries - Quick Reference Cheatsheet"
level: "All Levels"
estimated_time: "30 minutes review"
tags: ["trie", "prefix-tree", "cheatsheet", "templates", "patterns"]
---

# Tries - Quick Reference Cheatsheet

## 🎯 Core Concepts

### Trie Properties

1. **Prefix Tree**: Each path from root represents a prefix
2. **Shared Prefixes**: Common prefixes share same path
3. **Character per Edge**: Each edge represents one character
4. **Word Termination**: Special marker for complete words
5. **Root is Empty**: Root node contains no character

### When to Use Tries

- ✅ **Prefix operations** (autocomplete, spell check)
- ✅ **Multiple pattern matching**
- ✅ **Dictionary/vocabulary storage**
- ✅ **String processing with shared prefixes**
- ✅ **IP routing and network applications**

### When NOT to Use Tries

- ❌ **Simple key-value storage** (use hash tables)
- ❌ **Single pattern matching** (use KMP/Boyer-Moore)
- ❌ **Large alphabets** (memory inefficient)
- ❌ **No shared prefixes** (wasteful memory usage)

---

## 🔥 Essential Templates

### 1. Basic Trie Implementation

```python
class TrieNode:
    """Standard trie node structure."""
    def __init__(self):
        self.children = {}  # Or use [None] * 26 for lowercase letters
        self.is_end = False
        self.word = None    # Optional: store complete word

class Trie:
    """
    Basic trie with insert, search, and prefix operations.
    Time: O(m) for all operations where m = word/prefix length
    """
    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, word):
        """Insert word into trie."""
        if not word:
            return

        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if not node.is_end:
            node.is_end = True
            node.word = word
            self.size += 1

    def search(self, word):
        """Search for exact word."""
        if not word:
            return False

        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        """Check if any word starts with prefix."""
        if not prefix:
            return True

        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### 2. Trie with Prefix Collection

```python
def get_all_words_with_prefix(self, prefix):
    """
    Get all words starting with given prefix.
    Time: O(p + n) where p = prefix length, n = results count
    """
    if not prefix:
        return self._get_all_words()

    # Navigate to prefix node
    node = self.root
    for char in prefix:
        if char not in node.children:
            return []
        node = node.children[char]

    # Collect all words from this node
    words = []

    def dfs_collect(curr_node, curr_path):
        if curr_node.is_end:
            words.append(curr_path)

        for char, child in curr_node.children.items():
            dfs_collect(child, curr_path + char)

    dfs_collect(node, prefix)
    return words

def _get_all_words(self):
    """Get all words in trie."""
    return self.get_all_words_with_prefix("")
```

### 3. Trie with Deletion

```python
def delete(self, word):
    """
    Delete word from trie with node cleanup.
    Time: O(m), Space: O(m) for recursion
    """
    def _delete_helper(node, word, index):
        if index == len(word):
            if not node.is_end:
                return False  # Word doesn't exist

            node.is_end = False
            node.word = None

            # Return True if node can be deleted (no children)
            return len(node.children) == 0

        char = word[index]
        child = node.children.get(char)

        if child is None:
            return False  # Word doesn't exist

        should_delete_child = _delete_helper(child, word, index + 1)

        if should_delete_child:
            del node.children[char]
            # Return True if current node can be deleted
            return len(node.children) == 0 and not node.is_end

        return False

    if self.search(word):
        _delete_helper(self.root, word, 0)
        self.size -= 1
        return True
    return False
```

### 4. Wildcard Search Template

```python
def search_with_wildcards(self, pattern):
    """
    Search with wildcard support ('.' matches any character).
    Time: O(m × n) worst case where n = nodes at each level
    """
    def dfs_search(node, pattern, index):
        if index == len(pattern):
            return node.is_end

        char = pattern[index]

        if char == '.':
            # Wildcard: try all possible children
            for child in node.children.values():
                if dfs_search(child, pattern, index + 1):
                    return True
            return False
        else:
            # Regular character
            if char in node.children:
                return dfs_search(node.children[char], pattern, index + 1)
            return False

    return dfs_search(self.root, pattern, 0)
```

### 5. Trie with Frequency/Count

```python
class TrieNodeWithCount:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0          # Word frequency
        self.prefix_count = 0   # Words with this prefix

class TrieWithCount:
    def insert(self, word):
        """Insert word and update counts."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNodeWithCount()
            node = node.children[char]
            node.prefix_count += 1

        node.is_end = True
        node.count += 1

    def count_words_with_prefix(self, prefix):
        """Count words starting with prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.prefix_count

    def get_word_frequency(self, word):
        """Get frequency of specific word."""
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count if node.is_end else 0
```

### 6. Binary Trie Template

```python
class BinaryTrieNode:
    def __init__(self):
        self.children = {}  # 0 and 1 children
        self.count = 0      # Numbers ending here

class BinaryTrie:
    """
    Binary trie for XOR optimization problems.
    Stores numbers as binary representations.
    """
    def __init__(self):
        self.root = BinaryTrieNode()

    def insert(self, num):
        """Insert number into binary trie."""
        node = self.root
        for i in range(31, -1, -1):  # 32 bits, MSB first
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = BinaryTrieNode()
            node = node.children[bit]
        node.count += 1

    def find_max_xor(self, num):
        """Find maximum XOR with any number in trie."""
        node = self.root
        max_xor = 0

        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try opposite bit for maximum XOR
            toggle_bit = 1 - bit

            if toggle_bit in node.children:
                max_xor |= (1 << i)
                node = node.children[toggle_bit]
            else:
                node = node.children[bit]

        return max_xor
```

---

## 📋 Problem Pattern Recognition

### Pattern 1: Autocomplete/Suggestions

**Keywords**: prefix, autocomplete, suggestions, spell check
**Strategy**: Basic trie with prefix collection
**Template**: Trie with prefix collection

**Examples**:

- Search autocomplete
- Spell checker suggestions
- Word completion

### Pattern 2: Multiple String Matching

**Keywords**: multiple patterns, find all, word search
**Strategy**: Build trie of patterns, then search text
**Template**: Basic trie + DFS/backtracking

**Examples**:

- Word Search II
- Stream of Characters
- Multiple pattern matching

### Pattern 3: Word Composition

**Keywords**: concatenate, word break, formed by
**Strategy**: Trie + DP or backtracking for segmentation
**Template**: Trie with word segmentation

**Examples**:

- Word Break II
- Concatenated Words
- Word squares

### Pattern 4: Prefix Operations

**Keywords**: count with prefix, sum with prefix, longest prefix
**Strategy**: Store aggregate data in trie nodes
**Template**: Trie with count/aggregation

**Examples**:

- Map Sum Pairs
- Longest Common Prefix
- Count words with prefix

### Pattern 5: XOR Optimization

**Keywords**: maximum XOR, minimum XOR, bit manipulation
**Strategy**: Binary trie with bit-by-bit processing
**Template**: Binary trie

**Examples**:

- Maximum XOR of Two Numbers
- XOR queries on arrays
- Bit manipulation problems

### Pattern 6: Wildcard/Pattern Matching

**Keywords**: wildcard, pattern, '.' or '\*'
**Strategy**: DFS through trie with pattern logic
**Template**: Wildcard search

**Examples**:

- Add and Search Word
- Regular expression matching
- Pattern matching with wildcards

---

## ⚡ Quick Implementation Patterns

### Memory Optimization Patterns

```python
# Use array for small, fixed alphabet (26 lowercase letters)
class TrieNodeArray:
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

    def _get_index(self, char):
        return ord(char) - ord('a')

# Use dictionary for large/variable alphabet
class TrieNodeDict:
    def __init__(self):
        self.children = {}
        self.is_end = False

# Compressed trie for sparse data
class CompressedTrieNode:
    def __init__(self):
        self.children = {}  # edge_label -> node
        self.is_end = False
```

### Performance Optimization Patterns

```python
# Caching for frequent operations
class CachedTrie:
    def __init__(self):
        self.trie = Trie()
        self.prefix_cache = {}

    def get_words_with_prefix_cached(self, prefix):
        if prefix in self.prefix_cache:
            return self.prefix_cache[prefix]

        result = self.trie.get_all_words_with_prefix(prefix)
        self.prefix_cache[prefix] = result
        return result

# Early termination optimizations
def search_optimized(self, word):
    """Search with early termination."""
    if not word:
        return False

    node = self.root
    for i, char in enumerate(word):
        if char not in node.children:
            return False
        node = node.children[char]

        # Early termination if no words can be formed
        if i < len(word) - 1 and not node.children:
            return False

    return node.is_end
```

### Bulk Operations Patterns

```python
# Batch insertion with optimization
def batch_insert(self, words):
    """Optimized batch insertion."""
    # Sort words to improve cache locality
    words.sort()

    for word in words:
        self.insert(word)

# Batch search
def batch_search(self, words):
    """Search multiple words efficiently."""
    results = {}
    for word in words:
        results[word] = self.search(word)
    return results
```

---

## 🚨 Common Pitfalls & Fixes

### ❌ Mistake 1: Forgetting Word End Marker

```python
# WRONG: No end marker
def search_wrong(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return True  # Wrong! This matches prefixes too

# CORRECT: Check end marker
def search_correct(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end  # Correct!
```

### ❌ Mistake 2: Memory Leak in Deletion

```python
# WRONG: Not cleaning up empty nodes
def delete_wrong(self, word):
    node = self.root
    for char in word:
        node = node.children[char]
    node.is_end = False  # Leaves empty nodes

# CORRECT: Recursive cleanup
def delete_correct(self, word):
    def helper(node, word, index):
        if index == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            return len(node.children) == 0

        char = word[index]
        child = node.children.get(char)
        if child is None:
            return False

        should_delete = helper(child, word, index + 1)
        if should_delete:
            del node.children[char]
            return len(node.children) == 0 and not node.is_end
        return False

    helper(self.root, word, 0)
```

### ❌ Mistake 3: Inefficient Wildcard Handling

```python
# WRONG: Not pruning impossible paths
def search_wildcard_wrong(self, pattern):
    def dfs(node, index):
        if index == len(pattern):
            return node.is_end

        char = pattern[index]
        if char == '.':
            for child in node.children.values():
                if dfs(child, index + 1):  # Explores all paths
                    return True
        # ... rest of logic

# BETTER: Add pruning logic
def search_wildcard_better(self, pattern):
    def dfs(node, index):
        if index == len(pattern):
            return node.is_end

        # Early termination if remaining pattern can't be satisfied
        remaining = len(pattern) - index
        if not self._can_satisfy(node, remaining):
            return False

        # ... rest of logic with pruning
```

### ❌ Mistake 4: Wrong Character Mapping

```python
# WRONG: Assuming lowercase letters only
def get_index_wrong(self, char):
    return ord(char) - ord('a')  # Fails for uppercase/numbers

# CORRECT: Validate input or use dictionary
def get_index_correct(self, char):
    if 'a' <= char <= 'z':
        return ord(char) - ord('a')
    else:
        raise ValueError(f"Invalid character: {char}")

# OR use dictionary for general case
class GeneralTrie:
    def __init__(self):
        self.children = {}  # Handles any character
```

---

## 🔧 Complexity Quick Reference

| Operation       | Time Complexity | Space Complexity | Notes                  |
| --------------- | --------------- | ---------------- | ---------------------- |
| Insert          | O(m)            | O(m)             | m = word length        |
| Search          | O(m)            | O(1)             | Fast exact match       |
| Prefix Check    | O(p)            | O(1)             | p = prefix length      |
| Delete          | O(m)            | O(m)             | Recursive cleanup      |
| Get All Words   | O(total_chars)  | O(total_chars)   | DFS traversal          |
| Wildcard Search | O(m × branches) | O(m)             | Exponential worst case |

### Space Complexity Analysis

```python
# Worst case: O(ALPHABET_SIZE × N × M)
# - ALPHABET_SIZE: size of character set
# - N: number of words
# - M: average word length

# Best case: O(M) where M is length of longest word
# - When all words share maximum prefixes

# Typical case: Depends on prefix sharing
# - Real dictionaries have good prefix sharing
# - Random strings have poor prefix sharing
```

---

## 🎯 Interview Quick Tips

### 1. Recognition Phase (30 seconds)

- Look for prefix-related keywords
- Multiple string operations
- Autocomplete/suggestion features
- Word composition problems

### 2. Strategy Selection (1 minute)

- **Basic operations** → Standard trie
- **Prefix aggregation** → Trie with counts/sums
- **Wildcard matching** → DFS with backtracking
- **XOR optimization** → Binary trie
- **Word segmentation** → Trie + DP

### 3. Implementation Tips (5-10 minutes)

- Start with TrieNode structure
- Add problem-specific data to nodes
- Implement core operations first
- Add optimizations if needed

### 4. Testing Strategy

```python
# Essential test cases for trie problems:
test_cases = [
    "",                    # Empty string
    "a",                   # Single character
    ["app", "apple"],      # Prefix relationship
    ["cat", "cats"],       # One word prefix of another
    ["hello", "world"],    # No common prefix
    duplicate_words,       # Duplicate insertions
]
```

### 5. Optimization Opportunities

- **Memory**: Array vs dictionary based on alphabet size
- **Time**: Caching for frequent prefix operations
- **Space**: Compressed tries for sparse data
- **Functionality**: Store additional metadata in nodes

---

## 💡 Problem-Solving Mindset

### Questions to Ask Yourself:

1. **Do I need prefix operations?** (If yes, likely trie)
2. **Are there multiple patterns to match?** (Trie + search)
3. **Do I need to aggregate data by prefix?** (Trie with counts)
4. **Is this about bit manipulation?** (Binary trie)
5. **Do I need wildcard matching?** (DFS through trie)

### Implementation Strategy:

1. **Design node structure** based on requirements
2. **Choose character mapping** (array vs dictionary)
3. **Implement core operations** (insert, search, prefix)
4. **Add problem-specific logic** (aggregation, wildcards, etc.)
5. **Optimize for constraints** (memory, time, functionality)

---

_This cheatsheet covers 90% of trie-related interview questions. Master these patterns and you'll handle any trie problem with confidence!_
