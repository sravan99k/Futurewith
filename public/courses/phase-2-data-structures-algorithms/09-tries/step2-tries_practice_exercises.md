---
title: "Tries - Practice Problems"
level: "Intermediate to Advanced"
estimated_time: "6-8 hours"
tags: ["trie", "prefix-tree", "string-algorithms", "practice", "interview"]
---

# Tries - Practice Problems

## 🎯 Learning Objectives

- Master trie implementation through progressive practice
- Solve complex string processing problems efficiently
- Build intuition for trie-based optimizations
- Prepare for trie-related interview questions

## 🌟 Practice Philosophy

> "Tries are like organizing a massive library where each book title shares paths with others - the deeper the shared path, the more related the books. Master this organization, and you can find any book instantly!"

---

## 🔥 Easy Problems (Foundation Building)

### Problem 1: Basic Trie Implementation

**Classic Trie Data Structure**

```python
class TrieNode:
    """Basic trie node implementation."""
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    """
    Basic trie implementation for LeetCode-style problems.

    Operations:
    - insert(word): Insert word into trie
    - search(word): Return True if word exists
    - startsWith(prefix): Return True if prefix exists

    Time Complexity: O(m) for all operations where m = word/prefix length
    Space Complexity: O(ALPHABET_SIZE × N × M) worst case
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Insert word into trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        """Search for complete word."""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def startsWith(self, prefix):
        """Check if any word starts with prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# Test the implementation
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))   # True
print(trie.search("app"))     # False
print(trie.startsWith("app")) # True
trie.insert("app")
print(trie.search("app"))     # True
```

### Problem 2: Word Search II (Trie + Backtracking)

**LeetCode 212 - Multi-word board search**

```python
def findWords(board, words):
    """
    Find all words from word list that can be constructed from board.

    Strategy: Build trie from words, then DFS on board following trie paths
    Time: O(M×N×4^L + W×L) where M×N=board size, L=max word length, W=words count
    Space: O(W×L) for trie
    """
    # Build trie from words
    trie = {}
    for word in words:
        node = trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = word  # Store complete word at end

    def dfs(i, j, parent):
        """DFS on board following trie structure."""
        letter = board[i][j]
        curr_node = parent[letter]

        # Check if we found a complete word
        word = curr_node.pop('#', False)
        if word:
            result.append(word)

        # Mark current cell as visited
        board[i][j] = '*'

        # Explore all 4 directions
        for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
            ni, nj = i + di, j + dj

            if (0 <= ni < rows and 0 <= nj < cols and
                board[ni][nj] in curr_node):
                dfs(ni, nj, curr_node)

        # Restore current cell
        board[i][j] = letter

        # Optimization: remove empty trie branches
        if not curr_node:
            parent.pop(letter)

    result = []
    rows, cols = len(board), len(board[0])

    # Start DFS from each cell that has a valid first character
    for i in range(rows):
        for j in range(cols):
            if board[i][j] in trie:
                dfs(i, j, trie)

    return result

# Test
board = [
    ['o','a','a','n'],
    ['e','t','a','e'],
    ['i','h','k','r'],
    ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]
print(findWords(board, words))  # ['oath', 'eat']
```

### Problem 3: Replace Words

**LeetCode 648 - Dictionary root replacement**

```python
def replaceWords(dictionary, sentence):
    """
    Replace words with their dictionary roots (shortest prefix match).

    Example: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
    Output: "the cat was rat by the bat"

    Time: O(N + M×L) where N=dict words, M=sentence words, L=avg length
    Space: O(N×L) for trie
    """
    # Build trie from dictionary
    trie = {}
    for root in dictionary:
        node = trie
        for char in root:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = root  # Mark end of root word

    def find_root(word):
        """Find shortest root for given word."""
        node = trie
        for i, char in enumerate(word):
            if char not in node:
                return word  # No root found, return original
            node = node[char]
            if '#' in node:
                return node['#']  # Found root, return it
        return word  # Word is shorter than any root

    # Replace each word in sentence
    words = sentence.split()
    return ' '.join(find_root(word) for word in words)

# Test
dictionary = ["cat","bat","rat"]
sentence = "the cattle was rattled by the battery"
print(replaceWords(dictionary, sentence))
# Output: "the cat was rat by the bat"
```

### Problem 4: Longest Word in Dictionary

**LeetCode 720 - Build word character by character**

```python
def longestWord(words):
    """
    Find longest word that can be built one character at a time.

    Strategy: Build trie, then DFS to find longest buildable word
    Time: O(N×L) where N=words count, L=average length
    Space: O(N×L) for trie
    """
    # Build trie
    trie = {}
    for word in words:
        node = trie
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['word'] = word

    def dfs(node):
        """DFS to find longest buildable word."""
        # Can only continue if current position has a complete word
        # (except for root which has no word)
        if node != trie and 'word' not in node:
            return ""

        longest = node.get('word', "")

        # Try all children
        for char_node in node.values():
            if isinstance(char_node, dict):  # Skip 'word' key
                candidate = dfs(char_node)
                if len(candidate) > len(longest) or \
                   (len(candidate) == len(longest) and candidate < longest):
                    longest = candidate

        return longest

    return dfs(trie)

# Alternative iterative solution
def longestWordIterative(words):
    """
    Iterative solution using sorting and set.
    """
    words_set = set(words)
    words.sort(key=lambda x: (-len(x), x))  # Sort by length desc, then lexicographically

    for word in words:
        if all(word[:i] in words_set for i in range(1, len(word))):
            return word

    return ""

# Test
words = ["w","wo","wor","worl","world"]
print(longestWord(words))  # "world"

words2 = ["a","banana","app","appl","ap","apply","apple"]
print(longestWord(words2))  # "apple"
```

---

## 🔥 Medium Problems (Building Intuition)

### Problem 5: Add and Search Word (Wildcard Support)

**LeetCode 211 - Trie with wildcard matching**

```python
class WordDictionary:
    """
    Data structure supporting add and search with wildcard '.' support.

    Operations:
    - addWord(word): Add word to dictionary
    - search(word): Search word (may contain '.' as wildcard)
    """

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_end = False

    def __init__(self):
        self.root = self.TrieNode()

    def addWord(self, word):
        """Add word to trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        """
        Search word with wildcard support.
        '.' matches any single character.

        Time: O(M×N) worst case where M=word length, N=trie nodes
        """
        def dfs(node, index):
            if index == len(word):
                return node.is_end

            char = word[index]

            if char == '.':
                # Try all possible children for wildcard
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # Regular character match
                if char in node.children:
                    return dfs(node.children[char], index + 1)
                return False

        return dfs(self.root, 0)

# Test
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("dad")
wd.addWord("mad")
print(wd.search("pad"))  # False
print(wd.search("bad"))  # True
print(wd.search(".ad"))  # True
print(wd.search("b.."))  # True
```

### Problem 6: Map Sum Pairs

**LeetCode 677 - Prefix sum with trie**

```python
class MapSum:
    """
    Data structure that supports:
    - insert(key, val): Insert key-value pair (or update existing)
    - sum(prefix): Return sum of all values with given prefix
    """

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.value = 0  # Store value at this node
            self.sum = 0    # Store prefix sum

    def __init__(self):
        self.root = self.TrieNode()
        self.key_values = {}  # Track previous values for updates

    def insert(self, key, val):
        """
        Insert key-value pair with prefix sum update.

        Time: O(M) where M = key length
        """
        # Calculate delta for updates
        delta = val - self.key_values.get(key, 0)
        self.key_values[key] = val

        # Update trie with delta
        node = self.root
        node.sum += delta

        for char in key:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
            node.sum += delta

        node.value = val

    def sum(self, prefix):
        """
        Get sum of all values with given prefix.

        Time: O(M) where M = prefix length
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.sum

# Alternative implementation without prefix sums
class MapSumAlternative:
    def __init__(self):
        self.trie = {}
        self.values = {}

    def insert(self, key, val):
        self.values[key] = val
        node = self.trie
        for char in key:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True

    def sum(self, prefix):
        def dfs(node, current_key):
            total = 0
            if '#' in node and current_key in self.values:
                total += self.values[current_key]

            for char, child in node.items():
                if char != '#':
                    total += dfs(child, current_key + char)

            return total

        node = self.trie
        for char in prefix:
            if char not in node:
                return 0
            node = node[char]

        return dfs(node, prefix)

# Test
ms = MapSum()
ms.insert("apple", 3)
print(ms.sum("ap"))     # 3
ms.insert("app", 2)
print(ms.sum("ap"))     # 5
```

### Problem 7: Palindrome Pairs

**LeetCode 336 - Complex trie with reverse matching**

```python
def palindromePairs(words):
    """
    Find all palindrome pairs from word array.

    Strategy: Build trie of reversed words, then for each word check:
    1. If reverse exists in trie
    2. If word + suffix forms palindrome (suffix reverse exists)
    3. If prefix + word forms palindrome (prefix reverse exists)

    Time: O(N×M²) where N=words count, M=max word length
    Space: O(N×M) for trie
    """
    def is_palindrome(s):
        return s == s[::-1]

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word_index = -1  # Index of word ending here
            self.palindrome_indices = []  # Indices of words that form palindrome with path to here

    # Build trie with reversed words
    root = TrieNode()

    for i, word in enumerate(words):
        node = root
        # Add indices where remaining suffix is palindrome
        for j in range(len(word)):
            if is_palindrome(word[j:]):
                node.palindrome_indices.append(i)

            char = word[len(word) - 1 - j]  # Reverse order
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.word_index = i
        node.palindrome_indices.append(i)  # Empty suffix is always palindrome

    result = []

    # Search for palindrome pairs
    for i, word in enumerate(words):
        node = root

        # Case 1 & 2: Current word + something from trie
        for j in range(len(word)):
            # Case 1: Exact reverse found and remaining is palindrome
            if node.word_index != -1 and node.word_index != i and is_palindrome(word[j:]):
                result.append([i, node.word_index])

            if word[j] not in node.children:
                break
            node = node.children[word[j]]
        else:
            # Case 2: Whole word matched, check palindrome suffixes
            for idx in node.palindrome_indices:
                if idx != i:
                    result.append([i, idx])

    return result

# Optimized version with better palindrome checking
def palindromePairsOptimized(words):
    """
    Optimized version with precomputed palindrome information.
    """
    def manacher_palindrome_check(s):
        """Precompute all palindrome substrings using Manacher's algorithm."""
        # Implementation omitted for brevity
        # Returns boolean array indicating palindrome positions
        pass

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.word_index = -1
            self.palindrome_word_indices = []

    # Build reverse trie
    root = TrieNode()
    for i, word in enumerate(words):
        node = root
        for j in range(len(word)):
            # Check if suffix word[j:] is palindrome
            if word[j:] == word[j:][::-1]:
                node.palindrome_word_indices.append(i)

            char = word[len(word) - 1 - j]
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.word_index = i
        node.palindrome_word_indices.append(i)

    result = []
    for i, word in enumerate(words):
        node = root
        for j in range(len(word)):
            if (node.word_index >= 0 and node.word_index != i and
                word[j:] == word[j:][::-1]):
                result.append([i, node.word_index])

            if word[j] not in node.children:
                break
            node = node.children[word[j]]
        else:
            for idx in node.palindrome_word_indices:
                if idx != i:
                    result.append([i, idx])

    return result

# Test
words = ["abcd","dcba","lls","s","sssll"]
print(palindromePairs(words))  # [[0,1],[1,0],[3,2],[2,4]]
```

### Problem 8: Stream of Characters

**LeetCode 1032 - Online string matching**

```python
class StreamChecker:
    """
    Check if stream of characters contains any word from given dictionary.

    Strategy: Build suffix trie and match stream backwards
    """
    def __init__(self, words):
        """
        Initialize with word dictionary.

        Time: O(N×M) where N=words count, M=max length
        """
        self.trie = {}
        self.stream = []

        # Build trie with reversed words
        for word in words:
            node = self.trie
            for char in reversed(word):  # Insert reversed
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['#'] = True  # End marker

    def query(self, letter):
        """
        Process next character and return if any word is completed.

        Time: O(M) where M = max word length
        """
        self.stream.append(letter)

        # Search stream backwards in trie
        node = self.trie
        for i in range(len(self.stream) - 1, -1, -1):
            char = self.stream[i]
            if char not in node:
                break
            node = node[char]
            if '#' in node:
                return True

        return False

# Optimized version with length limit
class StreamCheckerOptimized:
    def __init__(self, words):
        self.trie = {}
        self.stream = []
        self.max_len = 0

        # Build trie and track max length
        for word in words:
            self.max_len = max(self.max_len, len(word))
            node = self.trie
            for char in reversed(word):
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['#'] = True

    def query(self, letter):
        self.stream.append(letter)

        # Keep only last max_len characters
        if len(self.stream) > self.max_len:
            self.stream.pop(0)

        # Search backwards
        node = self.trie
        for i in range(len(self.stream) - 1, -1, -1):
            char = self.stream[i]
            if char not in node:
                break
            node = node[char]
            if '#' in node:
                return True

        return False

# Test
sc = StreamChecker(["cd","f","kl"])
print(sc.query('a'))  # False
print(sc.query('b'))  # False
print(sc.query('c'))  # False
print(sc.query('d'))  # True (matched "cd")
```

---

## 🔥 Hard Problems (Advanced Techniques)

### Problem 9: Word Squares

**LeetCode 425 - Complex backtracking with trie**

```python
def wordSquares(words):
    """
    Find all word squares from given word list.

    A word square is a sequence of words that forms a square where
    kth row and kth column form the same word.

    Strategy: Use trie for efficient prefix lookup + backtracking
    Time: O(N×26^L) where N=words, L=word length
    """
    if not words or not words[0]:
        return []

    n = len(words[0])

    # Build prefix trie
    prefix_map = {}
    for word in words:
        for i in range(n):
            prefix = word[:i]
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(word)

    def get_words_with_prefix(prefix):
        return prefix_map.get(prefix, [])

    def backtrack(square):
        if len(square) == n:
            return [square[:]]  # Found complete square

        results = []
        step = len(square)

        # Build prefix for next word from current columns
        prefix = ""
        for i in range(step):
            prefix += square[i][step]

        # Try all words with this prefix
        for word in get_words_with_prefix(prefix):
            square.append(word)
            results.extend(backtrack(square))
            square.pop()

        return results

    result = []
    for word in words:
        result.extend(backtrack([word]))

    return result

# Optimized version with trie
class WordSquareSolver:
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.words = []  # Words with this prefix

    def __init__(self, words):
        self.n = len(words[0]) if words else 0
        self.trie = self.TrieNode()
        self.build_trie(words)

    def build_trie(self, words):
        """Build trie with word lists at each prefix."""
        for word in words:
            node = self.trie
            node.words.append(word)

            for char in word:
                if char not in node.children:
                    node.children[char] = self.TrieNode()
                node = node.children[char]
                node.words.append(word)

    def get_words_with_prefix(self, prefix):
        """Get all words with given prefix."""
        node = self.trie
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.words

    def solve(self):
        """Find all word squares."""
        result = []

        def backtrack(square):
            if len(square) == self.n:
                result.append(square[:])
                return

            step = len(square)
            prefix = "".join(square[i][step] for i in range(step))

            for word in self.get_words_with_prefix(prefix):
                square.append(word)
                backtrack(square)
                square.pop()

        # Start with each word
        for word in self.get_words_with_prefix(""):
            backtrack([word])

        return result

# Test
words = ["area","lead","wall","lady","ball"]
solver = WordSquareSolver(words)
print(solver.solve())
# Output: [["wall","area","lead","lady"],["ball","area","lead","lady"]]
```

### Problem 10: Concatenated Words

**LeetCode 472 - Word composition check**

```python
def findAllConcatenatedWords(words):
    """
    Find all words that can be formed by concatenating other words.

    Strategy: Build trie from shorter words, check if longer words can be segmented
    Time: O(N×M²) where N=words count, M=max length
    """
    if not words:
        return []

    # Sort by length to process shorter words first
    words.sort(key=len)

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False

    root = TrieNode()
    result = []

    def add_word(word):
        """Add word to trie."""
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    def can_form(word):
        """Check if word can be formed by concatenating existing words."""
        def dfs(index, count):
            if index == len(word):
                return count >= 2  # Must use at least 2 words

            node = root
            for i in range(index, len(word)):
                if word[i] not in node.children:
                    return False
                node = node.children[word[i]]

                if node.is_word and dfs(i + 1, count + 1):
                    return True

            return False

        return dfs(0, 0)

    for word in words:
        if can_form(word):
            result.append(word)
        add_word(word)

    return result

# DP-based solution for comparison
def findAllConcatenatedWordsDP(words):
    """
    DP solution: for each word, check if it can be segmented.
    """
    word_set = set(words)

    def can_segment(word):
        if not word:
            return False

        n = len(word)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and word[j:i] in word_set and word[j:i] != word:
                    dp[i] = True
                    break

        return dp[n]

    return [word for word in words if can_segment(word)]

# Test
words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
print(findAllConcatenatedWords(words))
# Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
```

### Problem 11: Design Search Autocomplete System

**LeetCode 642 - Real-world autocomplete system**

```python
class AutocompleteSystem:
    """
    Design search autocomplete system with:
    - Historical data with frequencies
    - Real-time input processing
    - Top 3 hot sentences for each prefix
    """

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.sentences = {}  # sentence -> frequency

    def __init__(self, sentences, times):
        """
        Initialize with historical data.

        Args:
            sentences: List of historical sentences
            times: List of corresponding frequencies
        """
        self.root = self.TrieNode()
        self.current_input = ""
        self.current_node = self.root

        # Build trie from historical data
        for sentence, frequency in zip(sentences, times):
            self.add_sentence(sentence, frequency)

    def add_sentence(self, sentence, frequency):
        """Add sentence to trie with frequency."""
        node = self.root

        # Add sentence to every prefix node
        for char in sentence:
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
            node.sentences[sentence] = node.sentences.get(sentence, 0) + frequency

        # Also add to final node
        node.sentences[sentence] = node.sentences.get(sentence, 0) + frequency

    def input(self, c):
        """
        Process input character and return top 3 suggestions.

        Returns:
            List of top 3 hot sentences, or empty list if c == '#'
        """
        if c == '#':
            # End of input, add current sentence to system
            if self.current_input:
                self.add_sentence(self.current_input, 1)

            # Reset for next query
            self.current_input = ""
            self.current_node = self.root
            return []

        # Add character to current input
        self.current_input += c

        # Navigate trie
        if c in self.current_node.children:
            self.current_node = self.current_node.children[c]

            # Get top 3 suggestions
            suggestions = list(self.current_node.sentences.items())
            suggestions.sort(key=lambda x: (-x[1], x[0]))  # Sort by frequency desc, then lexicographically

            return [sentence for sentence, _ in suggestions[:3]]
        else:
            # No valid path, create dead-end node
            self.current_node = self.TrieNode()
            return []

# Optimized version with lazy evaluation
class AutocompleteSystemOptimized:
    def __init__(self, sentences, times):
        self.sentence_freq = {}
        for sentence, freq in zip(sentences, times):
            self.sentence_freq[sentence] = freq

        self.current_input = ""

    def input(self, c):
        if c == '#':
            # Add current input to frequency map
            if self.current_input:
                self.sentence_freq[self.current_input] = \
                    self.sentence_freq.get(self.current_input, 0) + 1
            self.current_input = ""
            return []

        self.current_input += c

        # Find all sentences with current prefix
        candidates = []
        for sentence, freq in self.sentence_freq.items():
            if sentence.startswith(self.current_input):
                candidates.append((sentence, freq))

        # Sort and return top 3
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [sentence for sentence, _ in candidates[:3]]

# Test
ac = AutocompleteSystem(["i love you", "island","ironman", "i love leetcode"], [5,3,2,2])
print(ac.input('i'))  # ["i love you", "island","i love leetcode"]
print(ac.input(' '))  # ["i love you", "i love leetcode"]
print(ac.input('a'))  # []
print(ac.input('#'))  # [] (adds "i a" to system)
```

### Problem 12: Word Break II with Trie

**Enhanced word segmentation with trie optimization**

```python
def wordBreakII(s, wordDict):
    """
    Find all possible word break combinations using trie optimization.

    Returns all possible sentences formed by adding spaces.

    Time: O(N×2^N) worst case, but trie helps pruning
    """
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False

    # Build trie from word dictionary
    root = TrieNode()
    for word in wordDict:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

    # Memoization for optimization
    memo = {}

    def backtrack(index):
        """Find all possible segmentations starting from index."""
        if index in memo:
            return memo[index]

        if index == len(s):
            return [[]]  # Empty segmentation for end of string

        result = []
        node = root

        # Try all possible words starting at current index
        for end in range(index, len(s)):
            char = s[end]
            if char not in node.children:
                break  # No valid words with this prefix

            node = node.children[char]

            if node.is_word:
                # Found valid word, recurse for remaining string
                word = s[index:end + 1]
                rest_results = backtrack(end + 1)

                for rest in rest_results:
                    result.append([word] + rest)

        memo[index] = result
        return result

    # Convert word lists to sentences
    word_lists = backtrack(0)
    return [" ".join(words) for words in word_lists]

# Alternative implementation with dynamic programming
def wordBreakIIDP(s, wordDict):
    """
    DP approach with trie for efficient word checking.
    """
    word_set = set(wordDict)
    n = len(s)

    # dp[i] stores all possible segmentations for s[:i]
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]  # Base case: empty string

    for i in range(1, n + 1):
        for j in range(i):
            word = s[j:i]
            if word in word_set and dp[j]:
                # Add new segmentations
                for prev_segmentation in dp[j]:
                    dp[i].append(prev_segmentation + [word])

    return [" ".join(words) for words in dp[n]]

# Test
s = "catsanddog"
wordDict = ["cat","cats","and","sand","dog"]
print(wordBreakII(s, wordDict))
# Output: ["cats and dog","cat sand dog"]
```

---

## 🚀 Expert-Level Problems

### Problem 13: Maximum XOR of Two Numbers

**LeetCode 421 - Binary trie for XOR optimization**

```python
def findMaximumXOR(nums):
    """
    Find maximum XOR of any two numbers in array.

    Strategy: Build binary trie, for each number find complement path
    Time: O(N×32) = O(N), Space: O(N×32)
    """
    class TrieNode:
        def __init__(self):
            self.children = {}

    root = TrieNode()

    # Insert all numbers into binary trie
    for num in nums:
        node = root
        for i in range(31, -1, -1):  # 32 bits, MSB first
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]

    max_xor = 0

    # For each number, find maximum XOR
    for num in nums:
        node = root
        current_xor = 0

        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction for maximum XOR
            toggle_bit = 1 - bit

            if toggle_bit in node.children:
                current_xor |= (1 << i)
                node = node.children[toggle_bit]
            else:
                node = node.children[bit]

        max_xor = max(max_xor, current_xor)

    return max_xor

# Optimized bit manipulation approach
def findMaximumXORBitwise(nums):
    """
    Bit manipulation approach without explicit trie.
    Build answer bit by bit from MSB.
    """
    max_xor = 0
    mask = 0

    for i in range(31, -1, -1):
        mask |= (1 << i)
        prefixes = {num & mask for num in nums}

        temp = max_xor | (1 << i)

        # Check if we can achieve this bit in result
        for prefix in prefixes:
            if temp ^ prefix in prefixes:
                max_xor = temp
                break

    return max_xor

# Test
nums = [3, 10, 5, 25, 2, 8]
print(findMaximumXOR(nums))  # 28 (5 XOR 25)
```

### Problem 14: Count Unique Substrings

**Advanced string processing with suffix trie**

```python
def countUniqueSubstrings(s):
    """
    Count all unique substrings using suffix trie.

    Time: O(N²), Space: O(N²)
    """
    class TrieNode:
        def __init__(self):
            self.children = {}

    root = TrieNode()
    unique_count = 0

    # Build suffix trie
    n = len(s)
    for i in range(n):
        node = root
        for j in range(i, n):
            char = s[j]
            if char not in node.children:
                node.children[char] = TrieNode()
                unique_count += 1  # New unique substring found
            node = node.children[char]

    return unique_count

# Space-optimized using rolling hash
def countUniqueSubstringsHash(s):
    """
    Use rolling hash to count unique substrings more efficiently.
    """
    unique_substrings = set()
    n = len(s)

    for i in range(n):
        for j in range(i + 1, n + 1):
            unique_substrings.add(s[i:j])

    return len(unique_substrings)

# Most efficient: suffix array approach
def countUniqueSubstringsSuffix(s):
    """
    Using suffix array and LCP array for O(N log N) solution.
    """
    # Build suffix array (implementation omitted for brevity)
    # Count unique substrings using LCP array
    # Returns total_substrings - duplicate_substrings
    pass

# Test
s = "ababa"
print(countUniqueSubstrings(s))  # 9 unique substrings
```

---

## 🎯 Problem-Solving Strategies

### Pattern Recognition Guide

1. **Basic Trie Operations**
   - Insert, search, prefix matching
   - Template: Standard trie implementation

2. **Wildcard/Pattern Matching**
   - '.' wildcard support
   - DFS with backtracking through trie

3. **Word Segmentation**
   - Break string into valid words
   - Combine trie with DP or backtracking

4. **Prefix Sum/Aggregation**
   - Sum values for all words with prefix
   - Store aggregated data in trie nodes

5. **Binary/Numeric Tries**
   - XOR optimization problems
   - Use binary representation in trie

6. **Stream Processing**
   - Online string matching
   - Reverse trie for suffix matching

### Implementation Tips

1. **Choose Right Node Structure**

   ```python
   # For small alphabet
   children = [None] * 26

   # For large/dynamic alphabet
   children = {}

   # For memory optimization
   children = {} # Only create when needed
   ```

2. **Handle Edge Cases**

   ```python
   # Empty strings
   if not word: return False

   # Duplicate insertions
   if not node.is_end:
       node.is_end = True
       self.size += 1
   ```

3. **Optimize Memory Usage**

   ```python
   # Store complete words only at end nodes
   if node.is_end:
       node.word = word  # Only store at end

   # Use bit manipulation for small alphabets
   has_children = 0  # Bitmask for 26 letters
   ```

4. **Add Helpful Methods**
   ```python
   def get_all_words(self):
       """Get all words in trie"""
       pass

   def delete(self, word):
       """Remove word from trie"""
       pass

   def size(self):
       """Return number of words"""
       pass
   ```

---

## 📊 Complexity Analysis Summary

| Problem Type    | Time Insert | Time Search | Space           | Optimization Notes             |
| --------------- | ----------- | ----------- | --------------- | ------------------------------ |
| Basic Trie      | O(M)        | O(M)        | O(ALPHABET×N×M) | Use arrays for small alphabets |
| Wildcard Search | O(M)        | O(M×K)      | O(ALPHABET×N×M) | K = number of wildcards        |
| Word Break      | O(N×M)      | O(N²)       | O(N×M)          | DP memoization crucial         |
| Binary Trie     | O(log MAX)  | O(log MAX)  | O(N×log MAX)    | For XOR problems               |
| Compressed Trie | O(M)        | O(M)        | O(edges)        | Better for sparse data         |

---

## 🏆 Interview Success Tips

### 1. Problem Identification

- Look for prefix-related operations
- Multiple string matching requirements
- Autocomplete/suggestion features
- Word composition problems

### 2. Implementation Strategy

- Start with basic trie structure
- Add problem-specific data to nodes
- Consider optimization opportunities
- Handle edge cases explicitly

### 3. Common Optimizations

- **Memory**: Use arrays vs dictionaries
- **Time**: Add caching for frequent operations
- **Space**: Compressed tries for sparse data
- **Functionality**: Store additional data in nodes

### 4. Testing Approach

- Test with empty strings
- Test with single characters
- Test with overlapping prefixes
- Test edge cases (duplicates, etc.)

Remember: Tries excel at prefix operations and multi-pattern string matching. Choose them when you need fast prefix lookups or when building sophisticated string processing systems!

---

_Continue to Tries Cheatsheet for quick reference patterns and templates._
