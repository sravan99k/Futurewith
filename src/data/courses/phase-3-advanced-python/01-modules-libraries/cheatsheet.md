# Modules & Libraries Cheat Sheet

## Quick Reference

### Import Statements

```python
# Basic import
import math
import os
import sys

# Import specific items
from math import sqrt, pi
from collections import Counter, defaultdict

# Import with alias
import numpy as np
import pandas as pd
from collections import OrderedDict as OD

# Import all (generally avoid)
from module import *

# Relative imports
from .sibling import func
from ..parent import Class
from . import specific_module
```

### Package Creation

```python
# Required: __init__.py
# Can be empty or contain initialization code

# Package with public API
from .module1 import Class1, func1
from .module2 import Class2, func2

__all__ = ['Class1', 'Class2', 'func1', 'func2']
VERSION = '1.0.0'
```

### File Structure

```
mypackage/
    __init__.py          # Package marker & initialization
    __main__.py          # For 'python -m mypackage'
    module1.py           # Regular module
    subpackage/
        __init__.py
        submodule.py
```

---

## Essential Standard Library Modules

### collections

```python
from collections import Counter, defaultdict, OrderedDict, namedtuple, deque

# Counter - count occurrences
words = ['apple', 'banana', 'apple']
count = Counter(words)
print(count['apple'])  # 2

# defaultdict - auto-initialize missing keys
dd = defaultdict(list)
dd['fruits'].append('apple')  # No KeyError

# OrderedDict - remembers order (Python 3.7+ dicts do this natively)
od = OrderedDict()
od['a'] = 1
od['b'] = 2

# namedtuple - lightweight record
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)

# deque - double-ended queue
dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
```

### itertools

```python
import itertools

# Infinite iterators
for i in itertools.count(0, 2):  # 0, 2, 4, ...
    if i > 10: break

# Combinations
for combo in itertools.combinations([1, 2, 3, 4], 2):
    print(combo)

# Permutations
for perm in itertools.permutations('ABC', 2):
    print(perm)

# Chain - concatenate iterables
for item in itertools.chain([1, 2], [3, 4]):
    print(item)

# Groupby
data = [('a', 1), ('a', 2), ('b', 3)]
for key, group in itertools.groupby(data, lambda x: x[0]):
    print(key, list(group))

# Product - cartesian product
for x, y in itertools.product([1, 2], ['a', 'b']):
    print(x, y)
```

### functools

```python
from functools import reduce, lru_cache, partial, cmp_to_key

# reduce - accumulate values
result = reduce(lambda x, y: x + y, [1, 2, 3, 4])  # 10

# lru_cache - memoization
@lru_cache(maxsize=128)
def fib(n):
    if n <= 1: return n
    return fib(n-1) + fib(n-2)

# partial - preset arguments
def multiply(x, y): return x * y
double = partial(multiply, y=2)
print(double(5))  # 10

# cmp_to_key - custom sorting
words = ['hi', 'hello', 'hey']
sorted_words = sorted(words, key=cmp_to_key(lambda a, b: len(a) - len(b)))
```

### pathlib

```python
from pathlib import Path

p = Path('/home/user/docs')
file_path = p / 'report.txt'

# File operations
file_path.exists()
file_path.is_file()
file_path.is_dir()
file_path.suffix  # '.txt'
file_path.stem    # 'report'

# Read/write
content = file_path.read_text()
file_path.write_text('New content')

# Directory operations
p.mkdir(parents=True, exist_ok=True)
for item in p.iterdir():
    if item.is_file():
        print(f'File: {item}')
```

### re (Regular Expressions)

```python
import re

# Basic patterns
pattern = r'\d{3}-\d{3}-\d{4}'  # Phone number
email_pattern = r'[\w.-]+@[\w.-]+\.\w+'

# Matching
if re.match(pattern, '123-456-7890'):
    print('Match!')

# Finding all
matches = re.findall(r'\d+', '123 abc 456 def')

# Substitution
new_text = re.sub(r'\d+', 'XXX', 'Contact: 123-456-7890')

# Splitting
parts = re.split(r'[\s,]+', 'apple, banana cherry')

# Groups
match = re.search(r'(\d{3})-(\d{3})-(\d{4})', 'Call: 123-456-7890')
print(match.group(1))  # 123
```

### typing

```python
from typing import List, Dict, Set, Tuple, Optional, Union, Callable

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}!"

# Complex types
def process_items(items: List[int]) -> Dict[str, int]:
    pass

# Optional and Union
def find_user(user_id: int) -> Optional[Dict]:
    # Returns Dict or None
    pass

# Callable
def execute(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Type aliases
Matrix = List[List[float]]
Result = Union[str, int, None]
```

---

## Common Package Patterns

### Singleton Pattern

```python
# Method 1: Class-based
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Method 2: Module-level (Pythonic)
# In config.py
_config = None

def get_config():
    global _config
    if _config is None:
        _config = Configuration()
    return _config
```

### Factory Pattern

```python
class DataProcessor:
    @staticmethod
    def create(processor_type: str):
        if processor_type == 'csv':
            return CSVProcessor()
        elif processor_type == 'json':
            return JSONProcessor()
        elif processor_type == 'xml':
            return XMLProcessor()
        raise ValueError(f"Unknown type: {processor_type}")
```

### Facade Pattern

```python
# Complex system
class VideoConverter:
    def extract_audio(self, video): pass
    def transcode(self, video): pass
    def apply_effects(self, video): pass

# Simple interface
class VideoEditor:
    def __init__(self):
        self.converter = VideoConverter()
    
    def quick_edit(self, video):
        self.converter.extract_audio(video)
        self.converter.transcode(video)
        return video
```

---

## pip Commands

```bash
# Install packages
pip install package_name
pip install package_name==1.2.3
pip install package_name>=1.2.0
pip install -r requirements.txt

# Upgrade
pip install --upgrade package_name

# Uninstall
pip uninstall package_name

# List and search
pip list
pip show package_name
pip search package_name  # Deprecated, use PyPI instead

# Virtual environments
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows
deactivate
```

---

## pyproject.toml Structure

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my_package"
version = "1.0.0"
description = "A short description"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "black>=22.0.0"]

[project.scripts]
my-script = "my_package.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["my_package*"]
```

---

## Debugging Imports

```python
import sys

# Check where module is loaded from
import mymodule
print(mymodule.__file__)

# Check if module is already imported
print('mymodule' in sys.modules)

# Check sys.path
import pprint
pprint.pprint(sys.path)

# Handle import errors gracefully
try:
    import optional_module
except ImportError:
    optional_module = None
```

---

## Best Practices

### Do ✓

```python
# Organize imports at top of file
import os
import sys
from collections import defaultdict

# Use clear, descriptive names
data_processor.py
user_authentication.py

# Document modules with docstrings
"""Module for data processing utilities.

This module provides functions for cleaning,
transforming, and analyzing data.
"""

# Use __all__ for public API
__all__ = ['process_data', 'clean_text']
```

### Don't ✗

```python
# Don't import at bottom of file
def some_function():
    import some_module  # Bad!

# Don't use wildcard imports
from module import *  # Bad!

# Avoid circular imports
# file_a.py imports from file_b.py
# file_b.py imports from file_a.py  # Circular!

# Don't modify sys.path
import sys
sys.path.insert(0, '/random/path')  # Bad!
```

---

## Import Flow

```
When you type: import module

Python searches:
1. Current directory
2. sys.path (in order):
   - Directory containing input script
   - PYTHONPATH environment variable
   - Standard library directories
   - Site-packages directories
   - .zip archives in path

Then:
1. Creates module object
2. Executes module code
3. Caches in sys.modules
4. Returns module reference
```

---

## Quick Reference Table

| Pattern | Usage | Example |
|---------|-------|---------|
| `import X` | Import module | `import math` |
| `from X import Y` | Import specific item | `from os import path` |
| `from X import Y as Z` | Import with alias | `import pandas as pd` |
| `from .module import Y` | Relative import | `from .utils import helper` |
| `from ..parent import Y` | Parent package import | `from ..config import settings` |

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Module not in path | Add to sys.path or install package |
| `ImportError` | Circular import | Restructure or use lazy import |
| `AttributeError` | Wrong attribute name | Check module docs |
| `ImportError: cannot import name` | Name not in module | Check `__all__` or module exports |

---

## Performance Tips

1. **Lazy imports** for heavy modules
2. **Cache imports** - Python caches in sys.modules
3. **Use __all__** for faster wildcard imports
4. **Avoid circular imports** - restructure code
5. **Use built-ins** - implemented in C, faster than Python
