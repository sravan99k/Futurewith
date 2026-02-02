# Modules and Libraries Practice Questions

## Introduction

This practice guide provides hands-on exercises for creating, organizing, and distributing Python packages. Work through each exercise to build practical skills in module development and package management.

---

## Exercise 1: Creating Your First Module

**Objective:** Create a basic Python module with functions and classes.

**Task:** Create a module named `string_utils.py` that provides the following utilities:

```python
# string_utils.py

def reverse_string(s):
    """Reverse a given string."""
    pass

def is_palindrome(s):
    """Check if a string is a palindrome."""
    pass

def count_vowels(s):
    """Count the number of vowels in a string."""
    pass

def to_snake_case(s):
    """Convert a string to snake_case."""
    pass

class StringProcessor:
    """A class for processing strings."""
    
    def __init__(self, text):
        self.text = text
    
    def get_word_count(self):
        """Return the number of words."""
        pass
    
    def get_unique_chars(self):
        """Return unique characters."""
        pass
    
    def get_frequency(self):
        """Return character frequency dictionary."""
        pass
```

**Requirements:**

1. Implement all functions and methods
2. Handle edge cases (empty strings, None values)
3. Add proper docstrings to all functions
4. Write unit tests to verify functionality

**Starter Code:**

```python
def reverse_string(s):
    """Reverse a given string."""
    if s is None:
        raise ValueError("Input cannot be None")
    return s[::-1]

# Implement the rest...

if __name__ == "__main__":
    # Test your module
    utils = StringProcessor("Hello World")
    print(f"Word count: {utils.get_word_count()}")
```

**Deliverable:** Submit `string_utils.py` with complete implementation and a separate `test_string_utils.py` file with at least 5 test cases.

---

## Exercise 2: Creating a Package with Subpackages

**Objective:** Design a package structure with multiple subpackages.

**Task:** Create a data processing package with the following structure:

```
data_processor/
    __init__.py
    core/
        __init__.py
        processor.py
        validators.py
    io/
        __init__.py
        readers.py
        writers.py
    utils/
        __init__.py
        helpers.py
```

**Requirements:**

1. Create each file with proper content
2. Set up proper imports between modules
3. Configure `__init__.py` files appropriately
4. Write a main script that demonstrates the package

**Implementation Details:**

**data_processor/core/processor.py:**

```python
class DataProcessor:
    """Main data processor class."""
    
    def __init__(self, validator=None):
        self.validator = validator
        self.data = []
    
    def add_data(self, data):
        """Add data to be processed."""
        pass
    
    def process(self):
        """Process all data."""
        pass
    
    def get_results(self):
        """Return processed results."""
        pass
```

**data_processor/io/readers.py:**

```python
class CSVReader:
    """Read data from CSV files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
    
    def read(self):
        """Read and return data."""
        pass

class JSONReader:
    """Read data from JSON files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
    
    def read(self):
        """Read and return data."""
        pass
```

**data_processor/io/writers.py:**

```python
class CSVWriter:
    """Write data to CSV files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
    
    def write(self, data):
        """Write data to file."""
        pass

class JSONWriter:
    """Write data to JSON files."""
    
    def __init__(self, filepath):
        self.filepath = filepath
    
    def write(self, data):
        """Write data to file."""
        pass
```

**data_processor/utils/helpers.py:**

```python
def clean_data(data):
    """Clean and normalize data."""
    pass

def validate_schema(data, schema):
    """Validate data against a schema."""
    pass

def transform_data(data, transformation):
    """Apply transformation to data."""
    pass
```

**Deliverable:** Submit the complete package structure with all files implemented and tested.

---

## Exercise 3: Import System Mastery

**Objective:** Practice different import patterns and understand import mechanics.

**Task:** Analyze and fix the following code with import issues:

```python
# Problem 1: Circular Import
# file_a.py
from file_b import function_b

def function_a():
    return "Function A"

# file_b.py
from file_a import function_a

def function_b():
    return "Function B"

# Problem 2: Import not found
import non_existent_module

# Problem 3: Relative import error
from ..parent import parent_function
```

**Requirements:**

1. Identify all import issues
2. Fix circular imports using appropriate techniques
3. Demonstrate proper use of relative imports
4. Show how to handle missing modules gracefully

**Solution Template:**

```python
# Fix for circular import - Method 1: Restructure
# Move common imports to a third module
# common.py
def shared_function():
    pass

# file_a.py
from common import shared_function

def function_a():
    return "Function A"

# file_b.py
from common import shared_function

def function_b():
    return "Function B"

# Fix for circular import - Method 2: Lazy import
def function_a():
    from file_b import function_b
    return function_b()

# Handle missing module gracefully
try:
    import optional_module
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False
    optional_module = None

# Use conditional import
if HAS_OPTIONAL:
    result = optional_module.do_something()
```

**Deliverable:** Provide a working example showing proper import patterns with explanations.

---

## Exercise 4: Building a Reusable Utility Package

**Objective:** Create a professional-grade utility package.

**Task:** Build a package `python_utils` with the following features:

**Package Structure:**

```
python_utils/
    __init__.py
    __version__.py
    text/
        __init__.py
        cleaners.py
        validators.py
    collection/
        __init__.py
        counters.py
        mergers.py
    file/
        __init__.py
        readers.py
        path_utils.py
```

**Required Functionality:**

**text/cleaners.py:**

```python
def remove_whitespace(text):
    """Remove all whitespace from text."""
    pass

def normalize_whitespace(text):
    """Normalize multiple spaces to single space."""
    pass

def remove_special_chars(text, allowed=None):
    """Remove special characters, keeping only allowed."""
    pass

def truncate(text, max_length, suffix="..."):
    """Truncate text to max length with suffix."""
    pass

class TextCleaner:
    """Text cleaning pipeline."""
    
    def __init__(self):
        self.pipelines = []
    
    def add_step(self, func):
        """Add a cleaning step."""
        pass
    
    def clean(self, text):
        """Apply all cleaning steps."""
        pass
```

**text/validators.py:**

```python
def is_valid_email(email):
    """Validate email format."""
    pass

def is_valid_url(url):
    """Validate URL format."""
    pass

def is_valid_phone(phone):
    """Validate phone number format."""
    pass

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid, errors=None):
        pass
    
    def add_error(self, error):
        """Add an error message."""
        pass

class TextValidator:
    """Multi-field text validator."""
    
    def __init__(self):
        self.rules = {}
    
    def add_rule(self, field, rule_func):
        """Add validation rule for a field."""
        pass
    
    def validate(self, data):
        """Validate data against all rules."""
        pass
```

**collection/counters.py:**

```python
from collections import Counter
from typing import Any, List, Dict

class AdvancedCounter:
    """Enhanced counter with additional methods."""
    
    def __init__(self, iterable=None):
        pass
    
    def most_common_weighted(self, n=None, weights=None):
        """Get most common with weight considerations."""
        pass
    
    def get_frequency_percentage(self, item):
        """Get percentage of total for an item."""
        pass
    
    def merge(self, other):
        """Merge with another counter."""
        pass
    
    def subtract(self, other):
        """Subtract another counter."""
        pass

def count_by_key(items: List[Dict], key: str) -> Dict:
    """Count items by a specific key."""
    pass
```

**Deliverable:** Complete package with all modules implemented and comprehensive tests.

---

## Exercise 5: Package Distribution

**Objective:** Prepare a package for distribution on PyPI.

**Task:** Create distribution files for your `python_utils` package.

**Requirements:**

1. Create `setup.py` for legacy distribution
2. Create `pyproject.toml` for modern distribution
3. Create `MANIFEST.in` for inclusion rules
4. Write `README.md` with documentation
5. Create `LICENSE` file
6. Write `setup.cfg` or `pyproject.toml` with configuration

**Files to Create:**

**setup.py:**

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("python_utils/__version__.py", "r") as f:
    version = f.read().split("=")[1].strip().strip('"')

setup(
    name="python-utils",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="A collection of useful Python utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/python-utils",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        # List dependencies here
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "python-utils=python_utils.cli:main",
        ],
    },
)
```

**pyproject.toml:**

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "python-utils"
version = "0.1.0"
description = "A collection of useful Python utilities"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest",
    "black",
    "flake8",
    "twine",
    "wheel",
]

[project.scripts]
python-utils = "python_utils.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["python_utils*"]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Deliverable:** Complete distribution package with all configuration files.

---

## Exercise 6: Working with Standard Library Modules

**Objective:** Master key standard library modules through practical exercises.

**Task:** Create a comprehensive text processing tool using standard library modules.

**Requirements:**

**Using `collections`:**

```python
from collections import Counter, defaultdict, OrderedDict, namedtuple

# Create a word frequency analyzer
class WordFrequency:
    def __init__(self, text):
        self.text = text
        self.word_counts = Counter()
        self.char_counts = Counter()
    
    def analyze(self):
        """Analyze word and character frequencies."""
        pass
    
    def get_top_words(self, n=10):
        """Get top n most common words."""
        pass
    
    def get_least_common(self, n=5):
        """Get least common words."""
        pass
```

**Using `itertools`:**

```python
import itertools

class CombinationGenerator:
    """Generate and work with combinations."""
    
    @staticmethod
    def generate_combinations(items, r):
        """Generate all combinations of r items."""
        pass
    
    @staticmethod
    def generate_permutations(items, r=None):
        """Generate permutations of items."""
        pass
    
    @staticmethod
    def batch_iterable(iterable, batch_size):
        """Yield batches from an iterable."""
        pass
    
    @staticmethod
    def flatten(nested_iterables):
        """Flatten nested iterables."""
        pass
```

**Using `functools`:**

```python
from functools import reduce, lru_cache, partial
from operator import mul

class Calculator:
    """Functional calculator using functools."""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def factorial(n):
        """Calculate factorial with memoization."""
        pass
    
    @staticmethod
    def product(numbers):
        """Calculate product of numbers."""
        pass
    
    @staticmethod
    def compose(*functions):
        """Compose functions."""
        pass
    
    @staticmethod
    def partial_func(func, **fixed_kwargs):
        """Create partial function with fixed kwargs."""
        pass
```

**Deliverable:** Complete text processing tool with comprehensive documentation.

---

## Exercise 7: Creating a Plugin System

**Objective:** Build a modular plugin architecture using dynamic imports.

**Task:** Create a plugin system for a text processing application.

**Requirements:**

```python
# plugin_system/plugin_manager.py

class PluginManager:
    """Manage dynamic plugin loading."""
    
    def __init__(self):
        self.plugins = {}
        self.hooks = {}
    
    def register_plugin(self, name, plugin_class):
        """Register a plugin."""
        pass
    
    def load_plugins(self, plugin_dir):
        """Dynamically load all plugins from a directory."""
        pass
    
    def get_plugin(self, name):
        """Get a registered plugin."""
        pass
    
    def execute_hook(self, hook_name, *args, **kwargs):
        """Execute all plugins registered for a hook."""
        pass

# plugins/text_plugin.py
class TextPlugin:
    """Base class for text processing plugins."""
    
    name = "base"
    description = "Base text plugin"
    
    def process(self, text):
        """Process text - override in subclasses."""
        raise NotImplementedError

# plugins/uppercase_plugin.py
class UppercasePlugin(TextPlugin):
    name = "uppercase"
    description = "Convert text to uppercase"
    
    def process(self, text):
        """Convert text to uppercase."""
        return text.upper()

# plugins/reverse_plugin.py
class ReversePlugin(TextPlugin):
    name = "reverse"
    description = "Reverse text"
    
    def process(self, text):
        """Reverse the text."""
        return text[::-1]
```

**Deliverable:** Complete plugin system with documentation and examples.

---

## Exercise 8: Package Testing and Quality Assurance

**Objective:** Write comprehensive tests for your modules.

**Task:** Create a test suite for the `python_utils` package.

**Requirements:**

**test_string_utils.py:**

```python
import pytest
from python_utils.text.cleaners import remove_whitespace, normalize_whitespace
from python_utils.text.validators import is_valid_email

class TestTextCleaners:
    def test_remove_whitespace(self):
        """Test whitespace removal."""
        assert remove_whitespace("hello world") == "helloworld"
        assert remove_whitespace("  spaced  out  ") == "spacedout"
        assert remove_whitespace("") == ""
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_whitespace("hello   world") == "hello world"
        assert normalize_whitespace("  leading") == "leading"
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        assert remove_special_chars("hello@world.com", allowed=".") == "helloworld.com"

class TestTextValidators:
    def test_valid_email(self):
        """Test email validation."""
        assert is_valid_email("test@example.com") == True
        assert is_valid_email("invalid") == False
    
    def test_invalid_email(self):
        """Test invalid email detection."""
        assert is_valid_email("") == False
        assert is_valid_email("@example.com") == False
```

**pytest Configuration:**

```python
# pytest.ini or pyproject.toml section
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

**Deliverable:** Complete test suite with at least 20 test cases covering all package modules.

---

## Exercise 9: Dependency Management

**Objective:** Master dependency management and virtual environments.

**Task:** Set up professional development environment.

**Requirements:**

**Create requirements.txt:**

```
# Core dependencies
numpy>=1.20.0
pandas>=1.3.0
requests>=2.25.0

# Development dependencies
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.950
pre-commit>=2.17.0

# Documentation
sphinx>=4.5.0
```

**Create requirements-dev.txt:**

```
-r requirements.txt
-r requirements-dev.in
```

**Create virtual environment setup script:**

```bash
#!/bin/bash
# setup_dev_environment.sh

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

echo "Development environment ready!"
```

**Deliverable:** Complete dependency management setup with documentation.

---

## Exercise 10: Complete Package Project

**Objective:** Apply all learned concepts in a comprehensive project.

**Task:** Create a professional data analysis utility package.

**Package Structure:**

```
data_utils/
    __init__.py
    __version__.py
    README.md
    LICENSE
    setup.py
    pyproject.toml
    
    core/
        __init__.py
        analyzer.py
        transformer.py
        validator.py
    
    io/
        __init__.py
        csv_handler.py
        json_handler.py
    
    stats/
        __init__.py
        descriptive.py
        correlation.py
    
    tests/
        __init__.py
        test_analyzer.py
        test_transformer.py
        test_csv_handler.py
```

**Required Features:**

**core/analyzer.py:**

```python
from typing import List, Dict, Any
import statistics

class DataAnalyzer:
    """Comprehensive data analysis toolkit."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.columns = self._extract_columns()
    
    def _extract_columns(self) -> List[str]:
        """Extract column names from data."""
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all columns."""
        pass
    
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """Get statistics for a specific column."""
        pass
    
    def get_correlations(self) -> Dict[str, float]:
        """Calculate correlations between numeric columns."""
        pass
    
    def filter_by(self, column: str, condition) -> List[Dict]:
        """Filter data by condition."""
        pass
    
    def group_by(self, column: str) -> Dict[Any, List[Dict]]:
        """Group data by column value."""
        pass
```

**io/csv_handler.py:**

```python
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path

class CSVHandler:
    """Handle CSV file operations."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
    
    def read(self, has_header: bool = True) -> List[Dict[str, Any]]:
        """Read CSV file and return data."""
        pass
    
    def write(self, data: List[Dict[str, Any]], 
              columns: Optional[List[str]] = None) -> None:
        """Write data to CSV file."""
        pass
    
    def append(self, data: List[Dict[str, Any]]) -> None:
        """Append data to existing CSV file."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the CSV file."""
        pass
```

**Deliverable:** Complete package with all features implemented, tested, and documented.

---

## Exercise 11: Debugging Module Imports

**Objective:** Practice debugging import-related issues.

**Task:** Fix the following problematic code:

```python
# Problem 1: __all__ not respected
# module.py
def private_function():
    pass

def public_function():
    pass

__all__ = ['public_function']

# main.py
from module import *
print(private_function())  # This should fail!
```

**Problem 2: Module not found**

```python
# Attempting to import without proper path setup
try:
    import my_custom_module
except ImportError as e:
    print(f"Import error: {e}")
    # Fix this!
```

**Problem 3: Import shadowing**

```python
# utils.py (user's file)
def process():
    pass

# main.py
from collections import Counter
from utils import process  # This works

from utils import Counter  # This shadows collections.Counter!
```

**Deliverable:** Provide solutions with explanations for each problem.

---

## Exercise 12: Advanced Package Patterns

**Objective:** Learn advanced package design patterns.

**Task:** Implement a configurable package with plugins.

**Requirements:**

```python
# config_loader.py
import json
from typing import Dict, Any

class ConfigLoader:
    """Load and manage configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        pass
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value."""
        pass
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass
    
    def save(self) -> None:
        """Save configuration to file."""
        pass

# Package with runtime configuration
class ConfigurablePackage:
    """Package with runtime configuration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._components = {}
    
    def register_component(self, name: str, component):
        """Register a component."""
        pass
    
    def get_component(self, name: str):
        """Get a registered component."""
        pass
    
    def configure(self, **kwargs) -> None:
        """Update configuration."""
        pass
```

**Deliverable:** Complete implementation with examples.

---

## Bonus Exercise: Open Source Contribution

**Objective:** Experience real-world package development.

**Task:** Fork an open source project and make a contribution.

**Steps:**

1. Find a small open source Python project on GitHub
2. Read the contribution guidelines
3. Find a good first issue
4. Create a fork and branch
5. Implement the fix
6. Write tests
7. Create a pull request

**Deliverable:** Screenshot of your pull request and brief description of the changes made.

---

## Submission Guidelines

For each exercise:
1. Provide complete, runnable code
2. Include at least 3 test cases per function/class
3. Add comprehensive docstrings
4. Include example usage in comments
5. Document any assumptions made

**Grading Criteria:**
- Code Quality (40%): Clean, readable, well-documented
- Functionality (30%): All requirements met
- Testing (20%): Comprehensive test coverage
- Best Practices (10%): Follows Python conventions

**Expected Time:**
- Exercises 1-4: 2-3 hours each
- Exercises 5-8: 3-4 hours each
- Exercise 10: 4-5 hours
- Exercise 11-12: 2-3 hours each
