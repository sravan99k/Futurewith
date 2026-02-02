# Python Cheat Sheets & Quick Reference Guide

## 📚 Welcome to Python Programming!

This guide is designed to be your **universal companion** for Python programming, whether you're:

- 🎓 A complete beginner (age 8-100+)
- 🔰 Returning to programming after a break
- 🚀 Looking to refresh your knowledge
- 👨‍💻 An experienced developer needing a quick reference

### How to Use This Guide

- **📖 Beginners**: Start with Python Basics (Section 1) and work your way up
- **⚡ Quick Reference**: Jump directly to specific topics using the Table of Contents
- **🎯 Problem Solving**: Use the error resolution section when you get stuck
- **📈 Skill Building**: Progress through sections based on your comfort level

### Legend

- ✅ **Beginner Friendly** - Great for first-time programmers
- 🔧 **Common Pattern** - Used frequently in real projects
- ⚠️ **Important Note** - Pay attention to this!
- 💡 **Tip** - Helpful shortcut or best practice
- 🚨 **Advanced** - For experienced programmers

---

## Table of Contents

### 🌱 Foundation (Start Here!)

1. [Python Basics Quick Reference](#python-basics-quick-reference) - ✅ Beginner Friendly
2. [Data Structures Cheat Sheet](#data-structures-cheat-sheet) - Essential building blocks
3. [Control Structures Reference](#control-structures-reference) - Making decisions in code

### 🔧 Core Skills

4. [Functions & Lambda Cheat Sheet](#functions--lambda-cheat-sheet) - Reusable code blocks
5. [String Operations Reference](#string-operations-reference) - Working with text
6. [File Operations Quick Guide](#file-operations-quick-guide) - Reading and writing files

### 🚀 Advanced Applications

7. [Database Operations Reference](#database-operations-reference) - Storing data
8. [Web Development Libraries](#web-development-libraries) - Building web applications
9. [Data Science Libraries](#data-science-libraries) - Data analysis and visualization
10. [System Programming Reference](#system-programming-reference) - System interactions

### 💡 Professional Development

11. [GUI Development Reference](#gui-development-reference) - Creating desktop applications
12. [Testing & Debugging Checklist](#testing--debugging-checklist) - Ensuring code quality
13. [Common Patterns & Idioms](#common-patterns--idioms) - Python best practices
14. [Performance Optimization Tips](#performance-optimization-tips) - Making code faster
15. [Security Best Practices](#security-best-practices) - Writing secure code
16. [Error Resolution Guide](#error-resolution-guide) - Fixing common problems

---

---

## Python Basics Quick Reference

### 🎯 What You'll Learn

- How to store information (variables)
- Different types of data
- Basic operations
- How to write comments

### ✅ Variables & Assignment

> **Think of variables like labeled boxes where you store things!**

```python
# Basic assignment - like putting something in a box
x = 10                    # Store the number 10 in box 'x'
name = "Alice"           # Store the text "Alice" in box 'name'
is_valid = True          # Store True (yes/true) in box 'is_valid'

# Multiple assignment - filling multiple boxes at once
x, y, z = 1, 2, 3        # Put 1 in x, 2 in y, 3 in z
x = y = z = 0            # Put 0 in all three boxes

# Swapping - exchanging contents
x, y = y, x              # Swap what's in x and y

# Constants (things that don't change)
PI = 3.14159             # Mathematical constant
MAX_SIZE = 100           # Maximum size limit

# 🔧 Checking what's in your boxes
type(x)                  # Find out what type is in box x
isinstance(x, int)       # Check if x contains a number

# 🔄 Converting between types
int("42")               # Convert text "42" to number 42
float("3.14")           # Convert text "3.14" to decimal 3.14
str(42)                  # Convert number 42 to text "42"
bool(1)                  # Convert to True (anything non-zero is True)
```

💡 **Beginner Tip**: Variable names should describe what they store! Use `age` instead of `a`, `user_name` instead of `un`.

```python
# Basic assignment
x = 10
name = "Alice"
is_valid = True

# Multiple assignment
x, y, z = 1, 2, 3
x = y = z = 0

# Swapping
x, y = y, x

# Constants (by convention)
PI = 3.14159
MAX_SIZE = 100

# Type checking
type(x)           # Get type
isinstance(x, int)  # Check type

# Type conversion
int("42")         # 42
float("3.14")     # 3.14
str(42)           # "42"
bool(1)           # True
```

### 🧮 Operators

> **Operators are like math symbols you use to work with data!**

```python
# ➕ Arithmetic (math operations)
+     # Addition: 5 + 3 = 8
-     # Subtraction: 5 - 3 = 2
*     # Multiplication: 5 * 3 = 15
/     # Division: 6 / 3 = 2.0
//    # Floor Division: 7 // 3 = 2 (drops decimals)
%     # Modulo (remainder): 7 % 3 = 1
**    # Exponent (power): 2 ** 3 = 8

# 🔍 Comparison (asking questions)
==    # Equal to: 5 == 5 is True
!=    # Not equal to: 5 != 3 is True
<     # Less than: 3 < 5 is True
>     # Greater than: 5 > 3 is True
<=    # Less than or equal: 3 <= 5 is True
>=    # Greater than or equal: 5 >= 5 is True

# 🔀 Logical (combining yes/no questions)
and   # Both must be True: True and True = True
or    # Either can be True: True or False = True
not   # Flip the answer: not True = False

# 📦 Assignment (storing results)
=     # Store: x = 5
+=    # Add and store: x += 3  (same as x = x + 3)
-=    # Subtract and store: x -= 3
*=    # Multiply and store: x *= 3
/=    # Divide and store: x /= 3
```

💡 **Beginner Tip**: Use parentheses `()` to control the order of operations, just like in math!

⚠️ **Important**: `==` checks if things are equal, `=` puts a value somewhere!

### 💬 Comments & Docstrings

> **Comments are notes to your future self and others!**

```python
# Single line comment - Python ignores everything after #
# Use these to explain what your code does

"""
Multi-line docstring (triple quotes)
Great for explaining:
- What a function does
- What parameters it needs
- What it returns

def calculate_area(length, width):
    '''
    Calculate the area of a rectangle.

    Parameters:
        length (float): The length of the rectangle
        width (float): The width of the rectangle

    Returns:
        float: The area of the rectangle
    '''
    return length * width
"""

def greet_user(name):
    """Say hello to a user by name."""
    print(f"Hello, {name}!")

class Dog:
    """Represents a dog with a name and breed."""

    def __init__(self, name, breed):
        """Create a new dog."""
        self.name = name
        self.breed = breed

    def bark(self):
        """Make the dog bark."""
        print(f"{self.name} says: Woof!")
```

✅ **Best Practice**: Write comments that explain WHY, not WHAT your code does!
💡 **Tip**: Good variable and function names often eliminate the need for comments!

### 💬 Input/Output

> **How your program talks to the user!**

```python
# ⌨️ Getting Input from User
name = input("Enter your name: ")  # Shows message and waits for typing
print(f"Nice to meet you, {name}!")  # Greets the user

# Getting numbers (always returns text, so convert it!)
age = int(input("Enter your age: "))  # Convert text to number
height = float(input("Enter your height: "))  # Convert to decimal

# 📺 Showing Output
print("Hello, world!")              # Print text to screen
print(f"You are {age} years old")    # f-strings (Python 3.6+)
print("Name: {}, Age: {}".format(name, age))  # Older style
print("Welcome", name, "!", sep="-", end="\n")  # Custom separator & ending

# 📝 Saving to File
with open("output.txt", "w") as file:  # Open file for writing
    file.write("Hello, file!")
    file.write(f"Name: {name}\n")
```

⚠️ **Common Mistake**: `input()` always returns text (string), so convert it to numbers when needed!
💡 **Tip**: Use `sep` to change what's between items, `end` to change what comes after

---

## Data Structures Cheat Sheet

### 🎯 What You'll Learn

- Different containers for storing multiple pieces of data
- When to use each type
- Common operations for each

### 📋 Lists

> **Lists are like shopping lists - ordered collections of items you can change!**

```python
# 🛒 Creating Lists (Shopping Lists)
numbers = [1, 2, 3, 4, 5]           # List of numbers
fruits = ["apple", "banana", "orange"]  # List of text
mixed = [1, "hello", 3.14, True]   # Mixed types (not recommended!)
empty_list = []                    # Empty list
range_list = list(range(10))       # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 🔍 Accessing Items (Finding items in your list)
numbers[0]          # First item: 1
numbers[-1]         # Last item: 5
numbers[1:3]        # Items from index 1 to 3: [2, 3]
numbers[::2]        # Every other item: [1, 3, 5]

# ✏️ Modifying Lists (Changing your shopping list)
numbers.append(6)        # Add to end: [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)     # Insert at position: [0, 1, 2, 3, 4, 5, 6]
numbers.remove(3)        # Remove first occurrence of 3
numbers.pop()            # Remove and return last item
numbers.pop(0)           # Remove and return item at index 0

# 📊 List Information
len(numbers)             # How many items: 6
numbers.count(2)         # How many times 2 appears
numbers.index(3)         # Where is 3 first found?
3 in numbers             # Is 3 in the list? (True/False)

# 🔄 List Operations
numbers.sort()           # Sort in place: [0, 1, 2, 3, 4, 5, 6]
sorted(numbers)          # Return sorted copy
numbers.reverse()        # Reverse in place
numbers.clear()          # Remove all items: []

# ✨ List Comprehension (Creating lists efficiently)
squares = [x**2 for x in range(10)]           # [0, 1, 4, 9, 16, 81]
evens = [x for x in numbers if x % 2 == 0]   # [0, 2, 4, 6]
```

✅ **Beginner Tip**: Lists keep items in order and you can have duplicates!
⚠️ **Important**: Lists start counting from 0, not 1!

```python
# Creation
numbers = [1, 2, 3, 4, 5]
empty_list = []
range_list = list(range(10))

# Accessing
numbers[0]          # First element
numbers[-1]         # Last element
numbers[1:3]        # Sublist [2, 3]
numbers[::2]        # Every second element [1, 3, 5]

# Modifying
numbers.append(6)   # Add to end
numbers.insert(0, 0)  # Insert at index
numbers.remove(3)   # Remove first occurrence
numbers.pop()       # Remove and return last
numbers.pop(0)      # Remove and return index

# List operations
len(numbers)        # Length
numbers.count(3)    # Count occurrences
numbers.index(3)    # First index of value
3 in numbers        # Membership test

# List methods
numbers.sort()      # Sort in place
sorted(numbers)     # Return sorted copy
numbers.reverse()   # Reverse in place
numbers.clear()     # Remove all elements

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in numbers if x % 2 == 0]
```

### 📚 Dictionaries

> **Dictionaries are like real dictionaries - you look up a word (key) to get its definition (value)!**

```python
# 📖 Creating Dictionaries (Word Definitions)
person = {"name": "Alice", "age": 30}      # Key-value pairs
empty_dict = {}                             # Empty dictionary
dict_from_tuples = dict([("a", 1), ("b", 2)])  # From list of tuples

# 🔍 Accessing Values (Looking up definitions)
name = person["name"]          # Gets value, but error if key missing
name = person.get("name")      # Gets value, None if missing
name = person.get("name", "Unknown")  # Gets value, "Unknown" if missing

# ✏️ Modifying Dictionaries (Updating your dictionary)
person["age"] = 31             # Change existing value
person["city"] = "NYC"         # Add new key-value pair
person.update({"age": 32, "city": "SF"})  # Update multiple at once

# 🛠️ Dictionary Methods
person.keys()                   # Get all keys: dict_keys(['name', 'age', 'city'])
person.values()                 # Get all values: dict_values(['Alice', 31, 'NYC'])
person.items()                  # Get all pairs: dict_items([('name', 'Alice'), ...])
person.pop("city")             # Remove and return value
person.popitem()                # Remove and return last item
person.clear()                  # Remove all items: {}

# ✨ Dictionary Comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

✅ **Beginner Tip**: Use dictionaries when you need to look things up by name!
💡 **Tip**: Keys must be unique and immutable (strings, numbers, tuples)

```python
# Creation
person = {"name": "Alice", "age": 30}
empty_dict = {}
dict_from_tuples = dict([("a", 1), ("b", 2)])

# Accessing
name = person["name"]        # KeyError if missing
name = person.get("name")    # None if missing
name = person.get("name", "Unknown")  # Default value

# Modifying
person["age"] = 31           # Update value
person["city"] = "NYC"       # Add new key-value
person.update({"age": 32, "city": "SF"})  # Update multiple

# Dictionary methods
person.keys()        # All keys
person.values()      # All values
person.items()       # All key-value pairs
person.pop("city")   # Remove and return value
person.popitem()     # Remove and return last item
person.clear()       # Remove all items

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

### Sets

```python
# Creation
numbers = {1, 2, 3, 4, 5}
empty_set = set()
set_from_list = set([1, 2, 3])

# Operations
numbers.add(6)              # Add element
numbers.remove(3)           # Remove (KeyError if missing)
numbers.discard(3)          # Remove (no error if missing)
numbers.pop()               # Remove arbitrary element

# Set operations
a = {1, 2, 3}
b = {3, 4, 5}

a.union(b)                  # {1, 2, 3, 4, 5}
a.intersection(b)           # {3}
a.difference(b)             # {1, 2}
a.symmetric_difference(b)   # {1, 2, 4, 5}

a.issubset(b)               # Is a subset of b?
a.issuperset(b)             # Is a superset of b?
```

### Tuples

```python
# Creation
point = (3, 4)
single_item = (1,)          # Note the comma
empty_tuple = ()

# Accessing
x, y = point               # Unpacking
first = point[0]           # Indexing

# Tuple operations
len(point)                 # Length
x, y, z = (1, 2, 3)       # Multiple unpacking
```

---

## Control Structures Reference

### If Statements

```python
# Basic if
if condition:
    # code

# If-else
if condition:
    # code
else:
    # code

# If-elif-else
if condition1:
    # code
elif condition2:
    # code
else:
    # code

# Ternary operator
value = "yes" if condition else "no"
```

### Loops

```python
# For loops
for item in iterable:
    # code

for i in range(5):         # 0, 1, 2, 3, 4
for i in range(1, 6):      # 1, 2, 3, 4, 5
for i in range(0, 10, 2):  # 0, 2, 4, 6, 8

# Enumerate
for index, value in enumerate(items):
    # code

# Zip
for item1, item2 in zip(list1, list2):
    # code

# While loops
while condition:
    # code

# Loop control
break        # Exit loop
continue     # Skip to next iteration
else:        # Executes if loop completes normally
    # code
```

### Comprehensions

```python
# List comprehension
squares = [x**2 for x in range(10)]

# Dictionary comprehension
square_dict = {x: x**2 for x in range(5)}

# Set comprehension
unique_squares = {x**2 for x in [-2, -1, 0, 1, 2]}

# Generator expression
squares_gen = (x**2 for x in range(1000000))
```

---

## Functions & Lambda Cheat Sheet

### Function Definition

```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Multiple return values
def get_coordinates():
    return x, y, z

# Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# *args (variable positional arguments)
def sum_all(*args):
    return sum(args)

# **kwargs (variable keyword arguments)
def create_profile(**kwargs):
    return kwargs

# Combined
def flexible_function(a, b, *args, **kwargs):
    # a, b are positional
    # args is tuple of extra positional
    # kwargs is dict of extra keyword
```

### Lambda Functions

```python
# Basic lambda
square = lambda x: x**2

# With multiple parameters
add = lambda x, y: x + y

# Common uses
numbers.sort(key=lambda x: x)           # Sort by value
filtered = filter(lambda x: x > 0, nums)  # Filter numbers
mapped = map(lambda x: x*2, nums)       # Double numbers
```

### Decorators

```python
# Simple decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

@timer
def my_function():
    # function code
    pass

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def my_function():
    # function code
    pass
```

### Context Managers

```python
# Custom context manager
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileManager("data.txt", "r") as f:
    content = f.read()

# Using contextlib
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()
```

---

## String Operations Reference

### Basic Operations

```python
# Concatenation
"Hello" + " " + "World"          # "Hello World"
"Hello" * 3                      # "HelloHelloHello"

# String formatting
name = "Alice"
age = 25

# f-strings (Python 3.6+)
message = f"Hello, {name}! You are {age} years old."

# format() method
message = "Hello, {}! You are {} years old.".format(name, age)
message = "Hello, {name}! You are {age} years old.".format(name=name, age=age)

# % formatting (older style)
message = "Hello, %s! You are %d years old." % (name, age)
```

### String Methods

```python
# Case conversion
"hello".upper()          # "HELLO"
"HELLO".lower()          # "hello"
"hello".title()          # "Hello"
"hELLO".swapcase()       # "Hello"

# String cleaning
"  hello  ".strip()      # "hello" (remove whitespace)
"hello".lstrip()         # "hello" (remove left whitespace)
"hello".rstrip()         # "hello" (remove right whitespace)

# String searching
"hello world".find("world")     # 6 (index) or -1 if not found
"hello world".index("world")    # 6 (raises ValueError if not found)
"hello".startswith("he")        # True
"hello".endswith("lo")          # True
"hello".count("l")              # 2

# String replacement
"hello world".replace("world", "Python")  # "hello Python"
"hello".replace("l", "L", 1)              # "heLlo" (replace only first)

# String splitting
"hello,world".split(",")        # ["hello", "world"]
"hello\nworld".splitlines()     # ["hello", "world"]

# String joining
",".join(["a", "b", "c"])      # "a,b,c"
" ".join(["Hello", "World"])    # "Hello World"
```

### Regular Expressions

```python
import re

# Basic patterns
pattern = r'\d+'                # One or more digits
pattern = r'\w+'                # One or more word characters
pattern = r'[a-zA-Z]'           # Single letter
pattern = r'[0-9]{3}'           # Exactly 3 digits

# Common functions
re.search(pattern, text)        # Find first match
re.findall(pattern, text)       # Find all matches
re.sub(pattern, replacement, text)  # Replace matches

# Example usage
import re

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
email = re.search(email_pattern, text)
```

---

## File Operations Quick Guide

### Basic File Operations

```python
# Reading files
with open("file.txt", "r") as f:
    content = f.read()                    # Read entire file
    content = f.read(100)                 # Read first 100 characters
    line = f.readline()                   # Read one line
    lines = f.readlines()                 # Read all lines into list

# Writing files
with open("file.txt", "w") as f:
    f.write("Hello, world!")              # Write text
    f.writelines(["line1\n", "line2\n"]) # Write multiple lines

# File modes
"r"    # Read (default)
"w"    # Write (overwrites existing)
"a"    # Append
"r+"   # Read and write
"rb"   # Read binary
"wb"   # Write binary
```

### Path Operations

```python
import os
from pathlib import Path

# Basic path operations
os.path.exists("file.txt")       # Check if file/directory exists
os.path.isfile("file.txt")       # Check if it's a file
os.path.isdir("folder")          # Check if it's a directory
os.path.getsize("file.txt")      # Get file size in bytes

# Pathlib (recommended)
path = Path("folder/file.txt")
path.exists()
path.is_file()
path.is_dir()
path.stat()                      # File statistics

# Creating directories
os.makedirs("new_folder", exist_ok=True)  # Create with parents
path.mkdir(exist_ok=True)                 # Create single directory

# File copying/moving
import shutil
shutil.copy("source.txt", "dest.txt")     # Copy file
shutil.move("source.txt", "dest.txt")     # Move file
```

### CSV Operations

```python
import csv

# Reading CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # row is a list of strings
        process(row)

# Writing CSV
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])      # Write header
    writer.writerows([["Alice", 25], ["Bob", 30]])  # Write multiple rows

# Dictionary CSV
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # row is a dictionary
        print(row["Name"], row["Age"])
```

### JSON Operations

```python
import json

# Reading JSON
with open("data.json", "r") as f:
    data = json.load(f)           # Load as Python object

# Writing JSON
data = {"name": "Alice", "age": 25}
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)  # Write with formatting

# String operations
json_string = '{"name": "Alice", "age": 25}'
data = json.loads(json_string)    # Parse JSON string
json_str = json.dumps(data)       # Convert to JSON string
```

---

## Database Operations Reference

### SQLite3

```python
import sqlite3

# Connect to database
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Execute queries
cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
cursor.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
cursor.execute("SELECT * FROM users")
results = cursor.fetchall()

# Execute multiple queries
users = [("Bob",), ("Charlie",)]
cursor.executemany("INSERT INTO users (name) VALUES (?)", users)

# Transaction management
conn.commit()              # Save changes
conn.rollback()            # Undo changes
conn.close()               # Close connection

# With context manager
with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    # Database operations
    # Auto-committed or rolled back
```

### Pandas Database Operations

```python
import pandas as pd
import sqlalchemy

# Create SQLAlchemy engine
engine = sqlalchemy.create_engine("sqlite:///database.db")

# Read from database
df = pd.read_sql("SELECT * FROM users", engine)
df = pd.read_sql_table("users", engine)
df = pd.read_sql_query("SELECT * FROM users WHERE age > 25", engine)

# Write to database
df.to_sql("new_users", engine, if_exists="replace", index=False)

# Parameters in queries
query = "SELECT * FROM users WHERE age > :age"
df = pd.read_sql(query, engine, params={"age": 25})
```

---

## Web Development Libraries

### Requests

```python
import requests

# Basic GET request
response = requests.get("https://api.example.com/data")
print(response.status_code)    # 200
print(response.text)           # Response content
print(response.json())         # Parse JSON

# POST request
data = {"name": "Alice", "age": 25}
response = requests.post("https://api.example.com/users", json=data)

# Headers and parameters
headers = {"Authorization": "Bearer token"}
params = {"q": "search term", "page": 1}
response = requests.get("https://api.example.com/search",
                       headers=headers, params=params)

# Session (persistent connection)
session = requests.Session()
session.headers.update({"User-Agent": "MyApp/1.0"})
response = session.get("https://api.example.com/data")
```

### BeautifulSoup (Web Scraping)

```python
from bs4 import BeautifulSoup
import requests

# Get webpage
response = requests.get("https://example.com")
soup = BeautifulSoup(response.content, "html.parser")

# Find elements
soup.find("h1")                        # First h1 tag
soup.find_all("p")                     # All p tags
soup.find("div", class_="content")     # By class
soup.find("a", href=True)              # By attribute

# CSS selectors
soup.select(".class-name")             # By class
soup.select("#element-id")             # By ID
soup.select("div > p")                 # Direct children

# Extract data
title = soup.find("title").text
links = [a.get("href") for a in soup.find_all("a")]
```

---

## Data Science Libraries

### Pandas

```python
import pandas as pd

# DataFrame creation
df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": ["a", "b", "c"],
    "C": [1.1, 2.2, 3.3]
})

# Reading data
df = pd.read_csv("data.csv")
df = pd.read_excel("data.xlsx")
df = pd.read_json("data.json")

# Basic operations
df.head()                              # First 5 rows
df.tail()                              # Last 5 rows
df.info()                              # DataFrame info
df.describe()                          # Statistical summary
df.shape                               # Dimensions (rows, columns)
df.columns                             # Column names
df.dtypes                              # Data types

# Indexing and selection
df["A"]                                # Column A
df[["A", "B"]]                         # Multiple columns
df.iloc[0]                             # First row (integer location)
df.loc[0]                              # First row (label)
df.loc[0, "A"]                         # Specific value
df[df["A"] > 2]                        # Filter rows

# Data manipulation
df.sort_values("A")                    # Sort by column A
df.groupby("B").sum()                  # Group and aggregate
df.pivot_table(values="C", index="A", columns="B")  # Pivot table

# Missing data
df.dropna()                            # Remove rows with missing values
df.fillna(0)                           # Fill missing values with 0
df.isnull().sum()                      # Count missing values per column

# Writing data
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
```

### NumPy

```python
import numpy as np

# Array creation
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros(5)                    # Array of zeros
ones = np.ones((3, 3))                 # 2D array of ones
random = np.random.rand(5)             # Random numbers
range_arr = np.arange(0, 10, 2)        # Range with step

# Array operations
arr + 10                               # Add 10 to each element
arr * 2                                # Multiply by 2
arr.sum()                              # Sum of all elements
arr.mean()                             # Average
arr.std()                              # Standard deviation

# Array indexing
arr[0]                                 # First element
arr[1:3]                               # Subarray
arr[arr > 2]                           # Elements greater than 2

# Matrix operations
matrix = np.array([[1, 2], [3, 4]])
matrix.T                               # Transpose
np.dot(matrix, matrix)                 # Matrix multiplication
```

### Matplotlib

```python
import matplotlib.pyplot as plt

# Basic plotting
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])   # Line plot
plt.scatter([1, 2, 3, 4], [1, 4, 2, 3]) # Scatter plot
plt.bar([1, 2, 3, 4], [1, 4, 2, 3])    # Bar chart
plt.hist([1, 1, 2, 3, 3, 3, 4, 4, 5])  # Histogram

# Plot customization
plt.xlabel("X Label")
plt.ylabel("Y Label")
plt.title("My Plot")
plt.legend(["Line 1"])
plt.grid(True)

# Multiple plots
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)                   # 2x2 grid, first subplot
plt.plot(x, y1)
plt.subplot(2, 2, 2)                   # Second subplot
plt.plot(x, y2)

# Save plot
plt.savefig("plot.png", dpi=300, bbox_inches="tight")
plt.show()
```

---

## System Programming Reference

### OS Module

```python
import os

# File and directory operations
os.listdir(".")                        # List directory contents
os.getcwd()                            # Get current working directory
os.chdir("/path/to/directory")         # Change directory
os.makedirs("new_folder", exist_ok=True)  # Create directory tree

# File operations
os.remove("file.txt")                  # Delete file
os.rename("old.txt", "new.txt")        # Rename file
os.path.exists("file.txt")             # Check if path exists
os.path.isfile("file.txt")             # Check if it's a file
os.path.isdir("folder")                # Check if it's a directory

# Environment variables
os.environ["HOME"]                     # Get environment variable
os.getenv("HOME", "/default")          # Get with default value

# Path operations
os.path.join("folder", "file.txt")     # Join path components
os.path.dirname("/path/to/file.txt")   # Directory path
os.path.basename("/path/to/file.txt")  # Filename
os.path.splitext("file.txt")           # Split extension
```

### Subprocess

```python
import subprocess

# Run command
result = subprocess.run(["ls", "-la"],
                       capture_output=True,
                       text=True)
print(result.stdout)
print(result.stderr)
print(result.returncode)

# Using shell
subprocess.run("ls -la", shell=True)

# Other functions
subprocess.call(["echo", "Hello"])     # Like run but returns return code
subprocess.check_output(["date"])      # Like run but raises exception on error

# Popen (more control)
process = subprocess.Popen(["python", "script.py"],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
```

### Threading

```python
import threading
import time

# Basic threading
def worker():
    for i in range(5):
        print(f"Worker: {i}")
        time.sleep(1)

# Create and start thread
thread = threading.Thread(target=worker)
thread.start()

# Wait for thread to complete
thread.join()

# Thread with arguments
def worker(name, count):
    print(f"Worker {name}: {count}")

thread = threading.Thread(target=worker, args=("A", 10))
thread.start()
thread.join()

# Thread pool
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(worker, i) for i in range(10)]
    for future in futures:
        future.result()  # Wait for completion
```

---

## GUI Development Reference

### Tkinter Basics

```python
import tkinter as tk
from tkinter import ttk

# Create main window
root = tk.Tk()
root.title("My Application")
root.geometry("400x300")

# Labels
label = tk.Label(root, text="Hello, World!")
label.pack()

# Buttons
button = tk.Button(root, text="Click Me", command=click_handler)
button.pack()

# Entry (text input)
entry = tk.Entry(root, width=30)
entry.pack()

# Text widget
text_widget = tk.Text(root, height=5, width=40)
text_widget.pack()

# Checkbutton
var = tk.BooleanVar()
checkbutton = tk.Checkbutton(root, text="Option", variable=var)
checkbutton.pack()

# Radiobutton
var = tk.StringVar()
radio1 = tk.Radiobutton(root, text="Option 1", variable=var, value="1")
radio2 = tk.Radiobutton(root, text="Option 2", variable=var, value="2")
radio1.pack()
radio2.pack()

# Listbox
listbox = tk.Listbox(root)
listbox.pack()
listbox.insert(0, "Item 1")
listbox.insert(1, "Item 2")

# Menu
menubar = tk.Menu(root)
root.config(menu=menubar)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="New")
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

# Event handlers
def click_handler():
    label.config(text=f"Button clicked! Entry: {entry.get()}")

# Run the application
root.mainloop()
```

### Layout Managers

```python
# Pack (simplest)
widget.pack()                          # Pack at default position
widget.pack(side="left")               # Pack on left side
widget.pack(fill="x")                  # Fill horizontal space
widget.pack(expand=True)               # Expand to fill available space

# Grid (precise positioning)
widget.grid(row=0, column=0)           # Position at row 0, column 0
widget.grid(row=0, column=1, columnspan=2)  # Span 2 columns
widget.grid(row=1, column=0, sticky="ew")   # Stick to east-west

# Configure grid weights
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Frame (container for grouping widgets)
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)
```

---

## Testing & Debugging Checklist

### Testing with unittest

```python
import unittest

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
```

### Testing with pytest

```python
import pytest

def test_upper():
    assert 'foo'.upper() == 'FOO'

def test_isupper():
    assert 'FOO'.isupper() is True
    assert 'Foo'.isupper() is False

# Fixtures
@pytest.fixture
def sample_data():
    return {'name': 'Alice', 'age': 25}

def test_user(sample_data):
    assert sample_data['name'] == 'Alice'

# Parametrized tests
@pytest.mark.parametrize("input,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square(input, expected):
    assert input ** 2 == expected
```

### Debugging Techniques

```python
# Print debugging
print(f"Debug: variable = {variable}")

# Logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")

# pdb debugger
import pdb
pdb.set_trace()  # Set breakpoint
# Commands: n (next), s (step), c (continue), l (list), p variable

# Assertions
assert condition, "Error message"
assert x > 0, "x must be positive"

# Try-except for error handling
try:
    risky_operation()
except SpecificError as e:
    handle_error(e)
except Exception as e:
    handle_generic_error(e)
finally:
    cleanup_always()
```

### Common Debugging Tools

```python
# Timing code execution
import time
start = time.time()
# code to time
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# Memory usage
import tracemalloc
tracemalloc.start()
# your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")

# Profiling
import cProfile
cProfile.run('your_function()')

# IDE debugger breakpoints (set in IDE)
```

---

## Common Patterns & Idioms

### Iterator Pattern

```python
# Custom iterator
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# Usage
for num in Countdown(5):
    print(num)
```

### Context Manager Pattern

```python
# Resource management
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Usage
with FileManager("data.txt", "r") as f:
    content = f.read()
```

### Factory Pattern

```python
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError("Unknown animal type")

# Usage
dog = AnimalFactory.create_animal("dog")
print(dog.speak())
```

### Singleton Pattern

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Usage
s1 = Singleton()
s2 = Singleton()
assert s1 is s2  # True
```

### Observer Pattern

```python
class Subject:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def detach(self, observer):
        self.observers.remove(observer)

    def notify(self, message):
        for observer in self.observers:
            observer.update(message)

class Observer:
    def __init__(self, name):
        self.name = name

    def update(self, message):
        print(f"{self.name} received: {message}")

# Usage
subject = Subject()
observer1 = Observer("Observer 1")
observer2 = Observer("Observer 2")

subject.attach(observer1)
subject.attach(observer2)
subject.notify("Hello!")
```

---

## Performance Optimization Tips

### Memory Efficiency

```python
# Use generators for large datasets
def large_range():
    for i in range(1000000):
        yield i

# Instead of list comprehension
large_list = [i for i in range(1000000)]  # Uses lots of memory
large_generator = (i for i in range(1000000))  # Memory efficient

# Use __slots__ for many instances
class Point:
    __slots__ = ['x', 'y']  # Reduces memory usage
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Use collections for specialized containers
from collections import deque, defaultdict, Counter

# deque for efficient append/pop from both ends
queue = deque([1, 2, 3])
queue.appendleft(0)  # O(1) operation

# defaultdict for automatic default values
d = defaultdict(list)
d['key'].append('value')  # No KeyError

# Counter for counting hashable objects
counter = Counter([1, 2, 2, 3, 3, 3])
```

### CPU Efficiency

```python
# Use built-in functions (written in C)
nums = [1, 2, 3, 4, 5]
total = sum(nums)  # Faster than manual loop
max_val = max(nums)
min_val = min(nums)

# List comprehensions vs loops
# List comprehension is faster
squares = [x**2 for x in range(1000)]  # Fast
squares = []
for x in range(1000):                   # Slower
    squares.append(x**2)

# Use appropriate data structures
# Set for membership testing
large_list = list(range(1000000))
large_set = set(range(1000000))

1000000 in large_list  # Slow - O(n)
1000000 in large_set   # Fast - O(1)

# Use numpy for numerical operations
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = arr * 2  # Vectorized operation

# For loops are slow in Python
# Consider using vectorized operations with NumPy
```

### Caching and Memoization

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Manual caching
cache = {}
def expensive_function(x):
    if x in cache:
        return cache[x]
    result = x * x * x  # Expensive operation
    cache[x] = result
    return result
```

---

## Security Best Practices

### Input Validation

```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    # Check length
    if len(password) < 8:
        return False, "Password too short"

    # Check complexity
    if not re.search(r'[A-Z]', password):
        return False, "Missing uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Missing lowercase letter"
    if not re.search(r'\d', password):
        return False, "Missing number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Missing special character"

    return True, "Password is valid"

# Use parameterization to prevent SQL injection
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
```

### Data Sanitization

```python
import html

def sanitize_html(text):
    return html.escape(text)

# Remove dangerous characters
def sanitize_filename(filename):
    # Remove path traversal
    filename = filename.replace('..', '')
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    return filename

# Validate file uploads
import os
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.png', '.jpg'}

def validate_file_upload(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, "File type not allowed"

    # Check file size
    file_size = os.path.getsize(file_path)
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        return False, "File too large"

    return True, "File is valid"
```

### Cryptography

```python
import hashlib
import secrets
import base64
from cryptography.fernet import Fernet

# Hashing passwords (use bcrypt or argon2 in production)
def hash_password(password):
    salt = secrets.token_hex(32)
    password_hash = hashlib.pbkdf2_hmac('sha256',
                                       password.encode('utf-8'),
                                       salt.encode('utf-8'),
                                       100000)
    return base64.b64encode(salt + password_hash).decode('utf-8')

def verify_password(stored_hash, password):
    stored_bytes = base64.b64decode(stored_hash.encode('utf-8'))
    salt = stored_bytes[:32]
    stored_password_hash = stored_bytes[32:]

    password_hash = hashlib.pbkdf2_hmac('sha256',
                                       password.encode('utf-8'),
                                       salt,
                                       100000)
    return password_hash == stored_password_hash

# Generate secure tokens
token = secrets.token_urlsafe(32)

# Encrypt sensitive data
key = Fernet.generate_key()
cipher_suite = Fernet(key)
encrypted_data = cipher_suite.encrypt(b"Sensitive data")
decrypted_data = cipher_suite.decrypt(encrypted_data)
```

---

## Error Resolution Guide

### Common Error Types and Solutions

#### NameError

```python
# Error: name 'variable' is not defined
# Solution: Define the variable before using it

# Fix 1: Define the variable
x = 5
print(x)

# Fix 2: Check spelling
my_variable = 10
print(my_variiable)  # Typo
```

#### TypeError

```python
# Error: unsupported operand type(s) for +: 'int' and 'str'
# Solution: Ensure compatible types

# Fix 1: Convert types
age = 25
message = "I am " + str(age) + " years old"

# Fix 2: Use f-strings
message = f"I am {age} years old"
```

#### IndexError

```python
# Error: list index out of range
# Solution: Check list bounds

my_list = [1, 2, 3]
if len(my_list) > 3:
    print(my_list[3])
else:
    print("List is too short")
```

#### KeyError

```python
# Error: 2
# Solution: Use get() method or check if key exists

my_dict = {"a": 1, "b": 2}
value = my_dict.get("c", "default")  # Returns "default" instead of error
```

#### AttributeError

```python
# Error: 'list' object has no attribute 'append_left'
# Solution: Use correct method name

from collections import deque
my_deque = deque([1, 2, 3])
my_deque.appendleft(0)  # Correct method
```

#### ImportError

```python
# Error: No module named 'requests'
# Solution: Install the package

# In terminal: pip install requests
import requests
```

### Debugging Workflow

```python
# 1. Read the error message carefully
# 2. Identify the file and line number
# 3. Check the traceback for context
# 4. Use print() or logging to debug
# 5. Check variable types and values
# 6. Test in smaller pieces
# 7. Use debugger if needed

# Example debugging process
def debug_function(data):
    print(f"Debug: data = {data}")
    print(f"Debug: type(data) = {type(data)}")

    try:
        result = data['key']
        print(f"Debug: result = {result}")
        return result
    except KeyError as e:
        print(f"Debug: KeyError - {e}")
        print(f"Debug: available keys = {list(data.keys())}")
        return None
```

### Performance Debugging

```python
# Identify bottlenecks
import cProfile
cProfile.run('your_function()')

# Memory usage
import tracemalloc
tracemalloc.start()
# your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Memory usage: {current / 1024 / 1024:.1f} MB")

# Time measurement
import time
start_time = time.time()
# code to measure
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")
```

---

## 🌟 Congratulations!

You've reached the end of your Python Cheat Sheets journey! 🎉

### 🎆 What You've Learned

This guide has taken you through:

✅ **Python Basics** - Variables, operators, and basic operations  
📋 **Data Structures** - Lists, dictionaries, sets, and tuples  
🔀 **Control Structures** - Making decisions and loops  
🔧 **Functions** - Writing reusable code  
📝 **String & File Operations** - Working with text and files  
🚀 **Advanced Topics** - Libraries, optimization, and best practices

### 🔎 Next Steps

**🔰 If you're new to Python:**

1. Practice with the questions in the Assessment Guide
2. Try the Coding Challenges
3. Build a simple project (calculator, to-do list, etc.)
4. Join Python communities online

**🔄 If you're returning to Python:**

1. Review sections you haven't used recently
2. Focus on modern Python features (f-strings, type hints, etc.)
3. Practice with real-world examples
4. Consider contributing to open source

**💼 If you're preparing for interviews:**

1. Practice explaining concepts clearly
2. Work through all coding challenges
3. Study system design questions
4. Review security and performance sections

### 📚 Learning Resources

**🌐 Online Communities:**

- r/learnpython on Reddit
- Python.org community forums
- Stack Overflow for specific questions

**📖 Recommended Reading:**

- "Automate the Boring Stuff with Python" (free online)
- "Python Crash Course" by Eric Matthes
- Official Python documentation (docs.python.org)

**🎥 Video Learning:**

- Python.org tutorial videos
- YouTube channels (Corey Schafer, sentdex)
- Interactive coding platforms (codecademy.com)

### 🚀 Building Confidence

Remember:

- **✅ Everyone starts as a beginner** - Don't be intimidated!
- **🔄 Practice makes progress** - Code a little bit every day
- **💪 Mistakes are learning opportunities** - Debugging teaches you more than correct code
- **🤝 Ask for help** - The Python community is welcoming and supportive
- **🎆 Small steps lead to big achievements** - Build simple projects first

### 📝 Final Tips

1. **Keep this guide handy** - Bookmark it for quick reference
2. **Write code daily** - Even 15 minutes helps
3. **Explain concepts to others** - Teaching reinforces learning
4. **Read other people's code** - Learn from open source projects
5. **Don't give up** - Programming challenges are normal and part of the journey!

### 🎆 Your Python Journey Starts Now!

Whether you're:

- 🎓 A student learning your first programming language
- 🔄 A professional switching careers
- 💻 An experienced developer adding Python to your toolkit
- 🎯 Someone preparing for a technical interview

**You've got this!** 💪

This guide will be with you every step of the way. Come back whenever you need a refresher or want to tackle a new challenge.

**Happy coding!** 🚀

---

### 📜 Quick Reference Summary

**Most Common Operations:**

```python
# Lists
my_list.append(item)          # Add item
my_list.remove(item)          # Remove item
len(my_list)                  # Get length

# Dictionaries
my_dict['key'] = value        # Add/update
my_dict.get('key', default)   # Safe access
my_dict.keys()                # Get all keys

# Strings
my_string.lower()             # Lowercase
my_string.replace('a', 'b')   # Replace text
my_string.split(',')          # Split into list

# Files
with open('file.txt') as f:   # Open file safely
    content = f.read()        # Read content

# Loops
for item in list:             # Loop through items
    print(item)

for i in range(10):           # Loop with numbers
    print(i)
```

### ⚙️ Essential Commands

```python
# Type checking
type(variable)                # What type is it?
isinstance(value, int)        # Is it a number?

# Math operations
abs(-5)                       # Absolute value: 5
round(3.14159, 2)             # Round: 3.14
max([1, 2, 3])                # Maximum: 3
min([1, 2, 3])                # Minimum: 1
sum([1, 2, 3])                # Sum: 6

# String formatting
f"Hello {name}!"              # F-strings (preferred)
"Hello {}!".format(name)      # Format method

# List operations
sorted(my_list)               # Return sorted list
my_list.sort()                # Sort in place
list(reversed(my_list))       # Reverse list

# Checking membership
item in my_list               # Is item in list?
key in my_dict                # Is key in dict?
```

### 🚨 Common Errors & Quick Fixes

| Error              | Common Cause                      | Quick Fix                  |
| ------------------ | --------------------------------- | -------------------------- |
| `NameError`        | Using variable before defining it | Define variable first      |
| `TypeError`        | Wrong data type                   | Check types with `type()`  |
| `IndexError`       | Accessing invalid list index      | Check list length first    |
| `KeyError`         | Dictionary key doesn't exist      | Use `.get()` method        |
| `IndentationError` | Wrong spaces/tabs                 | Use consistent indentation |

### 🏆 You're Ready For:

- ✅ Building simple Python programs
- ✅ Solving coding interview questions
- ✅ Contributing to open source projects
- ✅ Taking online Python courses
- ✅ Building web applications with Flask/Django
- ✅ Data analysis with pandas
- ✅ Machine learning with scikit-learn

**Remember: Every expert was once a beginner. Your journey starts now!** 🌟
