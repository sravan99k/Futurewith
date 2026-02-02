# 🎒 Python Data Structures: A Student's Guide

_Master Lists, Tuples, Sets, Dictionaries & Strings - Made Simple!_

_This guide uses everyday school analogies to make Python data structures easy to understand!_

## Table of Contents

1. [Introduction to Data Structures](#1-introduction-to-data-structures)
2. [Lists: Ordered Mutable Sequences](#2-lists-ordered-mutable-sequences)
3. [Tuples: Ordered Immutable Sequences](#3-tuples-ordered-immutable-sequences)
4. [Sets: Unordered Unique Collections](#4-sets-unordered-unique-collections)
5. [Dictionaries: Key-Value Pairs](#5-dictionaries-key-value-pairs)
6. [Strings: Text Data Operations](#6-strings-text-data-operations)
7. [When to Use Each Data Structure](#7-when-to-use-each-data-structure)
8. [Advanced Operations & Methods](#8-advanced-operations--methods)
9. [Real-World Applications](#9-real-world-applications)
10. [Performance Considerations](#10-performance-considerations)
11. [Common Patterns & Best Practices](#11-common-patterns--best-practices)
12. [Practice Exercises](#12-practice-exercises)

---

## 1. Introduction to Data Structures

### What Are Data Structures?

**Think of data structures like different types of containers in your school:**

🎒 **Lists** = Your homework list (ordered, can check off tasks)
🏷️ **Tuples** = Your locker combination (fixed order, can't change)
📚 **Sets** = Unique clubs you're in (no duplicates allowed)
📖 **Dictionaries** = Your school directory (name → phone number)
📝 **Strings** = Individual words or phrases in your notes

### Why Different Data Structures?

**Real Scenario:** Using the wrong "container" at school can cause problems!

````python
# WRONG - Using list for unique student IDs
student_id_list = ["S001", "S001", "S002"]  # Duplicate ID?

# RIGHT - Using set for unique student IDs
student_ids = {"S001", "S002", "S003"}  # No duplicates!

# WRONG - Using lists for student lookup
student_names = ["Alice", "Bob", "Carol"]
student_grades = [95, 87, 92]
# How do you find Alice's grade? Must search through entire list!

# RIGHT - Using dictionary for student lookup
students = {
    "Alice": 95,
    "Bob": 87,
    "Carol": 92
}
print(students["Alice"])  # Direct lookup!

### Why Different Data Structures?

**Real Scenario:** Using wrong container = frustration!
```python
# WRONG - Using list for unique ingredients
shopping_list = ["apple", "apple", "banana"]  # Duplicate apples!

# RIGHT - Using set for unique items
available_items = {"apple", "banana", "orange"}  # No duplicates!

# WRONG - Using list for recipe lookup
recipe_names = ["pasta", "salad", "soup"]
recipe_instructions = ["boil water", "chop vegetables", "simmer"]
# How do you find "pasta" instructions? Must search through entire list!

# RIGHT - Using dictionary for recipe lookup
recipes = {
    "pasta": "boil water",
    "salad": "chop vegetables",
    "soup": "simmer"
}
print(recipes["pasta"])  # Direct lookup!
````

### Data Structure Hierarchy

```
📊 DATA STRUCTURES
├── 📝 Sequences (ordered collections)
│   ├── 📋 Lists (mutable, can change)
│   ├── 📦 Tuples (immutable, can't change)
│   └── 🔤 Strings (text, immutable)
├── 🔢 Sets (unordered, unique)
└── 🗝️ Dictionaries (key-value pairs)
```

---

## 2. Lists: Ordered Mutable Sequences

### What Are Lists?

**Think of lists like your class schedule or to-do list:**

- **Ordered** = 1st period, 2nd period, 3rd period (specific order)
- **Mutable** = Can add, remove, or change subjects as needed
- **Dynamic** = Your schedule can grow or shrink each semester

**Real School Examples:**

- 📅 Class schedule: ["Math", "English", "Science", "History"]
- ✅ Homework list: ["Math worksheet", "Read chapter 5", "Science lab"]
- 🍕 Cafeteria menu items: ["Pizza", "Burgers", "Salad", "Pasta"]

### Creating Lists

```python
# Different ways to create lists
empty_list = []                           # Empty class schedule
grade_scores = [85, 92, 78, 96, 89]       # Test scores
student_names = ["Alice", "Bob", "Carol", "David"]  # Class roster
mixed_types = [1, "Math", 3.14, True]     # Mixed homework data
period_subjects = [["Math", "Room 101"], ["English", "Room 102"]]  # Nested list

# Using list() constructor
class_periods = list(range(1, 8))         # [1, 2, 3, 4, 5, 6, 7] (7 periods)
letters_in_name = list("ALICE")           # ['A', 'L', 'I', 'C', 'E']

print(f"Grade scores: {grade_scores}")
print(f"Student names: {student_names}")
print(f"Class periods: {class_periods}")

# Real school examples
math_topics = ["Algebra", "Geometry", "Calculus", "Statistics"]
print(f"Math topics: {math_topics}")
print(f"First topic: {math_topics[0]}")  # Access first item
```

### List Indexing & Slicing

```python
subjects = ["Math", "English", "Science", "History", "Art", "PE"]

# Indexing (accessing single items)
print(subjects[0])      # First item: "Math"
print(subjects[-1])     # Last item: "PE"
print(subjects[2])      # Third item: "Science"

# Slicing (accessing portions like selecting periods)
morning_classes = subjects[0:3]       # ["Math", "English", "Science"] (1st-3rd period)
afternoon_classes = subjects[3:]      # ["History", "Art", "PE"] (4th-6th period)
every_other_class = subjects[::2]     # ["Math", "Science", "Art"] (every 2nd class)

# Negative slicing (counting from end)
last_three = subjects[-3:]            # ["History", "Art", "PE"] (last 3 classes)
all_except_last = subjects[:-1]       # ["Math", "English", "Science", "History", "Art"] (except PE)

print(f"Morning classes: {morning_classes}")
print(f"Every other: {every_other_class}")
print(f"Last three: {last_three}")

# Real school scenario - getting schedule portions
print(f"1st period: {subjects[0]}")    # Math
print(f"Last period: {subjects[-1]}")  # PE
```

### List Modification Methods

```python
homework = ["Math", "English"]    # Starting homework list

# Adding homework assignments
homework.append("Science")        # Add to end: ["Math", "English", "Science"]
homework.insert(1, "History")     # Insert at position 1: ["Math", "History", "English", "Science"]
homework.extend(["Art", "PE"])    # Add multiple: ["Math", "History", "English", "Science", "Art", "PE"]

# Completing (removing) homework
completed = homework.pop()        # Remove and return last item (PE)
homework.remove("History")        # Remove specific assignment
homework.clear()                  # Remove all assignments when done!

print(f"Completed: {completed}")
print(f"Remaining homework: {homework}")

# Real school example - managing class roster
class_roster = ["Alice", "Bob"]   # Starting with 2 students
class_roster.append("Carol")      # New student joins: ["Alice", "Bob", "Carol"]
class_roster.insert(1, "David")   # Student transfers in: ["Alice", "David", "Bob", "Carol"]

# Student leaves
absent_student = class_roster.pop()  # Last student absent today
print(f"Absent today: {absent_student}")
print(f"Present students: {class_roster}")
```

### List Operations & Methods

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# Sorting and reversing
numbers.sort()                    # [1, 1, 2, 3, 4, 5, 6, 9]
numbers.sort(reverse=True)        # [9, 6, 5, 4, 3, 2, 1, 1]
numbers.reverse()                 # Reverse in place

# Searching
index = numbers.index(4)          # Find index of value 4
count = numbers.count(1)          # Count occurrences of 1
exists = 5 in numbers             # Check if 5 exists

# List comprehension (powerful feature!)
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

print(f"Squares: {squares}")
print(f"Evens: {evens}")
```

---

## 3. Tuples: Ordered Immutable Sequences

### What Are Tuples?

**Think of tuples like your locker combination or GPS coordinates:**

- **Fixed** = Your locker combination doesn't change during the year
- **Ordered** = First number, second number, third number (order matters!)
- **Immutable** = Cannot be changed after setting

**Real School Examples:**

- 🔢 Locker combination: (42, 17, 8) - always in same order
- 📍 Classroom location: (Building A, Room 205) - coordinates that don't change
- 🎓 Graduation year: (2024, 6, 15) - month and day are fixed

### Creating Tuples

```python
# Different ways to create tuples
empty_tuple = ()                  # Empty tuple
single_item = (42,)              # Single item (comma required!)
coordinates = (10.5, 25.3)       # GPS coordinates
colors = ("red", "green", "blue") # Colors tuple
mixed = (1, "hello", 3.14, True)  # Mixed types

# Tuple unpacking
x, y = coordinates               # x = 10.5, y = 25.3
first, second, *rest = colors    # first = "red", second = "green", rest = ["blue"]

print(f"Coordinates: {coordinates}")
print(f"X: {x}, Y: {y}")
print(f"First: {first}, Rest: {rest}")
```

### Tuple Operations

```python
colors = ("red", "green", "blue", "red")

# Indexing and slicing (same as lists)
print(colors[0])                 # "red"
print(colors[-1])                # "red"
print(colors[1:3])               # ("green", "blue")

# Counting and finding
count_red = colors.count("red")  # 2
index_green = colors.index("green")  # 1

# Membership testing
exists = "blue" in colors        # True

# Concatenation
new_colors = colors + ("yellow", "purple")  # New tuple

print(f"Red appears {count_red} times")
print(f"New colors: {new_colors}")
```

### When to Use Tuples

**Use tuples when:**

- Data shouldn't change (coordinates, RGB values)
- Need hashable objects (for sets/dictionaries)
- Return multiple values from functions
- Better performance (faster than lists)

```python
# Perfect for coordinates
def get_location():
    return (40.7128, -74.0060)  # NYC coordinates

lat, lon = get_location()
print(f"Latitude: {lat}, Longitude: {lon}")

# Perfect for RGB colors
def create_color(r, g, b):
    return (r, g, b)

red_color = create_color(255, 0, 0)
print(f"Red color: {red_color}")
```

---

## 4. Sets: Unordered Unique Collections

### What Are Sets?

**Think of sets like your collection of clubs or unique student interests:**

- **No duplicates** = Each club appears only once in your list
- **Unordered** = Order of clubs doesn't matter (alphabetical vs. interest order)
- **Fast membership** = Quick to check if someone is in a club

**Real School Examples:**

- 🎭 Clubs you're in: {"Drama Club", "Chess Club", "Debate Team"}
- 🏷️ Student ID tags: Each student ID appears only once
- 📚 Available textbooks: {"Math Book", "Science Book", "English Book"}

### Creating Sets

```python
# Different ways to create sets
empty_set = set()                           # Empty set of interests
clubs = {"Drama Club", "Chess Club", "Debate Team"}  # Your clubs
grade_levels = {9, 10, 11, 12}             # High school grades (freshman-senior)

# From other collections (removing duplicates automatically)
duplicate_classes = ["Math", "Science", "Math", "English", "Science"]
unique_classes = set(duplicate_classes)    # {"Math", "Science", "English"}

duplicate_names = ["Alice", "Bob", "Alice", "Carol", "Bob"]
unique_names = set(duplicate_names)        # {"Alice", "Bob", "Carol"}

# Real school example - student interests
interests = ["Reading", "Sports", "Music", "Sports", "Art"]  # Duplicates!
unique_interests = set(interests)          # {"Reading", "Sports", "Music", "Art"}

print(f"Classes I take: {unique_classes}")
print(f"People in my class: {unique_names}")
print(f"My unique interests: {unique_interests}")
```

### Set Operations

```python
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Union (all items from both sets)
union = set1 | set2              # {1, 2, 3, 4, 5, 6, 7, 8}
union_method = set1.union(set2)  # Same result

# Intersection (common items)
intersection = set1 & set2       # {4, 5}
intersection_method = set1.intersection(set2)

# Difference (items in set1 but not set2)
difference = set1 - set2         # {1, 2, 3}
difference_method = set1.difference(set2)

# Symmetric difference (items in either set, not both)
sym_diff = set1 ^ set2           # {1, 2, 3, 6, 7, 8}

print(f"Union: {union}")
print(f"Intersection: {intersection}")
print(f"Difference: {difference}")
print(f"Symmetric difference: {sym_diff}")
```

### Set Methods

```python
skills = {"python", "java", "css"}

# Adding and removing
skills.add("javascript")         # Add single item
skills.update(["react", "node"]) # Add multiple items

# Remove with different behaviors
skills.discard("css")           # Remove if exists (no error)
skills.remove("python")         # Remove if exists (error if not)
popped = skills.pop()           # Remove and return random item
skills.clear()                  # Remove all items

# Check membership (very fast!)
has_python = "python" in skills # True/False

print(f"Skills: {skills}")
```

---

## 5. Dictionaries: Key-Value Pairs

### What Are Dictionaries?

**Think of dictionaries like your school directory or grade book:**

- **Key** = Student's name or ID (unique identifier)
- **Value** = Phone number, grade, or other information
- **Fast lookup** = Find information by student name quickly
- **No duplicates** = Each student appears once

**Real School Examples:**

- 📞 School directory: {"Alice Smith": "555-0123", "Bob Jones": "555-0456"}
- 📊 Grade book: {"Math": 95, "Science": 87, "English": 92}
- 🏫 Class roster: {"S001": "Alice", "S002": "Bob", "S003": "Carol"}

### Creating Dictionaries

```python
# Different ways to create dictionaries
empty_roster = {}                           # Empty class roster
student_info = {"name": "Alice", "grade": 10, "GPA": 3.8}  # Student info

# Using dict() constructor (teacher's preferred way)
classroom = dict(math=25, english=22, science=28)  # Students per subject
hallway = dict([("main_hall", 150), ("north_wing", 200)])  # From list of tuples

# Dictionary comprehension (student ID -> GPA)
gpa_lookup = {student_id: round(3.5 + i*0.1, 1) for i, student_id in enumerate(["S001", "S002", "S003"], 1)}
# {"S001": 3.5, "S002": 3.6, "S003": 3.7}

# Real school examples
class_schedule = {
    "1st": "Math - Room 101",
    "2nd": "English - Room 102",
    "3rd": "Science - Room 103"
}

student_grades = {
    "Alice": {"Math": 95, "Science": 87},
    "Bob": {"Math": 89, "Science": 92}
}

print(f"Student info: {student_info}")
print(f"Class schedule: {class_schedule}")
print(f"Alice's Math grade: {student_grades['Alice']['Math']}")
```

### Dictionary Operations

```python
student = {"name": "Alice", "age": 20, "grade": "A", "courses": ["Math", "Science"]}

# Accessing values
name = student["name"]          # "Alice" (raises error if key doesn't exist)
age = student.get("age", 0)     # 20 (returns default if key doesn't exist)

# Adding and updating
student["gpa"] = 3.8           # Add new key-value
student.update({"age": 21, "major": "Computer Science"})  # Update multiple

# Removing items
grade = student.pop("grade")    # Remove and return value
last_item = student.popitem()   # Remove and return last item
student.clear()                 # Remove all items

print(f"Student info: {student}")
```

### Dictionary Methods

```python
student = {"name": "Alice", "age": 20, "major": "CS"}

# Getting keys, values, and items
keys = student.keys()           # dict_keys(['name', 'age', 'major'])
values = student.values()       # dict_values(['Alice', 20, 'CS'])
items = student.items()         # dict_items([('name', 'Alice'), ('age', 20), ('major', 'CS')])

# Iterating
for key in student:
    print(f"{key}: {student[key]}")

for key, value in student.items():
    print(f"{key}: {value}")

# Dictionary methods
copy = student.copy()           # Create shallow copy
has_name = "name" in student    # Check if key exists

print(f"Keys: {list(keys)}")
print(f"Values: {list(values)}")
```

### Nested Dictionaries

```python
# Database-like structure
students = {
    "student1": {
        "name": "Alice",
        "grades": {"Math": "A", "Science": "B"},
        "contact": {"email": "alice@email.com", "phone": "555-0123"}
    },
    "student2": {
        "name": "Bob",
        "grades": {"Math": "B", "Science": "A"},
        "contact": {"email": "bob@email.com", "phone": "555-0456"}
    }
}

# Access nested data
alice_math_grade = students["student1"]["grades"]["Math"]
alice_email = students["student1"]["contact"]["email"]

print(f"Alice's Math grade: {alice_math_grade}")
print(f"Alice's email: {alice_email}")
```

---

## 6. Strings: Text Data Operations

### String Basics (Working with Text Like Your Essays!)

**Think of strings like text in your homework assignments:**

- Words, sentences, paragraphs
- Can be processed, searched, and formatted
- Similar to lists but for characters

**Real School Examples:**

- 📝 Essay title: "The Benefits of Renewable Energy"
- 📚 Subject name: "Advanced Mathematics"
- 📧 Student email: "student@school.edu"

```python
# Different ways to create strings
essay_title = "The Impact of Climate Change"    # Essay title
subject_name = 'Advanced Mathematics'           # Class name (single quotes)
multiline_note = """Important Dates:
Midterm: March 15
Final: May 20
Project Due: April 10"""                       # Notes with line breaks
email_address = "student@school.edu"            # Student email

# String indexing (accessing characters like positions in text)
grade = "A+"
print(grade[0])          # "A" (first character)
print(grade[-1])         # "+" (last character)

# String slicing (getting parts of text)
student_name = "AliceJohnson"
first_name = student_name[:5]      # "Alice" (first 5 letters)
last_name = student_name[5:]       # "Johnson" (from position 5 to end)
initials = student_name[::2]       # "Aieo" (every 2nd character)

print(f"First name: {first_name}")
print(f"Last name: {last_name}")
print(f"Every 2nd letter: {initials}")
```

### String Methods

```python
essay_text = "  The Benefits of Renewable Energy  "

# Whitespace handling (cleaning up your essays!)
clean_essay = essay_text.strip()      # "The Benefits of Renewable Energy" (remove extra spaces)
left_trim = essay_text.lstrip()       # "The Benefits of Renewable Energy  " (remove left spaces)
right_trim = essay_text.rstrip()      # "  The Benefits of Renewable Energy" (remove right spaces)

# Case conversion (formatting your title)
uppercase_title = essay_text.upper()       # "  THE BENEFITS OF RENEWABLE ENERGY  "
lowercase_paragraph = essay_text.lower()   # "  the benefits of renewable energy  "
title_case = essay_text.title()            # "  The Benefits Of Renewable Energy  "
sentence_case = essay_text.capitalize()    # "  the benefits of renewable energy  "

# String searching and replacing
find_word = essay_text.find("Renewable")  # 15 (returns -1 if not found)
replace_word = essay_text.replace("Energy", "Power")  # "  The Benefits of Renewable Power  "
count_occurrences = essay_text.count("e")  # Count letter 'e'

# Real school example - processing student names
student_name = "  ALICE JOHNSON  "
clean_name = student_name.strip().title()  # "Alice Johnson" (clean and proper case)

print(f"Clean essay: '{clean_essay}'")
print(f"Student name: '{clean_name}'")
print(f"Find 'Renewable': {find_word}")
```

### String Splitting and Joining

```python
sentence = "Python is awesome and easy to learn"

# Splitting
words = sentence.split()        # ["Python", "is", "awesome", "and", "easy", "to", "learn"]
csv_line = "name,age,city,email"
csv_parts = csv_line.split(",") # ["name", "age", "city", "email"]

# Joining
word_list = ["Hello", "from", "Python"]
joined = " ".join(word_list)    # "Hello from Python"
csv_joined = ",".join(csv_parts) # "name,age,city,email"

print(f"Words: {words}")
print(f"Joined: {joined}")
```

### String Formatting

```python
name = "Alice"
age = 25
grade = "A"

# Old style formatting
old_style = "Name: %s, Age: %d, Grade: %s" % (name, age, grade)

# format() method
format_style = "Name: {}, Age: {}, Grade: {}".format(name, age, grade)
format_positions = "Name: {0}, Age: {1}, Grade: {2}".format(name, age, grade)
format_named = "Name: {n}, Age: {a}, Grade: {g}".format(n=name, a=age, g=grade)

# f-strings (Python 3.6+)
f_string = f"Name: {name}, Age: {age}, Grade: {grade}"
f_advanced = f"Name: {name.upper()}, Age: {age + 5}, Grade: {grade}"

print(f"f-string: {f_string}")
print(f"Advanced: {f_advanced}")
```

---

## 7. 🎓 Real School Scenarios & Examples

### Scenario 1: Managing Your Class Schedule

```python
# Your daily schedule - LIST (ordered, can change)
daily_schedule = ["Math", "English", "Science", "History", "PE"]
print(f"1st period: {daily_schedule[0]}")  # Access by position

# Add a new class
daily_schedule.append("Art")
print(f"Updated schedule: {daily_schedule}")

# Your locker combination - TUPLE (fixed, can't change)
locker_combo = (42, 17, 8)  # Always in this exact order
print(f"Locker combo: {locker_combo}")

# Clubs you join - SET (unique, no duplicates)
joined_clubs = {"Drama", "Chess", "Drama", "Debate"}  # Automatically removes duplicate "Drama"
print(f"Actual clubs: {joined_clubs}")

# Student grade book - DICTIONARY (name -> grade lookup)
grades = {
    "Alice": 95,
    "Bob": 87,
    "Carol": 92
}
print(f"Alice's grade: {grades['Alice']}")  # Direct lookup!
```

### Scenario 2: Organizing Your Study Groups

```python
# Study group members - LIST (ordered, can add/remove)
math_group = ["Alice", "Bob", "Carol"]
math_group.append("David")  # Add new member
math_group.remove("Bob")    # Remove someone

# Group assignments by subject - DICTIONARY
study_groups = {
    "Math": ["Alice", "Bob"],
    "Science": ["Carol", "David"],
    "English": ["Alice", "Carol", "David"]
}

# Unique subjects being studied - SET (automatically removes duplicates)
all_subjects = set()
for subject in study_groups:
    all_subjects.add(subject)

print(f"Studying: {all_subjects}")

# Check if someone is in a subject - FAST lookup with sets
has_alice_math = "Alice" in study_groups["Math"]
print(f"Alice studies math: {has_alice_math}")
```

### Scenario 3: Keeping Track of Homework

```python
# Today's homework - LIST (ordered by priority or deadline)
homework_today = [
    "Math worksheet (due tomorrow)",
    "Read chapter 5 (due Friday)",
    "Science lab report (due next week)"
]

# Assignment details - DICTIONARY (assignment -> details)
assignment_details = {
    "Math worksheet": {
        "subject": "Math",
        "pages": "42-45",
        "due_date": "tomorrow"
    },
    "Read chapter 5": {
        "subject": "English",
        "pages": "120-150",
        "due_date": "Friday"
    }
}

# Subjects you have homework for - SET (unique subjects)
homework_subjects = {"Math", "English", "Science"}

print(f"First homework: {homework_today[0]}")
print(f"Math worksheet details: {assignment_details['Math worksheet']}")
```

### Scenario 4: Cafeteria Menu System

```python
# Today's menu items - LIST (ordered by serving line)
todays_menu = ["Pizza", "Burgers", "Salad", "Pasta", "Soup"]

# Menu prices - DICTIONARY (item -> price)
menu_prices = {
    "Pizza": 3.50,
    "Burgers": 4.00,
    "Salad": 2.50,
    "Pasta": 3.75,
    "Soup": 2.25
}

# Available ingredients - SET (no duplicates)
available_ingredients = {
    "tomatoes", "cheese", "lettuce", "beef",
    "tomatoes", "cheese"  # Duplicate automatically removed
}

# Your lunch order - LIST (can modify order)
my_order = ["Pizza", "Salad"]
my_order.insert(1, "Soup")  # Add soup as second item

# Check what's available
print(f"Available: {available_ingredients}")
print(f"Order total: ${menu_prices[my_order[0]] + menu_prices[my_order[1]] + menu_prices[my_order[2]]}")

# Find vegetarian options
vegetarian_items = ["Salad", "Pasta", "Soup"]  # Assuming these are vegetarian
vegetarian_available = [item for item in todays_menu if item in vegetarian_items]
print(f"Vegetarian options: {vegetarian_available}")
```

---

## 8. When to Use Each Data Structure

### Decision Matrix (School Edition)

| Use Case           | Best Choice | Why                                            |
| ------------------ | ----------- | ---------------------------------------------- |
| **Class schedule** | List        | Need specific order, may change classes        |
| **Locker combo**   | Tuple       | Fixed numbers, never change                    |
| **Your clubs**     | Set         | No duplicate memberships, order doesn't matter |
| **Grade book**     | Dictionary  | Lookup grades by student name                  |
| **GPS location**   | Tuple       | Immutable coordinates                          |
| **Subject list**   | Dictionary  | Store different info per subject               |

### Quick Reference Guide (School Examples)

```python
# Class schedule - ORDER MATTERS, classes change
daily_classes = ["Math", "English", "Science", "History"]
daily_classes.append("PE")  # Add last period
print(daily_classes[0])     # First period class

# Locker combo - FIXED values, shouldn't change
my_combo = (42, 17, 8)
# my_combo[0] = 100  # ERROR! Tuples are immutable

# Your clubs - NO duplicates, fast lookup
my_clubs = {"Drama", "Chess", "Debate"}
my_clubs.add("Math Club")  # Join new club
has_chess = "Chess" in my_clubs  # Very fast O(1) lookup

# Grade book - Key-value relationships
student_grades = {
    "Alice": {"Math": 95, "Science": 87},
    "Bob": {"Math": 89, "Science": 92}
}
print(student_grades["Alice"]["Math"])  # Direct lookup by student name

# Assignment list - Ordered, can change priority
assignments = ["Math homework", "Read chapter", "Science lab"]
assignments[0] = "Math homework (URGENT)"  # Can modify priority
print(assignments[:2])           # Top 2 priorities
```

---

## 9. Advanced Operations & Methods

### List Comprehensions (Advanced)

```python
# Basic list comprehensions
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]  # [1, 4, 9, 16, 25]
evens = [x for x in numbers if x % 2 == 0]  # [2, 4]

# Nested comprehensions
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
# [[1, 2, 3], [2, 4, 6], [3, 6, 9]]

# With conditions
temperature_data = [25, 18, 30, 15, 22]
hot_days = [temp for temp in temperature_data if temp > 25]
cold_days = [temp if temp > 15 else "Cold" for temp in temperature_data]

print(f"Squares: {squares}")
print(f"Hot days: {hot_days}")
print(f"Temperature status: {cold_days}")
```

### Dictionary Comprehensions (Advanced)

```python
# Dictionary comprehensions
squares_dict = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
filtered_dict = {k: v for k, v in squares_dict.items() if v > 5}

# Transforming existing data
grades = {"Alice": 85, "Bob": 92, "Charlie": 78}
letter_grades = {name: "A" if score >= 90 else "B" if score >= 80 else "C"
                for name, score in grades.items()}

print(f"Squares: {squares_dict}")
print(f"Letter grades: {letter_grades}")
```

### Set Comprehensions

```python
# Set comprehensions (similar to list comprehensions)
numbers = [1, 2, 2, 3, 3, 4, 5, 5, 5]
unique_squares = {x**2 for x in numbers}  # {1, 4, 9, 16, 25}
even_squares = {x**2 for x in numbers if x % 2 == 0}  # {4, 16}

# From strings
text = "hello world"
unique_chars = {char for char in text if char.isalpha()}

print(f"Unique squares: {unique_squares}")
print(f"Unique characters: {unique_chars}")
```

### Advanced Slicing

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# All possible slices
first_half = numbers[:5]          # [0, 1, 2, 3, 4]
second_half = numbers[5:]         # [5, 6, 7, 8, 9]
middle = numbers[2:8]             # [2, 3, 4, 5, 6, 7]
every_second = numbers[::2]       # [0, 2, 4, 6, 8]
every_third = numbers[::3]        # [0, 3, 6, 9]
reversed_list = numbers[::-1]     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

# Modify with slices
numbers[1:4] = [10, 20, 30]       # [0, 10, 20, 30, 4, 5, 6, 7, 8, 9]
numbers[::2] = [100, 200, 300]    # [100, 10, 200, 30, 300, 5, 100, 7, 200, 9]

print(f"Numbers: {numbers}")
```

---

## 10. Real-World Applications

### Application 1: Student Management System (School Edition)

```python
class StudentDatabase:
    def __init__(self):
        self.students = {}
        self.courses = set()

    def add_student(self, student_id, name, age):
        """Add a new student"""
        self.students[student_id] = {
            "name": name,
            "age": age,
            "courses": set(),
            "grades": {}
        }

    def enroll_course(self, student_id, course):
        """Enroll student in a course"""
        if student_id in self.students:
            self.students[student_id]["courses"].add(course)
            self.courses.add(course)

    def assign_grade(self, student_id, course, grade):
        """Assign grade to student for a course"""
        if student_id in self.students:
            self.students[student_id]["grades"][course] = grade

    def get_student_info(self, student_id):
        """Get complete student information"""
        if student_id in self.students:
            student = self.students[student_id]
            return {
                "name": student["name"],
                "age": student["age"],
                "enrolled_courses": list(student["courses"]),
                "grades": student["grades"]
            }
        return None

    def get_course_enrollments(self):
        """Get all course enrollments"""
        enrollments = {}
        for student_id, student in self.students.items():
            for course in student["courses"]:
                if course not in enrollments:
                    enrollments[course] = []
                enrollments[course].append(student_id)
        return enrollments

    def find_students_by_course(self, course):
        """Find all students enrolled in a specific course"""
        return [sid for sid, student in self.students.items()
                if course in student["courses"]]

# Test the system
db = StudentDatabase()

# Add students
db.add_student("S001", "Alice Johnson", 20)
db.add_student("S002", "Bob Smith", 21)
db.add_student("S003", "Carol Davis", 19)

# Enroll in courses
db.enroll_course("S001", "Math")
db.enroll_course("S001", "Physics")
db.enroll_course("S002", "Math")
db.enroll_course("S003", "Physics")

# Assign grades
db.assign_grade("S001", "Math", "A")
db.assign_grade("S001", "Physics", "B")
db.assign_grade("S002", "Math", "A")

# Get information
alice_info = db.get_student_info("S001")
print(f"Alice info: {alice_info}")

# Find students in Math
math_students = db.find_students_by_course("Math")
print(f"Students in Math: {math_students}")

# Course enrollments
enrollments = db.get_course_enrollments()
print(f"Course enrollments: {enrollments}")
```

### Application 2: Inventory Management

```python
class InventoryManager:
    def __init__(self):
        self.products = {}  # Product ID -> Product info
        self.categories = set()  # Available categories
        self.low_stock_threshold = 10

    def add_product(self, product_id, name, category, price, stock):
        """Add a new product"""
        self.products[product_id] = {
            "name": name,
            "category": category,
            "price": price,
            "stock": stock,
            "transactions": []
        }
        self.categories.add(category)

    def update_stock(self, product_id, quantity, transaction_type):
        """Update stock and record transaction"""
        if product_id in self.products:
            product = self.products[product_id]
            product["stock"] += quantity
            transaction = {
                "type": transaction_type,
                "quantity": quantity,
                "new_stock": product["stock"]
            }
            product["transactions"].append(transaction)
            return True
        return False

    def get_low_stock_products(self):
        """Get products with low stock"""
        return [pid for pid, product in self.products.items()
                if product["stock"] <= self.low_stock_threshold]

    def get_products_by_category(self, category):
        """Get all products in a specific category"""
        return [pid for pid, product in self.products.items()
                if product["category"] == category]

    def get_total_inventory_value(self):
        """Calculate total inventory value"""
        return sum(product["price"] * product["stock"]
                  for product in self.products.values())

    def get_category_statistics(self):
        """Get statistics by category"""
        stats = {}
        for product in self.products.values():
            category = product["category"]
            if category not in stats:
                stats[category] = {"count": 0, "total_value": 0, "total_stock": 0}

            stats[category]["count"] += 1
            stats[category]["total_value"] += product["price"] * product["stock"]
            stats[category]["total_stock"] += product["stock"]

        return stats

# Test the inventory system
inventory = InventoryManager()

# Add products
inventory.add_product("P001", "Laptop", "Electronics", 999.99, 15)
inventory.add_product("P002", "Coffee Mug", "Kitchen", 12.99, 50)
inventory.add_product("P003", "T-Shirt", "Clothing", 24.99, 8)
inventory.add_product("P004", "Headphones", "Electronics", 199.99, 12)

# Update stock
inventory.update_stock("P001", -5, "sale")      # Sold 5 laptops
inventory.update_stock("P003", 20, "restock")   # Restocked t-shirts
inventory.update_stock("P004", -8, "sale")      # Sold 8 headphones

# Get information
low_stock = inventory.get_low_stock_products()
print(f"Low stock products: {low_stock}")

electronics = inventory.get_products_by_category("Electronics")
print(f"Electronics: {electronics}")

total_value = inventory.get_total_inventory_value()
print(f"Total inventory value: ${total_value:.2f}")

# Category statistics
category_stats = inventory.get_category_statistics()
print(f"Category statistics: {category_stats}")
```

### Application 3: Text Analyzer

```python
class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.words = self.text.lower().split()
        self.characters = len(text)
        self.characters_no_spaces = len(text.replace(" ", ""))

    def get_word_frequency(self):
        """Get frequency of each word"""
        frequency = {}
        for word in self.words:
            # Remove punctuation
            clean_word = word.strip(".,!?;:")
            frequency[clean_word] = frequency.get(clean_word, 0) + 1
        return frequency

    def get_character_frequency(self):
        """Get frequency of each character (letters only)"""
        frequency = {}
        for char in self.text.lower():
            if char.isalpha():
                frequency[char] = frequency.get(char, 0) + 1
        return frequency

    def get_sentences(self):
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if s.strip()]

    def get_unique_words(self):
        """Get list of unique words"""
        return list(set(self.words))

    def get_longest_words(self, n=5):
        """Get n longest words"""
        sorted_words = sorted(self.words, key=len, reverse=True)
        return sorted_words[:n]

    def get_shortest_words(self, n=5):
        """Get n shortest words"""
        sorted_words = sorted(self.words, key=len)
        return sorted_words[:n]

    def get_average_word_length(self):
        """Calculate average word length"""
        if not self.words:
            return 0
        total_length = sum(len(word) for word in self.words)
        return total_length / len(self.words)

    def get_readability_metrics(self):
        """Calculate basic readability metrics"""
        sentences = self.get_sentences()
        words = len(self.words)
        characters = self.characters_no_spaces

        if len(sentences) == 0 or words == 0:
            return {"avg_words_per_sentence": 0, "avg_chars_per_word": 0}

        avg_words_per_sentence = words / len(sentences)
        avg_chars_per_word = characters / words

        return {
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_chars_per_word": avg_chars_per_word
        }

    def generate_report(self):
        """Generate complete text analysis report"""
        print("=" * 50)
        print("TEXT ANALYSIS REPORT")
        print("=" * 50)

        # Basic metrics
        print(f"Characters (with spaces): {self.characters}")
        print(f"Characters (no spaces): {self.characters_no_spaces}")
        print(f"Words: {len(self.words)}")
        print(f"Sentences: {len(self.get_sentences())}")
        print(f"Average word length: {self.get_average_word_length():.2f}")

        # Word frequency
        print(f"\nTop 5 most frequent words:")
        word_freq = self.get_word_frequency()
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for word, freq in top_words:
            print(f"  {word}: {freq}")

        # Longest and shortest words
        print(f"\nLongest words: {self.get_longest_words(3)}")
        print(f"Shortest words: {self.get_shortest_words(3)}")

        # Readability metrics
        readability = self.get_readability_metrics()
        print(f"\nReadability:")
        print(f"  Avg words per sentence: {readability['avg_words_per_sentence']:.1f}")
        print(f"  Avg characters per word: {readability['avg_chars_per_word']:.1f}")

# Test the text analyzer
sample_text = """
Python is an interpreted, high-level programming language for general-purpose programming.
Created by Guido van Rossum and first released in 1991, Python has a design philosophy
that emphasizes code readability, notably using significant whitespace. It provides
constructs that enable clear programming on both small and large scales. Python interpreters
are available for many operating systems. CPython, the reference implementation of Python,
is open source software and has a community-based development model.
"""

analyzer = TextAnalyzer(sample_text)
analyzer.generate_report()
```

---

## 11. Performance Considerations

### Time Complexity

| Operation           | List   | Tuple | Set  | Dictionary |
| ------------------- | ------ | ----- | ---- | ---------- |
| **Access by index** | O(1)   | O(1)  | N/A  | N/A        |
| **Access by key**   | O(n)   | O(n)  | O(1) | O(1)       |
| **Insertion**       | O(1)\* | O(n)  | O(1) | O(1)       |
| **Deletion**        | O(n)   | O(n)  | O(1) | O(1)       |
| **Search**          | O(n)   | O(n)  | O(1) | O(1)       |
| **Membership**      | O(n)   | O(n)  | O(1) | O(1)       |

\*O(1) at end, O(n) at beginning/middle

### Choosing the Right Data Structure

```python
# Performance examples
import time

# Large dataset
data = list(range(100000))

# List search (O(n))
start_time = time.time()
50000 in data
list_time = time.time() - start_time

# Set search (O(1))
data_set = set(data)
start_time = time.time()
50000 in data_set
set_time = time.time() - start_time

print(f"List search: {list_time:.6f} seconds")
print(f"Set search: {set_time:.6f} seconds")
print(f"Set is {list_time/set_time:.1f}x faster!")

# Memory usage
import sys
list_size = sys.getsizeof(data)
set_size = sys.getsizeof(data_set)

print(f"List memory: {list_size} bytes")
print(f"Set memory: {set_size} bytes")
```

---

## 12. Common Patterns & Best Practices

### Pattern 1: Data Processing Pipeline

```python
def process_student_data(raw_data):
    """Process raw student data using multiple data structures"""

    # Convert to list for ordered processing
    students = list(raw_data)

    # Use dictionary for fast lookup by ID
    student_dict = {student["id"]: student for student in students}

    # Use set for unique course codes
    all_courses = set()
    for student in students:
        all_courses.update(student["courses"])

    # Use list comprehension for transformed data
    processed_students = [
        {
            "id": student["id"],
            "name": student["name"].title(),
            "courses_count": len(student["courses"]),
            "gpa": student.get("grades", {}).get("gpa", 0.0)
        }
        for student in students
    ]

    return {
        "students": processed_students,
        "student_lookup": student_dict,
        "courses": list(all_courses)
    }
```

### Pattern 2: Frequency Counter

```python
def word_frequency_analysis(text):
    """Analyze word frequency using dictionary"""
    words = text.lower().split()

    # Count frequency using dictionary
    frequency = {}
    for word in words:
        clean_word = word.strip(".,!?;:")
        frequency[clean_word] = frequency.get(clean_word, 0) + 1

    # Sort by frequency
    sorted_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    return {
        "frequency": frequency,
        "most_common": sorted_frequency[:10],
        "unique_words": len(frequency),
        "total_words": sum(frequency.values())
    }
```

### Pattern 3: Data Grouping

```python
def group_students_by_grade(students):
    """Group students by grade using nested dictionaries"""
    grade_groups = {}

    for student in students:
        grade = student["grade"]
        if grade not in grade_groups:
            grade_groups[grade] = []
        grade_groups[grade].append(student)

    return grade_groups

def group_courses_by_department(courses):
    """Group courses by department using defaultdict"""
    from collections import defaultdict

    dept_groups = defaultdict(list)
    for course in courses:
        dept_groups[course["department"]].append(course)

    return dict(dept_groups)
```

---

## 13. Practice Exercises

### Exercise 1: School Locker Management System

**Create a locker management system using different data structures**

```python
class SchoolLockerSystem:
    def __init__(self):
        self.lockers = {}            # Locker number -> student info
        self.locker_combinations = {}  # Locker number -> combo
        self.available_lockers = set()  # Available locker numbers
        self.student_lockers = {}    # Student ID -> locker number

    def add_locker(self, locker_num, combination):
        """Add a new locker with combination"""
        self.lockers[locker_num] = {"student": None, "books": []}
        self.locker_combinations[locker_num] = combination
        self.available_lockers.add(locker_num)

    def assign_locker(self, student_id, locker_num):
        """Assign locker to student"""
        if locker_num in self.lockers and locker_num in self.available_lockers:
            self.lockers[locker_num]["student"] = student_id
            self.available_lockers.remove(locker_num)
            self.student_lockers[student_id] = locker_num
            return True
        return False

    def add_books_to_locker(self, locker_num, books):
        """Add books to locker"""
        if locker_num in self.lockers:
            self.lockers[locker_num]["books"].extend(books)
            return True
        return False

    def get_student_locker(self, student_id):
        """Get student's locker info"""
        locker_num = self.student_lockers.get(student_id)
        if locker_num:
            return self.lockers[locker_num]
        return None

    def get_available_lockers(self):
        """Get all available lockers"""
        return list(self.available_lockers)

    def calculate_locker_utilization(self):
        """Calculate locker usage percentage"""
        total_lockers = len(self.lockers)
        assigned_lockers = total_lockers - len(self.available_lockers)
        return (assigned_lockers / total_lockers) * 100 if total_lockers > 0 else 0

# Test the locker system
locker_system = SchoolLockerSystem()

# Add lockers
locker_system.add_locker(101, (25, 10, 30))
locker_system.add_locker(102, (15, 35, 20))
locker_system.add_locker(103, (40, 5, 25))

# Assign lockers to students
locker_system.assign_locker("S001", 101)
locker_system.assign_locker("S002", 102)

# Add books
locker_system.add_books_to_locker(101, ["Math textbook", "English novel"])
locker_system.add_books_to_locker(102, ["Science workbook", "History notes"])

# Get information
s001_locker = locker_system.get_student_locker("S001")
available = locker_system.get_available_lockers()
utilization = locker_system.calculate_locker_utilization()

print(f"S001's locker: {s001_locker}")
print(f"Available lockers: {available}")
print(f"Locker utilization: {utilization:.1f}%")
```

### Exercise 2: Student Grade Tracker

**Track student grades and subjects using multiple data structures**

```python
class StudentGradeSystem:
    def __init__(self):
        self.students = {}            # Student ID -> student info
        self.subjects = set()         # Available subjects
        self.grades = {}              # (student_id, subject) -> grade
        self.assignments = {}         # Assignment name -> max points

    def add_student(self, student_id, name, grade_level):
        """Add a student"""
        self.students[student_id] = {
            "name": name,
            "grade_level": grade_level,
            "subjects": set()
        }

    def add_subject(self, subject_name):
        """Add a subject"""
        self.subjects.add(subject_name)

    def record_assignment(self, assignment_name, subject, max_points):
        """Record assignment info"""
        self.assignments[assignment_name] = {"subject": subject, "max_points": max_points}
        self.add_subject(subject)

    def submit_grade(self, student_id, assignment, score):
        """Submit a grade"""
        if assignment in self.assignments and student_id in self.students:
            self.grades[(student_id, assignment)] = score
            subject = self.assignments[assignment]["subject"]
            self.students[student_id]["subjects"].add(subject)
            return True
        return False

    def get_student_gpa(self, student_id):
        """Calculate student's GPA"""
        if student_id not in self.students:
            return None

        student_grades = []
        for (sid, assignment), score in self.grades.items():
            if sid == student_id:
                max_points = self.assignments[assignment]["max_points"]
                percentage = (score / max_points) * 100
                student_grades.append(percentage)

        return sum(student_grades) / len(student_grades) if student_grades else None

    def get_subject_rankings(self, subject):
        """Get student rankings for a subject"""
        rankings = []
        for (student_id, assignment), score in self.grades.items():
            if self.assignments[assignment]["subject"] == subject:
                max_points = self.assignments[assignment]["max_points"]
                percentage = (score / max_points) * 100
                rankings.append((student_id, percentage))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_class_average(self, subject):
        """Get class average for a subject"""
        scores = []
        for (student_id, assignment), score in self.grades.items():
            if self.assignments[assignment]["subject"] == subject:
                max_points = self.assignments[assignment]["max_points"]
                percentage = (score / max_points) * 100
                scores.append(percentage)

        return sum(scores) / len(scores) if scores else None

# Test the grade system
grade_system = StudentGradeSystem()

# Add students
grade_system.add_student("S001", "Alice Johnson", 10)
grade_system.add_student("S002", "Bob Smith", 10)
grade_system.add_student("S003", "Carol Davis", 11)

# Add subjects and assignments
grade_system.record_assignment("Quiz 1", "Math", 100)
grade_system.record_assignment("Midterm", "Math", 200)
grade_system.record_assignment("Essay", "English", 150)

# Submit grades
grade_system.submit_grade("S001", "Quiz 1", 85)
grade_system.submit_grade("S001", "Midterm", 165)
grade_system.submit_grade("S002", "Quiz 1", 92)
grade_system.submit_grade("S002", "Midterm", 180)
grade_system.submit_grade("S003", "Quiz 1", 78)
grade_system.submit_grade("S003", "Essay", 135)

# Get statistics
alice_gpa = grade_system.get_student_gpa("S001")
math_rankings = grade_system.get_subject_rankings("Math")
math_average = grade_system.get_class_average("Math")

print(f"Alice's GPA: {alice_gpa:.1f}%")
print(f"Math rankings: {math_rankings}")
print(f"Math class average: {math_average:.1f}%")
```

### Exercise 3: School Event Planner

**Plan school events using lists and dictionaries**

```python
class SchoolEventPlanner:
    def __init__(self):
        self.events = {}              # Event name -> event details
        self.participants = {}        # Event name -> list of participants
        self.available_rooms = set()  # Available rooms
        self.event_schedule = []      # Chronological event list

    def add_room(self, room_number, capacity):
        """Add an available room"""
        self.available_rooms.add((room_number, capacity))

    def create_event(self, event_name, date, time, room):
        """Create a new event"""
        if event_name not in self.events:
            self.events[event_name] = {
                "date": date,
                "time": time,
                "room": room,
                "description": "",
                "organizer": ""
            }
            self.participants[event_name] = []
            self.event_schedule.append(event_name)
            return True
        return False

    def add_participant(self, event_name, student_name, student_id):
        """Add participant to event"""
        if event_name in self.events:
            participant = {
                "name": student_name,
                "id": student_id,
                "checked_in": False
            }
            self.participants[event_name].append(participant)
            return True
        return False

    def check_in_participant(self, event_name, student_id):
        """Check in a participant"""
        if event_name in self.participants:
            for participant in self.participants[event_name]:
                if participant["id"] == student_id:
                    participant["checked_in"] = True
                    return True
        return False

    def get_event_attendance(self, event_name):
        """Get attendance statistics for an event"""
        if event_name not in self.participants:
            return None

        participants = self.participants[event_name]
        total = len(participants)
        checked_in = sum(1 for p in participants if p["checked_in"])

        return {
            "total_registered": total,
            "checked_in": checked_in,
            "attendance_rate": (checked_in / total) * 100 if total > 0 else 0
        }

    def get_events_by_date(self, date):
        """Get all events on a specific date"""
        return [event for event, details in self.events.items()
                if details["date"] == date]

    def get_room_usage(self):
        """Get room usage statistics"""
        room_usage = {}
        for event, details in self.events.items():
            room = details["room"]
            room_usage[room] = room_usage.get(room, 0) + 1
        return room_usage

# Test the event planner
planner = SchoolEventPlanner()

# Add rooms
planner.add_room("Auditorium", 300)
planner.add_room("Gym", 500)
planner.add_room("Library", 50)

# Create events
planner.create_event("Talent Show", "2024-03-15", "7:00 PM", "Auditorium")
planner.create_event("Basketball Game", "2024-03-16", "6:00 PM", "Gym")
planner.create_event("Book Club", "2024-03-17", "3:30 PM", "Library")

# Add participants
planner.add_participant("Talent Show", "Alice Johnson", "S001")
planner.add_participant("Talent Show", "Bob Smith", "S002")
planner.add_participant("Basketball Game", "Carol Davis", "S003")
planner.add_participant("Book Club", "Alice Johnson", "S001")

# Check in participants
planner.check_in_participant("Talent Show", "S001")
planner.check_in_participant("Talent Show", "S002")

# Get event information
talent_show_attendance = planner.get_event_attendance("Talent Show")
march_events = planner.get_events_by_date("2024-03-15")
room_usage = planner.get_room_usage()

print(f"Talent Show attendance: {talent_show_attendance}")
print(f"March 15 events: {march_events}")
print(f"Room usage: {room_usage}")
```

---

---

## 🎯 Key Takeaways - You're Now a Data Structure Expert!

### School Data Structure Mastery:

✅ **Lists**: Ordered, mutable sequences - perfect for class schedules and homework lists  
✅ **Tuples**: Ordered, immutable sequences - perfect for locker combinations  
✅ **Sets**: Unordered, unique collections - perfect for your clubs and interests  
✅ **Dictionaries**: Key-value pairs - perfect for grade books and student directories  
✅ **Strings**: Text data with powerful methods - perfect for essays and assignments

### When to Use Each (School Edition):

✅ **Lists** = Need order + modification (class schedules, homework lists)  
✅ **Tuples** = Fixed data + performance (locker combos, coordinates)  
✅ **Sets** = No duplicates + fast lookup (clubs, unique interests)  
✅ **Dictionaries** = Key-value relationships (grade books, student info)  
✅ **Strings** = Working with text (essay titles, assignments)

### Advanced Operations (Your Superpowers!):

✅ **List/Dictionary/Set Comprehensions** - Transform data like magic ✨  
✅ **Slicing** - Extract exactly what you need from lists  
✅ **Method Chaining** - Combine operations efficiently  
✅ **Nested Structures** - Store complex school information

### Performance Tips (For School Projects):

✅ **Big O Notation** - Understanding how fast your code runs  
✅ **Memory Usage** - Choosing the right data structure  
✅ **Search Performance** - Sets are super fast for "is this student in..." questions

### 💪 Real-World School Applications:

✅ **Student Management Systems** - Track grades, attendance, schedules  
✅ **Event Planning** - Organize school activities efficiently  
✅ **Grade Tracking** - Calculate GPAs and rankings  
✅ **Locker Systems** - Manage assignments and combinations  
✅ **Study Group Organization** - Group students by subjects

### 🎓 You're Ready For:

- Building school management apps
- Creating grade tracking systems
- Organizing school events
- Managing student databases
- Any data-heavy school project!

### 🏆 Next Steps:

Now that you master data structures, you're ready for:

- **Object-Oriented Programming** (Classes & Objects)
- **File Handling** (Reading/writing school data)
- **Web Development** (School websites & apps)
- **Database Management** (Advanced student systems)

---

_🎉 Congratulations! You've just learned one of the most important concepts in Python programming. Data structures are everywhere in real applications - from social media to games to school systems. You're now equipped to build amazing projects!_

**Keep coding, keep learning, and remember: Every expert was once a beginner!** 🚀
