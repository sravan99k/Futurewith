# 📚 Python Data Structures Practice Questions - Universal Edition

_Master Lists, Tuples, Sets, Dictionaries & Strings with Real-World Examples_

## Question Categories:

1. [List Basics (Questions 1-25)](#list-basics-questions-1-25)
2. [List Operations & Methods (Questions 26-50)](#list-operations--methods-questions-26-50)
3. [Tuples (Questions 51-65)](#tuples-questions-51-65)
4. [Sets (Questions 66-85)](#sets-questions-66-85)
5. [Dictionaries (Questions 86-115)](#dictionaries-questions-86-115)
6. [Strings (Questions 116-140)](#strings-questions-116-140)
7. [Data Structure Combinations (Questions 141-160)](#data-structure-combinations-questions-141-160)
8. [Advanced Applications (Questions 161-180)](#advanced-applications-questions-161-180)
9. [Challenge Problems (Questions 181-200)](#challenge-problems-questions-181-200)

---

## List Basics (Questions 1-25)

### Question 1: Create and Access Item Lists

**Create a list of 5 items in your shopping cart and print the first and last item**

**Answer:**

```python
shopping_cart = ["Milk", "Bread", "Eggs", "Cheese", "Apples"]
print(f"First item: {shopping_cart[0]}")
print(f"Last item: {shopping_cart[-1]}")
```

### Question 2: List Slicing

**Create a list of weekdays and slice to get the first three workdays**

**Answer:**

```python
workdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
first_three = workdays[:3]  # Monday through Wednesday
print(f"First three workdays: {first_three}")  # ['Monday', 'Tuesday', 'Wednesday']
```

### Question 3: List with Mixed Types

**Create a list containing customer name (string), age (integer), rating (float), and is_premium (boolean)**

**Answer:**

```python
customer_info = ["Sarah Connor", 28, 4.5, True]
print(customer_info)
```

### Question 4: Empty List and Adding Items

**Start with an empty shopping list and add 5 items using append()**

**Answer:**

```python
shopping_list = []
shopping_list.append("Milk")
shopping_list.append("Bread")
shopping_list.append("Eggs")
shopping_list.append("Cheese")
shopping_list.append("Apples")
print(f"Shopping list: {shopping_list}")
```

### Question 5: List Concatenation

**Create two lists: morning tasks and afternoon tasks, then combine them**

**Answer:**

```python
morning_tasks = ["Check email", "Review calendar", "Team meeting"]
afternoon_tasks = ["Project work", "Client call", "Documentation"]
all_tasks = morning_tasks + afternoon_tasks
print(f"Today's tasks: {all_tasks}")  # ['Check email', 'Review calendar', 'Team meeting', 'Project work', 'Client call', 'Documentation']
```

### Question 6: List Repetition

**Repeat the lunch period names 3 times to show the school week schedule**

**Answer:**

```python
lunch_period = ["12:00 PM - 12:30 PM"]
weekly_lunch = lunch_period * 5
print(f"Lunch schedule: {weekly_lunch}")
# ['12:00 PM - 12:30 PM', '12:00 PM - 12:30 PM', '12:00 PM - 12:30 PM', '12:00 PM - 12:30 PM', '12:00 PM - 12:30 PM']
```

### Question 7: Nested Lists

**Create a 3x3 class schedule grid using nested lists**

**Answer:**

```python
class_schedule = [
    ["Math", "Science", "English"],
    ["History", "PE", "Art"],
    ["Music", "Spanish", "Study Hall"]
]
print(f"Morning class: {class_schedule[0][1]}")  # Science
```

### Question 8: List Length

**Find the length of various school lists**

**Answer:**

```python
empty_list = []
single_class = ["Math"]
multiple_classes = ["Math", "Science", "English", "History", "Art"]
print(f"No classes: {len(empty_list)}")
print(f"One class: {len(single_class)}")
print(f"Today's classes: {len(multiple_classes)}")
```

### Question 9: List Membership

**Check if students are in the club list**

**Answer:**

```python
chess_club_members = ["Alice", "Bob", "Charlie", "Diana"]
print("Alice in chess club:", "Alice" in chess_club_members)    # True
print("Eve in chess club:", "Eve" in chess_club_members)       # False
```

### Question 10: Negative Indexing

**Use negative indices to access class periods**

**Answer:**

```python
daily_classes = ["Math", "English", "Science", "History", "PE", "Lunch", "Art"]
print(f"Last class: {daily_classes[-1]}")  # Art
print(f"Before last class: {daily_classes[-3]}")  # PE
```

### Question 11: List Index Method

**Find the position of a student in the attendance list**

**Answer:**

```python
attendance = ["Alice", "Bob", "Charlie", "Diana", "Bob"]
bob_position = attendance.index("Bob")
print(f"Bob is at position: {bob_position}")

# Find all positions of Bob
all_bob_positions = [i for i, student in enumerate(attendance) if student == "Bob"]
print(f"Bob is present at positions: {all_bob_positions}")
```

### Question 12: Count Method

**Count how many students scored A grades**

**Answer:**

```python
test_grades = ["A", "B", "A", "C", "A", "B", "A", "D"]
grade_a_count = test_grades.count("A")
print(f"Students with A grade: {grade_a_count}")
```

### Question 13: Insert Method

**Insert a new student between two existing students in the roster**

**Answer:**

```python
seating_chart = ["Alice", "Charlie", "Diana"]
seating_chart.insert(1, "Bob")
print(f"Updated seating: {seating_chart}")  # ['Alice', 'Bob', 'Charlie', 'Diana']
```

### Question 14: Remove vs Pop

**Difference between remove() and pop() in student management**

**Answer:**

```python
# remove() - removes first occurrence of a student name
class_attendance = ["Alice", "Bob", "Charlie", "Bob"]
class_attendance.remove("Bob")  # Removes first "Bob"
print(f"After remove: {class_attendance}")  # ['Alice', 'Charlie', 'Bob']

# pop() - removes and returns the last student
class_roster = ["Alice", "Bob", "Charlie"]
last_student = class_roster.pop()
print(f"Removed student: {last_student}")  # Charlie
print(f"Remaining students: {class_roster}")    # ['Alice', 'Bob']
```

### Question 15: Reverse List

**Reverse a list of students to show the reverse attendance order**

**Answer:**

```python
morning_line = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

# Method 1: reverse()
line_copy = morning_line.copy()
line_copy.reverse()
print(f"Using reverse(): {line_copy}")

# Method 2: slicing
line_copy = morning_line.copy()
reversed_line = line_copy[::-1]
print(f"Using slicing: {reversed_line}")
```

### Question 16: List Sorting

**Sort students by name alphabetically and by grade numerically**

**Answer:**

```python
student_names = ["Charlie", "Alice", "Bob", "Diana"]
student_names.sort()
print(f"Names A-Z: {student_names}")

# Sort by grade
student_grades = [85, 92, 78, 96]
student_grades.sort()
print(f"Grades low to high: {student_grades}")

# Sort grades descending
student_grades = [85, 92, 78, 96]
student_grades.sort(reverse=True)
print(f"Grades high to low: {student_grades}")

# Create new sorted list
student_grades = [85, 92, 78, 96]
sorted_grades = sorted(student_grades)
print(f"New sorted list: {sorted_grades}")
```

### Question 17: List Comprehension Basics

**Create a list of student IDs using list comprehension**

**Answer:**

```python
# Traditional way
student_ids = []
for i in range(5):
    student_ids.append(f"STU{i:03d}")
print(f"Traditional: {student_ids}")

# List comprehension
student_ids_comp = [f"STU{i:03d}" for i in range(5)]
print(f"Comprehension: {student_ids_comp}")
```

### Question 18: List with Range

**Create class period numbers using range()**

**Answer:**

```python
# Periods 1-8
all_periods = list(range(1, 9))
print(f"Periods: {all_periods}")

# Even periods
even_periods = list(range(2, 9, 2))
print(f"Even periods: {even_periods}")

# Reverse countdown
countdown = list(range(5, 0, -1))
print(f"Countdown to summer: {countdown}")
```

### Question 19: Modifying List Elements

**Change class subjects in a schedule**

**Answer:**

```python
morning_schedule = ["Math", "English", "Science", "History"]
print(f"Original: {morning_schedule}")

morning_schedule[0] = "Advanced Math"
morning_schedule[-1] = "World History"
print(f"Modified: {morning_schedule}")

# Replace multiple periods
morning_schedule[1:3] = ["Literature", "Biology"]
print(f"After replacement: {morning_schedule}")
```

### Question 20: List Clearing

**Clear completed homework items**

**Answer:**

```python
homework_list = ["Math", "Science", "English", "History", "Art"]
print(f"Original: {homework_list}")

# Method 1: clear()
homework_copy = homework_list.copy()
homework_copy.clear()
print(f"After clear(): {homework_copy}")

# Method 2: slice assignment
homework_copy = homework_list.copy()
homework_copy[:] = []
print(f"After slice assignment: {homework_copy}")

# Method 3: del statement
homework_copy = homework_list.copy()
del homework_copy[:]
print(f"After del: {homework_copy}")
```

### Question 21: List Copying

**Create a copy of the student roster for backup**

**Answer:**

```python
original_roster = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

# Method 1: copy()
backup1 = original_roster.copy()

# Method 2: slice
backup2 = original_roster[:]

# Method 3: list() constructor
backup3 = list(original_roster)

# Method 4: multiplication
backup4 = original_roster * 1

print(f"Original: {original_roster}")
print(f"Backup 1: {backup1}")
print(f"Backup 2: {backup2}")
print(f"Backup 3: {backup3}")
print(f"Backup 4: {backup4}")
```

### Question 22: Extend vs Append

**Difference between extend() and append() with class lists**

**Answer:**

```python
freshmen = ["Alice", "Bob", "Charlie"]
sophomores = ["Diana", "Eve", "Frank"]

# append() adds the entire list as one element
freshmen.append(sophomores)
print(f"After append: {freshmen}")  # ['Alice', 'Bob', 'Charlie', ['Diana', 'Eve', 'Frank']]

# extend() adds each element individually
freshmen = ["Alice", "Bob", "Charlie"]
freshmen.extend(sophomores)
print(f"After extend: {freshmen}")  # ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
```

### Question 23: List of Lists Operations

**Work with class sections**

**Answer:**

```python
grade_10_classes = [
    ["Math - Section A", "English - Section A", "Science - Section A"],
    ["Math - Section B", "English - Section B", "Science - Section B"],
    ["Math - Section C", "English - Section C", "Science - Section C"]
]

# Access elements
print(f"Math Section B: {grade_10_classes[1][0]}")        # Math - Section B
print(f"English Section A: {grade_10_classes[0][1]}")     # English - Section A

# Access entire section
print(f"All of Section A: {grade_10_classes[0]}")

# Get all math classes
all_math_classes = [section[0] for section in grade_10_classes]
print(f"All math sections: {all_math_classes}")
```

### Question 24: List Comparison

**Compare two class attendance lists**

**Answer:**

```python
morning_attendance = ["Alice", "Bob", "Charlie", "Diana"]
afternoon_attendance = ["Alice", "Bob", "Charlie", "Diana"]
next_day_attendance = ["Diana", "Charlie", "Bob", "Alice"]

print(f"Same day: {morning_attendance == afternoon_attendance}")  # True (same students, same order)
print(f"Different day: {morning_attendance == next_day_attendance}")  # False (same students, different order)

# Check if all students are present (order doesn't matter)
all_present = all(student in next_day_attendance for student in morning_attendance)
print(f"All students present (order doesn't matter): {all_present}")
```

### Question 25: List Min/Max/Sum

**Find statistics for test scores**

**Answer:**

```python
math_test_scores = [85, 92, 78, 89, 95, 87, 83, 90, 88, 91]
print(f"Lowest score: {min(math_test_scores)}")
print(f"Highest score: {max(math_test_scores)}")
print(f"Total points: {sum(math_test_scores)}")
print(f"Average score: {sum(math_test_scores) / len(math_test_scores):.2f}")

# Find student names (indices) with min/max
lowest_index = math_test_scores.index(min(math_test_scores))
highest_index = math_test_scores.index(max(math_test_scores))
print(f"Index of lowest score: {lowest_index}")
print(f"Index of highest score: {highest_index}")
```

---

## List Operations & Methods (Questions 26-50)

### Question 26: Advanced List Comprehension

**Create a list of even student ID numbers using comprehension**

**Answer:**

```python
even_student_ids = [id_num for id_num in range(1000, 1020) if id_num % 2 == 0]
print(even_student_ids)  # [1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018]

# Traditional way
even_ids_traditional = []
for id_num in range(1000, 1020):
    if id_num % 2 == 0:
        even_ids_traditional.append(id_num)
print(even_ids_traditional)
```

### Question 27: List Comprehension with Conditions

**Find subject names longer than 6 characters**

**Answer:**

```python
subjects = ["Math", "English", "Science", "History", "Art", "Music", "Physical Education"]
long_subjects = [subject for subject in subjects if len(subject) > 6]
print(long_subjects)  # ['English', 'Science', 'History', 'Physical Education']
```

### Question 28: Nested List Comprehension

**Create a class seating arrangement matrix**

**Answer:**

```python
# 4x5 classroom layout (4 rows, 5 seats per row)
classroom = [[f"Student_{i}_{j}" for j in range(1, 6)] for i in range(1, 5)]
for row in classroom:
    print(row)

# Output:
# ['Student_1_1', 'Student_1_2', 'Student_1_3', 'Student_1_4', 'Student_1_5']
# ['Student_2_1', 'Student_2_2', 'Student_2_3', 'Student_2_4', 'Student_2_5']
# ['Student_3_1', 'Student_3_2', 'Student_3_3', 'Student_3_4', 'Student_3_5']
# ['Student_4_1', 'Student_4_2', 'Student_4_3', 'Student_4_4', 'Student_4_5']
```

### Question 29: Filter with List Comprehension

**Get scores below 80 and make them priority for tutoring**

**Answer:**

```python
class_scores = [85, 65, 90, 72, 88, 78, 95, 68, 82, 76]
priority_tutoring = [score for score in class_scores if score < 80]
print(f"Students needing tutoring: {priority_tutoring}")  # [65, 72, 78, 68, 76]
```

### Question 30: Flatten Nested List

**Flatten a 2D list of class sections into a single list**

**Answer:**

```python
class_sections = [
    ["Math A", "Math B", "Math C"],
    ["English A", "English B"],
    ["Science A", "Science B", "Science C", "Science D"]
]

# Method 1: Using nested loops
all_classes = []
for section in class_sections:
    all_classes.extend(section)
print(f"Method 1: {all_classes}")

# Method 2: Using list comprehension
all_classes_comp = [class_name for section in class_sections for class_name in section]
print(f"Method 2: {all_classes_comp}")
```

### Question 31: List of Tuples from Lists

**Create pairs of student names and their grades**

**Answer:**

```python
student_names = ["Alice", "Bob", "Charlie", "Diana"]
student_grades = [85, 92, 78, 96]

# Method 1: Using zip()
student_grade_pairs = list(zip(student_names, student_grades))
print(f"Zipped: {student_grade_pairs}")

# Method 2: Using list comprehension
pairs = [(name, grade) for name, grade in zip(student_names, student_grades)]
print(f"Comprehension: {pairs}")
```

### Question 32: Remove Duplicates from List

**Remove duplicate student names while preserving order**

**Answer:**

```python
duplicate_students = ["Alice", "Bob", "Charlie", "Alice", "Diana", "Bob", "Eve", "Charlie"]

# Method 1: Using set (order not preserved)
unique_students_set = list(set(duplicate_students))
print(f"Using set: {unique_students_set}")

# Method 2: Using set and dict (preserves order in Python 3.7+)
unique_students_ordered = list(dict.fromkeys(duplicate_students))
print(f"Ordered unique: {unique_students_ordered}")

# Method 3: Manual method (always works)
unique_students_manual = []
for student in duplicate_students:
    if student not in unique_students_manual:
        unique_students_manual.append(student)
print(f"Manual: {unique_students_manual}")
```

### Question 33: List Intersection and Union

**Find students in both math club AND science club (intersection) and all students in either club (union)**

**Answer:**

```python
math_club = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
science_club = ["Charlie", "Diana", "Eve", "Frank", "Grace"]

# Convert to sets for efficient operations
math_set = set(math_club)
science_set = set(science_club)

# Intersection (students in both clubs)
both_clubs = list(math_set & science_set)
print(f"Students in both clubs: {both_clubs}")

# Union (students in either club)
either_club = list(math_set | science_set)
print(f"Students in either club: {either_club}")
```

### Question 34: List Rotation

**Rotate daily class schedule by k periods**

**Answer:**

```python
def rotate_schedule(lst, k):
    """Rotate schedule by k periods to the right"""
    n = len(lst)
    k = k % n  # Handle case where k > n
    return lst[-k:] + lst[:-k]

# Test
daily_schedule = ["Math", "English", "Science", "History", "PE", "Lunch", "Art"]
rotated_schedule = rotate_schedule(daily_schedule, 2)
print(f"Original: {daily_schedule}")
print(f"Rotated by 2: {rotated_schedule}")  # Art, Math, English, Science, History, PE, Lunch
```

### Question 35: Find Duplicate Elements

**Find students who submitted the same assignment twice**

**Answer:**

```python
assignment_submissions = ["Alice", "Bob", "Charlie", "Alice", "Diana", "Bob", "Eve", "Alice"]

# Find duplicates
submitted_students = set()
duplicate_students = set()

for student in assignment_submissions:
    if student in submitted_students:
        duplicate_students.add(student)
    else:
        submitted_students.add(student)

print(f"Duplicate submissions: {list(duplicate_students)}")

# With count information
from collections import Counter
submission_counts = Counter(assignment_submissions)
duplicates_with_count = {student: count for student, count in submission_counts.items() if count > 1}
print(f"Duplicates with counts: {duplicates_with_count}")
```

### Question 36: List Chunking

**Split homework list into chunks of size 3 for group assignments**

**Answer:**

```python
def chunk_homework(lst, size):
    """Split homework into chunks of given size"""
    return [lst[i:i+size] for i in range(0, len(lst), size)]

# Test
weekly_homework = ["Math", "Science", "English", "History", "Art", "Music", "PE"]
homework_groups = chunk_homework(weekly_homework, 3)
print(homework_groups)  # [['Math', 'Science', 'English'], ['History', 'Art', 'Music'], ['PE']]
```

### Question 37: Find Missing Student IDs

**Find missing student ID numbers in sequence**

**Answer:**

```python
def find_missing_ids(present_ids, start_id, end_id):
    """Find missing student IDs in sequence"""
    expected_ids = set(range(start_id, end_id + 1))
    present_ids_set = set(present_ids)
    missing_ids = sorted(expected_ids - present_ids_set)
    return missing_ids

# Test
present_student_ids = [1001, 1002, 1004, 1007, 1008, 1010]
missing = find_missing_ids(present_student_ids, 1001, 1010)
print(f"Missing IDs: {missing}")  # [1003, 1005, 1006, 1009]
```

### Question 38: List Shuffling

**Shuffle student names for random presentation order**

**Answer:**

```python
import random

presentation_order = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

# Method 1: shuffle() - shuffles in place
order_copy1 = presentation_order.copy()
random.shuffle(order_copy1)
print(f"Shuffled order: {order_copy1}")

# Method 2: sample() - creates new shuffled list
order_copy2 = presentation_order.copy()
shuffled_order = random.sample(order_copy2, len(order_copy2))
print(f"Sampled order: {shuffled_order}")
```

### Question 39: Frequency Counter

**Count frequency of subjects students signed up for**

**Answer:**

```python
from collections import Counter

subject_preferences = ["Math", "Science", "Math", "English", "Science", "Math", "Art", "Science"]

# Using Counter
preference_counts = Counter(subject_preferences)
print(f"Subject preferences: {preference_counts}")

# Manual method
manual_counts = {}
for subject in subject_preferences:
    manual_counts[subject] = manual_counts.get(subject, 0) + 1
print(f"Manual counts: {manual_counts}")
```

### Question 40: List Partitioning

**Partition students into those who passed and those who need to retake**

**Answer:**

```python
test_results = [85, 45, 90, 62, 88, 35, 92, 58, 78, 40]

# Method 1: Using list comprehensions
passed_students = [score for score in test_results if score >= 60]
need_retake = [score for score in test_results if score < 60]
print(f"Passed: {passed_students}")
print(f"Need retake: {need_retake}")

# Method 2: Using partition function
def partition_students(results):
    passed = []
    need_help = []
    for score in results:
        if score >= 60:
            passed.append(score)
        else:
            need_help.append(score)
    return passed, need_help

passed, retake = partition_students(test_results)
print(f"Passed students: {passed}")
print(f"Students needing help: {retake}")
```

### Question 41: Merge Sorted Lists

**Merge two sorted lists of student names alphabetically**

**Answer:**

```python
def merge_sorted_names(list1, list2):
    """Merge two sorted name lists into one sorted list"""
    merged = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    # Add remaining names
    merged.extend(list1[i:])
    merged.extend(list2[j:])

    return merged

# Test
class_a_names = sorted(["Alice", "Charlie", "Eve"])
class_b_names = sorted(["Bob", "Diana"])
merged_names = merge_sorted_names(class_a_names, class_b_names)
print(f"Merged: {merged_names}")
```

### Question 42: Find Pairs That Sum to Target

**Find pairs of students whose combined attendance days equal a target**

**Answer:**

```python
def find_attendance_pairs(days_present, target_days):
    """Find all pairs that sum to target_days"""
    pairs = []
    seen_days = set()

    for days in days_present:
        partner_days = target_days - days
        if partner_days in seen_days:
            pairs.append((partner_days, days))
        seen_days.add(days)

    return pairs

# Test
student_attendance_days = [10, 15, 8, 12, 13, 5, 20]
target = 23
pairs = find_attendance_pairs(student_attendance_days, target)
print(f"Pairs summing to {target}: {pairs}")
```

### Question 43: List Compression

**Compress consecutive duplicate homework subject assignments**

**Answer:**

```python
def compress_homework(subjects):
    """Compress consecutive duplicate subjects"""
    if not subjects:
        return []

    compressed = [subjects[0]]

    for subject in subjects[1:]:
        if subject != compressed[-1]:
            compressed.append(subject)

    return compressed

# Test
original_homework = ["Math", "Math", "Math", "Science", "Science", "Art", "Math", "Math"]
compressed = compress_homework(original_homework)
print(f"Original: {original_homework}")
print(f"Compressed: {compressed}")
```

### Question 44: List Expansion

**Expand compressed list of students by group size**

**Answer:**

```python
def expand_groupings(compressed):
    """Expand compressed list with student counts"""
    # For run-length encoding: [(student, count), ...]
    def expand_rle(rle_data):
        expanded = []
        for student, count in rle_data:
            expanded.extend([student] * count)
        return expanded

    # Test with group assignments
    group_assignments = [("Team A", 3), ("Team B", 4), ("Team C", 2)]
    expanded = expand_rle(group_assignments)
    print(f"Group data: {group_assignments}")
    print(f"Expanded: {expanded}")
```

### Question 45: List Permutations

**Generate all possible orders for student presentation groups**

**Answer:**

```python
import itertools

def get_presentation_orders(students):
    """Get all possible presentation orders"""
    return list(itertools.permutations(students))

# Test
presentation_group = ["Alice", "Bob", "Charlie"]
orders = get_presentation_orders(presentation_group)
print(f"Possible orders for {presentation_group}:")
for order in orders:
    print(order)
```

### Question 46: List Combinations

**Generate all possible groups of 3 students from the class**

**Answer:**

```python
import itertools

def get_student_groups(students, group_size):
    """Get all possible student groups of size group_size"""
    return list(itertools.combinations(students, group_size))

# Test
class_roster = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
groups_of_3 = get_student_groups(class_roster, 3)
groups_of_2 = get_student_groups(class_roster, 2)
print(f"Groups of 3: {groups_of_3}")
print(f"Groups of 2: {groups_of_2}")
```

### Question 47: List Cartesian Product

**Create all possible student-subject combinations**

**Answer:**

```python
import itertools

def student_subject_combinations(students, subjects):
    """Get Cartesian product of students and subjects"""
    return list(itertools.product(students, subjects))

# Test
students = ["Alice", "Bob", "Charlie"]
subjects = ["Math", "Science", "Art"]
combinations = student_subject_combinations(students, subjects)
print(f"Student-Subject combinations:")
for combo in combinations:
    print(f"{combo[0]} -> {combo[1]}")
```

### Question 48: List Grouping

**Group students by their grade level**

**Answer:**

```python
from collections import defaultdict

def group_by_grade_level(student_ids, grades):
    """Group students by their grade level"""
    groups = defaultdict(list)
    for student_id, grade in zip(student_ids, grades):
        groups[grade].append(student_id)
    return dict(groups)

# Test
student_ids = ["STU001", "STU002", "STU003", "STU004", "STU005"]
grades = [9, 10, 9, 11, 10]
grouped = group_by_grade_level(student_ids, grades)
print(f"Grouped by grade level: {grouped}")
```

### Question 49: List Deduplication

**Remove duplicate club memberships while preserving original order**

**Answer:**

```python
def remove_duplicate_memberships(lst):
    """Remove duplicates while preserving order"""
    seen = set()
    seen_add = seen.add
    return [student for student in lst if not (student in seen or seen_add(student))]

# Test
club_memberships = ["Alice", "Bob", "Charlie", "Alice", "Diana", "Bob", "Eve", "Charlie"]
unique_memberships = remove_duplicate_memberships(club_memberships)
print(f"Original: {club_memberships}")
print(f"Unique: {unique_memberships}")
```

### Question 50: List Operations Benchmark

**Compare performance of different list operations for large school data**

**Answer:**

```python
import time

def benchmark_school_operations():
    """Benchmark various list operations on large school data"""
    large_student_list = list(range(10000))

    # Benchmark membership test
    start_time = time.time()
    9999 in large_student_list
    list_membership_time = time.time() - start_time

    # Compare with set
    student_set = set(large_student_list)
    start_time = time.time()
    9999 in student_set
    set_membership_time = time.time() - start_time

    print(f"List membership test: {list_membership_time:.6f} seconds")
    print(f"Set membership test: {set_membership_time:.6f} seconds")
    print(f"Set is {list_membership_time/set_membership_time:.1f}x faster!")

benchmark_school_operations()
```

---

## Tuples (Questions 51-65)

### Question 51: Create and Access Tuples

**Create a tuple representing student coordinates in classroom**

**Answer:**

```python
# Student seating coordinates (row, column)
student_positions = {
    "Alice": (1, 3),
    "Bob": (2, 1),
    "Charlie": (3, 4),
    "Diana": (2, 3)
}

print(f"Alice sits at: {student_positions['Alice']}")

# Access individual coordinates
alice_row, alice_column = student_positions["Alice"]
print(f"Alice: Row {alice_row}, Column {alice_column}")
```

### Question 52: Tuple Immutability

**Demonstrate that student records cannot be changed**

**Answer:**

```python
student_record = ("Alice Johnson", 16, 3.85)
print(f"Original record: {student_record}")

# This will cause an error:
# student_record[1] = 17  # TypeError: 'tuple' object does not support item assignment

# Instead, create a new record for next year
new_record = (student_record[0], 17, 3.90)
print(f"Updated record: {new_record}")

# Or create a new record with different GPA
updated_gpa = (student_record[0], student_record[1], 3.90)
print(f"GPA update: {updated_gpa}")
```

### Question 53: Single Element Tuple

**Create single-element tuples for unique identifiers**

**Answer:**

```python
# Wrong way - this creates an integer, not tuple
class_id_wrong = (101)
print(f"Wrong: {class_id_wrong}, type: {type(class_id_wrong)}")

# Correct way - comma is required
class_id_correct = (101,)
print(f"Correct: {class_id_correct}, type: {type(class_id_correct)}")

# Empty tuple
empty_roster = ()
print(f"Empty roster: {empty_roster}, type: {type(empty_roster)}")
```

### Question 54: Tuple Methods

**Use count() and index() with student class data**

**Answer:**

```python
class_periods = ("Math", "English", "Science", "Math", "History", "Math")

# Count occurrences
math_count = class_periods.count("Math")
print(f"'Math' appears {math_count} times")

# Find index of first occurrence
english_index = class_periods.index("English")
print(f"'English' is at period: {english_index}")

# Try to find index of non-existent subject
try:
    art_index = class_periods.index("Art")
except ValueError as e:
    print(f"Art not found: {e}")
```

### Question 55: Tuple Unpacking

**Unpack student information tuples**

**Answer:**

```python
# Basic unpacking
student_info = ("Sarah Connor", 16, "Grade 10", "Science Club")
name, age, grade_level, club = student_info
print(f"{name} is {age} years old in {grade_level} and in {club}")

# Unpacking with *
test_scores = (85, 92, 88, 95, 90)
first_score, *middle_scores, last_score = test_scores
print(f"First: {first_score}, Middle: {middle_scores}, Last: {last_score}")

# Unpacking with _ for ignored values
coordinates_3d = (10, 20, 30)
x, y, _ = coordinates_3d  # Ignore z-coordinate
print(f"X: {x}, Y: {y}")
```

### Question 56: Named Tuples

**Create and use named tuples for student records**

**Answer:**

```python
from collections import namedtuple

# Define named tuple for student records
Student = namedtuple('Student', ['name', 'age', 'grade', 'gpa'])

# Create student instances
student1 = Student("Alice Johnson", 16, 10, 3.85)
student2 = Student("Bob Smith", 17, 11, 3.92)

print(f"Student 1: {student1}")
print(f"Name: {student1.name}")
print(f"GPA: {student1.gpa}")

# Convert to dictionary
student_dict = student1._asdict()
print(f"As dict: {student_dict}")
```

### Question 57: Tuple Concatenation and Multiplication

**Combine and repeat class schedules**

**Answer:**

```python
morning_schedule = ("Math", "English", "Science")
afternoon_schedule = ("History", "PE", "Art")

# Concatenation
full_schedule = morning_schedule + afternoon_schedule
print(f"Full schedule: {full_schedule}")

# Multiplication
lunch_break = ("Lunch",)
extended_lunch = lunch_break * 5
print(f"Extended lunch schedule: {extended_lunch}")

# Nested tuples
daily_blocks = (morning_schedule, afternoon_schedule, ("Study Hall",))
print(f"Daily blocks: {daily_blocks}")
print(f"Morning block: {daily_blocks[0]}")
```

### Question 58: Tuple Comparison

**Compare student grade tuples**

**Answer:**

```python
student1_grades = (85, 92, 88, 90)
student2_grades = (85, 92, 88, 90)
student3_grades = (85, 92, 88, 91)
student4_grades = (85, 93, 88, 90)

print(f"Same student: {student1_grades == student2_grades}")  # True
print(f"Student 1 vs 3: {student1_grades < student3_grades}")    # True (lexicographic comparison)
print(f"Student 3 vs 4: {student3_grades > student4_grades}")    # False

# Comparison is element by element
grades_a = (85, 90)
grades_b = (85, 90, 88)  # Shorter tuple is considered smaller
print(f"grades_a < grades_b: {grades_a < grades_b}")  # True
```

### Question 59: Tuple to List and Vice Versa

**Convert between tuples and lists for mutable operations**

**Answer:**

```python
# Tuple to list
class_schedule = ("Math", "English", "Science", "History", "PE")
schedule_list = list(class_schedule)
print(f"Tuple to list: {schedule_list}")

# List to tuple
homework_list = ["Math problems", "Science lab", "Essay", "History reading"]
homework_tuple = tuple(homework_list)
print(f"List to tuple: {homework_tuple}")

# Modify list, then convert back
schedule_list.append("Art")
back_to_tuple = tuple(schedule_list)
print(f"Modified and converted back: {back_to_tuple}")
```

### Question 60: Tuple as Dictionary Keys

**Use tuples as dictionary keys for classroom coordinates**

**Answer:**

```python
# Classroom seats can be stored with tuple coordinates as keys
classroom_seats = {
    (0, 0): "Alice Johnson",
    (0, 1): "Bob Smith",
    (1, 0): "Charlie Brown",
    (1, 1): "Diana Prince",
    (2, 0): "Eve Wilson",
    (2, 1): "Frank Miller"
}

print(f"Seat (1, 0): {classroom_seats[(1, 0)]}")

# Grade ranges with tuple keys
grade_ranges = {
    (90, 100): "A",
    (80, 89): "B",
    (70, 79): "C",
    (60, 69): "D",
    (0, 59): "F"
}

print(f"Grade for 85%: {grade_ranges.get((80, 89))}")
```

### Question 61: Tuple in Sets

**Use tuples in sets for unique student-coordinate combinations**

**Answer:**

```python
# Unique student positions (rows, columns)
assigned_positions = {
    (0, 0),
    (1, 1),
    (2, 2),
    (0, 0),  # Duplicate - will be ignored
    (3, 3)
}

print(f"Unique positions: {assigned_positions}")

# Check if position exists
position_exists = (1, 1) in assigned_positions
print(f"Position (1, 1) assigned: {position_exists}")

# Add new position
assigned_positions.add((4, 4))
print(f"After adding (4, 4): {assigned_positions}")
```

### Question 62: Tuple Return from Functions

**Use tuples to return multiple student statistics**

**Answer:**

```python
def calculate_class_stats(scores):
    """Return multiple statistics as tuple"""
    if not scores:
        return (0, 0, 0, 0)

    minimum = min(scores)
    maximum = max(scores)
    total = sum(scores)
    average = total / len(scores)

    return (minimum, maximum, total, average)

# Test the function
math_scores = [85, 92, 78, 88, 95, 87, 83, 90]
min_score, max_score, total, avg = calculate_class_stats(math_scores)

print(f"Class statistics: min={min_score}, max={max_score}, total={total}, avg={avg:.2f}")

# Return min and max separately
def get_score_range(scores):
    return (min(scores), max(scores))

minimum, maximum = get_score_range(math_scores)
print(f"Score range: {minimum} - {maximum}")
```

### Question 63: Tuple Sorting

**Sort lists of student tuples by different criteria**

**Answer:**

```python
# List of students as tuples (name, grade, age)
class_roster = [
    ("Alice Johnson", 85, 16),
    ("Bob Smith", 92, 17),
    ("Charlie Brown", 78, 16),
    ("Diana Prince", 96, 17),
    ("Eve Wilson", 88, 16)
]

# Sort by name (first element)
by_name = sorted(class_roster)
print(f"Sorted by name: {by_name}")

# Sort by grade (second element)
by_grade = sorted(class_roster, key=lambda x: x[1])
print(f"Sorted by grade: {by_grade}")

# Sort by age (third element)
by_age = sorted(class_roster, key=lambda x: x[2])
print(f"Sorted by age: {by_age}")

# Sort by multiple criteria (grade descending, then name)
by_grade_desc_name = sorted(class_roster, key=lambda x: (-x[1], x[0]))
print(f"Sorted by grade desc, then name: {by_grade_desc_name}")
```

### Question 64: Tuple Comprehension

**Create tuples using comprehension**

**Answer:**

```python
# Generator expression converted to tuple
student_ids = tuple(f"STU{i:03d}" for i in range(1, 6))
print(f"Student IDs: {student_ids}")

# Even period numbers as tuple
period_numbers = tuple(period for period in range(1, 9) if period % 2 == 0)
print(f"Even periods: {period_numbers}")

# Classroom coordinates as nested tuples
classroom_coords = tuple((row, col) for row in range(3) for col in range(4))
print(f"Classroom coordinates: {classroom_coords}")
```

### Question 65: Tuple Memory Efficiency

**Compare memory usage of tuples vs lists for student data**

**Answer:**

```python
import sys

# Compare memory usage for student records
student_list = ["Alice", 16, 3.85, "Grade 10", True]
student_tuple = ("Alice", 16, 3.85, "Grade 10", True)

list_size = sys.getsizeof(student_list)
tuple_size = sys.getsizeof(student_tuple)

print(f"Student list size: {list_size} bytes")
print(f"Student tuple size: {tuple_size} bytes")
print(f"Tuple is {list_size - tuple_size} bytes smaller")

# For larger class data
large_class_list = list(range(1000))
large_class_tuple = tuple(range(1000))

large_list_size = sys.getsizeof(large_class_list)
large_tuple_size = sys.getsizeof(large_class_tuple)

print(f"Large class list size: {large_list_size} bytes")
print(f"Large class tuple size: {large_tuple_size} bytes")

# Why tuples are more memory efficient:
# 1. Immutable - no need for methods to modify
# 2. Better memory allocation for fixed data
# 3. Can be cached for small values
# 4. Ideal for student records and configuration data
```

---

## Sets (Questions 66-85)

### Question 66: Create and Access Sets

**Create sets for unique student clubs and check membership**

**Answer:**

```python
# Set of unique clubs in school
active_clubs = {"Chess Club", "Drama Club", "Math Club", "Science Club", "Art Club"}
print(f"Active clubs: {active_clubs}")

# Check if a club exists
print(f"Chess Club exists: {'Chess Club' in active_clubs}")
print(f"Cooking Club exists: {'Cooking Club' in active_clubs}")

# Add a new club
active_clubs.add("Coding Club")
print(f"After adding Coding Club: {active_clubs}")

# Remove a club
active_clubs.remove("Art Club")
print(f"After removing Art Club: {active_clubs}")
```

### Question 67: Set Operations - Union

**Combine students from two different club events**

**Answer:**

```python
chess_tournament = {"Alice", "Bob", "Charlie", "Diana"}
math_olympiad = {"Bob", "Charlie", "Eve", "Frank"}

# Union - all students who participated in either event
all_participants = chess_tournament | math_olympiad
print(f"All participants: {all_participants}")

# Using union() method
all_participants_method = chess_tournament.union(math_olympiad)
print(f"Using union method: {all_participants_method}")
```

### Question 68: Set Operations - Intersection

**Find students who attended both club meetings**

**Answer:**

```python
monday_meeting = {"Alice", "Bob", "Charlie", "Diana"}
wednesday_meeting = {"Alice", "Bob", "Eve", "Frank"}

# Intersection - students who attended both meetings
both_days = monday_meeting & wednesday_meeting
print(f"Attended both days: {both_days}")

# Using intersection() method
both_days_method = monday_meeting.intersection(wednesday_meeting)
print(f"Using intersection method: {both_days_method}")

# Students who only attended Monday
only_monday = monday_meeting - wednesday_meeting
print(f"Only Monday: {only_monday}")
```

### Question 69: Set Operations - Difference

**Find students who are in only one club**

**Answer:**

```python
science_club = {"Alice", "Bob", "Charlie", "Diana", "Eve"}
robotics_club = {"Charlie", "Diana", "Frank", "Grace", "Henry"}

# Only in science club
only_science = science_club - robotics_club
print(f"Only in science club: {only_science}")

# Only in robotics club
only_robotics = robotics_club - science_club
print(f"Only in robotics club: {only_robotics}")

# Students in exactly one club
exactly_one_club = only_science | only_robotics
print(f"In exactly one club: {exactly_one_club}")
```

### Question 70: Set Operations - Symmetric Difference

**Find students who attended exactly one club meeting**

**Answer:**

```python
morning_session = {"Alice", "Bob", "Charlie", "Diana"}
afternoon_session = {"Alice", "Charlie", "Eve", "Frank"}

# Students who attended exactly one session (not both)
exactly_one = morning_session ^ afternoon_session
print(f"Attended exactly one session: {exactly_one}")

# Using symmetric_difference() method
exactly_one_method = morning_session.symmetric_difference(afternoon_session)
print(f"Using symmetric_difference method: {exactly_one_method}")

# Verify: total unique students minus those in both
total_unique = morning_session | afternoon_session
both_sessions = morning_session & afternoon_session
verification = total_unique - both_sessions
print(f"Verification: {verification}")
```

### Question 71: Set Subset and Superset

**Check if all students in one class are in another**

**Answer:**

```python
honors_math = {"Alice", "Bob", "Charlie", "Diana"}
advanced_math = {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"}

# Is honors_math a subset of advanced_math?
print(f"Honors math is subset of advanced math: {honors_math.issubset(advanced_math)}")

# Is advanced_math a superset of honors_math?
print(f"Advanced math is superset of honors math: {advanced_math.issubset(honors_math)}")

# Proper subset (strict subset, not equal)
print(f"Honors math is proper subset: {honors_math < advanced_math}")
print(f"Advanced math is proper superset: {advanced_math > honors_math}")
```

### Question 72: Set Disjoint

**Check if two clubs have no common members**

**Answer:**

```python
music_club = {"Alice", "Bob", "Charlie"}
sports_club = {"Diana", "Eve", "Frank", "Grace"}

# Check if clubs are disjoint (no common members)
disjoint = music_club.isdisjoint(sports_club)
print(f"Music and Sports clubs are disjoint: {disjoint}")

# Now check with a club that might overlap
art_club = {"Alice", "Frank", "Grace"}
overlap_check = music_club.isdisjoint(art_club)
print(f"Music and Art clubs are disjoint: {overlap_check}")

print(f"Common members: {music_club & art_club}")
```

### Question 73: Set Comprehension

**Create sets using comprehension for student grades**

**Answer:**

```python
# Set of students who scored above 85
class_scores = [85, 92, 78, 88, 95, 72, 89, 76, 91, 84]
high_scorers = {score for score in class_scores if score > 85}
print(f"Scores above 85: {high_scorers}")

# Set of even student IDs
student_ids = range(1001, 1011)
even_ids = {id_num for id_num in student_ids if id_num % 2 == 0}
print(f"Even student IDs: {even_ids}")

# Set from existing list with modification
student_names = ["alice", "bob", "charlie", "diana"]
name_set = {name.upper() for name in student_names}
print(f"Names in uppercase: {name_set}")
```

### Question 74: Frozenset

**Use immutable frozenset for constant school configurations**

**Answer:**

```python
# Allowed grade levels (can't be changed)
ALLOWED_GRADES = frozenset([9, 10, 11, 12])
print(f"Allowed grades: {ALLOWED_GRADES}")

# Set of required subjects
REQUIRED_SUBJECTS = frozenset(["Math", "English", "Science", "History"])
print(f"Required subjects: {REQUIRED_SUBJECTS}")

# Can still perform set operations
all_subjects = {"Math", "English", "Science", "History", "Art", "PE", "Music"}
elective_subjects = all_subjects - REQUIRED_SUBJECTS
print(f"Elective subjects: {elective_subjects}")

# Frozensets can be in sets
subject_combinations = {
    frozenset(["Math", "Science"]),
    frozenset(["English", "History"]),
    frozenset(["Art", "Music"])
}
print(f"Subject combinations: {subject_combinations}")
```

### Question 75: Set Union Methods

**Compare different ways to union multiple class sets**

**Answer:**

```python
grade_9 = {"Alice", "Bob", "Charlie"}
grade_10 = {"Diana", "Eve", "Frank"}
grade_11 = {"Grace", "Henry"}
grade_12 = {"Ivy", "Jack"}

# Method 1: Using | operator iteratively
all_students = grade_9 | grade_10 | grade_11 | grade_12
print(f"All students (operator): {all_students}")

# Method 2: Using union() method with multiple sets
all_students_method = grade_9.union(grade_10, grade_11, grade_12)
print(f"All students (method): {all_students_method}")

# Method 3: Using update() method (modifies original)
grade_9_copy = grade_9.copy()
grade_9_copy.update(grade_10, grade_11, grade_12)
print(f"All students (update): {grade_9_copy}")
```

### Question 76: Set Intersection Methods

**Find students present in all class periods**

**Answer:**

```python
period_1 = {"Alice", "Bob", "Charlie", "Diana"}
period_2 = {"Bob", "Charlie", "Diana", "Eve"}
period_3 = {"Charlie", "Diana", "Eve", "Frank"}

# Students present in all three periods
all_periods = period_1 & period_2 & period_3
print(f"Students in all periods: {all_periods}")

# Using intersection() method
all_periods_method = period_1.intersection(period_2, period_3)
print(f"Using intersection method: {all_periods_method}")

# Students in at least two periods
at_least_two = (period_1 & period_2) | (period_1 & period_3) | (period_2 & period_3)
print(f"Students in at least two periods: {at_least_two}")
```

### Question 77: Set Filtering

**Filter students based on multiple criteria using sets**

**Answer:**

```python
all_students = {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"}
math_students = {"Alice", "Bob", "Charlie", "Diana"}
science_students = {"Bob", "Charlie", "Eve", "Frank"}
honor_roll = {"Alice", "Charlie", "Eve", "Grace"}

# Students in both math and science
both_math_science = math_students & science_students
print(f"In both math and science: {both_math_science}")

# Students in math but not science
math_only = math_students - science_students
print(f"Math only: {math_only}")

# Honor roll students in advanced subjects
advanced_honor = honor_roll & (math_students | science_students)
print(f"Honor roll in advanced subjects: {advanced_honor}")
```

### Question 78: Set Cardinatity

**Work with set sizes for classroom management**

**Answer:**

```python
class_a = {"Alice", "Bob", "Charlie", "Diana", "Eve"}
class_b = {"Frank", "Grace", "Henry", "Ivy"}
class_c = {"Alice", "Charlie", "Eve", "Jack", "Kate"}

print(f"Class A size: {len(class_a)}")
print(f"Class B size: {len(class_b)}")
print(f"Class C size: {len(class_c)}")

# Total unique students
all_classes = class_a | class_b | class_c
print(f"Total unique students: {len(all_classes)}")

# Students in multiple classes
in_two_classes = (class_a & class_b) | (class_a & class_c) | (class_b & class_c)
print(f"Students in multiple classes: {len(in_two_classes)}")

# Students in exactly one class
exactly_one = all_classes - in_two_classes
print(f"Students in exactly one class: {len(exactly_one)}")
```

### Question 79: Set Type Conversion

**Convert between lists, tuples, and sets for student data**

**Answer:**

```python
# List to set (removes duplicates)
student_list = ["Alice", "Bob", "Alice", "Charlie", "Bob", "Diana"]
unique_students = set(student_list)
print(f"Original list: {student_list}")
print(f"Unique set: {unique_students}")

# Set back to list (order not preserved)
unique_list = list(unique_students)
print(f"Back to list: {unique_list}")

# Tuple to set and back
student_tuple = ("Alice", "Bob", "Charlie")
student_set = set(student_tuple)
tuple_from_set = tuple(student_set)
print(f"Tuple: {student_tuple}")
print(f"Set: {student_set}")
print(f"Back to tuple: {tuple_from_set}")

# When order matters, preserve it
ordered_list = []
for student in student_list:
    if student not in ordered_list:
        ordered_list.append(student)
print(f"Unique list (preserving order): {ordered_list}")
```

### Question 80: Set Performance

**Compare set vs list membership testing for large student databases**

**Answer:**

```python
import time

def benchmark_membership():
    # Large dataset
    all_students = list(range(10000))
    student_set = set(all_students)

    # Test membership
    test_student = 5000

    # List membership (slower)
    start_time = time.time()
    test_student in all_students
    list_time = time.time() - start_time

    # Set membership (faster)
    start_time = time.time()
    test_student in student_set
    set_time = time.time() - start_time

    print(f"List membership test: {list_time:.6f} seconds")
    print(f"Set membership test: {set_time:.6f} seconds")
    print(f"Set is {list_time/set_time:.1f}x faster!")

    # For 100 membership tests
    start_time = time.time()
    for _ in range(100):
        test_student in all_students
    list_100_time = time.time() - start_time

    start_time = time.time()
    for _ in range(100):
        test_student in student_set
    set_100_time = time.time() - start_time

    print(f"100 list tests: {list_100_time:.6f} seconds")
    print(f"100 set tests: {set_100_time:.6f} seconds")

benchmark_membership()
```

### Question 81: Set with Custom Objects

**Handle complex student data in sets**

**Answer:**

```python
# Students as tuples (name, grade_level)
student_records = {
    ("Alice Johnson", 10),
    ("Bob Smith", 11),
    ("Charlie Brown", 10),
    ("Diana Prince", 11),
    ("Alice Johnson", 10)  # Duplicate
}

print(f"Unique student records: {len(student_records)}")
print(f"Students: {student_records}")

# Find students by grade level
grade_10_students = {record for record in student_records if record[1] == 10}
print(f"Grade 10 students: {grade_10_students}")

# Student ID and name pairs
student_data = {
    ("STU001", "Alice Johnson"),
    ("STU002", "Bob Smith"),
    ("STU003", "Charlie Brown"),
    ("STU001", "Alice Johnson")  # Duplicate ID
}

unique_student_data = set(student_data)
print(f"Unique student data: {unique_student_data}")
```

### Question 82: Set Operations Chaining

**Chain multiple set operations for complex queries**

**Answer:**

```python
math_club = {"Alice", "Bob", "Charlie", "Diana", "Eve"}
science_club = {"Bob", "Charlie", "Eve", "Frank", "Grace"}
chess_club = {"Alice", "Charlie", "Frank", "Henry"}
debate_club = {"Diana", "Eve", "Grace", "Henry", "Ivy"}

# Find students in exactly two clubs
exactly_two_clubs = (
    (math_club & science_club) |
    (math_club & chess_club) |
    (math_club & debate_club) |
    (science_club & chess_club) |
    (science_club & debate_club) |
    (chess_club & debate_club)
) - (
    math_club & science_club & chess_club |
    math_club & science_club & debate_club |
    math_club & chess_club & debate_club |
    science_club & chess_club & debate_club
)

print(f"Students in exactly two clubs: {exactly_two_clubs}")

# Students in all clubs
all_clubs = math_club & science_club & chess_club & debate_club
print(f"Students in all clubs: {all_clubs}")
```

### Question 83: Set Partitioning

**Partition students into groups using set operations**

**Answer:**

```python
all_students = {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"}
morning_class = {"Alice", "Bob", "Charlie"}
afternoon_class = {"Diana", "Eve", "Frank", "Grace"}

# Verify partition (every student in exactly one group)
is_partition = len(all_students) == len(morning_class | afternoon_class)
print(f"Is proper partition: {is_partition}")

print(f"All students accounted for: {morning_class | afternoon_class == all_students}")
print(f"No overlap: {len(morning_class & afternoon_class) == 0}")

# Create more complex partitioning
classifications = {
    "beginners": {"Alice", "Bob"},
    "intermediate": {"Charlie", "Diana"},
    "advanced": {"Eve", "Frank", "Grace"}
}

# Verify no overlaps
total_classified = set()
for group in classifications.values():
    total_classified |= group

print(f"All students classified: {total_classified == all_students}")

# Students in multiple skill levels (advanced students)
advanced_only = classifications["advanced"] - classifications["beginners"] - classifications["intermediate"]
print(f"Advanced only students: {advanced_only}")
```

### Question 84: Set Symmetric Difference Applications

**Use symmetric difference for exclusive group membership**

**Answer:**

```python
week1_attendance = {"Alice", "Bob", "Charlie", "Diana"}
week2_attendance = {"Alice", "Bob", "Eve", "Frank"}
week3_attendance = {"Charlie", "Diana", "Eve", "Frank"}

# Students whose attendance changed between weeks
attendance_changed_1_2 = week1_attendance ^ week2_attendance
print(f"Attendance changed (week1 -> week2): {attendance_changed_1_2}")

attendance_changed_2_3 = week2_attendance ^ week3_attendance
print(f"Attendance changed (week2 -> week3): {attendance_changed_2_3}")

# Students who attended exactly one week
exactly_one_week = (week1_attendance ^ week2_attendance) | (week2_attendance ^ week3_attendance) | (week1_attendance ^ week3_attendance)
exactly_one_week = exactly_one_week - (week1_attendance & week2_attendance & week3_attendance)
print(f"Attended exactly one week: {exactly_one_week}")

# Students who attended consecutive weeks
consecutive_weeks = (week1_attendance & week2_attendance) | (week2_attendance & week3_attendance)
print(f"Attended consecutive weeks: {consecutive_weeks}")
```

### Question 85: Set Challenge - Student Scheduling

**Complex scheduling problem using multiple set operations**

**Answer:**

```python
def solve_scheduling_conflict():
    """Find students who can take advanced classes"""

    # Prerequisites
    algebra_1 = {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"}
    algebra_2_candidates = {"Bob", "Charlie", "Diana", "Frank", "Grace", "Henry"}
    geometry_candidates = {"Alice", "Charlie", "Diana", "Eve", "Frank", "Ivy"}

    # Grade requirements
    high_grade_students = {"Charlie", "Diana", "Grace"}
    honor_students = {"Alice", "Charlie", "Eve"}

    # Find students who can take algebra 2
    can_take_algebra_2 = algebra_2_candidates & high_grade_students
    print(f"Students ready for Algebra 2: {can_take_algebra_2}")

    # Find students who can take geometry
    can_take_geometry = geometry_candidates & honor_students
    print(f"Students ready for Geometry: {can_take_geometry}")

    # Students who can take both
    can_take_both = can_take_algebra_2 & can_take_geometry
    print(f"Students who can take both: {can_take_both}")

    # Students who can take at least one
    can_take_one = can_take_algebra_2 | can_take_geometry
    print(f"Students who can take at least one: {can_take_one}")

    # Students who need prerequisite help
    needs_help = algebra_1 - can_take_one
    print(f"Students needing prerequisite support: {needs_help}")

solve_scheduling_conflict()
```

---

## Dictionaries (Questions 86-115)

### Question 86: Create and Access Dictionaries

**Create a student grade book dictionary**

**Answer:**

```python
# Student grades dictionary
grade_book = {
    "Alice Johnson": 92,
    "Bob Smith": 85,
    "Charlie Brown": 78,
    "Diana Prince": 96,
    "Eve Wilson": 88
}

print(f"Grade book: {grade_book}")

# Access individual grades
print(f"Alice's grade: {grade_book['Alice Johnson']}")
print(f"Bob's grade: {grade_book.get('Bob Smith')}")

# Get with default for non-existent student
print(f"Frank's grade: {grade_book.get('Frank Miller', 'Not enrolled')}")

# Access all keys, values, and items
print(f"Students: {list(grade_book.keys())}")
print(f"Grades: {list(grade_book.values())}")
print(f"Student-grade pairs: {list(grade_book.items())}")
```

### Question 87: Dictionary Methods

**Use various dictionary methods for school data management**

**Answer:**

```python
student_info = {
    "STU001": {"name": "Alice", "grade": 10, "gpa": 3.85},
    "STU002": {"name": "Bob", "grade": 11, "gpa": 3.92},
    "STU003": {"name": "Charlie", "grade": 10, "gpa": 3.78}
}

# get() method with default
print(f"Student info: {student_info.get('STU004', 'Student not found')}")

# keys(), values(), items()
print(f"All student IDs: {list(student_info.keys())}")
print(f"All names: {list(student_info.values())}")
print(f"All student records: {list(student_info.items())}")

# update() method - add or modify students
student_info["STU004"] = {"name": "Diana", "grade": 9, "gpa": 3.95}
print(f"After adding Diana: {len(student_info)} students")

# pop() method - remove student
removed_student = student_info.pop("STU002", "Student not found")
print(f"Removed: {removed_student}")
print(f"Remaining students: {len(student_info)}")

# popitem() method - remove last added item
last_student = student_info.popitem()
print(f"Last student removed: {last_student}")
```

### Question 88: Nested Dictionaries

**Create a complex school information system**

**Answer:**

```python
school_data = {
    "Lincoln High School": {
        "students": {
            "STU001": {"name": "Alice Johnson", "grade": 10, "clubs": ["Chess", "Math"]},
            "STU002": {"name": "Bob Smith", "grade": 11, "clubs": ["Sports", "Drama"]},
            "STU003": {"name": "Charlie Brown", "grade": 9, "clubs": ["Science", "Art"]}
        },
        "teachers": {
            "T001": {"name": "Mrs. Davis", "subject": "Math", "years": 15},
            "T002": {"name": "Mr. Wilson", "subject": "Science", "years": 8}
        },
        "classes": {
            "MATH101": {"teacher": "T001", "students": ["STU001", "STU002"], "period": 1},
            "SCI101": {"teacher": "T002", "students": ["STU002", "STU003"], "period": 2}
        }
    }
}

# Access nested data
print(f"School name: {school_data['Lincoln High School']['students']['STU001']['name']}")
print(f"Alice's clubs: {school_data['Lincoln High School']['students']['STU001']['clubs']}")

# Iterate through students
for student_id, info in school_data["Lincoln High School"]["students"].items():
    print(f"{info['name']} (Grade {info['grade']}) - Clubs: {', '.join(info['clubs'])}")
```

### Question 89: Dictionary Comprehension

**Create dictionaries using comprehension for student data**

**Answer:**

```python
student_names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
student_grades = [92, 85, 78, 96, 88]

# Create grade dictionary
grade_dict = {name: grade for name, grade in zip(student_names, student_grades)}
print(f"Grade dictionary: {grade_dict}")

# Create GPA dictionary with grade calculation
test_scores = [85, 92, 78, 89, 95, 87, 83, 90]
student_gpas = {f"Student_{i+1}": score/20 for i, score in enumerate(test_scores)}
print(f"GPA dictionary: {student_gpas}")

# Filter dictionary comprehension
passing_students = {name: grade for name, grade in grade_dict.items() if grade >= 80}
print(f"Passing students: {passing_students}")

# Transform values
grade_letters = {name: "A" if grade >= 90 else "B" if grade >= 80 else "C"
                for name, grade in grade_dict.items()}
print(f"Letter grades: {grade_letters}")
```

### Question 90: Dictionary Merging

**Combine multiple class dictionaries**

**Answer:**

```python
class_a_grades = {"Alice": 92, "Bob": 85, "Charlie": 78}
class_b_grades = {"Diana": 96, "Eve": 88, "Frank": 91}
class_c_grades = {"Alice": 90, "Bob": 87, "Grace": 84}

# Method 1: Using update()
all_grades = class_a_grades.copy()
all_grades.update(class_b_grades)
all_grades.update(class_c_grades)
print(f"Using update(): {all_grades}")

# Method 2: Using dictionary unpacking (Python 3.5+)
all_grades_unpack = {**class_a_grades, **class_b_grades, **class_c_grades}
print(f"Using unpacking: {all_grades_unpack}")

# Handle conflicts - keep first occurrence
all_grades_first = class_a_grades | class_b_grades | class_c_grades
print(f"Keep first: {all_grades_first}")

# Handle conflicts - keep last occurrence
all_grades_last = {**class_c_grades, **class_b_grades, **class_a_grades}
print(f"Keep last: {all_grades_last}")

# Merge with conflict resolution
def smart_merge(*dicts, resolve_conflict='max'):
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key in result:
                if resolve_conflict == 'max':
                    result[key] = max(result[key], value)
                elif resolve_conflict == 'min':
                    result[key] = min(result[key], value)
                elif resolve_conflict == 'avg':
                    # Track occurrences for averaging
                    result[key] = [result[key], value]
                else:
                    result[key] = value
            else:
                result[key] = value

    # Handle averaging if requested
    if resolve_conflict == 'avg':
        averaged_result = {}
        for key, value in result.items():
            if isinstance(value, list):
                averaged_result[key] = sum(value) / len(value)
            else:
                averaged_result[key] = value
        result = averaged_result

    return result

merged_grades = smart_merge(class_a_grades, class_b_grades, class_c_grades)
print(f"Smart merge (max): {merged_grades}")
```

### Question 91: Dictionary Grouping

**Group students by grade level**

**Answer:**

```python
students = {
    "Alice": 10,
    "Bob": 11,
    "Charlie": 10,
    "Diana": 9,
    "Eve": 11,
    "Frank": 9,
    "Grace": 12
}

# Group students by grade level
from collections import defaultdict

# Method 1: Using defaultdict
students_by_grade = defaultdict(list)
for student, grade in students.items():
    students_by_grade[grade].append(student)

print(f"Using defaultdict: {dict(students_by_grade)}")

# Method 2: Using regular dict with setdefault
students_by_grade_regular = {}
for student, grade in students.items():
    students_by_grade_regular.setdefault(grade, []).append(student)

print(f"Using setdefault: {students_by_grade_regular}")

# Method 3: Dictionary comprehension
unique_grades = set(students.values())
students_by_grade_comp = {grade: [student for student, g in students.items() if g == grade]
                         for grade in unique_grades}

print(f"Using comprehension: {students_by_grade_comp}")

# Sort by grade level
sorted_by_grade = dict(sorted(students_by_grade.items()))
print(f"Sorted by grade: {sorted_by_grade}")
```

### Question 92: Dictionary Inversion

**Swap keys and values in grade dictionary**

**Answer:**

```python
student_grades = {"Alice": 92, "Bob": 85, "Charlie": 78, "Diana": 96, "Eve": 88}

# Simple inversion (but loses duplicates)
grade_to_student = {grade: student for student, grade in student_grades.items()}
print(f"Simple inversion: {grade_to_student}")

# Handle duplicates by creating lists
from collections import defaultdict

grade_to_students = defaultdict(list)
for student, grade in student_grades.items():
    grade_to_students[grade].append(student)

print(f"Handle duplicates: {dict(grade_to_students)}")

# Reverse with value transformation
student_to_gpa = {student: grade/20 for student, grade in student_grades.items()}
gpa_to_student = {gpa: student for student, gpa in student_to_gpa.items()}
print(f"GPA conversion: {gpa_to_student}")

# Multi-level inversion for nested data
student_subjects = {
    "Alice": {"Math": 92, "Science": 88},
    "Bob": {"Math": 85, "Science": 91}
}

# Invert to subject -> student -> grade
subject_student_grades = {}
for student, subjects in student_subjects.items():
    for subject, grade in subjects.items():
        if subject not in subject_student_grades:
            subject_student_grades[subject] = {}
        subject_student_grades[subject][student] = grade

print(f"Subject inversion: {subject_student_grades}")
```

### Question 93: Dictionary Filtering

**Filter student records based on multiple criteria**

**Answer:**

```python
student_records = {
    "Alice Johnson": {"grade": 10, "gpa": 3.85, "attendance": 95},
    "Bob Smith": {"grade": 11, "gpa": 3.92, "attendance": 88},
    "Charlie Brown": {"grade": 10, "gpa": 3.78, "attendance": 92},
    "Diana Prince": {"grade": 9, "gpa": 3.95, "attendance": 97},
    "Eve Wilson": {"grade": 11, "gpa": 3.65, "attendance": 85}
}

# Filter by GPA
honor_roll = {name: record for name, record in student_records.items() if record["gpa"] >= 3.8}
print(f"Honor roll (GPA >= 3.8): {honor_roll}")

# Filter by multiple criteria
high_performers = {name: record for name, record in student_records.items()
                  if record["gpa"] >= 3.8 and record["attendance"] >= 90}
print(f"High performers: {high_performers}")

# Filter by grade level
grade_10_students = {name: record for name, record in student_records.items() if record["grade"] == 10}
print(f"Grade 10 students: {grade_10_students}")

# Get specific fields from filtered results
grade_11_names = [name for name, record in student_records.items() if record["grade"] == 11]
print(f"Grade 11 names: {grade_11_names}")

# Filter with complex conditions
needs_attention = {name: record for name, record in student_records.items()
                  if record["gpa"] < 3.7 or record["attendance"] < 90}
print(f"Students needing attention: {needs_attention}")
```

### Question 94: Dictionary Sorting

**Sort dictionaries by keys and values**

**Answer:**

```python
student_grades = {"Alice": 92, "Bob": 85, "Charlie": 78, "Diana": 96, "Eve": 88}

# Sort by student name (keys)
sorted_by_name = dict(sorted(student_grades.items()))
print(f"Sorted by name: {sorted_by_name}")

# Sort by grade (values)
sorted_by_grade = dict(sorted(student_grades.items(), key=lambda x: x[1]))
print(f"Sorted by grade (ascending): {sorted_by_grade}")

# Sort by grade descending
sorted_by_grade_desc = dict(sorted(student_grades.items(), key=lambda x: x[1], reverse=True))
print(f"Sorted by grade (descending): {sorted_by_grade_desc}")

# Sort by grade, then by name for ties
student_data = {"Alice": 85, "Bob": 92, "Charlie": 85, "Diana": 92}
sorted_complex = dict(sorted(student_data.items(), key=lambda x: (-x[1], x[0])))
print(f"Sorted by grade desc, then name: {sorted_complex}")

# Get top N students
top_3_students = dict(sorted(student_grades.items(), key=lambda x: x[1], reverse=True)[:3])
print(f"Top 3 students: {top_3_students}")

# Sort nested dictionary
student_records = {
    "Alice": {"grade": 10, "gpa": 3.85},
    "Bob": {"grade": 11, "gpa": 3.92},
    "Charlie": {"grade": 10, "gpa": 3.78}
}

sorted_by_gpa = dict(sorted(student_records.items(), key=lambda x: x[1]["gpa"], reverse=True))
print(f"Students sorted by GPA: {sorted_by_gpa}")
```

### Question 95: Dictionary Update Operations

**Update and modify dictionary values**

**Answer:**

```python
student_grades = {"Alice": 92, "Bob": 85, "Charlie": 78}

# Update single value
student_grades["Alice"] = 95
print(f"Alice's updated grade: {student_grades}")

# Update multiple values
student_grades.update({"Bob": 88, "Charlie": 82})
print(f"After batch update: {student_grades}")

# Update with conditional logic
def curved_grades(grades, curve_amount=5):
    """Add curve to grades, max 100"""
    updated = {}
    for name, grade in grades.items():
        new_grade = min(100, grade + curve_amount)
        updated[name] = new_grade
    return updated

curved = curved_grades(student_grades, 7)
print(f"After curving: {curved}")

# Update nested dictionaries
student_records = {
    "Alice": {"math": 92, "science": 88},
    "Bob": {"math": 85, "science": 91}
}

# Update specific nested value
student_records["Alice"]["math"] = 95
print(f"Alice's updated math grade: {student_records}")

# Update entire nested dictionary
student_records["Bob"].update({"math": 87, "science": 93})
print(f"Bob's updated record: {student_records}")

# Set default values for missing keys
for student in student_records:
    student_records[student].setdefault("art", 85)

print(f"Added default art grades: {student_records}")
```

### Question 96: Dictionary Counter

**Use dictionaries as counters for school data**

**Answer:**

```python
from collections import Counter

# Count student absences
student_absences = [
    "Alice", "Bob", "Charlie", "Alice", "Diana", "Bob", "Alice", "Eve"
]

absence_counter = Counter(student_absences)
print(f"Absence counts: {absence_counter}")

# Most absent students
most_absent = absence_counter.most_common(2)
print(f"Most absent students: {most_absent}")

# Students with no absences
all_students = {"Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"}
absent_students = set(absence_counter.keys())
perfect_attendance = all_students - absent_students
print(f"Perfect attendance: {perfect_attendance}")

# Manual counting without Counter
grade_distribution = {}
test_grades = ["A", "B", "A", "C", "A", "B", "A", "D"]

for grade in test_grades:
    grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

print(f"Grade distribution (manual): {grade_distribution}")

# Using Counter for the same
grade_counter = Counter(test_grades)
print(f"Grade distribution (Counter): {grade_counter}")

# Count subject preferences
subjects = ["Math", "Science", "Math", "English", "Science", "Math", "Art"]
subject_counts = Counter(subjects)
print(f"Subject preferences: {subject_counts}")
print(f"Most popular subject: {subject_counts.most_common(1)}")
```

### Question 97: Dictionary Memory Management

**Efficiently handle large student databases**

**Answer:**

```python
import sys

# Compare memory usage of different data structures
student_grades_dict = {
    f"Student_{i:04d}": 70 + (i % 31) for i in range(10000)
}

# Using tuple for (name, grade) pairs
grade_pairs = [(f"Student_{i:04d}", 70 + (i % 31)) for i in range(10000)]

dict_size = sys.getsizeof(student_grades_dict)
list_size = sys.getsizeof(grade_pairs)

print(f"Dictionary size: {dict_size} bytes")
print(f"List size: {list_size} bytes")

# Dictionary key optimization
# Instead of long keys, use IDs
student_ids = list(range(10000))
grade_lookup = {i: 70 + (i % 31) for i in student_ids}

id_dict_size = sys.getsizeof(grade_lookup)
print(f"Integer key dictionary: {id_dict_size} bytes")

# Use __slots__ for custom objects (if needed)
class Student:
    __slots__ = ['id', 'name', 'grade']
    def __init__(self, student_id, name, grade):
        self.id = student_id
        self.name = name
        self.grade = grade

# Test performance
import time

# Lookup performance test
def test_lookup_performance():
    # Dictionary lookup
    start_time = time.time()
    for _ in range(100000):
        _ = student_grades_dict.get("Student_0500")
    dict_time = time.time() - start_time

    # List lookup (linear search)
    start_time = time.time()
    for _ in range(100000):
        for name, grade in grade_pairs:
            if name == "Student_0500":
                break
    list_time = time.time() - start_time

    print(f"Dictionary lookup: {dict_time:.4f} seconds")
    print(f"List lookup: {list_time:.4f} seconds")
    print(f"Dictionary is {list_time/dict_time:.1f}x faster!")

test_lookup_performance()
```

### Question 98: Dictionary Serialization

**Convert dictionaries to/from JSON for data storage**

**Answer:**

```python
import json

# Student data as dictionary
student_database = {
    "school_year": "2023-2024",
    "students": {
        "STU001": {
            "name": "Alice Johnson",
            "grade": 10,
            "gpa": 3.85,
            "clubs": ["Chess Club", "Math Club"]
        },
        "STU002": {
            "name": "Bob Smith",
            "grade": 11,
            "gpa": 3.92,
            "clubs": ["Sports", "Drama Club"]
        }
    }
}

# Convert to JSON string
json_string = json.dumps(student_database, indent=2)
print("JSON string:")
print(json_string)

# Convert back to dictionary
parsed_data = json.loads(json_string)
print(f"Parsed student name: {parsed_data['students']['STU001']['name']}")

# Save to file
with open("student_data.json", "w") as file:
    json.dump(student_database, file, indent=2)

# Read from file
with open("student_data.json", "r") as file:
    loaded_data = json.load(file)
    print(f"Loaded data: {loaded_data}")

# Handle special data types
special_data = {
    "class_schedule": ("Math", "Science", "English"),  # Tuple becomes list
    "student_set": {"Alice", "Bob"},  # Set becomes list
    "graduation_date": None  # None remains None
}

json_special = json.dumps(special_data, default=str)
print(f"Special data JSON: {json_special}")
```

### Question 99: Dictionary Performance Optimization

**Optimize dictionary operations for school systems**

**Answer:**

```python
import time

# Large student database
student_db = {f"STU{i:04d}": {"name": f"Student_{i}", "grade": 9 + (i % 4), "gpa": 3.0 + (i % 100)/100}
              for i in range(10000)}

def benchmark_operations():
    # Benchmark different operations

    # 1. Key access
    start_time = time.time()
    for _ in range(10000):
        _ = student_db.get("STU0500")
    key_access_time = time.time() - start_time

    # 2. Value modification
    start_time = time.time()
    for _ in range(10000):
        student_db["STU0001"]["gpa"] = 4.0
    value_mod_time = time.time() - start_time

    # 3. New key insertion
    start_time = time.time()
    for i in range(1000):
        student_db[f"NEW{i:04d}"] = {"name": f"New_Student_{i}", "grade": 12, "gpa": 3.5}
    insertion_time = time.time() - start_time

    # 4. Dictionary iteration
    start_time = time.time()
    for student_id, data in student_db.items():
        _ = data["grade"]
    iteration_time = time.time() - start_time

    print(f"Key access: {key_access_time:.4f} seconds")
    print(f"Value modification: {value_mod_time:.4f} seconds")
    print(f"Key insertion: {insertion_time:.4f} seconds")
    print(f"Iteration: {iteration_time:.4f} seconds")

benchmark_operations()

# Optimize by caching frequently accessed data
def create_grade_cache(student_db):
    """Create a cache for grade lookups"""
    grade_cache = {}
    for student_id, data in student_db.items():
        grade = data["grade"]
        if grade not in grade_cache:
            grade_cache[grade] = []
        grade_cache[grade].append(student_id)
    return grade_cache

grade_index = create_grade_cache(student_db)
print(f"Students in grade 10: {len(grade_index.get(10, []))}")

# Use dictionary views for efficient operations
student_keys = student_db.keys()
student_values = student_db.values()
student_items = student_db.items()

# These are dynamic views, not copies
print(f"Number of students: {len(student_keys)}")
print(f"First student name: {next(iter(student_values))['name']}")
```

### Question 100: Dictionary Chain Operations

**Chain dictionary operations for complex school queries**

**Answer:**

```python
from collections import defaultdict, Counter

# Complex school data
school_data = {
    "students": {
        "STU001": {"name": "Alice", "grade": 10, "clubs": ["Chess", "Math"]},
        "STU002": {"name": "Bob", "grade": 11, "clubs": ["Sports"]},
        "STU003": {"name": "Charlie", "grade": 10, "clubs": ["Math", "Science"]},
        "STU004": {"name": "Diana", "grade": 9, "clubs": ["Art", "Drama"]},
    },
    "grades": {
        "STU001": {"math": 92, "science": 88, "english": 85},
        "STU002": {"math": 78, "science": 91, "english": 89},
        "STU003": {"math": 95, "science": 92, "english": 87},
        "STU004": {"math": 85, "science": 89, "english": 93},
    }
}

def find_advanced_students():
    """Find students eligible for advanced courses"""
    # Filter students by grade and club membership
    advanced_candidates = {}
    for student_id, info in school_data["students"].items():
        if info["grade"] >= 10 and "Math" in info["clubs"]:
            # Get their grades
            grades = school_data["grades"].get(student_id, {})
            # Check if they have high math grades
            if grades.get("math", 0) >= 90:
                advanced_candidates[student_id] = {
                    "name": info["name"],
                    "math_grade": grades["math"],
                    "all_grades": grades
                }
    return advanced_candidates

advanced_students = find_advanced_students()
print(f"Advanced math students: {advanced_students}")

def calculate_class_statistics():
    """Calculate comprehensive class statistics"""
    all_grades = []
    subject_totals = defaultdict(list)

    for student_id, grades in school_data["grades"].items():
        for subject, grade in grades.items():
            subject_totals[subject].append(grade)
            all_grades.append(grade)

    statistics = {
        "overall": {
            "average": sum(all_grades) / len(all_grades),
            "highest": max(all_grades),
            "lowest": min(all_grades)
        },
        "by_subject": {}
    }

    for subject, grades in subject_totals.items():
        statistics["by_subject"][subject] = {
            "average": sum(grades) / len(grades),
            "highest": max(grades),
            "lowest": min(grades),
            "count": len(grades)
        }

    return statistics

stats = calculate_class_statistics()
print(f"Class statistics: {stats}")

def find_student_conflicts():
    """Find students with conflicting schedules or grades"""
    grade_ranges = defaultdict(list)
    club_conflicts = defaultdict(list)

    for student_id, info in school_data["students"].items():
        grade = info["grade"]
        grade_ranges[grade].append(student_id)

        # Check for many clubs (potential conflict)
        if len(info["clubs"]) > 2:
            club_conflicts[student_id] = info["clubs"]

    return {
        "grade_distribution": dict(grade_ranges),
        "heavy_club_members": club_conflicts
    }

conflicts = find_student_conflicts()
print(f"Schedule analysis: {conflicts}")
```

---

## Strings (Questions 116-140)

### Question 116: String Creation and Access

**Work with student name strings and formatting**

**Answer:**

```python
# String creation
student_name = "Alice Johnson"
class_name = 'Lincoln High School'
school_motto = """Excellence in Education"""

print(f"Student: {student_name}")
print(f"School: {class_name}")
print(f"Motto: {school_motto}")

# String access and slicing
print(f"First character: {student_name[0]}")
print(f"Last character: {student_name[-1]}")
print(f"First name: {student_name[:5]}")
print(f"Last name: {student_name[6:]}")

# String length
print(f"Name length: {len(student_name)}")
print(f"School length: {len(class_name)}")

# String with special characters
student_id = "STU001"
email = "alice.johnson@school.edu"
print(f"Student ID: {student_id}")
print(f"Email: {email}")
```

### Question 117: String Formatting

**Format student grade reports and certificates**

**Answer:**

```python
student_name = "Alice Johnson"
student_grade = 92
student_gpa = 3.85
class_average = 87.5

# Method 1: f-strings (Python 3.6+)
grade_report = f"""
Grade Report
Student: {student_name}
Current Grade: {student_grade}%
GPA: {student_gpa:.2f}
Class Average: {class_average:.1f}%
Status: {"Honor Roll" if student_grade >= 90 else "Satisfactory"}
"""
print(grade_report)

# Method 2: format() method
certificate = "Certificate of Achievement\nStudent: {}\nGrade: {}\nDate: {}".format(
    student_name, student_grade, "2023-12-15"
)
print(certificate)

# Method 3: % formatting (older style)
old_style = "Student: %s, Grade: %d, GPA: %.2f" % (student_name, student_grade, student_gpa)
print(old_style)

# Advanced formatting
test_scores = [85, 92, 88, 95, 90]
formatted_scores = "Test Scores: " + ", ".join(f"{score}" for score in test_scores)
print(formatted_scores)

# Padding and alignment
roster = f"{'Name':<15} {'Grade':>5} {'Rank':>5}"
print(roster)
print("-" * 25)
for i, name in enumerate(["Alice", "Bob", "Charlie"], 1):
    print(f"{name:<15} {90+i:>5} {i:>5}")
```

### Question 118: String Methods - Case

**Convert student names to different cases**

**Answer:**

```python
student_name = "alice johnson"

# Case conversion methods
print(f"Original: {student_name}")
print(f"Capitalized: {student_name.capitalize()}")  # First letter uppercase
print(f"Title case: {student_name.title()}")        # Each word capitalized
print(f"Upper case: {student_name.upper()}")
print(f"Lower case: {student_name.lower()}")
print(f"Swap case: {student_name.swapcase()}")

# Real-world applications
email = "ALICE.JOHNSON@SCHOOL.EDU"
print(f"Email (lower): {email.lower()}")

student_names = ["alice johnson", "BOB SMITH", "Charlie Brown"]
print("Proper names:")
for name in student_names:
    print(f"  {name.title()}")

# Validation
def validate_student_name(name):
    """Validate if name is properly formatted"""
    return name == name.title()

names_to_check = ["Alice Johnson", "bob smith", "CHARLIE BROWN"]
for name in names_to_check:
    is_valid = validate_student_name(name)
    print(f"{name}: {'✓ Valid' if is_valid else '✗ Needs formatting'}")
```

### Question 119: String Methods - Search

**Search within student records and text**

**Answer:**

```python
student_bio = "Alice Johnson is a senior at Lincoln High School. She excels in mathematics and science, and is captain of the chess club."

# Basic search methods
print(f"Contains 'chess': {'chess' in student_bio.lower()}")
print(f"Contains 'math': {'math' in student_bio.lower()}")

# Find methods
print(f"First 'is' at position: {student_bio.find('is')}")
print(f"Last 'is' at position: {student_bio.rfind('is')}")

# Count occurrences
word_count = student_bio.lower().count('e')
print(f"Letter 'e' appears {word_count} times")

# Startswith and endswith
print(f"Starts with 'Alice': {student_bio.startswith('Alice')}")
print(f"Ends with period: {student_bio.endswith('.')}")

# Practical applications
def find_student_mentions(text, student_list):
    """Find all mentions of students in text"""
    mentions = []
    text_lower = text.lower()
    for student in student_list:
        if student.lower() in text_lower:
            mentions.append(student)
    return mentions

student_names = ["Alice", "Bob", "Charlie", "Diana"]
mentions = find_student_mentions(student_bio, student_names)
print(f"Students mentioned: {mentions}")

# Search with different cases
def case_insensitive_search(text, pattern):
    """Case-insensitive pattern matching"""
    return pattern.lower() in text.lower()

test_text = "Alice Johnson - Honor Student"
print(f"Contains 'alice': {case_insensitive_search(test_text, 'alice')}")
```

### Question 120: String Methods - Modification

**Modify student information strings**

**Answer:**

```python
student_info = "   Alice Johnson, Grade 10, Math Club   "

# Whitespace methods
print(f"Original: '{student_info}'")
print(f"Stripped: '{student_info.strip()}'")
print(f"Lstrip: '{student_info.lstrip()}'")
print(f"Rstrip: '{student_info.rstrip()}'")

# Replace methods
class_list = "Math, Science, English, History, Art"
print(f"Original: {class_list}")
print(f"Replace comma with dash: {class_list.replace(',', ' - ')}")

# Split and join
subjects = class_list.split(', ')
print(f"Split subjects: {subjects}")
reconstructed = ' + '.join(subjects)
print(f"Rejoined: {reconstructed}")

# Partition and split
grade_info = "Student: Alice Johnson, Grade: 10, GPA: 3.85"
before, separator, after = grade_info.partition(', ')
print(f"Before: {before}")
print(f"After: {after}")

# Multiple replacements
student_id = "STU001-Alice-Johnson-2023"
cleaned_id = student_id.replace('-', '_').lower()
print(f"Cleaned ID: {cleaned_id}")

# Translation table for complex replacements
grade_translation = str.maketrans('ABCD', 'VWXY')
encoded_grades = "ABCD".translate(grade_translation)
print(f"Encoded grades: {encoded_grades}")

# Real-world example: parsing student records
def parse_student_record(record):
    """Parse comma-separated student record"""
    fields = [field.strip() for field in record.split(',')]
    return {
        'name': fields[0],
        'grade': fields[1],
        'gpa': fields[2]
    }

record = "Alice Johnson, 10, 3.85"
parsed = parse_student_record(record)
print(f"Parsed: {parsed}")
```

### Question 121: String Validation

**Validate student input data**

**Answer:**

```python
def validate_student_email(email):
    """Validate student email format"""
    if '@' not in email:
        return False, "Missing @ symbol"
    if '.' not in email:
        return False, "Missing domain"
    if email.count('@') != 1:
        return False, "Invalid @ usage"
    local, domain = email.split('@')
    if len(local) == 0 or len(domain) < 4:
        return False, "Invalid local or domain"
    return True, "Valid email"

# Test email validation
test_emails = [
    "alice@school.edu",
    "bob.smith@highschool.org",
    "charlie@",
    "@domain.com",
    "invalid.email"
]

print("Email Validation:")
for email in test_emails:
    is_valid, message = validate_student_email(email)
    print(f"  {email}: {'✓' if is_valid else '✗'} {message}")

def validate_student_id(student_id):
    """Validate student ID format"""
    # Should be STU followed by 3 digits
    if len(student_id) != 6:
        return False, "ID must be 6 characters"
    if not student_id.startswith('STU'):
        return False, "Must start with 'STU'"
    try:
        number = int(student_id[3:])
        if number < 0:
            return False, "Number must be non-negative"
    except ValueError:
        return False, "Last 3 characters must be digits"
    return True, "Valid ID"

test_ids = ["STU001", "STU123", "STU99", "STU1A2", "STU999"]
print("\nStudent ID Validation:")
for student_id in test_ids:
    is_valid, message = validate_student_id(student_id)
    print(f"  {student_id}: {'✓' if is_valid else '✗'} {message}")

def validate_grade(grade):
    """Validate grade percentage"""
    try:
        grade_float = float(grade)
        if 0 <= grade_float <= 100:
            return True, f"Valid grade: {grade_float}%"
        else:
            return False, "Grade must be between 0 and 100"
    except ValueError:
        return False, "Grade must be a number"

test_grades = ["95", "87.5", "105", "-5", "A", "85.2"]
print("\nGrade Validation:")
for grade in test_grades:
    is_valid, message = validate_grade(grade)
    print(f"  {grade}: {'✓' if is_valid else '✗'} {message}")
```

### Question 122: String Encryption for Student Data

**Simple encryption for sensitive student information**

**Answer:**

```python
import string

# Simple Caesar cipher for student IDs
def caesar_cipher(text, shift):
    """Encrypt text using Caesar cipher"""
    alphabet = string.ascii_uppercase + string.ascii_lowercase
    shifted = alphabet[shift:] + alphabet[:shift]
    table = str.maketrans(alphabet, shifted)
    return text.translate(table)

def caesar_decipher(encrypted_text, shift):
    """Decrypt Caesar cipher text"""
    alphabet = string.ascii_uppercase + string.ascii_lowercase
    shifted = alphabet[-shift:] + alphabet[:-shift]
    table = str.maketrans(alphabet, shifted)
    return encrypted_text.translate(table)

# Encrypt student IDs
student_ids = ["STU001", "STU002", "STU003"]
shift = 3

print("Original IDs:", student_ids)
encrypted_ids = [caesar_cipher(student_id, shift) for student_id in student_ids]
print("Encrypted IDs:", encrypted_ids)

decrypted_ids = [caesar_decipher(encrypted_id, shift) for encrypted_id in encrypted_ids]
print("Decrypted IDs:", decrypted_ids)

# XOR cipher for passwords
def xor_cipher(text, key):
    """Simple XOR cipher"""
    result = []
    for i, char in enumerate(text):
        key_char = key[i % len(key)]
        result.append(chr(ord(char) ^ ord(key_char)))
    return ''.join(result)

def xor_decipher(encrypted_text, key):
    """Decrypt XOR cipher"""
    return xor_cipher(encrypted_text, key)  # XOR is symmetric

# Encrypt passwords
passwords = ["password123", "secure456", "student789"]
key = "SCHOOL"

print("\nOriginal passwords:", passwords)
encrypted_passwords = [xor_cipher(password, key) for password in passwords]
print("Encrypted passwords:", encrypted_passwords)

decrypted_passwords = [xor_decipher(encrypted_pw, key) for encrypted_pw in encrypted_passwords]
print("Decrypted passwords:", decrypted_passwords)

# Base64 encoding (simpler, for non-sensitive data)
import base64

def encode_data(data):
    """Encode data using Base64"""
    return base64.b64encode(data.encode()).decode()

def decode_data(encoded_data):
    """Decode Base64 data"""
    return base64.b64decode(encoded_data.encode()).decode()

student_notes = "Alice excels in mathematics"
encoded_notes = encode_data(student_notes)
decoded_notes = decode_data(encoded_notes)

print(f"\nOriginal: {student_notes}")
print(f"Base64 encoded: {encoded_notes}")
print(f"Decoded back: {decoded_notes}")
```

### Question 123: String Pattern Matching

**Find patterns in student data**

**Answer:**

```python
import re

# Student bio data
student_bios = [
    "Alice Johnson - Senior, Math major, Chess Club captain",
    "Bob Smith - Junior, Science major, Soccer team",
    "Charlie Brown - Sophomore, Arts major, Drama club president",
    "Diana Prince - Senior, Engineering track, Robotics team leader",
    "Eve Wilson - Freshman, Undecided major, Swimming team"
]

# Pattern matching examples
print("=== Pattern Matching Examples ===")

# Find all student names (capitalized words before dash)
for bio in student_bios:
    name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', bio)
    if name_match:
        print(f"Name: {name_match.group(1)}")

print("\n=== Grade Level Detection ===")
# Find grade levels
grade_pattern = r'(Senior|Junior|Sophomore|Freshman)'

for bio in student_bios:
    grade_match = re.search(grade_pattern, bio)
    if grade_match:
        print(f"{bio[:30]}... - {grade_match.group(1)}")

print("\n=== Major Detection ===")
# Find majors (word before "major")
major_pattern = r'(\w+) major'

for bio in student_bios:
    major_match = re.search(major_pattern, bio)
    if major_match:
        print(f"Major: {major_match.group(1)}")

print("\n=== Activity Extraction ===")
# Extract activities
activity_pattern = r'([A-Z][a-zA-Z ]+(?:captain|president|team leader|team))'

for bio in student_bios:
    activities = re.findall(activity_pattern, bio)
    if activities:
        print(f"Activities: {', '.join(activities)}")

# Advanced pattern matching
def extract_student_info(bio):
    """Extract structured information from student bio"""
    info = {}

    # Extract name
    name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', bio)
    info['name'] = name_match.group(1) if name_match else "Unknown"

    # Extract grade level
    grade_match = re.search(r'(Senior|Junior|Sophomore|Freshman)', bio)
    info['grade'] = grade_match.group(1) if grade_match else "Unknown"

    # Extract major
    major_match = re.search(r'(\w+) major', bio)
    info['major'] = major_match.group(1) if major_match else "Undecided"

    # Extract activities
    activity_pattern = r'([A-Z][a-zA-Z ]+(?:captain|president|team leader|team))'
    info['activities'] = re.findall(activity_pattern, bio)

    return info

print("\n=== Structured Extraction ===")
for bio in student_bios:
    info = extract_student_info(bio)
    print(f"{info['name']}: Grade {info['grade']}, {info['major']} major")
```

### Question 124: String Comparison

**Compare student names and codes**

**Answer:**

```python
# String comparison examples
print("=== String Comparison Examples ===")

# Basic string comparison
name1 = "Alice Johnson"
name2 = "alice johnson"
name3 = "Bob Smith"

print(f"name1 == name2: {name1 == name2}")        # False (case sensitive)
print(f"name1.lower() == name2: {name1.lower() == name2}")  # True

# Lexicographic comparison
print(f"'A' < 'B': {'A' < 'B'}")  # True
print(f"'Alice' < 'Bob': {'Alice' < 'Bob'}")  # True
print(f"'Anne' < 'Bob': {'Anne' < 'Bob'}")  # True

# Student grade comparison
grades = ["A", "B+", "A-", "B", "A+"]
print(f"\nOriginal grades: {grades}")

# Sort grades (lexicographic)
sorted_grades = sorted(grades)
print(f"Sorted grades (lexicographic): {sorted_grades}")

# Custom grade sorting
grade_order = {"A+": 4, "A": 3.7, "A-": 3.3, "B+": 3.0, "B": 2.7, "B-": 2.3, "C+": 2.0, "C": 1.7}

def grade_sort_key(grade):
    return grade_order.get(grade, 0)

sorted_by_value = sorted(grades, key=grade_sort_key, reverse=True)
print(f"Sorted by value: {sorted_by_value}")

# Case-insensitive comparison for sorting
student_names = ["alice", "Bob", "charlie", "Diana"]
sorted_names = sorted(student_names, key=str.lower)
print(f"Names sorted case-insensitive: {sorted_names}")

# String similarity comparison
def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

# Test similarity
names_to_compare = ["Alice Johnson", "Alice Johnsn", "Alice Jhonson", "Bob Smith"]
reference = "Alice Johnson"

print(f"\n=== Name Similarity Comparison ===")
print(f"Reference: {reference}")
for name in names_to_compare:
    distance = levenshtein_distance(reference.lower(), name.lower())
    similarity = 1 - (distance / max(len(reference), len(name)))
    print(f"{name}: Distance={distance}, Similarity={similarity:.2%}")
```

### Question 125: String Memory Management

**Optimize string operations for large student databases**

**Answer:**

```python
import sys
import time

# String memory comparison
student_name = "Alice Johnson"
student_name_multiline = """Alice
Johnson"""

# Memory usage
name_size = sys.getsizeof(student_name)
multiline_size = sys.getsizeof(student_name_multiline)

print(f"Single line name size: {name_size} bytes")
print(f"Multiline name size: {multiline_size} bytes")

# String interning (Python automatically interns some strings)
a = "hello"
b = "hello"
c = "".join(['h', 'e', 'l', 'l', 'o'])

print(f"\na is b: {a is b}")  # True (interned)
print(f"a is c: {a is c}")   # False (different objects)

# String concatenation efficiency
def test_string_concat():
    """Test different string concatenation methods"""

    # Method 1: Using + operator
    start_time = time.time()
    result = ""
    for i in range(10000):
        result += f"Student_{i}, "
    concat_time = time.time() - start_time

    # Method 2: Using join()
    start_time = time.time()
    students = [f"Student_{i}" for i in range(10000)]
    result = ", ".join(students)
    join_time = time.time() - start_time

    print(f"Concatenation with +: {concat_time:.4f} seconds")
    print(f"Concatenation with join(): {join_time:.4f} seconds")
    print(f"join() is {concat_time/join_time:.1f}x faster!")

test_string_concat()

# String builder pattern
class StringBuilder:
    """Efficient string builder"""
    def __init__(self):
        self.parts = []

    def append(self, text):
        self.parts.append(str(text))

    def to_string(self):
        return "".join(self.parts)

def build_student_list():
    """Build student list using StringBuilder"""
    builder = StringBuilder()
    builder.append("Student List:\n")

    students = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    for i, student in enumerate(students, 1):
        builder.append(f"{i}. {student}\n")

    return builder.to_string()

student_list = build_student_list()
print("\nGenerated student list:")
print(student_list)

# String pool optimization
def optimize_student_records():
    """Optimize memory for student records"""
    # Instead of storing full names repeatedly, use interning
    names = ["Alice Johnson"] * 1000

    # Check if all names point to same object
    unique_objects = len(set(id(name) for name in names))
    total_names = len(names)

    print(f"Total names: {total_names}")
    print(f"Unique objects: {unique_objects}")

    if unique_objects == 1:
        print("✓ Strings are interned (memory efficient)")
    else:
        print("✗ Strings are not interned")

optimize_student_records()
```

### Question 126: String Internationalization

**Handle student names from different cultures**

**Answer:**

```python
# International student names
international_names = [
    "Alice Johnson",
    "李明 (Li Ming)",
    "José García",
    "François Dubois",
    "Anna Müller",
    "Иван Петров (Ivan Petrov)",
    "العلي محمد (Al Ali Mohammed)",
    "Σοφία Παπαδοπούλου (Sophia Papadopoulos)"
]

print("=== International Student Names ===")
for name in international_names:
    print(f"Name: {name}")

# Character analysis
print("\n=== Character Analysis ===")
for name in international_names:
    char_count = len(name)
    print(f"{name}: {char_count} characters")

# Unicode handling
def normalize_unicode(text):
    """Normalize unicode strings"""
    import unicodedata

    # Normalize to NFKD form (compatibility decomposition)
    normalized = unicodedata.normalize('NFKD', text)

    # Try to encode to ASCII (ignoring non-ASCII characters)
    try:
        ascii_version = normalized.encode('ascii', 'ignore').decode('ascii')
        return ascii_version
    except:
        return normalized

print("\n=== Unicode Normalization ===")
for name in international_names:
    normalized = normalize_unicode(name)
    print(f"Original: {name}")
    print(f"Normalized: {normalized}")
    print()

# Case folding for international comparison
def case_fold_comparison(name1, name2):
    """Compare names using case folding"""
    folded1 = name1.casefold()
    folded2 = name2.casefold()
    return folded1 == folded2

test_names = [
    ("José García", "JOSÉ GARCÍA"),
    ("François Dubois", "francois dubois"),
    ("Anna Müller", "anna mueller")
]

print("=== Case-insensitive Comparison ===")
for name1, name2 in test_names:
    result = case_fold_comparison(name1, name2)
    print(f"{name1} == {name2}: {result}")

# String encoding/decoding
def handle_encoding(name):
    """Handle different string encodings"""
    # Encode to bytes
    utf8_bytes = name.encode('utf-8')

    # Decode back
    decoded = utf8_bytes.decode('utf-8')

    return {
        'original': name,
        'utf8_bytes': utf8_bytes,
        'decoded': decoded,
        'byte_length': len(utf8_bytes)
    }

print("\n=== Encoding Analysis ===")
for name in international_names[:3]:  # Test first 3 names
    encoding_info = handle_encoding(name)
    print(f"Original: {encoding_info['original']}")
    print(f"UTF-8 bytes: {encoding_info['utf8_bytes']}")
    print(f"Byte length: {encoding_info['byte_length']}")
    print(f"Match: {encoding_info['original'] == encoding_info['decoded']}")
    print()
```

### Question 127: String Performance Analysis

**Analyze performance of string operations on large datasets**

**Answer:**

```python
import time
import random
import string

def generate_random_student_data(count=10000):
    """Generate random student data for performance testing"""
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    last_names = ["Johnson", "Smith", "Brown", "Wilson", "Davis", "Miller", "Garcia", "Rodriguez"]

    students = []
    for i in range(count):
        first = random.choice(first_names)
        last = random.choice(last_names)
        grade = random.randint(9, 12)
        students.append(f"{first} {last}, Grade {grade}")

    return students

def benchmark_string_operations():
    """Benchmark various string operations"""
    student_data = generate_random_student_data(5000)

    # Test 1: String searching
    search_term = "Alice"

    start_time = time.time()
    alice_students = [student for student in student_data if search_term in student]
    search_time = time.time() - start_time

    # Test 2: String splitting
    start_time = time.time()
    split_data = []
    for student in student_data:
        parts = student.split(', ')
        if len(parts) == 2:
            name, grade_info = parts
            grade = grade_info.split(' ')[1]
            split_data.append((name, grade))
    split_time = time.time() - start_time

    # Test 3: String formatting
    start_time = time.time()
    formatted_data = []
    for student in student_data[:1000]:  # Smaller dataset for formatting
        formatted_data.append(f"STUDENT: {student}")
    format_time = time.time() - start_time

    # Test 4: String replacement
    start_time = time.time()
    modified_data = [student.replace('Grade', 'Yr') for student in student_data]
    replace_time = time.time() - start_time

    print(f"=== String Operation Performance (5000 students) ===")
    print(f"Search for '{search_term}': {search_time:.4f}s ({len(alice_students)} matches)")
    print(f"Split operation: {split_time:.4f}s")
    print(f"String formatting (1000 items): {format_time:.4f}s")
    print(f"String replacement: {replace_time:.4f}s")

benchmark_string_operations()

def test_string_vs_list_operations():
    """Compare string operations vs list operations"""
    student_names = ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", "Eve Wilson"]

    # String approach
    student_string = ", ".join(student_names)

    # List approach
    student_list = student_names.copy()

    # Test membership
    search_name = "Bob Smith"

    # String membership test
    start_time = time.time()
    for _ in range(10000):
        result = search_name in student_string
    string_time = time.time() - start_time

    # List membership test
    start_time = time.time()
    for _ in range(10000):
        result = search_name in student_list
    list_time = time.time() - start_time

    print(f"\n=== String vs List Membership Test ===")
    print(f"String membership (10000 tests): {string_time:.4f}s")
    print(f"List membership (10000 tests): {list_time:.4f}s")
    print(f"List is {string_time/list_time:.1f}x faster")

test_string_vs_list_operations()

def memory_efficient_string_processing():
    """Process strings efficiently for large datasets"""
    # Generate large dataset
    large_student_list = generate_random_student_data(10000)

    # Method 1: Create all formatted strings at once (memory intensive)
    start_time = time.time()
    formatted_strings = [f"Student: {name}" for name in large_student_list]
    all_at_once_time = time.time() - start_time
    memory_usage_all = sys.getsizeof(formatted_strings)

    # Method 2: Generator approach (memory efficient)
    start_time = time.time()
    formatted_generator = (f"Student: {name}" for name in large_student_list)
    generator_time = time.time() - start_time

    print(f"\n=== Memory-Efficient Processing ===")
    print(f"Create all strings at once: {all_at_once_time:.4f}s")
    print(f"Generator approach: {generator_time:.4f}s")
    print(f"Generator memory usage: {sys.getsizeof(formatted_generator)} bytes")

    # Test generator usage
    first_5 = list(formatted_generator)[:5]
    print(f"First 5 from generator: {first_5}")

memory_efficient_string_processing()
```

### Question 128: String Template System

**Create templates for student reports and certificates**

**Answer:**

```python
from string import Template

# Student certificate template
certificate_template = Template("""
CERTIFICATE OF ACHIEVEMENT

This certifies that

$student_name

has successfully completed

$course_name

with a grade of $final_grade

Date: $completion_date
Instructor: $instructor_name
""")

# Student report template
report_template = Template("""
STUDENT REPORT CARD
====================
Student: $student_name (ID: $student_id)
Grade Level: $grade_level
Semester: $semester
Academic Year: $academic_year

COURSE GRADES:
$course_grades

GPA: $gpa
Attendance: $attendance%
Rank: $class_rank

Comments: $comments
""")

# Generate certificates
def generate_certificate(student_name, course_name, final_grade, completion_date, instructor_name):
    """Generate a student certificate"""
    certificate = certificate_template.substitute(
        student_name=student_name,
        course_name=course_name,
        final_grade=final_grade,
        completion_date=completion_date,
        instructor_name=instructor_name
    )
    return certificate

# Generate report card
def generate_report_card(student_name, student_id, grade_level, semester, academic_year,
                        course_grades, gpa, attendance, class_rank, comments):
    """Generate a student report card"""
    # Format course grades
    grades_text = ""
    for course, grade in course_grades.items():
        grades_text += f"  {course}: {grade}\n"

    report = report_template.substitute(
        student_name=student_name,
        student_id=student_id,
        grade_level=grade_level,
        semester=semester,
        academic_year=academic_year,
        course_grades=grades_text,
        gpa=gpa,
        attendance=attendance,
        class_rank=class_rank,
        comments=comments
    )
    return report

# Test certificate generation
print("=== STUDENT CERTIFICATE ===")
certificate = generate_certificate(
    student_name="Alice Johnson",
    course_name="Advanced Mathematics",
    final_grade="A+ (97%)",
    completion_date="December 15, 2023",
    instructor_name="Dr. Sarah Davis"
)
print(certificate)

# Test report card generation
print("\n=== STUDENT REPORT CARD ===")
report = generate_report_card(
    student_name="Alice Johnson",
    student_id="STU001",
    grade_level="12",
    semester="Fall 2023",
    academic_year="2023-2024",
    course_grades={
        "Mathematics": "A+",
        "Science": "A",
        "English Literature": "A-",
        "History": "B+",
        "Art": "A"
    },
    gpa="3.85",
    attendance="98%",
    class_rank="15",
    comments="Excellent performance in advanced coursework. Shows strong leadership in group projects."
)
print(report)

# Safe substitution (won't throw error for missing variables)
def safe_certificate_template():
    """Template with safe substitution"""
    template = Template("Student: $student_name, Grade: $grade, School: $school")

    # Dictionary with some missing keys
    data = {"student_name": "Bob Smith", "grade": "A"}

    # Safe substitute (use safe_substitute instead of substitute)
    result = template.safe_substitute(data)
    return result

print("\n=== Safe Substitution ===")
safe_result = safe_certificate_template()
print(f"Safe substitution: {safe_result}")
```

### Question 129: String Data Structure Integration

**Integrate string operations with other data structures**

**Answer:**

```python
# String operations with lists
student_names = ["Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince"]

# Join names for display
roster_display = " | ".join(student_names)
print(f"Class roster: {roster_display}")

# Split back to individual names
names_from_string = roster_display.split(" | ")
print(f"Split back: {names_from_string}")

# String operations with dictionaries
grade_book = {
    "Alice Johnson": 92,
    "Bob Smith": 85,
    "Charlie Brown": 78,
    "Diana Prince": 96
}

# Create formatted grade report
def create_grade_report(grade_dict):
    """Create formatted grade report from dictionary"""
    report_lines = []
    report_lines.append("GRADE REPORT")
    report_lines.append("=" * 20)

    for name, grade in grade_dict.items():
        report_lines.append(f"{name:<20} {grade:>3}")

    return "\n".join(report_lines)

grade_report = create_grade_report(grade_book)
print(f"\n{grade_report}")

# String operations with sets
unique_first_names = set()
for name in student_names:
    first_name = name.split()[0]
    unique_first_names.add(first_name)

print(f"\nUnique first names: {', '.join(sorted(unique_first_names))}")

# Nested string operations
student_data_complex = {
    "students": [
        {"name": "Alice Johnson", "email": "alice@school.edu", "id": "STU001"},
        {"name": "Bob Smith", "email": "bob@school.edu", "id": "STU002"},
        {"name": "Charlie Brown", "email": "charlie@school.edu", "id": "STU003"}
    ]
}

# Extract and format student information
def extract_student_info(student_data):
    """Extract and format student information"""
    email_list = []
    id_list = []

    for student in student_data["students"]:
        email_list.append(student["email"])
        id_list.append(student["id"])

    return {
        "emails": ", ".join(email_list),
        "student_ids": " ".join(id_list),
        "name_count": len(student_data["students"])
    }

extracted_info = extract_student_info(student_data_complex)
print(f"\nExtracted info: {extracted_info}")

# String-based sorting and filtering
def filter_and_sort_students(grade_dict, min_grade=80):
    """Filter students by grade and return sorted list"""
    # Filter students with sufficient grades
    qualified_students = {name: grade for name, grade in grade_dict.items() if grade >= min_grade}

    # Sort by grade (descending), then by name
    sorted_students = sorted(qualified_students.items(), key=lambda x: (-x[1], x[0]))

    return sorted_students

qualified = filter_and_sort_students(grade_book, 85)
print(f"\nStudents with grade >= 85 (sorted):")
for name, grade in qualified:
    print(f"  {name}: {grade}")

# Multi-level string operations
class_schedule = {
    "Monday": ["Math", "Science", "English", "History"],
    "Tuesday": ["Science", "Math", "PE", "Art"],
    "Wednesday": ["English", "History", "Math", "Science"],
    "Thursday": ["Art", "PE", "Science", "Math"],
    "Friday": ["History", "English", "Math", "Science"]
}

def format_schedule(schedule_dict):
    """Format weekly schedule"""
    schedule_lines = []
    schedule_lines.append("WEEKLY CLASS SCHEDULE")
    schedule_lines.append("=" * 25)

    for day, classes in schedule_dict.items():
        class_list = " -> ".join(classes)
        schedule_lines.append(f"{day:<10}: {class_list}")

    return "\n".join(schedule_lines)

formatted_schedule = format_schedule(class_schedule)
print(f"\n{formatted_schedule}")
```

### Question 130: String Challenge - Student Profile Parser

**Create a comprehensive string parser for student profiles**

**Answer:**

```python
import re

class StudentProfileParser:
    """Parse complex student profile strings"""

    def __init__(self):
        # Patterns for parsing
        self.name_pattern = r'Name:\s*([A-Za-z\s]+)'
        self.id_pattern = r'ID:\s*([A-Z0-9]+)'
        self.grade_pattern = r'Grade:\s*(\d+)'
        self.gpa_pattern = r'GPA:\s*([0-9.]+)'
        self.email_pattern = r'Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        self.phone_pattern = r'Phone:\s*([0-9\-\(\)\s]+)'
        self.club_pattern = r'Club[s]?:\s*([A-Za-z,\s]+)'
        self.address_pattern = r'Address:\s*([0-9\sA-Za-z,\.]+)'

    def parse_profile(self, profile_string):
        """Parse complete student profile"""
        profile = {}

        # Parse each field
        name_match = re.search(self.name_pattern, profile_string, re.IGNORECASE)
        profile['name'] = name_match.group(1).strip() if name_match else None

        id_match = re.search(self.id_pattern, profile_string, re.IGNORECASE)
        profile['id'] = id_match.group(1).strip() if id_match else None

        grade_match = re.search(self.grade_pattern, profile_string)
        profile['grade'] = int(grade_match.group(1)) if grade_match else None

        gpa_match = re.search(self.gpa_pattern, profile_string)
        profile['gpa'] = float(gpa_match.group(1)) if gpa_match else None

        email_match = re.search(self.email_pattern, profile_string, re.IGNORECASE)
        profile['email'] = email_match.group(1) if email_match else None

        phone_match = re.search(self.phone_pattern, profile_string)
        profile['phone'] = phone_match.group(1).strip() if phone_match else None

        club_match = re.search(self.club_pattern, profile_string, re.IGNORECASE)
        if club_match:
            clubs_text = club_match.group(1).strip()
            profile['clubs'] = [club.strip() for club in clubs_text.split(',')]
        else:
            profile['clubs'] = []

        address_match = re.search(self.address_pattern, profile_string, re.IGNORECASE)
        profile['address'] = address_match.group(1).strip() if address_match else None

        return profile

    def validate_profile(self, profile):
        """Validate parsed profile data"""
        validation_results = {}

        # Name validation
        if profile['name']:
            name_parts = profile['name'].split()
            validation_results['name'] = len(name_parts) >= 2
        else:
            validation_results['name'] = False

        # ID validation
        if profile['id']:
            validation_results['id'] = len(profile['id']) >= 3 and profile['id'].startswith(('STU', 'ID'))
        else:
            validation_results['id'] = False

        # Grade validation
        validation_results['grade'] = profile['grade'] is not None and 1 <= profile['grade'] <= 12

        # GPA validation
        validation_results['gpa'] = profile['gpa'] is not None and 0.0 <= profile['gpa'] <= 4.0

        # Email validation
        if profile['email']:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            validation_results['email'] = bool(re.match(email_pattern, profile['email']))
        else:
            validation_results['email'] = False

        return validation_results

    def format_profile_report(self, profile, validation):
        """Format profile for display"""
        report_lines = []
        report_lines.append("STUDENT PROFILE REPORT")
        report_lines.append("=" * 25)

        # Basic info
        report_lines.append(f"Name: {profile['name'] or 'N/A'}")
        report_lines.append(f"ID: {profile['id'] or 'N/A'}")
        report_lines.append(f"Grade: {profile['grade'] or 'N/A'}")
        report_lines.append(f"GPA: {profile['gpa'] or 'N/A'}")
        report_lines.append("")

        # Contact info
        report_lines.append("CONTACT INFORMATION:")
        report_lines.append(f"  Email: {profile['email'] or 'N/A'}")
        report_lines.append(f"  Phone: {profile['phone'] or 'N/A'}")
        report_lines.append(f"  Address: {profile['address'] or 'N/A'}")
        report_lines.append("")

        # Clubs
        report_lines.append("CLUB MEMBERSHIP:")
        if profile['clubs']:
            for club in profile['clubs']:
                report_lines.append(f"  - {club}")
        else:
            report_lines.append("  None listed")
        report_lines.append("")

        # Validation status
        report_lines.append("VALIDATION STATUS:")
        for field, is_valid in validation.items():
            status = "✓ PASS" if is_valid else "✗ FAIL"
            report_lines.append(f"  {field.upper()}: {status}")

        return "\n".join(report_lines)

# Test the parser
parser = StudentProfileParser()

sample_profiles = [
    """
    Student Profile:
    Name: Alice Johnson
    ID: STU001
    Grade: 11
    GPA: 3.85
    Email: alice.johnson@school.edu
    Phone: (555) 123-4567
    Address: 123 Main St, Springfield, IL 62701
    Clubs: Chess Club, Math Team, Drama Club
    """,

    """
    Profile Data:
    Name: Bob Smith
    ID: STU002
    Grade: 10
    GPA: 3.62
    Email: bob@school.edu
    Phone: 555-987-6543
    Address: 456 Oak Ave, Springfield, IL 62702
    Clubs: Soccer, Science Club
    """,

    """
    Student Record:
    Name: Charlie Brown
    ID: ID003
    Grade: 12
    GPA: 3.95
    Email: charlie.brown@school.edu
    Clubs: Art Club, Photography, Yearbook
    """
]

print("=== STUDENT PROFILE PARSER ===")
for i, profile_text in enumerate(sample_profiles, 1):
    print(f"\n--- PROFILE {i} ---")
    parsed_profile = parser.parse_profile(profile_text)
    validation_results = parser.validate_profile(parsed_profile)
    formatted_report = parser.format_profile_report(parsed_profile, validation_results)
    print(formatted_report)
```

---

## 🎯 Quick Summary

### What You've Mastered So Far:

**Lists:**
✅ Creation, indexing, slicing, methods  
✅ List comprehensions and advanced operations  
✅ Performance considerations and school use cases

**Tuples:**
✅ Immutability and when to use for fixed data  
✅ Unpacking and named tuples for student records  
✅ Memory efficiency and performance

**Sets:**
✅ Set operations (union, intersection, difference)  
✅ Membership testing and performance  
✅ Comprehensions and practical school applications

**Dictionaries:**
✅ Key-value operations and methods  
✅ Nested structures and comprehension  
✅ Real-world school data management

**Strings:**
✅ Creation, formatting, and manipulation  
✅ Validation and pattern matching  
✅ Performance optimization and internationalization

### Next Steps:

- Learn data structure combinations (Questions 141-160)
- Tackle advanced applications (Questions 161-180)
- Challenge problems (Questions 181-200)

_Keep coding and experimenting with these data structures!_ 🚀
