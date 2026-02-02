# 📋 Python Fundamentals Quick Reference

_Student's Backpack Companion - Quick & Easy!_ 🎒

> **TL;DR**: This guide covers everything you need for Python basics in school projects. Keep it handy!

## 🚀 What Can You Build?

### Python Projects Perfect for School:

- **📊 Grade Calculators**: Automatically calculate GPA and final grades
- **📝 Quiz Generators**: Create random math or spelling quizzes
- **📚 Study Tools**: Track reading time and homework progress
- **🏆 Honor Roll Checker**: Multiple condition testing
- **⏰ Schedule Manager**: Plan study time and deadlines
- **📈 Grade Tracker**: Monitor progress in all subjects

## 💻 Installation & Setup

### Check If Python Is Installed

```bash
python --version    # Check Python version (should show 3.x)
python -V          # Alternative version check
python             # Open interactive mode (type 'exit()' to quit)
```

### Python Version Breakdown: 3.12.2

- **3**: Major version (big changes between versions)
- **12**: Minor version (new features added)
- **2**: Patch version (bug fixes)

### Your Python Files

- **File Extension**: .py (every Python file ends with this)
- **Example**: grade_calculator.py, homework_tracker.py, quiz_game.py
- **Run**: python filename.py (replaces filename with actual name)

---

## Basic Syntax Rules

### 🔥 Indentation (MOST IMPORTANT!)

```python
✅ CORRECT - Students get extra credit
if grade >= 90:
    print("A grade!")
    print("Great job!")

❌ WRONG - Program crashes!
if grade >= 90:
print("A grade!")  # ERROR! Missing spaces
```

### Comments (Write Notes to Yourself!)

```python
# This line explains what you did
# Single line - Use # at the start

"""
Multi-line notes about your code
Use this for longer explanations
"""
```

### Case Sensitivity (Very Important!)

```python
student_name = "Alice"    # Different variable!
Student_Name = "Bob"      # Different variable!
STUDENT_NAME = "Charlie"  # Different variable!

# Python treats these as 3 completely different variables!
```

---

## Variable Naming Rules

### ✅ GOOD Names (Clear & Descriptive)

```python
student_name = "Alice"        # Easy to understand
student_age = 16              # What does this store?
final_grade = 95.5            # Final course grade
homework_score = 88           # Homework grade
attendance_days = 180         # Days present
is_homework_done = True       # True/False answer
```

### ❌ BAD Names (Avoid These!)

```python
2nd_place = 2                 # ❌ Can't start with number
student-name = "Bob"          # ❌ Hyphens not allowed
class = "Math"                # ❌ "class" is a Python word
x = "Alice"                   # ❌ Not descriptive
myvar123 = 95                 # ❌ What does this mean?
```

---

## Data Types Quick Reference

### 📊 Numbers (Math & Grades)

```python
student_age = 16              # Integer (whole number)
gpa = 3.75                    # Float (decimal number)
temperature = -5              # Can be negative
homework_score = 100          # Whole number scores
```

### 📝 Strings (Text & Names)

```python
student_name = "Alice Johnson"     # Names with quotes
subject = 'Mathematics'             # Single or double quotes OK
teacher_note = """Great work
on your project!"""                 # Multi-line text
```

### ✅❌ Booleans (True/False Answers)

```python
is_homework_done = True            # Student completed homework
passed_exam = False                # Student needs to retake
is_honor_roll = True               # Honor roll student?
attended_class = True              # Was present today?
```

### 🚫 None (Empty/Nothing)

```python
# When you have no value yet
student_grade = None              # Grade not assigned yet
homework_file = None              # File not uploaded yet
```

---

## Operators Quick Reference

### ➕➖ Arithmetic Operators (Math Operations)

```python
homework_score + quiz_score     # Addition
exam_grade - bonus_points        # Subtraction
homework_count * points_per_hw   # Multiplication
total_points / num_assignments   # Division
total_points // homework_count   # Whole number division
student_grade % 10               # Remainder (modulus)
2 ** 3                           # Exponent (2^3 = 8)
```

### 📏 Comparison Operators (True/False Questions)

```python
grade == 95          # Equal to? (Is grade exactly 95?)
grade != 90          # Not equal to? (Is grade NOT 90?)
grade < 80           # Less than? (Grade under 80?)
grade > 90           # Greater than? (Grade over 90?)
grade <= 100         # Less than or equal to?
grade >= 60          # Greater than or equal to? (Passing grade?)
```

### 🧠 Logical Operators (Combining Conditions)

```python
(is_homework_done and is_quiz_taken)    # Both must be True
(passed_exam or extra_credit)           # At least one True
not is_absent_today                     # Opposite of the condition
```

### 📝 Assignment Operators (Change & Update)

```python
grade = 85           # Set grade to 85
grade += 5           # Same as: grade = grade + 5 (now 90)
grade -= 10          # Same as: grade = grade - 10 (now 80)
grade *= 2           # Same as: grade = grade * 2 (now 170)
grade /= 2           # Same as: grade = grade / 2 (now 85)
```

---

## String Methods Cheat Sheet

### 🔤 Common String Operations (Working with Text)

```python
student_name = "  alice Johnson  "

student_name.strip()         # Remove extra spaces: "alice Johnson"
student_name.upper()         # ALL CAPS: "  ALICE JOHNSON  "
student_name.lower()         # all lowercase: "  alice johnson  "
student_name.replace("Alice", "Bob")  # Replace name: "  bob Johnson  "
len(student_name)            # Count characters: 16
student_name.split()         # Split into words: ['alice', 'Johnson']
```

### 📍 String Indexing (Get Specific Letters)

```python
word = "Python"
word[0]      # 'P' (first letter)
word[1]      # 'y' (second letter)
word[-1]     # 'n' (last letter)
word[0:3]    # 'Pyt' (first 3 letters)
word[2:5]    # 'tho' (letters 2-4)
```

### 🔗 String Concatenation (Combine Text)

```python
"Hello " + "World!"          # "Hello World!"
"Grade: " + str(95)          # Need str() for numbers: "Grade: 95"
"Hi " * 3                    # "Hi Hi Hi " (repeat text)
```

---

## 🔍 Type Checking & Conversion

### Check Data Type (What Kind of Data?)

```python
type(25)                # <class 'int'> (whole number)
type("Alice")           # <class 'str'> (text/string)
type(3.75)              # <class 'float'> (decimal number)
type(True)              # <class 'bool'> (True/False)
type(None)              # <class 'NoneType'> (empty/nothing)

# Better way to check types:
isinstance(25, int)           # True (is this an integer?)
isinstance("Alice", str)      # True (is this text?)
isinstance(3.75, float)       # True (is this a decimal?)
```

### Convert Between Types (Change Data Types!)

```python
# Converting TO numbers (for calculations)
int("95")            # String "95" → Integer 95
float("3.75")        # String "3.75" → Float 3.75
int(3.75)            # Float 3.75 → Integer 3 (loses decimal!)

# Converting TO text (for display)
str(95)              # Integer 95 → String "95"
str(3.75)            # Float 3.75 → String "3.75"
str(True)            # Boolean True → String "True"

# Converting TO boolean (True/False)
bool(1)              # Any non-zero number → True
bool(0)              # Zero → False
bool("hello")        # Non-empty string → True
bool("")             # Empty string → False

# Real school examples:
grade_input = input("Enter your grade: ")     # Always returns STRING!
grade_number = int(grade_input)              # Convert to number for math
gpa_text = str(3.8)                          # Convert to text for display
```

---

## Input/Output (Talk to Your Program!)

### 📝 User Input (Get Information)

```python
student_name = input("Enter your name: ")         # Get student's name
student_age = int(input("Enter your age: "))      # Get age as number
gpa = float(input("Enter your GPA: "))            # Get GPA as decimal
is_honor_student = input("Honor roll? (y/n): ").lower() == "y"
```

### 🖨️ Display Output (Show Results)

```python
print("Welcome to Grade Calculator!")     # Simple message
print("Name:", student_name, "Age:", student_age)  # Multiple items
print(f"Hello {student_name}!")           # Easy formatting
print(f"Your GPA is: {gpa:.2f}")          # Round to 2 decimals
print("Grade:", "A" if gpa >= 3.5 else "B")  # Conditional display
```

---

## 🔑 Common Python Keywords (Reserved Words!)

### Words You CAN'T Use as Variable Names

```python
# 🔀 Making Decisions (Conditional Logic)
if, elif, else          # if/else decisions

# 🔄 Repeating Things (Loops)
for, while              # Repeat code multiple times
in, range              # For loop helpers

# 📦 Creating Functions & Classes
def, class              # Define functions and classes
return, yield          # Give back results

# ✅❌ True/False Values
True, False, None      # Boolean values and empty

# 🧠 Logic Operations
and, or, not           # Combine True/False conditions
is, in                 # Check relationships

# 📁 File Operations
import, from, as       # Bring in external code
with                   # Work with files

# 🚨 Error Handling
try, except, finally, raise  # Handle errors gracefully

# 🎮 Loop Control
break, continue        # Control loop flow
pass                   # Do nothing (placeholder)
del                    # Delete variables
global, nonlocal       # Variable scope control
assert                 # Debugging checks
lambda                 # Quick functions
```

### 🚨 How to Remember Keywords

- **IF decisions**: if, elif, else
- **LOOP helpers**: for, while, in, range
- **TRUE/FALSE**: True, False, None
- **MATH LOGIC**: and, or, not
- **MAKE THINGS**: def, class, return
- **HANDLE ERRORS**: try, except, finally

---

## 🚨 Error Messages (Common Student Mistakes!)

### Types of Errors You'll See

```python
# SyntaxError - Missing colon
if grade > 90      # Need colon after if statement!
    print("A grade")

# IndentationError - Wrong spacing
if grade > 90:
print("A grade")   # This line needs 4 spaces/tab!

# NameError - Variable not defined
print(student_name)   # You never created 'student_name'!

# TypeError - Mixing text and numbers
gpa = "3.5" + 1       # Can't add string and number!

# ValueError - Wrong type of input
grade = int("ninety") # Can't convert "ninety" to a number!

# ZeroDivisionError - Can't divide by zero
average = total_points / 0  # Teachers can't divide by zero!

# IndexError - Array/list too small
grades = [85, 90, 78]
print(grades[10])     # Only 3 grades, but asking for 11th!
```

## 🔧 Quick Troubleshooting Checklist

### When Code Doesn't Work:

1. **🔍 Check indentation** - Are all indented lines lined up?
2. **📝 Check spelling** - Variable names, keywords, functions
3. **❓ Check syntax** - Colons after if/for/while/def?
4. **🔢 Check data types** - Mixing strings and numbers?
5. **📍 Check variable scope** - Variable defined where used?
6. **📖 Read the error message** - It tells you the line number!

### Debug Commands (Use These!)

```python
print(variable)          # Show variable value
print(type(variable))    # Check data type
len(variable)            # Check length
variable.upper()         # Test string methods
# Add these temporarily to find bugs!
```

### Quick Fixes for Common Problems:

- **"NameError"** → Check spelling of variable names
- **"IndentationError"** → Fix spacing (4 spaces per level)
- **"SyntaxError"** → Check colons, quotes, brackets
- **"TypeError"** → Check if mixing strings and numbers
- **"ValueError"** → Check if input type matches expected type

---

---

## 📐 Mathematical Operations

### Order of Operations (PEMDAS)

```python
# 1. Parentheses first
(2 + 3) * 4    # 5 * 4 = 20

# 2. Exponents next
2 + 3 * 4 ** 2 # 2 + 3 * 16 = 2 + 48 = 50

# 3. Multiplication/Division (left to right)
2 + 3 * 4      # 2 + 12 = 14

# 4. Addition/Subtraction last
10 - 5 + 3     # 5 + 3 = 8

# Real school example:
# Grade calculation: 90% of homework + 10% of extra credit
final_grade = (homework_score * 0.9) + (extra_credit * 0.1)
```

### Math Shortcuts (Make Your Code Cleaner!)

```python
# Instead of: count = count + 1
count += 1

# Instead of: total = total * 2
total *= 2

# Instead of: x = x / 3
x /= 3

# Real examples:
gpa += 0.1              # Add bonus points
total_points *= 1.05    # Apply 5% bonus
homework_count -= 1     # Remove incomplete assignment
```

---

## ✨ Best Practices (Write Better Code!)

### Code Style (Make It Readable!)

```python
# ✅ Use descriptive names (Future you will thank you!)
student_name = "Alice Johnson"
homework_grade = 95
is_honor_student = True

# ❌ Avoid unclear names
x = "Alice Johnson"
grade = 95
h = True

# ✅ Add helpful comments for complex calculations
# Calculate final grade: 40% homework, 30% quizzes, 30% exams
final_grade = (homework * 0.4) + (quizzes * 0.3) + (exams * 0.3)

# ✅ Keep lines reasonable length (around 80 characters max)
# Good: grade = calculate_final_grade(homework, quizzes, exams)
# Bad:  grade = very_long_function_name_with_lots_of_parameters(param1, param2, param3, param4, param5, param6)

# ✅ Use consistent spacing around operators
✅ age = 25 + grade_point_average
❌ age=25+grade_point_average
```

### Variable Principles (Smart Naming!)

```python
# ✅ Use meaningful names that explain the data
student_age = 16                    # Clear what this represents
math_grade = 88                     # Grade for math class
attendance_rate = 0.95              # 95% attendance
homework_completed = True           # Boolean (yes/no) answer

# ❌ Avoid single letters (except in loops)
a = 16                              # Not clear what this is
g = 88                              # What kind of grade?
r = 0.95                            # What rate?

# ✅ Group related data together
student_name = "Alice"
student_age = 16
student_gpa = 3.8
is_honor_student = True

# ✅ Use prefixes/suffixes for clarity
total_points = 450
max_points = 500
average_score = 85.5
grade_percentage = 90.0
```

---

## 🎯 Quick Examples (Perfect for School Projects!)

### 📊 Grade Calculator (School Project!)

```python
# Get assignment scores
homework = float(input("Homework score (0-100): "))
quiz = float(input("Quiz score (0-100): "))
exam = float(input("Exam score (0-100): "))

# Calculate final grade
final_grade = (homework * 0.3) + (quiz * 0.2) + (exam * 0.5)

print(f"Final Grade: {final_grade:.1f}")

if final_grade >= 90:
    print("Grade: A 🎉")
elif final_grade >= 80:
    print("Grade: B 😊")
elif final_grade >= 70:
    print("Grade: C 😐")
elif final_grade >= 60:
    print("Grade: D 😞")
else:
    print("Grade: F 📚")
```

### 🏆 Student Honor Roll Checker

```python
# Get student info
name = input("Student name: ")
gpa = float(input("GPA (0.0-4.0): "))
attendance_rate = float(input("Attendance %: "))

# Check honor roll requirements
honor_roll = gpa >= 3.5 and attendance_rate >= 95

print(f"\n{name}'s Status:")
print(f"GPA: {gpa}")
print(f"Attendance: {attendance_rate}%")

if honor_roll:
    print("✅ Honor Roll Student!")
else:
    print("❌ Not on Honor Roll")
    if gpa < 3.5:
        print("   Reason: GPA too low")
    if attendance_rate < 95:
        print("   Reason: Attendance too low")
```

### 🧮 Math Quiz Generator

```python
import random

# Generate random math problem
num1 = random.randint(1, 10)
num2 = random.randint(1, 10)
operator = random.choice(['+', '-', '*'])

# Create the problem
if operator == '+':
    correct_answer = num1 + num2
    problem = f"{num1} + {num2}"
elif operator == '-':
    correct_answer = num1 - num2
    problem = f"{num1} - {num2}"
else:
    correct_answer = num1 * num2
    problem = f"{num1} * {num2}"

# Ask student
print(f"Solve: {problem}")
student_answer = int(input("Your answer: "))

# Check answer
if student_answer == correct_answer:
    print("✅ Correct! Great job!")
else:
    print(f"❌ Wrong! The answer was {correct_answer}")
```

### 📚 Reading Time Calculator

```python
# Get book info
pages = int(input("How many pages in the book? "))
reading_speed = float(input("How fast do you read (pages per hour)? "))

# Calculate reading time
hours_needed = pages / reading_speed

print(f"\n📖 Book Analysis:")
print(f"Pages: {pages}")
print(f"Reading speed: {reading_speed} pages/hour")

if hours_needed < 1:
    minutes = int(hours_needed * 60)
    print(f"⏱️ Reading time: {minutes} minutes")
else:
    print(f"⏱️ Reading time: {hours_needed:.1f} hours")

# Suggest reading schedule
days_available = int(input("How many days to finish? "))
pages_per_day = pages / days_available
print(f"📅 Read {pages_per_day:.1f} pages per day to finish on time!")
```

---

## 🧠 Memory Helpers (Never Forget!)

### Mnemonics (Memory Tricks)

- **PEMDAS**: Please Excuse My Dear Aunt Sally (Order of operations: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction)
- **T-F-N**: True, False, None (Remember capital letters!)
- **IDNIT**: Indentation Does Not Ignore Things! (4 spaces = 1 level)
- **CQA**: Close Quote Always (Strings need matching quotes)
- **COLON**: Conditions Need Colons (if, for, while, def need colons)

### Quick Memory Tests

- **String test**: Does it have quotes? "text" or 'text'
- **Number test**: No quotes for math: 25 or 3.14
- **Boolean test**: True/False with capital T/F
- **Variable test**: No spaces, can't start with number
- **Indentation test**: Every indented line needs consistent spacing

### 🔥 Pro Student Tips

```python
# Tip 1: Use meaningful variable names
❌ x = 95
✅ math_grade = 95

# Tip 2: Add comments for complex logic
final_grade = (homework * 0.3) + (quiz * 0.2) + (exam * 0.5)  # 30% homework, 20% quiz, 50% exam

# Tip 3: Test your code with different inputs
# Try: 0, negative numbers, very large numbers

# Tip 4: Use f-strings for cleaner output
print(f"Student {name} has GPA {gpa:.2f}")  # Much cleaner!

# Tip 5: Handle edge cases
if total_students > 0:
    class_average = total_points / total_students
else:
    print("No students to calculate average")
```

## 🚨 Common Student Coding Mistakes

### What NOT to Do (Learn from These!)

```python
# ❌ Forgetting quotes around text
student_name = Alice        # ERROR! Needs quotes

# ❌ Using spaces in variable names
student name = "John"       # ERROR! No spaces allowed

# ❌ Wrong comparison operator
if grade = 95:              # ERROR! Use == for comparison

# ❌ Mixing data types
result = "Score: " + 95     # ERROR! Convert to string first

# ❌ Wrong indentation
if grade > 90:
print("A grade")            # ERROR! Need 4 spaces

# ❌ Forgetting colon
if grade > 90               # ERROR! Need colon
    print("A grade")
```

### ✅ How to Fix Common Errors

```python
# Fix 1: Add quotes
student_name = "Alice"      # ✅ Correct

# Fix 2: Use underscores
student_name = "John"       # ✅ Correct

# Fix 3: Use == for comparison
if grade == 95:             # ✅ Correct

# Fix 4: Convert to string
result = "Score: " + str(95) # ✅ Correct

# Fix 5: Add proper indentation
if grade > 90:
    print("A grade")        # ✅ Correct

# Fix 6: Add colon
if grade > 90:              # ✅ Correct
    print("A grade")
```

## 🎮 Fun Practice Ideas

### Try These Mini-Projects:

1. **Grade Tracker**: Track grades for all subjects
2. **GPA Calculator**: Calculate semester GPA
3. **Attendance Tracker**: Monitor class attendance
4. **Study Timer**: Calculate study time needed
5. **Quiz Score Analyzer**: Analyze quiz performance
6. **Honor Roll Checker**: Multiple conditions checker

### Challenge Yourself:

- Add error handling for invalid inputs
- Create colorful output with emojis
- Build a menu-driven program
- Save data to files
- Create functions for repeated tasks

---

## 📚 Final Reminders

### Before You Submit Your Code:

- [ ] Variable names are descriptive
- [ ] Code has comments explaining complex parts
- [ ] Indentation is consistent (4 spaces)
- [ ] All strings have quotes
- [ ] All conditions have colons
- [ ] You tested with different inputs
- [ ] Error messages are clear and helpful

### When You Get Stuck:

1. **Read the error message** - It tells you exactly what's wrong!
2. **Check indentation** - Most errors are spacing issues
3. **Verify variable names** - Spelling and case matter
4. **Test smaller pieces** - Break complex code into smaller parts
5. **Use print statements** - Check what values your variables have

---

_Keep this reference handy while coding! Bookmark it for quick access._ 🔖

> 💡 **Pro Tip**: Start with simple projects and gradually add complexity. Every expert programmer started exactly where you are now!

**Happy Coding! 🎉**
