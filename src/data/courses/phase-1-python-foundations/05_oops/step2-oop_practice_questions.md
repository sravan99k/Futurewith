# 🎓 Python Object-Oriented Programming (OOP) Practice Questions

## Universal Edition - Learn OOP with REAL-World Examples!

Hey there, future coder! 👋 Welcome to the most practical way to learn programming - using examples from YOUR daily life!

**Why learn OOP with boring bank accounts when you can use CUSTOMERS, PRODUCTS, and BUSINESSES?** 🎯

In this guide, you'll master Python OOP using examples from the world you know best:

- 👨‍💼 **Customers** - Track orders, preferences, loyalty status
- 👩‍💼 **Employees** - Manage schedules, performance, departments
- 📦 **Products** - Inventory tracking, pricing, categories
- 🏢 **Companies** - Branches, teams, organizational structure
- 🛒 **Shopping** - Carts, payments, order processing
- 🎯 **Projects** - Tasks, deadlines, team collaboration
- 🏥 **Healthcare** - Patient records, appointments, billing
- 🚗 **Transportation** - Vehicles, routes, scheduling

**Perfect for:**

- ✅ **Beginners** - New to programming concepts
- ✅ **Students** - Learning CS fundamentals or preparing for interviews
- ✅ **Professionals** - Career changers or skill development
- ✅ **Anyone** - Who learns better with relatable examples

**What makes this guide effective:**

- 🚀 **100% Real-World Focused** - Every example from practical scenarios
- 📈 **Zero to Hero** - Starts easy, ends with real business systems
- 🎮 **Interactive Projects** - Build applications you use in daily life
- 💡 **Clear Explanations** - Complex concepts explained simply
- 🎯 **Career Relevant** - Technical skills with practical applications
- 🏆 **Challenge Yourself** - Go beyond basics with real projects

Ready to become an OOP master while building practical applications? Let's go! 🚀✨

## 📋 Your Learning Roadmap - From Beginner to OOP Pro!

| Level               | What You'll Build     | Real-World Examples                          | Time to Complete |
| ------------------- | --------------------- | -------------------------------------------- | ---------------- |
| 🌱 **Beginner**     | Basic object tracking | Simple customer or product classes           | 1-2 hours        |
| 🌿 **Intermediate** | Business logic        | Order processing, inventory management       | 2-3 hours        |
| 🌳 **Advanced**     | System design         | Multi-level organizations, complex workflows | 3-4 hours        |
| 🚀 **Expert**       | Real applications     | Complete business management systems         | 4-6 hours        |
| 🏆 **Master**       | Portfolio projects    | Your own business application                | Weekend project  |

## 🗂️ Complete Table of Contents

### 📚 Part 1: Getting Started - Classes & Objects

- **Question 1-5**: Meet your first objects! Simple classes and object creation
- **Focus**: What is a class? Creating customer/product objects, basic methods

### 🔧 Part 2: Building Skills - Methods & Encapsulation

- **Question 6-10**: Make your school tools smart
- **Focus**: Constructor tricks, data validation, keeping data safe

### 🎯 Part 3: Leveling Up - Inheritance & Polymorphism

- **Question 11-13**: School club hierarchies and smart systems
- **Focus**: Parent/child classes, making different clubs work together

### 🧠 Part 4: Expert Mode - Advanced Patterns

- **Question 14-18**: Professional programming techniques
- **Focus**: Multiple inheritance, special methods, design patterns

### 🏫 Part 5: Real School Projects

- **Project 1**: Library Management System (books, students, fines!)
- **Project 2**: School Cafeteria Ordering System
- **Project 3**: Student Transportation Tracker
- **Project 4**: School Event Planning Platform

### 💼 Part 6: Interview Preparation

- **Question 19-23**: Technical questions explained simply
- **Focus**: What interviewers ask + how to answer with school examples

### 🎮 Part 7: Epic Challenges

- **Challenge 1**: Complete School Management System
- **Challenge 2**: Virtual Classroom Platform
- **Challenge 3**: School Sports League Manager
- **Challenge 4**: Student Social Network (bonus!)

---

## 🆘 Student Troubleshooting Guide - "It Doesn't Work!"

### Common Errors & How to Fix Them 🤖

**❌ "NameError: name 'Student' is not defined"**

- **What happened**: You tried to use Student before creating the class
- **Fix**: Make sure your `class Student:` comes BEFORE you try to create objects
- **School analogy**: Like trying to enroll in math class before the school creates the math course!

**❌ "TypeError: Student() takes no arguments"**

- **What happened**: Your class doesn't have an `__init__` method, or you're passing wrong number of arguments
- **Fix**: Add `def __init__(self, name, grade):` to your class
- **School analogy**: Like trying to fill out a form with the wrong number of fields!

**❌ "AttributeError: 'Student' object has no attribute 'grade'"**

- **What happened**: You tried to access a variable that doesn't exist in the object
- **Fix**: Make sure you set `self.grade = grade` in your `__init__` method
- **School analogy**: Like asking a student for their locker number when they haven't been assigned one yet!

**❌ "IndentationError: expected an indented block"**

- **What happened**: Methods inside your class aren't properly indented
- **Fix**: Indent everything inside the class with 4 spaces
- **School analogy**: Like having organized folders vs. dumping everything loose!

### 💡 Debugging Like a Detective 🕵️

**Step 1: Read the Error Message**

- Python errors tell you exactly what's wrong and where
- The line number shows you exactly where the problem is
- Don't panic - errors are your friends! They help you learn!

**Step 2: Check Your Class Definition**

- Is your class defined properly with `class ClassName:`?
- Are your methods indented correctly inside the class?
- Does your `__init__` method take the right parameters?

**Step 3: Test Small Pieces**

- Create one simple object first
- Add complexity gradually
- Use `print()` statements to see what's happening

### 🎯 Quick Debugging Questions to Ask Yourself

1. **Did I spell everything correctly?**
   - Case matters! `student.name` vs `Student.Name`

2. **Did I create the object first?**
   - `my_student = Student("Name", 9)` must come before `my_student.introduce()`

3. **Am I using the right number of arguments?**
   - If `__init__(self, name, grade)` needs 2 args, pass exactly 2

4. **Is the variable actually set?**
   - Make sure you assigned values in `__init__` or elsewhere

### 🔧 Debugging Tools for Students

**Use `print()` like a detective:**

```python
class Student:
    def __init__(self, name, grade):
        self.name = name  # This should print the name
        print(f"Creating student: {name}")  # Debug line
        self.grade = grade  # This should print the grade
        print(f"Grade: {grade}")  # Debug line
```

**Check what type things are:**

```python
print(type(my_student))  # Should say <class '__main__.Student'>
print(isinstance(my_student, Student))  # Should say True
```

### 🎓 Learning from Mistakes (It's OK to Make Them!)

**Every programmer makes mistakes:**

- The best developers make TONS of mistakes
- Each error teaches you something new
- You'll recognize errors faster as you practice

**Keep a "Mistake Journal":**

- Write down errors you encountered
- Note what caused them and how you fixed them
- Review before starting new projects

### 📱 Quick Reference Card - Student-Friendly

| Problem              | Quick Check                                                 |
| -------------------- | ----------------------------------------------------------- |
| Class not found      | Is `class Name:` defined above where you use it?            |
| Wrong number of args | Does `__init__(self, x, y)` match `ClassName(a, b)`?        |
| Attribute missing    | Did you set `self.attribute = value` in `__init__`?         |
| Indentation error    | Is everything inside class indented 4 spaces?               |
| Method not working   | Did you call it with `object.method()` not `object.method`? |

### 🏆 Become a Bug Detective

**Pro Tip**: When you get an error, DON'T PANIC!

1. Take a deep breath
2. Read the error message carefully
3. Check the line number it mentions
4. Use `print()` to debug step by step
5. Ask for help if you're stuck for more than 15 minutes

**Remember**: Debugging is like solving a puzzle. Every bug you fix makes you stronger! 💪

---

**💡 Pro Tip:** Don't just read - CODE ALONG! Create your own versions as you go. Make them about YOUR school! When you get stuck, use this troubleshooting guide!

---

## Basic Level Questions - Classes and Objects

### Question 1: Simple Class Creation - Meet Your First Student!

Create a `Student` class with attributes: name, grade_level, and student_id. Create a student object and display their information.

**Solution:**

```python
class Student:
    def __init__(self, name, grade_level, student_id):
        self.name = name
        self.grade_level = grade_level
        self.student_id = student_id

# Create student object
my_student = Student("Sarah Johnson", 9, "S2024001")

# Print details
print(f"Name: {my_student.name}")
print(f"Grade Level: {my_student.grade_level}")
print(f"Student ID: {my_student.student_id}")

# Output:
# Name: Sarah Johnson
# Grade Level: 9
# Student ID: S2024001
```

**Why this works:** The `__init__` method is like a special setup function that automatically runs when creating a new student object, setting up their basic information. Think of it as filling out a student's enrollment form!

### Question 2: Adding Methods to Classes - Student Introductions!

Add a `introduce()` method to the Student class that makes students introduce themselves.

**Solution:**

```python
class Student:
    def __init__(self, name, grade_level, student_id):
        self.name = name
        self.grade_level = grade_level
        self.student_id = student_id

    def introduce(self):
        print(f"Hi! I'm {self.name}")
        print(f"I'm in grade {self.grade_level}")
        print(f"My student ID is {self.student_id}")
        print("Nice to meet you! 😊")
        print("-" * 30)

# Create and introduce students
student1 = Student("Alex Chen", 10, "S2024156")
student1.introduce()

# Output:
# Hi! I'm Alex Chen
# I'm in grade 10
# My student ID is S2024156
# Nice to meet you! 😊
# ------------------------------
```

**Why this works:** Methods are like abilities or actions that students can perform. They use `self` to access the student's own information, just like you know your own name and grade!

### 💡 Pro Student Tip: Making Code Your Own

**What you just learned:**

- A class is like a template or form for creating students
- Each student object has their own unique information
- Methods let students "do things" like introduce themselves

**Try this yourself:**

- Change the student names to your classmates
- Add new attributes like favorite subject or lunch period
- Create a method for students to "do homework" or "go to lunch"

### Question 3: Multiple Students - A Whole Class!

Create three different student objects and make them all introduce themselves.

**Solution:**

```python
class Student:
    def __init__(self, name, grade_level, student_id):
        self.name = name
        self.grade_level = grade_level
        self.student_id = student_id

    def introduce(self):
        print(f"Hi! I'm {self.name}")
        print(f"I'm in grade {self.grade_level}")
        print(f"My student ID is {self.student_id}")
        print("-" * 30)

# Create multiple students in a class
class_students = [
    Student("Emma Wilson", 9, "S2024001"),
    Student("Marcus Johnson", 9, "S2024002"),
    Student("Zoe Rodriguez", 9, "S2024003")
]

# Make all students introduce themselves
print("🎓 Welcome to 9th Grade Class A!")
print("=" * 40)
for student in class_students:
    student.introduce()

# Output:
# 🎓 Welcome to 9th Grade Class A!
# ========================================
# Hi! I'm Emma Wilson
# I'm in grade 9
# My student ID is S2024001
# ------------------------------
# Hi! I'm Marcus Johnson
# I'm in grade 9
# My student ID is S2024002
# ------------------------------
# Hi! I'm Zoe Rodriguez
# I'm in grade 9
# My student ID is S2024003
# ------------------------------
```

**Why this works:** Each student object is unique - even though they're all from the same Student class, each has their own name, grade, and ID. Just like in a real classroom, every student is different!

### Question 4: School Calculator Class

Create a `SchoolCalculator` class that students can use for homework calculations.

**Solution:**

```python
class SchoolCalculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b != 0:
            return a / b
        return "Error: Cannot divide by zero!"

    def calculate_percentage(self, obtained, total):
        if total != 0:
            return (obtained / total) * 100
        return "Error: Total cannot be zero"

    def find_average(self, numbers):
        if numbers:
            return sum(numbers) / len(numbers)
        return "Error: No numbers provided"

# Test the school calculator
calc = SchoolCalculator()
print("🧮 School Calculator for Homework!")
print("=" * 35)

# Basic math
print(f"5 + 3 = {calc.add(5, 3)}")            # 5 + 3 = 8
print(f"10 - 4 = {calc.subtract(10, 4)}")     # 10 - 4 = 6
print(f"6 * 7 = {calc.multiply(6, 7)}")       # 6 * 7 = 42
print(f"15 / 3 = {calc.divide(15, 3)}")       # 15 / 3 = 5.0

# School-specific calculations
test_scores = [85, 92, 78, 96, 88]
print(f"Test average: {calc.find_average(test_scores):.2f}")  # Test average: 87.80
print(f"Percentage: {calc.calculate_percentage(85, 100)}%")  # Percentage: 85.0%

print(f"\nError handling: {calc.divide(10, 0)}")
```

**Why this works:** Methods are like tools in your calculator. Just like a real calculator has buttons for different operations, our class has methods for different calculations. You can reuse it for all your math homework!

### Question 5: Full-Featured Student Class

Create a `Student` class with subjects and grades, including methods to check academic performance.

**Solution:**

```python
class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.subjects = {}  # Dictionary to store subject: grade pairs
        self.gpa = 0.0

    def add_grade(self, subject, grade):
        """Add a grade for a subject"""
        self.subjects[subject] = grade

    def calculate_gpa(self):
        """Calculate Grade Point Average"""
        if self.subjects:
            self.gpa = sum(self.subjects.values()) / len(self.subjects)
            return round(self.gpa, 2)
        return 0.0

    def is_passing(self, subject=None):
        """Check if passing (grade >= 60)"""
        if subject:
            return self.subjects.get(subject, 0) >= 60
        # Check if passing all subjects
        return all(grade >= 60 for grade in self.subjects.values())

    def get_letter_grade(self, grade=None):
        """Convert numeric grade to letter grade"""
        if grade is None:
            grade = self.gpa

        if grade >= 90:
            return "A"
        elif grade >= 80:
            return "B"
        elif grade >= 70:
            return "C"
        elif grade >= 60:
            return "D"
        else:
            return "F"

    def get_report_card(self):
        """Generate a complete report card"""
        print(f"\n📊 REPORT CARD FOR: {self.name}")
        print(f"Student ID: {self.student_id}")
        print("=" * 40)

        for subject, grade in self.subjects.items():
            letter = self.get_letter_grade(grade)
            status = "✅ PASS" if grade >= 60 else "❌ FAIL"
            print(f"{subject:15} | {grade:3} | {letter} | {status}")

        gpa = self.calculate_gpa()
        overall_letter = self.get_letter_grade(gpa)
        print("=" * 40)
        print(f"GPA: {gpa} ({overall_letter})")
        return gpa

# Test the enhanced student class
student1 = Student("Sarah Kim", "S202401")
student1.add_grade("Math", 92)
student1.add_grade("Science", 88)
student1.add_grade("English", 95)
student1.add_grade("History", 87)

student2 = Student("Jake Martinez", "S202402")
student2.add_grade("Math", 58)
student2.add_grade("Science", 72)
student2.add_grade("English", 45)
student2.add_grade("History", 68)

# Display report cards
student1.get_report_card()
student2.get_report_card()

# Performance checks
print(f"\n{student1.name} passing all subjects: {student1.is_passing()}")
print(f"{student2.name} passing all subjects: {student2.is_passing()}")
print(f"{student2.name} passing Math: {student2.is_passing('Math')}")
```

**Why this works:** Methods can perform complex calculations using object data. This student class tracks multiple subjects, calculates GPA, and provides detailed academic reports - just like a real school report card!

---

## Intermediate Level Questions - Methods, Constructors, Encapsulation

### Question 6: Enhanced Student Class with Validation

Improve the Student class to validate that student_id is valid and grades are between 0-100.

**Solution:**

```python
class Student:
    def __init__(self, name, student_id, grade_level=9):
        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Student name must be a non-empty string")
        if not student_id or not isinstance(student_id, str):
            raise ValueError("Student ID must be a non-empty string")
        if not isinstance(grade_level, int) or grade_level < 1 or grade_level > 12:
            raise ValueError("Grade level must be an integer between 1 and 12")

        self.name = name
        self.student_id = student_id
        self.grade_level = grade_level
        self.subjects = {}
        self.attendance = 0
        self.total_days = 0

    def __str__(self):
        return f"Student: {self.name} (ID: {self.student_id}, Grade {self.grade_level})"

    def add_grade(self, subject, grade):
        """Add a grade with validation (0-100)"""
        if not isinstance(grade, (int, float)):
            raise ValueError("Grade must be a number")
        if grade < 0 or grade > 100:
            raise ValueError("Grade must be between 0 and 100")

        self.subjects[subject] = grade
        return f"Added {subject}: {grade} for {self.name}"

    def mark_attendance(self, present=True):
        """Mark attendance for a day"""
        self.total_days += 1
        if present:
            self.attendance += 1
            return f"{self.name} marked present"
        return f"{self.name} marked absent"

    def get_attendance_percentage(self):
        """Calculate attendance percentage"""
        if self.total_days == 0:
            return 0
        return (self.attendance / self.total_days) * 100

    def is_excellent_student(self):
        """Check if student has A average and good attendance"""
        if not self.subjects:
            return False
        avg_grade = sum(self.subjects.values()) / len(self.subjects)
        attendance_pct = self.get_attendance_percentage()
        return avg_grade >= 90 and attendance_pct >= 95

# Test with valid data
try:
    student1 = Student("Emily Chen", "S2024501", 10)
    print(student1)  # Student: Emily Chen (ID: S2024501, Grade 10)

    # Add grades
    print(student1.add_grade("Math", 95))
    print(student1.add_grade("Science", 92))
    print(student1.add_grade("English", 88))

    # Mark attendance
    for i in range(20):
        student1.mark_attendance(present=True)
    for i in range(1):
        student1.mark_attendance(present=False)

    print(f"Attendance: {student1.get_attendance_percentage():.1f}%")
    print(f"Excellent student: {student1.is_excellent_student()}")

except ValueError as e:
    print(f"Error: {e}")

# Test with invalid data
try:
    invalid_student = Student("", "S000", 15)
except ValueError as e:
    print(f"Error caught: {e}")  # Error caught: Grade level must be between 1 and 12

try:
    student1.add_grade("History", 150)  # Invalid grade
except ValueError as e:
    print(f"Error caught: {e}")  # Error caught: Grade must be between 0 and 100
```

**Why this works:** Constructor and method validation ensures student data is always valid. Just like schools have rules for student IDs and grade ranges, our class validates all inputs to prevent errors!

### Question 7: School Locker System with Encapsulation

Create a SchoolLocker class with private combination and methods to store/retrieve items securely.

**Solution:**

```python
class SchoolLocker:
    def __init__(self, locker_number, student_name, combination):
        self.locker_number = locker_number
        self.student_name = student_name
        self.__combination = combination  # Private attribute - like a real locker!
        self.__items = []  # Private storage
        self.is_open = False

    def unlock(self, entered_combination):
        """Unlock the locker with the correct combination"""
        if entered_combination == self.__combination:
            self.is_open = True
            return f"Locker {self.locker_number} unlocked! Welcome, {self.student_name}"
        else:
            return "❌ Wrong combination! Access denied."

    def lock(self):
        """Lock the locker"""
        self.is_open = False
        return f"Locker {self.locker_number} locked securely."

    def store_item(self, item):
        """Store an item (only when unlocked)"""
        if not self.is_open:
            return "❌ Cannot store item - locker is locked!"

        self.__items.append(item)
        return f"✅ Stored '{item}' in locker {self.locker_number}"

    def get_item(self, item_name):
        """Retrieve an item (only when unlocked)"""
        if not self.is_open:
            return "❌ Cannot access items - locker is locked!"

        if item_name in self.__items:
            self.__items.remove(item_name)
            return f"✅ Retrieved '{item_name}' from locker {self.locker_number}"
        else:
            return f"❌ Item '{item_name}' not found in locker"

    def list_items(self):
        """List all items (only when unlocked)"""
        if not self.is_open:
            return "❌ Cannot view items - locker is locked!"

        if not self.__items:
            return f"Locker {self.locker_number} is empty"

        items_list = "\n".join([f"  • {item}" for item in self.__items])
        return f"Items in locker {self.locker_number}:\n{items_list}"

    def get_locker_info(self):
        """Get basic locker information (safe to access anytime)"""
        return f"Locker #{self.locker_number} - Assigned to: {self.student_name}"

# Test the school locker system
locker = SchoolLocker(127, "Alice Johnson", "23-15-7")

print("🔐 School Locker System")
print("=" * 40)
print(locker.get_locker_info())

# Try to access without unlocking
print(f"\n📝 Without unlocking:")
print(locker.list_items())  # Should fail

# Unlock with correct combination
print(f"\n🔑 Unlocking locker:")
print(locker.unlock("23-15-7"))

# Store items
print(f"\n📦 Storing items:")
print(locker.store_item("Math textbook"))
print(locker.store_item("Lunch box"))
print(locker.store_item("Gym shoes"))

# View items
print(f"\n📋 Viewing contents:")
print(locker.list_items())

# Retrieve item
print(f"\n🎒 Getting math book:")
print(locker.get_item("Math textbook"))

# Lock and try to access again
print(f"\n🔒 Locking locker:")
print(locker.lock())
print(locker.list_items())  # Should fail

# Try wrong combination
print(f"\n❌ Testing wrong combination:")
wrong_locker = SchoolLocker(128, "Bob Smith", "10-20-30")
print(wrong_locker.unlock("99-99-99"))

# Cannot access private attributes directly
try:
    print(f"\n🚫 Trying to access private combination:")
    print(locker.__combination)
except AttributeError as e:
    print(f"Private attribute access blocked: {e}")
```

**Why this works:** Private attributes (with double underscores) are like the locker's combination - students can't see or change them directly! This is called encapsulation, just like real lockers protect your belongings. Only the correct methods can access the private data!

### Question 8: School Building and Classroom Classes

Create a SchoolRoom class with capacity and features, then create Classroom that inherits from it.

**Solution:**

```python
class SchoolRoom:
    def __init__(self, room_number, capacity):
        self.room_number = room_number
        self.capacity = capacity
        self.current_occupancy = 0

    def enter_room(self, people):
        """People entering the room"""
        if self.current_occupancy + people <= self.capacity:
            self.current_occupancy += people
            return f"{people} people entered. Room occupancy: {self.current_occupancy}"
        else:
            return f"Room full! Max capacity: {self.capacity}"

    def leave_room(self, people):
        """People leaving the room"""
        if self.current_occupancy >= people:
            self.current_occupancy -= people
            return f"{people} people left. Room occupancy: {self.current_occupancy}"
        else:
            return "Cannot have negative occupancy!"

    def get_available_space(self):
        """Calculate available space"""
        return self.capacity - self.current_occupancy

    def area(self):
        """Calculate room area (simplified)"""
        # Assume each person needs 25 sq ft
        return self.capacity * 25

    def __str__(self):
        return f"Room {self.room_number} (Capacity: {self.capacity})"

class Classroom(SchoolRoom):
    def __init__(self, room_number, capacity, subject):
        # Call parent constructor
        super().__init__(room_number, capacity)
        self.subject = subject
        self.has_projector = False
        self.has_whiteboard = True
        self.desks = capacity

    def set_projector(self, has_projector=True):
        """Install or remove projector"""
        self.has_projector = has_projector
        return f"Projector {'installed' if has_projector else 'removed'}"

    def get_room_type(self):
        """Get specialized room information"""
        projector_status = "Yes" if self.has_projector else "No"
        return f"{self.subject} Classroom with projector: {projector_status}"

    def __str__(self):
        return f"Classroom {self.room_number} - {self.subject} (Capacity: {self.capacity})"

# Test the school room system
print("🏫 School Room Management System")
print("=" * 50)

# Create regular room
gym = SchoolRoom("G101", 50)
print(gym)
print(gym.enter_room(25))  # 25 people entered. Room occupancy: 25
print(gym.enter_room(30))  # 30 people entered. Room occupancy: 55
print(gym.get_available_space())  # Available space: -5 (over capacity!)

# Create specialized classrooms
math_classroom = Classroom("M205", 30, "Mathematics")
science_lab = Classroom("S310", 24, "Chemistry")

print(f"\n📚 {math_classroom}")
print(f"Room type: {math_classroom.get_room_type()}")
print(math_classroom.set_projector(True))
print(math_classroom.enter_room(28))  # Students entering

print(f"\n🧪 {science_lab}")
print(f"Room type: {science_lab.get_room_type()}")
print(science_lab.enter_room(20))  # Students entering

# Test inheritance
print(f"\n🔄 Inheritance Test:")
print(f"Math classroom is a SchoolRoom: {isinstance(math_classroom, SchoolRoom)}")
print(f"Math classroom area: {math_classroom.area()} sq ft")  # Inherited method!
print(f"Available space in math room: {math_classroom.get_available_space()}")

# Test method overriding
print(f"\n📊 Room Details:")
print(f"Gym: {gym}")  # Uses SchoolRoom __str__
print(f"Math: {math_classroom}")  # Uses Classroom __str__
```

**Why this works:** Inheritance allows Classroom to inherit all the basic room functionality from SchoolRoom while adding specialized features like subjects and projectors. A Classroom IS a SchoolRoom, but with extra features!

### Question 9: Teacher Class with School-Specific Methods

Create a Teacher class with methods for classroom management and salary updates.

**Solution:**

```python
class Teacher:
    def __init__(self, name, teacher_id, subject, salary):
        self.name = name
        self.teacher_id = teacher_id
        self.subject = subject
        self.salary = salary
        self.classes = []
        self.students_taught = 0

    def add_class(self, class_name, grade_level, class_size):
        """Add a class to teach"""
        class_info = {
            "name": class_name,
            "grade_level": grade_level,
            "class_size": class_size,
            "students": []
        }
        self.classes.append(class_info)
        return f"Added {class_name} (Grade {grade_level}, {class_size} students)"

    def enroll_student(self, class_name, student_name):
        """Enroll a student in a specific class"""
        for class_info in self.classes:
            if class_info["name"] == class_name:
                if len(class_info["students"]) < class_info["class_size"]:
                    class_info["students"].append(student_name)
                    self.students_taught += 1
                    return f"✅ {student_name} enrolled in {class_name}"
                else:
                    return f"❌ {class_name} is full!"
        return f"❌ Class {class_name} not found"

    def give_grade(self, class_name, student_name, grade):
        """Assign a grade to a student"""
        if not isinstance(grade, (int, float)) or grade < 0 or grade > 100:
            return "❌ Grade must be a number between 0 and 100"
        return f"✅ Grade {grade} assigned to {student_name} in {class_name}"

    def get_annual_salary(self):
        """Calculate annual salary"""
        return self.salary * 12

    def give_raise(self, percentage):
        """Give teacher a salary raise"""
        if 0 < percentage <= 100:
            old_salary = self.salary
            self.salary += self.salary * (percentage / 100)
            return f"🎉 {self.name} received {percentage}% raise! ${old_salary:,.0f} → ${self.salary:,.0f}"
        return "❌ Invalid raise percentage (must be 1-100%)"

    def display_info(self):
        """Display teacher information"""
        return f"Teacher: {self.name} (ID: {self.teacher_id}) - {self.subject} - ${self.salary:,.0f}"

    def get_teaching_load(self):
        """Get total teaching load"""
        total_students = sum(len(cls["students"]) for cls in self.classes)
        return f"Teaching {len(self.classes)} classes, {total_students} students total"

    def get_class_summary(self):
        """Get summary of all classes"""
        if not self.classes:
            return "No classes assigned yet"

        summary = f"\n📚 Classes for {self.name}:\n"
        summary += "=" * 40
        for cls in self.classes:
            summary += f"\n📖 {cls['name']} (Grade {cls['grade_level']})\n"
            summary += f"   Capacity: {cls['class_size']}\n"
            summary += f"   Enrolled: {len(cls['students'])}\n"
            if cls['students']:
                summary += f"   Students: {', '.join(cls['students'][:3])}"
                if len(cls['students']) > 3:
                    summary += f" ... and {len(cls['students']) - 3} more"
                summary += "\n"
        return summary

# Test the teacher class
teacher1 = Teacher("Ms. Johnson", "T001", "Mathematics", 55000)
teacher2 = Teacher("Mr. Lopez", "T002", "Science", 52000)

print("👩‍🏫 Teacher Management System")
print("=" * 50)

print(teacher1.display_info())
print(teacher1.give_raise(5))
print(f"Annual salary: ${teacher1.get_annual_salary():,.0f}")

# Add classes
print(f"\n📚 Adding Classes:")
print(teacher1.add_class("Algebra I", 9, 25))
print(teacher1.add_class("Geometry", 10, 30))

# Enroll students
print(f"\n👨‍🎓 Enrolling Students:")
print(teacher1.enroll_student("Algebra I", "Alice Johnson"))
print(teacher1.enroll_student("Algebra I", "Bob Smith"))
print(teacher1.enroll_student("Algebra I", "Carol Wilson"))
print(teacher1.enroll_student("Geometry", "David Brown"))

# Assign grades
print(f"\n📝 Assigning Grades:")
print(teacher1.give_grade("Algebra I", "Alice Johnson", 95))
print(teacher1.give_grade("Algebra I", "Bob Smith", 87))

# Display teaching load
print(f"\n📊 Teaching Load:")
print(teacher1.get_teaching_load())
print(teacher1.get_class_summary())

# Test salary raise
print(f"\n💰 Salary Management:")
print(teacher1.give_raise(8))  # Valid raise
print(teacher1.give_raise(150))  # Invalid raise
```

**Why this works:** Methods can modify object state (like class lists and student enrollments) and perform calculations (salary, annual earnings). This teacher class manages all aspects of a teacher's job - classes, students, grades, and salary!

### Question 10: Temperature Class with Properties

Create a Temperature class with properties for Celsius and Fahrenheit conversions.

**Solution:**

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if -273.15 <= value <= 1000:
            self._celsius = value
        else:
            raise ValueError("Temperature out of valid range (-273.15°C to 1000°C)")

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

    @property
    def kelvin(self):
        return self._celsius + 273.15

    def __str__(self):
        return f"{self._celsius}°C ({self.fahrenheit:.1f}°F, {self.kelvin:.2f}K)"

# Test temperature conversions
temp1 = Temperature(25)
print(temp1)                         # 25°C (77.0°F, 298.15K)

print(f"Celsius: {temp1.celsius}")    # Celsius: 25
print(f"Fahrenheit: {temp1.fahrenheit:.1f}")  # Fahrenheit: 77.0
print(f"Kelvin: {temp1.kelvin:.2f}")  # Kelvin: 298.15

# Setting different units
temp1.fahrenheit = 86  # Sets Celsius to 30
print(f"After setting F to 86: {temp1.celsius}°C")  # After setting F to 86: 30.0°C

# Invalid temperature
try:
    temp2 = Temperature(-300)
except ValueError as e:
    print(f"Error: {e}")  # Error: Temperature out of valid range (-273.15°C to 1000°C)
```

**Why this works:** Properties create getters and setters that look like attributes but allow validation and calculations.

---

## 🎯 Making OOP Super Relatable - School Life Analogies

### Why OOP is Like Your School Life 🏫

**Classes = Student Registration Forms**

- A Student class is like the registration form EVERY new student fills out
- All forms have the same fields (name, grade, ID) but different information
- Each student gets their own completed form (object)

**Objects = Individual Students**

- You're an object created from the Student class
- Your best friend is another object from the same class
- You both have names, grades, and IDs, but they're different!

**Methods = Student Abilities**

- `introduce()` = What you say when meeting someone new
- `get_locker_number()` = Finding where you store your stuff
- `go_to_class()` = Moving from place to place during the day

**Inheritance = Special Student Programs**

- RegularStudent class has basic school features
- HonorRollStudent inherits from RegularStudent but gets special privileges
- StudentAthlete inherits from RegularStudent but adds sports features

**Encapsulation = Student Privacy**

- Your grades are private (encapsulated) - only you and teachers can see them
- Other students can't just change your GPA (protected data)
- But you can "expose" your good grades when applying to colleges (public methods)

**Polymorphism = Different Ways to "Go to School"**

- Some students walk (WalkStudent)
- Some take the bus (BusStudent)
- Some get driven by parents (CarStudent)
- They all have `go_to_school()` but do it differently!

### 📱 Real School Apps That Use These Concepts

Think about apps your school might use:

**Student Portal App:**

- Login system (authentication)
- Grade display (data encapsulation)
- Course schedule (polymorphism - different course types)

**School Cafeteria App:**

- Menu items (different food classes)
- Order system (method calls)
- Payment processing (inheritance from payment base class)

**Attendance Tracker:**

- Student check-in (method invocation)
- Teacher dashboard (data aggregation)
- Parent notifications (observer pattern)

---

## Advanced Level Questions - Inheritance, Polymorphism, Abstract Classes

### Question 11: School Club Hierarchy

Create an abstract SchoolClub class, then create ChessClub, DramaClub, and ScienceClub classes that inherit from it.

**Solution:**

```python
from abc import ABC, abstractmethod

class SchoolClub(ABC):
    def __init__(self, club_name, advisor, meeting_day):
        self.club_name = club_name
        self.advisor = advisor
        self.meeting_day = meeting_day
        self.members = []
        self.activities = []

    @abstractmethod
    def hold_meeting(self):
        pass

    @abstractmethod
    def organize_event(self):
        pass

    def add_member(self, student_name, grade):
        """Add a student to the club"""
        member = {"name": student_name, "grade": grade}
        self.members.append(member)
        return f"✅ {student_name} joined {self.club_name}!"

    def get_member_count(self):
        return len(self.members)

    def introduce_club(self):
        return f"We're the {self.club_name}, advised by {self.advisor}"

class ChessClub(SchoolClub):
    def __init__(self, advisor, meeting_day="Wednesday"):
        super().__init__("Chess Club", advisor, meeting_day)
        self.chess_boards = 10
        self.tournaments_won = 0

    def hold_meeting(self):
        return f"♟️  {self.club_name} meeting: Practice games and strategy lessons!"

    def organize_event(self):
        self.tournaments_won += 1
        return f"🏆 Organized chess tournament! Total tournaments won: {self.tournaments_won}"

    def play_tournament(self):
        return f"⚔️  {self.club_name} is competing in a tournament!"

    def teach_opening(self, opening_name):
        return f"📚 Teaching {opening_name} opening to club members"

class DramaClub(SchoolClub):
    def __init__(self, advisor, meeting_day="Thursday"):
        super().__init__("Drama Club", advisor, meeting_day)
        self.plays_performed = 0
        self.auditions_open = True

    def hold_meeting(self):
        return f"🎭 {self.club_name} meeting: Rehearsals and script readings!"

    def organize_event(self):
        self.plays_performed += 1
        return f"🎪 Organized drama performance! Total plays performed: {self.plays_performed}"

    def audition(self, student_name):
        if self.auditions_open:
            return f"🎬 {student_name} auditioned for our next play!"
        else:
            return f"❌ Auditions are closed for {student_name}"

    def get_costume_count(self):
        return len(self.members) * 2  # Approximate costume needs

class ScienceClub(SchoolClub):
    def __init__(self, advisor, meeting_day="Tuesday"):
        super().__init__("Science Club", advisor, meeting_day)
        self.experiments_conducted = 0
        self.lab_equipment = ["Microscopes", "Test tubes", "Safety goggles"]

    def hold_meeting(self):
        return f"🔬 {self.club_name} meeting: Lab safety and experiment planning!"

    def organize_event(self):
        self.experiments_conducted += 1
        return f"🧪 Organized science fair! Total experiments: {self.experiments_conducted}"

    def conduct_experiment(self, experiment_name):
        return f"🧫 Conducting {experiment_name} experiment with the team!"

    def get_lab_report(self):
        return f"📊 Lab Report: {self.experiments_conducted} experiments completed"

# Test the club hierarchy
print("🏫 School Club Management System")
print("=" * 50)

# Create clubs
chess_club = ChessClub("Ms. Parker")
drama_club = DramaClub("Mr. Rodriguez")
science_club = ScienceClub("Dr. Kim")

clubs = [chess_club, drama_club, science_club]

# Test club functionality
print("\n📋 Club Introductions:")
for club in clubs:
    print(f"{club.introduce_club()}")

print("\n👥 Adding Members:")
print(chess_club.add_member("Alice Johnson", 10))
print(chess_club.add_member("Bob Smith", 9))
print(drama_club.add_member("Carol Davis", 11))
print(science_club.add_member("David Brown", 10))

print("\n🎯 Club Meetings:")
print(chess_club.hold_meeting())
print(drama_club.hold_meeting())
print(science_club.hold_meeting())

print("\n🎉 Organizing Events:")
print(chess_club.organize_event())
print(drama_club.organize_event())
print(science_club.organize_event())

print("\n⚡ Club-Specific Activities:")
print(chess_club.play_tournament())
print(drama_club.audition("Emma Wilson"))
print(science_club.conduct_experiment("Volcano Chemistry"))

print("\n📊 Club Statistics:")
print(f"{chess_club.club_name}: {chess_club.get_member_count()} members")
print(f"{drama_club.club_name}: {chess_club.get_member_count()} members")
print(f"{science_club.club_name}: {science_club.get_member_count()} members")

# Cannot create abstract SchoolClub object
try:
    abstract_club = SchoolClub("Generic Club", "Mr. Unknown", "Friday")
except TypeError as e:
    print(f"\n❌ Cannot create abstract club: {e}")
```

**Why this works:** Abstract classes enforce that all clubs implement required methods (meetings and events), while allowing each club type to have specialized features. Just like real school clubs, they all follow the same basic structure but have unique activities!

### Question 12: School Building Hierarchy with Polymorphism

Create different school building types and use polymorphism to calculate total capacity.

**Solution:**

```python
from abc import ABC, abstractmethod

class SchoolBuilding(ABC):
    @abstractmethod
    def capacity(self):
        pass

    @abstractmethod
    def floor_area(self):
        pass

    def get_building_info(self):
        return f"A building with capacity for {self.capacity()} people"

class Classroom(SchoolBuilding):
    def __init__(self, room_number, length, width, desks_per_row=5, rows=5):
        self.room_number = room_number
        self.length = length
        self.width = width
        self.desks_per_row = desks_per_row
        self.rows = rows

    def capacity(self):
        return self.desks_per_row * self.rows

    def floor_area(self):
        return self.length * self.width

    def get_occupancy_rate(self, current_students):
        return (current_students / self.capacity()) * 100

    def __str__(self):
        return f"Classroom {self.room_number} ({self.length}x{self.width})"

class Gymnasium(SchoolBuilding):
    def __init__(self, name, length, width, height):
        self.name = name
        self.length = length
        self.width = width
        self.height = height
        self.max_spectators = 200  # For basketball games

    def capacity(self):
        # Students + teachers + spectators
        return self.max_spectators + 100

    def floor_area(self):
        return self.length * self.width

    def get_sport_capacity(self, sport_type):
        if sport_type.lower() == "basketball":
            return 10  # players
        elif sport_type.lower() == "volleyball":
            return 12
        else:
            return 20

    def __str__(self):
        return f"Gymnasium {self.name} ({self.length}x{self.width})"

class Cafeteria(SchoolBuilding):
    def __init__(self, name, length, width, tables=20, seats_per_table=8):
        self.name = name
        self.length = length
        self.width = width
        self.tables = tables
        self.seats_per_table = seats_per_table

    def capacity(self):
        return self.tables * self.seats_per_table

    def floor_area(self):
        return self.length * self.width

    def get_lunch_period_capacity(self, periods=4):
        return self.capacity() // periods

    def __str__(self):
        return f"Cafeteria {self.name} ({self.length}x{self.width})"

# Polymorphic function to calculate total school capacity
def calculate_total_school_capacity(buildings):
    total_capacity = 0
    total_area = 0
    print("🏫 School Building Analysis")
    print("=" * 50)

    for building in buildings:
        capacity = building.capacity()
        area = building.floor_area()
        total_capacity += capacity
        total_area += area
        print(f"{building}")
        print(f"   Capacity: {capacity} people")
        print(f"   Floor Area: {area:,.0f} sq ft")
        print(f"   {building.get_building_info()}")
        print()

    return total_capacity, total_area

# Create school buildings
buildings = [
    Classroom("M101", 30, 25, 6, 5),  # Math classroom
    Classroom("S205", 35, 30, 7, 6),  # Science lab
    Gymnasium("Main Gym", 80, 60, 25),
    Cafeteria("Student Center", 60, 40, 25, 10)
]

print("🏫 Welcome to Lincoln High School!")
print("=" * 50)

total_capacity, total_area = calculate_total_school_capacity(buildings)

print(f"📊 TOTALS:")
print(f"Total Building Capacity: {total_capacity:,} people")
print(f"Total Floor Space: {total_area:,.0f} sq ft")

# Demonstrate polymorphism with specific methods
print(f"\n🎯 Specialized Building Features:")
math_room = buildings[0]
print(f"Math room occupancy at 85%: {math_room.get_occupancy_rate(math_room.capacity() * 0.85):.1f}%")

gym = buildings[2]
print(f"Gym basketball capacity: {gym.get_sport_capacity('basketball')} players")
print(f"Gym volleyball capacity: {gym.get_sport_capacity('volleyball')} players")

cafeteria = buildings[3]
print(f"Cafeteria lunch periods: {cafeteria.get_lunch_period_capacity()} students per period")
```

**Why this works:** Polymorphism lets us treat all buildings uniformly - we can calculate total capacity without knowing if it's a classroom, gym, or cafeteria. Each building type has its own way of calculating capacity, but the interface is the same!

### Question 13: Vehicle Fleet Management

Create a vehicle hierarchy with different types of vehicles.

**Solution:**

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.fuel_level = 100
        self.odometer = 0

    @abstractmethod
    def fuel_consumption_rate(self):
        pass

    def drive(self, miles):
        if self.fuel_level > 0:
            fuel_used = miles / self.fuel_consumption_rate()
            if fuel_used <= self.fuel_level:
                self.fuel_level -= fuel_used
                self.odometer += miles
                return f"Drove {miles} miles. Fuel left: {self.fuel_level:.1f}"
            else:
                max_miles = self.fuel_level * self.fuel_consumption_rate()
                return f"Not enough fuel. Can drive max {max_miles:.1f} miles"
        return "No fuel left"

    def refuel(self, amount):
        self.fuel_level += amount
        if self.fuel_level > 100:
            self.fuel_level = 100
        return f"Refueled. Current fuel: {self.fuel_level:.1f}"

class Car(Vehicle):
    def fuel_consumption_rate(self):
        return 25  # miles per gallon

class Truck(Vehicle):
    def fuel_consumption_rate(self):
        return 15  # miles per gallon

    def haul_load(self, weight):
        if weight <= 1000:
            return f"Truck hauling {weight} lbs"
        return "Truck overloaded"

class Motorcycle(Vehicle):
    def fuel_consumption_rate(self):
        return 50  # miles per gallon

    def wheelie(self):
        return "Doing a wheelie!"

# Test vehicle fleet
fleet = [
    Car("Toyota", "Camry", 2020),
    Truck("Ford", "F-150", 2019),
    Motorcycle("Harley-Davidson", "Sportster", 2021)
]

for i, vehicle in enumerate(fleet, 1):
    print(f"\n=== Vehicle {i}: {vehicle.year} {vehicle.make} {vehicle.model} ===")
    print(vehicle.drive(100))  # Drive 100 miles
    print(vehicle.refuel(20))  # Refuel
    print(vehicle.drive(200))  # Drive 200 miles

# Specific vehicle features
truck = fleet[1]
motorcycle = fleet[2]
print(f"\n{truck.haul_load(800)}")      # Truck hauling 800 lbs
print(f"{motorcycle.wheelie()}")        # Doing a wheelie!
```

**Why this works:** Inheritance with abstract methods ensures all vehicles implement required behavior while allowing specialized features.

### Question 14: E-commerce Product Hierarchy

Create different types of products using inheritance and polymorphism.

**Solution:**

```python
from abc import ABC, abstractmethod

class Product(ABC):
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.total_sold = 0

    @abstractmethod
    def calculate_discount(self, percentage):
        pass

    def sell(self, quantity):
        if self.stock >= quantity:
            self.stock -= quantity
            self.total_sold += quantity
            revenue = self.price * quantity
            return f"Sold {quantity} {self.name}(s). Revenue: ${revenue:.2f}"
        return f"Insufficient stock. Available: {self.stock}"

    def restock(self, quantity):
        self.stock += quantity
        return f"Restocked {quantity} {self.name}(s). New stock: {self.stock}"

class Electronics(Product):
    def __init__(self, name, price, stock, warranty_months):
        super().__init__(name, price, stock)
        self.warranty_months = warranty_months

    def calculate_discount(self, percentage):
        return self.price * (percentage / 100)

    def __str__(self):
        return f"{self.name} (Electronics) - ${self.price}"

class Clothing(Product):
    def __init__(self, name, price, stock, size, material):
        super().__init__(name, price, stock)
        self.size = size
        self.material = material

    def calculate_discount(self, percentage):
        # Clothing has higher discount rates
        return self.price * (percentage / 100) * 1.5

    def __str__(self):
        return f"{self.name} (Clothing) - ${self.price} - Size: {self.size}"

class Food(Product):
    def __init__(self, name, price, stock, expiration_date):
        super().__init__(name, price, stock)
        self.expiration_date = expiration_date

    def calculate_discount(self, percentage):
        # Food has limited discount due to perishability
        return self.price * (percentage / 100) * 0.8

    def is_expired(self, current_date):
        return current_date > self.expiration_date

    def __str__(self):
        return f"{self.name} (Food) - ${self.price}"

# Test e-commerce system
products = [
    Electronics("Laptop", 1200, 5, 24),
    Clothing("T-Shirt", 25, 50, "L", "Cotton"),
    Food("Milk", 3, 20, "2025-11-01")
]

total_revenue = 0
print("=== E-Commerce Sales ===")

for product in products:
    print(f"\nProduct: {product}")

    # Calculate discount
    discount = product.calculate_discount(10)
    print(f"10% discount: ${discount:.2f}")

    # Make sales
    sale_result = product.sell(3 if "Food" in str(product) else 1)
    print(sale_result)

    # Calculate revenue
    if "Food" in str(product):
        total_revenue += 3 * 3  # 3 units at $3 each
    elif "Clothing" in str(product):
        total_revenue += 1 * 25  # 1 unit at $25 each
    else:
        total_revenue += 1 * 1200  # 1 unit at $1200 each

print(f"\nTotal Revenue: ${total_revenue:.2f}")
```

**Why this works:** Polymorphism allows treating different product types uniformly while each product calculates discounts according to its business rules.

---

## Expert Level Questions - Multiple Inheritance, Special Methods, Design Patterns

### Question 15: Multiple Inheritance with Flying and Swimming Animals

Create animals with multiple capabilities using multiple inheritance.

**Solution:**

```python
class Swimmer:
    def swim(self):
        return "I can swim underwater"

    def dive(self):
        return "I can dive deep"

class Flyer:
    def fly(self):
        return "I can soar through the sky"

    def land(self):
        return "I can land gracefully"

class Walker:
    def walk(self):
        return "I can walk on land"

    def run(self):
        return "I can run fast"

# Multiple inheritance - combining abilities
class Duck(Swimmer, Flyer, Walker):
    def __init__(self, name, feather_color):
        self.name = name
        self.feather_color = feather_color

    def quack(self):
        return f"{self.name} says: Quack!"

    def get_all_abilities(self):
        return [
            self.swim(),
            self.dive(),
            self.fly(),
            self.land(),
            self.walk(),
            self.run(),
            self.quack()
        ]

class Penguin(Swimmer, Walker):
    def __init__(self, name):
        self.name = name

    def slide(self):
        return f"{self.name} is sliding on belly"

    def get_all_abilities(self):
        return [
            self.swim(),
            self.dive(),
            self.walk(),
            self.run(),
            self.slide()
        ]

class Bat(Flyer, Walker):
    def __init__(self, name, wingspan):
        self.name = name
        self.wingspan = wingspan

    def hang_upside_down(self):
        return f"{self.name} is hanging upside down"

    def get_all_abilities(self):
        return [
            self.fly(),
            self.land(),
            self.walk(),
            self.run(),
            self.hang_upside_down()
        ]

# Test multiple inheritance
print("=== Duck (Swimmer + Flyer + Walker) ===")
duck = Duck("Donald", "Yellow")
abilities = duck.get_all_abilities()
for ability in abilities:
    print(f"- {ability}")

print(f"\n=== Penguin (Swimmer + Walker) ===")
penguin = Penguin("Chilly")
abilities = penguin.get_all_abilities()
for ability in abilities:
    print(f"- {ability}")

print(f"\n=== Bat (Flyer + Walker) ===")
bat = Batman("Bruce", "1.5m") if 'Batman' in str(type(bat)) else Bat("Bruce", "1.5m")
abilities = bat.get_all_abilities()
for ability in abilities:
    print(f"- {ability}")

# Check method resolution order
print(f"\nDuck MRO: {[cls.__name__ for cls in Duck.__mro__]}")
print(f"Penguin MRO: {[cls.__name__ for cls in Penguin.__mro__]}")
print(f"Bat MRO: {[cls.__name__ for cls in Bat.__mro__]}")
```

**Why this works:** Multiple inheritance allows classes to combine features from multiple parent classes, creating flexible combinations of behaviors.

### Question 16: Special Methods and Custom Collections

Create a custom collection class with special methods.

**Solution:**

```python
class MagicList:
    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def __delitem__(self, index):
        del self._data[index]

    def __contains__(self, item):
        return item in self._data

    def __add__(self, other):
        if isinstance(other, MagicList):
            new_list = MagicList()
            new_list._data = self._data + other._data
            return new_list
        return NotImplemented

    def __str__(self):
        return f"MagicList({self._data})"

    def __repr__(self):
        return f"MagicList({self._data!r})"

    def __iter__(self):
        return iter(self._data)

    def append(self, item):
        self._data.append(item)

    def sort(self, reverse=False):
        self._data.sort(reverse=reverse)
        return self

    def reverse(self):
        self._data.reverse()
        return self

# Test the magic list
magic_list = MagicList()

print("=== Testing MagicList ===")

# Adding items
magic_list.append(3)
magic_list.append(1)
magic_list.append(4)
magic_list.append(1)
magic_list.append(5)

print(f"Original: {magic_list}")                    # MagicList([3, 1, 4, 1, 5])
print(f"Length: {len(magic_list)}")                 # Length: 5
print(f"Contains 4? {4 in magic_list}")             # Contains 4? True
print(f"First item: {magic_list[0]}")               # First item: 3

# Modifying items
magic_list[0] = 10
print(f"After change: {magic_list}")                # MagicList([10, 1, 4, 1, 5])

# Sorting and reversing
magic_list.sort()
print(f"After sort: {magic_list}")                  # MagicList([1, 1, 4, 5, 10])
magic_list.reverse()
print(f"After reverse: {magic_list}")               # MagicList([10, 5, 4, 1, 1])

# Adding two MagicLists
list1 = MagicList()
list1.append(1)
list1.append(2)

list2 = MagicList()
list2.append(3)
list2.append(4)

combined = list1 + list2
print(f"Combined: {combined}")                      # MagicList([1, 2, 3, 4])

# Iteration
print("Iteration:")
for item in magic_list:
    print(f"  {item}")
```

**Why this works:** Special methods make objects behave like built-in Python types, allowing natural syntax for common operations.

### Question 17: Singleton Pattern for Database Connection

Implement a singleton pattern for database connection management.

**Solution:**

```python
class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating new database connection...")
            cls._instance = super().__new__(cls)
            cls._instance.connected = False
            cls._instance.connection_string = None
            cls._instance.queries_executed = 0
        return cls._instance

    def connect(self, connection_string):
        if not self.connected:
            self.connection_string = connection_string
            self.connected = True
            print(f"Connected to: {connection_string}")
            return True
        print(f"Already connected to: {self.connection_string}")
        return False

    def disconnect(self):
        if self.connected:
            self.connected = False
            self.connection_string = None
            print("Disconnected from database")
            return True
        print("Not connected to any database")
        return False

    def execute_query(self, query):
        if not self.connected:
            print("Not connected to database")
            return False

        self.queries_executed += 1
        print(f"Executing query #{self.queries_executed}: {query}")
        return True

    def get_status(self):
        return {
            "connected": self.connected,
            "connection_string": self.connection_string,
            "queries_executed": self.queries_executed
        }

# Test singleton behavior
print("=== Testing Singleton Database Connection ===")

# Create multiple "instances"
conn1 = DatabaseConnection()
conn2 = DatabaseConnection()
conn3 = DatabaseConnection()

print(f"Same instance? {conn1 is conn2 is conn3}")  # True

# Connect with first connection
conn1.connect("postgresql://localhost:5432/mydb")

# All connections show same status
print(f"conn2 status: {conn2.get_status()}")
print(f"conn3 status: {conn3.get_status()}")

# Execute queries from any connection
conn1.execute_query("SELECT * FROM users")
conn2.execute_query("INSERT INTO users VALUES (1, 'Alice')")
conn3.execute_query("UPDATE users SET name='Bob' WHERE id=1")

print(f"Final status: {conn1.get_status()}")
```

**Why this works:** Singleton pattern ensures only one instance exists, useful for shared resources like database connections.

### Question 18: Observer Pattern for Event Management

Implement observer pattern for a news notification system.

**Solution:**

```python
class NewsAgency:
    def __init__(self, name):
        self.name = name
        self._subscribers = []
        self._news = []

    def subscribe(self, subscriber):
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)
            print(f"{subscriber.name} subscribed to {self.name}")

    def unsubscribe(self, subscriber):
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            print(f"{subscriber.name} unsubscribed from {self.name}")

    def publish_news(self, headline, content):
        news_item = {
            "headline": headline,
            "content": content,
            "agency": self.name,
            "timestamp": "2025-10-29"
        }
        self._news.append(news_item)
        self._notify_subscribers(news_item)

    def _notify_subscribers(self, news_item):
        for subscriber in self._subscribers:
            subscriber.update(news_item)

class NewsSubscriber:
    def __init__(self, name):
        self.name = name
        self.received_news = []

    def update(self, news_item):
        self.received_news.append(news_item)
        print(f"{self.name} received: {news_item['headline']}")

# Test observer pattern
print("=== News Observer System ===")

# Create news agencies
cnn = NewsAgency("CNN")
bbc = NewsAgency("BBC")

# Create subscribers
alice = NewsSubscriber("Alice")
bob = NewsSubscriber("Bob")
charlie = NewsSubscriber("Charlie")

# Subscribe to agencies
cnn.subscribe(alice)
cnn.subscribe(bob)
bbc.subscribe(bob)
bbc.subscribe(charlie)

print("\n=== Publishing News ===")

# Publish news
cnn.publish_news("Breaking: Python 4.0 Released", "The latest version brings amazing features!")
bbc.publish_news("Tech Update: AI Advances", "New developments in artificial intelligence")

print(f"\n=== Alice's News ({len(alice.received_news)} items) ===")
for news in alice.received_news:
    print(f"- {news['headline']}")

print(f"\n=== Bob's News ({len(bob.received_news)} items) ===")
for news in bob.received_news:
    print(f"- {news['headline']}")

# Unsubscribe and test
cnn.unsubscribe(alice)
print("\n=== After Alice unsubscribes ===")
cnn.publish_news("Update: Security Patch", "Important security update released")
```

**Why this works:** Observer pattern allows objects to automatically receive updates when other objects change state, useful for notification systems.

---

## Real-World Projects

### Project 1: School Library Management System

Create a complete school library management system with books, students, and borrowing functionality.

**Solution:**

```python
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

class LibraryItem(ABC):
    def __init__(self, item_id, title, subject_area, year):
        self.item_id = item_id
        self.title = title
        self.subject_area = subject_area  # Math, Science, English, etc.
        self.year = year
        self.is_available = True
        self.borrowed_by = None
        self.due_date = None

    @abstractmethod
    def get_type(self):
        pass

    def borrow(self, student_id):
        if self.is_available:
            self.is_available = False
            self.borrowed_by = student_id
            self.due_date = datetime.now() + timedelta(days=14)  # 2 weeks for school
            return True
        return False

    def return_item(self):
        if not self.is_available:
            self.is_available = True
            self.borrowed_by = None
            self.due_date = None
            return True
        return False

    def is_overdue(self):
        return not self.is_available and datetime.now() > self.due_date

    def days_until_due(self):
        if self.is_available:
            return None
        delta = self.due_date - datetime.now()
        return delta.days

class Textbook(LibraryItem):
    def __init__(self, item_id, title, subject_area, year, grade_level, isbn):
        super().__init__(item_id, title, subject_area, year)
        self.grade_level = grade_level  # Grade 9, 10, 11, 12
        self.isbn = isbn

    def get_type(self):
        return "Textbook"

    def is_appropriate_for_grade(self, student_grade):
        return self.grade_level == student_grade

class Novel(LibraryItem):
    def __init__(self, item_id, title, author, genre, reading_level):
        # Novels don't have subject areas like textbooks
        super().__init__(item_id, title, "Literature", 2020)
        self.author = author
        self.genre = genre
        self.reading_level = reading_level  # Easy, Medium, Hard

    def get_type(self):
        return "Novel"

    def get_book_summary(self):
        return f"{self.title} by {self.author} - {self.genre} genre"

class ReferenceBook(LibraryItem):
    def __init__(self, item_id, title, subject_area, edition):
        super().__init__(item_id, title, subject_area, 2023)
        self.edition = edition
        self.must_return_same_day = True  # Reference books can't leave library

    def get_type(self):
        return "Reference Book"

    def borrow(self, student_id):
        # Reference books can only be used in library
        return False

class Student:
    def __init__(self, student_id, name, grade_level):
        self.student_id = student_id
        self.name = name
        self.grade_level = grade_level
        self.borrowed_items = []
        self.library_fines = 0.0

    def can_borrow(self):
        """Check if student can borrow more items (limit of 3 for students)"""
        return len(self.borrowed_items) < 3

    def borrow_item(self, item):
        if self.can_borrow() and item.borrow(self.student_id):
            self.borrowed_items.append(item)
            return True
        return False

    def return_item(self, item_id):
        for item in self.borrowed_items[:]:
            if item.item_id == item_id:
                item.return_item()
                self.borrowed_items.remove(item)
                return True
        return False

    def calculate_late_fees(self):
        """Calculate late fees for overdue items"""
        total_fine = 0.0
        for item in self.borrowed_items:
            if item.is_overdue():
                days_late = (datetime.now() - item.due_date).days
                total_fine += days_late * 0.25  # 25 cents per day
        self.library_fines = total_fine
        return total_fine

    def get_borrowing_summary(self):
        """Get summary of borrowed items"""
        if not self.borrowed_items:
            return f"{self.name} has no borrowed items"

        summary = f"\n{self.name}'s Borrowed Items:\n"
        summary += "-" * 40
        for item in self.borrowed_items:
            if item.is_overdue():
                days = (datetime.now() - item.due_date).days
                summary += f"\n❌ {item.title} ({item.get_type()}) - {days} days OVERDUE"
            else:
                days = item.due_date - datetime.now()
                summary += f"\n✅ {item.title} ({item.get_type()}) - Due in {days.days} days"
        return summary

class SchoolLibrary:
    def __init__(self, school_name):
        self.school_name = school_name
        self.items = {}
        self.students = {}
        self.transactions = []

    def add_library_item(self, item):
        self.items[item.item_id] = item
        print(f"📚 Added {item.get_type()}: {item.title}")

    def register_student(self, student):
        self.students[student.student_id] = student
        print(f"✅ Registered student: {student.name} (Grade {student.grade_level})")

    def lend_item(self, student_id, item_id):
        student = self.students.get(student_id)
        item = self.items.get(item_id)

        if not student:
            return "❌ Student not found"
        if not item:
            return "❌ Item not found"
        if not item.is_available:
            return "❌ Item is not available"

        # Check if textbook is appropriate for student's grade
        if item.get_type() == "Textbook" and not item.is_appropriate_for_grade(student.grade_level):
            return "❌ Textbook not appropriate for student's grade level"

        if student.borrow_item(item):
            self.transactions.append({
                "type": "borrow",
                "student_id": student_id,
                "item_id": item_id,
                "date": datetime.now(),
                "due_date": item.due_date
            })
            return f"✅ {student.name} borrowed '{item.title}'"
        else:
            return "❌ Student cannot borrow more items (limit: 3)"

    def return_item(self, student_id, item_id):
        student = self.students.get(student_id)

        if not student:
            return "❌ Student not found"

        # Calculate late fees
        late_fees = student.calculate_late_fees()
        if student.return_item(item_id):
            self.transactions.append({
                "type": "return",
                "student_id": student_id,
                "item_id": item_id,
                "date": datetime.now(),
                "late_fees": late_fees
            })
            fee_message = f" Late fees: ${late_fees:.2f}" if late_fees > 0 else " No late fees"
            return f"✅ Item returned successfully{fee_message}"
        return "❌ Return failed"

# Build school library system
print("🏫 School Library Management System")
print("=" * 50)

library = SchoolLibrary("Lincoln High School")

# Add library items
items = [
    Textbook("T001", "Algebra I", "Math", 2023, 9, "978-0134689154"),
    Textbook("T002", "Biology", "Science", 2023, 9, "978-0134093418"),
    Novel("N001", "To Kill a Mockingbird", "Harper Lee", "Classic Fiction", "Medium"),
    ReferenceBook("R001", "Encyclopedia Britannica", "Reference", 15)
]

for item in items:
    library.add_library_item(item)

# Register students
students = [
    Student("S2024001", "Emma Johnson", 9),
    Student("S2024002", "Marcus Chen", 10)
]

for student in students:
    library.register_student(student)

print("\n📚 Library Operations:")
print(library.lend_item("S2024001", "T001"))  # Emma borrows Algebra book
print(library.lend_item("S2024001", "T002"))  # Emma borrows Biology book
print(library.lend_item("S2024001", "N001"))  # Emma borrows novel
print(library.lend_item("S2024001", "T002"))  # Emma tries to borrow more (should fail)
print(library.lend_item("S2024002", "R001"))  # Marcus tries to borrow reference (should fail - not allowed)

print("\n📖 Student Borrowing Status:")
print(students[0].get_borrowing_summary())
print(students[1].get_borrowing_summary())

print("\n🔄 Returning Items:")
print(library.return_item("S2024001", "T001"))
```

**Why this works:** This school-specific library system demonstrates how OOP concepts apply to real school scenarios, with features like grade-appropriate textbooks, borrowing limits, and late fee calculations that schools actually use!

### Project 2: Banking System with Accounts and Transactions

**Solution:**

```python
from datetime import datetime
from abc import ABC, abstractmethod

class Transaction:
    def __init__(self, transaction_type, amount, description=""):
        self.transaction_type = transaction_type  # 'deposit', 'withdrawal', 'transfer'
        self.amount = amount
        self.description = description
        self.timestamp = datetime.now()

    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.transaction_type.title()}: ${self.amount:.2f}"

class Account(ABC):
    def __init__(self, account_number, holder_name, initial_balance=0):
        self.account_number = account_number
        self.holder_name = holder_name
        self._balance = initial_balance
        self.transactions = []

    @property
    def balance(self):
        return self._balance

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            transaction = Transaction('deposit', amount, 'Account deposit')
            self.transactions.append(transaction)
            return True
        return False

    @abstractmethod
    def withdraw(self, amount):
        pass

    def get_transaction_history(self):
        return self.transactions

    def __str__(self):
        return f"Account {self.account_number}: {self.holder_name} - Balance: ${self._balance:.2f}"

class CheckingAccount(Account):
    def __init__(self, account_number, holder_name, initial_balance=0, overdraft_limit=500):
        super().__init__(account_number, holder_name, initial_balance)
        self.overdraft_limit = overdraft_limit

    def withdraw(self, amount):
        if amount > 0 and amount <= (self._balance + self.overdraft_limit):
            self._balance -= amount
            transaction = Transaction('withdrawal', amount, 'ATM withdrawal')
            self.transactions.append(transaction)
            return True
        return False

class SavingsAccount(Account):
    def __init__(self, account_number, holder_name, initial_balance=0, interest_rate=0.02):
        super().__init__(account_number, holder_name, initial_balance)
        self.interest_rate = interest_rate
        self.minimum_balance = 100  # Minimum balance requirement

    def withdraw(self, amount):
        if amount > 0 and (self._balance - amount) >= self.minimum_balance:
            self._balance -= amount
            transaction = Transaction('withdrawal', amount, 'Savings withdrawal')
            self.transactions.append(transaction)
            return True
        return False

    def add_interest(self):
        interest = self._balance * self.interest_rate
        self._balance += interest
        transaction = Transaction('interest', interest, 'Interest payment')
        self.transactions.append(transaction)
        return interest

class CreditCard(Account):
    def __init__(self, account_number, holder_name, credit_limit=5000):
        super().__init__(account_number, holder_name, 0)
        self.credit_limit = credit_limit
        self._available_credit = credit_limit

    @property
    def balance(self):
        return self.credit_limit - self._available_credit  # Negative for credit

    def withdraw(self, amount):  # This is actually a charge for credit cards
        if amount > 0 and amount <= self._available_credit:
            self._available_credit -= amount
            transaction = Transaction('charge', amount, 'Credit card charge')
            self.transactions.append(transaction)
            return True
        return False

    def make_payment(self, amount):
        if amount > 0 and amount <= self.balance:
            self._available_credit += amount
            transaction = Transaction('payment', -amount, 'Credit card payment')
            self.transactions.append(transaction)
            return True
        return False

class Bank:
    def __init__(self, name):
        self.name = name
        self.accounts = {}

    def create_account(self, account_type, account_number, holder_name, **kwargs):
        if account_number in self.accounts:
            return False, "Account already exists"

        if account_type.lower() == "checking":
            account = CheckingAccount(account_number, holder_name, **kwargs)
        elif account_type.lower() == "savings":
            account = SavingsAccount(account_number, holder_name, **kwargs)
        elif account_type.lower() == "credit":
            account = CreditCard(account_number, holder_name, **kwargs)
        else:
            return False, "Invalid account type"

        self.accounts[account_number] = account
        return True, f"{account_type.title()} account created"

    def transfer_money(self, from_account, to_account, amount):
        from_acc = self.accounts.get(from_account)
        to_acc = self.accounts.get(to_account)

        if not from_acc or not to_acc:
            return False, "Account not found"

        if from_acc.withdraw(amount):
            to_acc.deposit(amount)
            # Record transfer transaction
            from_acc.transactions.append(
                Transaction('transfer', -amount, f'Transfer to {to_account}')
            )
            to_acc.transactions.append(
                Transaction('transfer', amount, f'Transfer from {from_account}')
            )
            return True, f"Transferred ${amount:.2f} from {from_account} to {to_account}"

        return False, "Transfer failed - insufficient funds"

    def get_account_summary(self, account_number):
        account = self.accounts.get(account_number)
        if not account:
            return None

        return {
            "account": account,
            "balance": account.balance,
            "transaction_count": len(account.transactions),
            "recent_transactions": account.transactions[-5:]  # Last 5 transactions
        }

# Build banking system
bank = Bank("First National Bank")

# Create accounts
accounts_created = [
    bank.create_account("checking", "CHK001", "Alice Johnson", initial_balance=1000, overdraft_limit=500),
    bank.create_account("savings", "SVG001", "Alice Johnson", initial_balance=5000, interest_rate=0.03),
    bank.create_account("checking", "CHK002", "Bob Smith", initial_balance=800, overdraft_limit=300),
    bank.create_account("credit", "CRC001", "Alice Johnson", credit_limit=3000)
]

for success, message in accounts_created:
    print(message)

# Banking operations
print("\n=== Banking Operations ===")

checking = bank.accounts["CHK001"]
savings = bank.accounts["SVG001"]
credit = bank.accounts["CRC001"]

# Deposit and withdraw
print(checking.deposit(500))      # True
print(checking.withdraw(300))     # True
print(savings.add_interest())     # Interest amount

# Transfer money
success, message = bank.transfer_money("CHK001", "SVG001", 200)
print(message)

# Credit card operations
print(credit.withdraw(1000))      # Charge $1000
print(credit.make_payment(500))   # Pay $500

# Account summaries
print("\n=== Account Summary ===")
summary = bank.get_account_summary("CHK001")
if summary:
    account = summary["account"]
    print(f"\nAccount: {account}")
    print(f"Balance: ${summary['balance']:.2f}")
    print(f"Recent transactions:")
    for transaction in summary["recent_transactions"]:
        print(f"  {transaction}")
```

**Why this works:** This banking system demonstrates complex inheritance hierarchies, multiple account types, transaction management, and real-world financial operations.

### Project 3: School Cafeteria Management System 🍕

Create a system for managing cafeteria food, student accounts, and lunch orders.

**Solution:**

```python
from datetime import datetime, timedelta

class FoodItem:
    def __init__(self, name, price, calories, category):
        self.name = name
        self.price = price
        self.calories = calories
        self.category = category  # Main, Side, Drink, Dessert
        self.is_available = True

    def __str__(self):
        return f"{self.name} - ${self.price:.2f} ({self.calories} cal)"

class StudentAccount:
    def __init__(self, student_id, student_name, initial_balance=0):
        self.student_id = student_id
        self.student_name = student_name
        self._balance = initial_balance
        self.orders = []
        self.daily_limit = 15.00  # Max $15 per day

    @property
    def balance(self):
        return self._balance

    def add_money(self, amount):
        if amount > 0:
            self._balance += amount
            return f"Added ${amount:.2f}. New balance: ${self._balance:.2f}"
        return "Invalid amount"

    def can_purchase(self, amount):
        today = datetime.now().date()
        today_spent = sum(order['total'] for order in self.orders
                         if order['date'].date() == today)
        return (self._balance >= amount and
                (today_spent + amount) <= self.daily_limit)

    def purchase(self, food_item, quantity=1):
        if not food_item.is_available:
            return f"❌ {food_item.name} is not available"

        total_cost = food_item.price * quantity
        if not self.can_purchase(total_cost):
            if self._balance < total_cost:
                return "❌ Insufficient funds"
            return f"❌ Would exceed daily limit of ${self.daily_limit}"

        self._balance -= total_cost
        order = {
            'item': food_item.name,
            'quantity': quantity,
            'total': total_cost,
            'date': datetime.now(),
            'calories': food_item.calories * quantity
        }
        self.orders.append(order)
        return f"✅ Purchased {quantity}x {food_item.name} for ${total_cost:.2f}"

class Cafeteria:
    def __init__(self, name):
        self.name = name
        self.menu = []
        self.student_accounts = {}

    def add_food_item(self, food_item):
        self.menu.append(food_item)
        print(f"📋 Added to menu: {food_item}")

    def register_student(self, student_account):
        self.student_accounts[student_account.student_id] = student_account
        print(f"✅ Registered: {student_account.student_name}")

    def get_menu_by_category(self, category):
        return [item for item in self.menu if item.category == category and item.is_available]

    def process_order(self, student_id, item_name, quantity=1):
        student = self.student_accounts.get(student_id)
        if not student:
            return "❌ Student not found"

        # Find food item
        food_item = next((item for item in self.menu if item.name == item_name), None)
        if not food_item:
            return "❌ Item not found"

        return student.purchase(food_item, quantity)

    def get_daily_report(self):
        """Generate today's sales report"""
        today = datetime.now().date()
        total_sales = 0
        item_counts = {}

        for account in self.student_accounts.values():
            for order in account.orders:
                if order['date'].date() == today:
                    total_sales += order['total']
                    item_counts[order['item']] = item_counts.get(order['item'], 0) + order['quantity']

        report = f"\n📊 {self.name} - Daily Report ({today})\n"
        report += "=" * 50
        report += f"\n💰 Total Sales: ${total_sales:.2f}\n"

        if item_counts:
            report += "\n🔥 Top Items:\n"
            for item, count in sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                report += f"  • {item}: {count} sold\n"
        else:
            report += "\n😢 No sales today\n"

        return report

# Test the cafeteria system
print("🍕 Lincoln High School Cafeteria System")
print("=" * 50)

# Setup cafeteria
cafeteria = Cafeteria("Lincoln High Cafeteria")

# Add menu items
menu_items = [
    FoodItem("Cheese Pizza", 3.50, 300, "Main"),
    FoodItem("Chicken Nuggets", 4.00, 350, "Main"),
    FoodItem("Caesar Salad", 3.25, 200, "Main"),
    FoodItem("French Fries", 2.00, 250, "Side"),
    FoodItem("Apple", 1.00, 80, "Side"),
    FoodItem("Chocolate Milk", 1.50, 150, "Drink"),
    FoodItem("Orange Juice", 1.75, 120, "Drink"),
    FoodItem("Cookie", 1.25, 200, "Dessert")
]

for item in menu_items:
    cafeteria.add_food_item(item)

# Register students
students = [
    StudentAccount("S001", "Emma Johnson", 25.00),
    StudentAccount("S002", "Marcus Chen", 15.50),
    StudentAccount("S003", "Zoe Rodriguez", 30.00)
]

for student in students:
    cafeteria.register_student(student)

print("\n🍽️ Student Purchases:")
print(cafeteria.process_order("S001", "Cheese Pizza", 2))
print(cafeteria.process_order("S001", "Chocolate Milk"))
print(cafeteria.process_order("S002", "Chicken Nuggets"))
print(cafeteria.process_order("S003", "Caesar Salad"))
print(cafeteria.process_order("S003", "Cookie"))

print("\n📱 Menu by Category:")
print(f"\n🥪 Main Dishes:")
for item in cafeteria.get_menu_by_category("Main"):
    print(f"  {item}")

print(f"\n🥤 Drinks:")
for item in cafeteria.get_menu_by_category("Drink"):
    print(f"  {item}")

# Daily report
print(cafeteria.get_daily_report())

# Show student balances
print("\n💳 Student Account Balances:")
for student_id, student in cafeteria.student_accounts.items():
    print(f"{student.student_name}: ${student.balance:.2f}")
```

**Why this works:** This cafeteria system shows real money management, daily limits, food categories, and reporting - exactly what schools need! Students learn about financial constraints, item management, and data tracking.

### Project 4: School Bus Transportation System 🚌

Create a system to track school buses, routes, student pickup, and delays.

**Solution:**

```python
from datetime import datetime, time

class BusStop:
    def __init__(self, stop_id, name, address):
        self.stop_id = stop_id
        self.name = name
        self.address = address
        self.students_waiting = []

    def add_student(self, student_name):
        if student_name not in self.students_waiting:
            self.students_waiting.append(student_name)
            return f"✅ {student_name} added to {self.name} stop"
        return f"⚠️ {student_name} already at this stop"

    def remove_student(self, student_name):
        if student_name in self.students_waiting:
            self.students_waiting.remove(student_name)
            return f"✅ {student_name} picked up from {self.name}"
        return f"❌ {student_name} not found at {self.name}"

    def get_waiting_count(self):
        return len(self.students_waiting)

class Bus:
    def __init__(self, bus_number, driver_name, capacity):
        self.bus_number = bus_number
        self.driver_name = driver_name
        self.capacity = capacity
        self.route = []
        self.current_students = []
        self.is_running = False
        self.delay_minutes = 0
        self.current_stop_index = 0

    def assign_route(self, bus_stops):
        self.route = bus_stops
        return f"🚌 Bus {self.bus_number} assigned route with {len(bus_stops)} stops"

    def start_route(self):
        if not self.route:
            return "❌ No route assigned"

        self.is_running = True
        self.current_stop_index = 0
        return f"🚌 Bus {self.bus_number} starting route with {self.driver_name}"

    def arrive_at_stop(self):
        if not self.is_running or self.current_stop_index >= len(self.route):
            return "❌ Bus not running or at end of route"

        current_stop = self.route[self.current_stop_index]
        students_picked_up = 0

        # Pick up waiting students
        for student in current_stop.students_waiting[:]:
            if len(self.current_students) < self.capacity:
                self.current_students.append(student)
                current_stop.remove_student(student)
                students_picked_up += 1

        self.current_stop_index += 1

        status = f"📍 Bus {self.bus_number} at {current_stop.name}\n"
        status += f"   Picked up: {students_picked_up} students\n"
        status += f"   Total onboard: {len(self.current_students)}/{self.capacity}"

        if self.delay_minutes > 0:
            status += f"\n   ⚠️ Running {self.delay_minutes} minutes late"

        return status

    def report_delay(self, minutes):
        self.delay_minutes += minutes
        return f"⏰ Bus {self.bus_number} delayed by {minutes} minutes (Total delay: {self.delay_minutes} min)"

    def arrive_at_school(self):
        if not self.is_running:
            return "❌ Bus not running"

        self.is_running = False
        delivered = len(self.current_students)
        self.current_students = []
        self.current_stop_index = 0

        return f"🏫 Bus {self.bus_number} arrived at school! Delivered {delivered} students"

class SchoolBusSystem:
    def __init__(self, school_name):
        self.school_name = school_name
        self.buses = {}
        self.bus_stops = {}
        self.student_assignments = {}

    def add_bus(self, bus):
        self.buses[bus.bus_number] = bus
        print(f"🚌 Added Bus {bus.bus_number} - Driver: {bus.driver_name}")

    def add_bus_stop(self, bus_stop):
        self.bus_stops[bus_stop.stop_id] = bus_stop
        print(f"📍 Added Bus Stop: {bus_stop.name}")

    def assign_student_to_stop(self, student_name, stop_id):
        if stop_id not in self.bus_stops:
            return "❌ Bus stop not found"

        self.student_assignments[student_name] = stop_id
        return self.bus_stops[stop_id].add_student(student_name)

    def get_morning_route_status(self):
        """Get status of all buses in the morning"""
        status = f"\n🚌 {self.school_name} - Morning Bus Status\n"
        status += "=" * 50

        for bus_number, bus in self.buses.items():
            status += f"\n🚌 Bus {bus_number}:\n"
            status += f"   Driver: {bus.driver_name}\n"
            status += f"   Route: {len(bus.route)} stops\n"
            status += f"   Status: {'Running' if bus.is_running else 'Not started'}\n"

            if bus.route:
                if bus.is_running and bus.current_stop_index < len(bus.route):
                    next_stop = bus.route[bus.current_stop_index]
                    status += f"   Next Stop: {next_stop.name} ({next_stop.get_waiting_count()} waiting)\n"
                elif not bus.is_running:
                    status += f"   First Stop: {bus.route[0].name} ({bus.route[0].get_waiting_count()} waiting)\n"

        return status

    def simulate_morning_route(self):
        """Simulate the morning bus routes"""
        print(f"\n🌅 {self.school_name} - Morning Bus Simulation")
        print("=" * 60)

        for bus_number, bus in self.buses.items():
            if bus.route:
                print(f"\n🚌 Bus {bus_number} Route:")
                print(bus.start_route())

                # Visit each stop
                for _ in range(len(bus.route)):
                    print(bus.arrive_at_stop())

                print(bus.arrive_at_school())

# Test the bus system
print("🚌 Lincoln High School Transportation System")
print("=" * 50)

# Create bus system
transport = SchoolBusSystem("Lincoln High School")

# Create bus stops
stops = [
    BusStop("STOP001", "Sunset Apartments", "123 Sunset Blvd"),
    BusStop("STOP002", "Oak Street", "456 Oak St"),
    BusStop("STOP003", "Pine Community", "789 Pine Dr"),
    BusStop("STOP004", "Maple Heights", "321 Maple Ave")
]

for stop in stops:
    transport.add_bus_stop(stop)

# Create buses
buses = [
    Bus("BUS101", "Mrs. Johnson", 45),
    Bus("BUS202", "Mr. Smith", 40)
]

for bus in buses:
    transport.add_bus(bus)

# Assign routes
route1 = [stops[0], stops[1]]  # Bus 101 route
route2 = [stops[2], stops[3]]  # Bus 202 route

print(buses[0].assign_route(route1))
print(buses[1].assign_route(route2))

# Assign students to stops
students = [
    ("Emma Johnson", "STOP001"),
    ("Marcus Chen", "STOP001"),
    ("Zoe Rodriguez", "STOP002"),
    ("David Brown", "STOP002"),
    ("Alice Smith", "STOP003"),
    ("Bob Wilson", "STOP004")
]

print("\n👨‍🎓 Student Assignments:")
for student, stop_id in students:
    print(transport.assign_student_to_stop(student, stop_id))

# Simulate delays
print(f"\n⏰ Adding delays:")
print(buses[0].report_delay(5))  # 5 minute delay
print(buses[1].report_delay(10))  # 10 minute delay

# Get system status
print(transport.get_morning_route_status())

# Simulate the routes
transport.simulate_morning_route()

print(f"\n📊 Final Summary:")
print(f"Total Students Transported: {len(transport.student_assignments)}")
print(f"Total Bus Stops: {len(transport.bus_stops)}")
print(f"Total Buses: {len(transport.buses)}")
```

**Why this works:** This bus system demonstrates route management, real-time tracking, delay handling, and capacity limits - all real challenges schools face with student transportation!

---

## Interview-Style Questions

### Question 19: Explain the Difference Between Class and Instance Variables

**Answer:**

```python
class Example:
    class_variable = "I belong to the class"  # Class variable

    def __init__(self, instance_variable):
        self.instance_variable = instance_variable  # Instance variable

# Class variable is shared across all instances
obj1 = Example("Object 1")
obj2 = Example("Object 2")

print(Example.class_variable)    # I belong to the class
print(obj1.class_variable)       # I belong to the class
print(obj2.class_variable)       # I belong to the class

# Modifying class variable affects all instances
Example.class_variable = "Modified"
print(obj1.class_variable)       # Modified
print(obj2.class_variable)       # Modified

# Instance variables are unique to each object
print(obj1.instance_variable)    # Object 1
print(obj2.instance_variable)    # Object 2

# Modifying instance variable affects only that instance
obj1.instance_variable = "Modified Object 1"
print(obj1.instance_variable)    # Modified Object 1
print(obj2.instance_variable)    # Object 2 (unchanged)
```

**Key Points:**

- **Class variables** are defined in the class but outside any methods
- **Instance variables** are defined inside `__init__` using `self`
- Class variables are shared; instance variables are unique
- Modify class variables with `ClassName.variable`
- Modify instance variables with `instance.variable`

### Question 20: What is the Purpose of `super()`?

**Answer:**

```python
class Parent:
    def __init__(self, name):
        self.name = name
        print(f"Parent constructor: {name}")

    def speak(self):
        return f"{self.name} says hello"

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # Call parent constructor
        self.age = age
        print(f"Child constructor: {name}, age {age}")

    def speak(self):
        parent_speech = super().speak()  # Call parent method
        return f"{parent_speech} and I'm {self.age} years old"

# Without super(), we'd have to hardcode the parent class
class ChildWithoutSuper(Parent):
    def __init__(self, name, age):
        Parent.__init__(self, name)  # Works but not flexible
        self.age = age

# Test
child = Child("Alice", 10)
print(child.speak())  # Alice says hello and I'm 10 years old

# Key benefits of super():
# 1. Avoids hardcoding parent class name
# 2. Works with multiple inheritance
# 3. Follows method resolution order (MRO)
# 4. More maintainable code
```

### Question 21: How Does Python's Method Resolution Order (MRO) Work?

**Answer:**

```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):  # Inherits from B first, then C
    pass

# Python uses C3 linearization algorithm for MRO
print("MRO for D:", [cls.__name__ for cls in D.__mro__])
# Output: MRO for D: ['D', 'B', 'C', 'A', 'object']

d = D()
print(d.method())  # Calls B's method (first in MRO)

# MRO Rules:
# 1. Child classes come before parent classes
# 2. Order of parent classes is preserved
# 3. No class appears more than once in MRO
# 4. Follows depth-first, left-to-right search
```

### Question 22: Explain the Difference Between Composition and Inheritance

**Answer:**

```python
# Inheritance: "is-a" relationship
class Animal:
    def eat(self):
        return "Eating"

class Dog(Animal):  # Dog IS an Animal
    def bark(self):
        return "Barking"

# Composition: "has-a" relationship
class Engine:
    def start(self):
        return "Engine started"

class Car:
    def __init__(self):
        self.engine = Engine()  # Car HAS an Engine

    def start_car(self):
        return self.engine.start()

# When to use each:
# Inheritance: When subclasses are specialized versions of parent
# Composition: When objects are made up of other objects

# Inheritance example
dog = Dog()
print(dog.eat())   # "Eating" (inherited from Animal)
print(dog.bark())  # "Barking"

# Composition example
car = Car()
print(car.start_car())  # "Engine started"

# Best practice: Favor composition over inheritance
# Why? More flexible, easier to test, less coupling
```

### Question 23: What are Abstract Base Classes and When to Use Them?

**Answer:**

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

    def describe(self):
        return f"Shape with area {self.area():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# Cannot instantiate abstract class
try:
    shape = Shape()  # TypeError
except TypeError as e:
    print(f"Error: {e}")

# Can instantiate concrete implementations
rect = Rectangle(5, 3)
print(rect.area())       # 15
print(rect.perimeter())  # 16
print(rect.describe())   # "Shape with area 15.00"

# Use cases for ABCs:
# 1. Define interfaces that subclasses must implement
# 2. Provide common functionality to subclasses
# 3. Prevent instantiation of base classes
# 4. Create plugin architectures
```

---

## 🎯 School Integration Challenges

### Challenge 1: Complete School Management System

Create a comprehensive system with students, teachers, courses, and grades.

**Solution Overview:**

```python
# This would involve:
# 1. Person base class (for students, teachers, staff)
# 2. Student and Teacher subclasses
# 3. Course and ClassPeriod hierarchy
# 4. GradeBook class managing all grades
# 5. School class coordinating everything

# Key concepts used:
# - Multiple inheritance
# - Polymorphism
# - Encapsulation
# - Abstract classes
# - Complex object relationships
```

### Challenge 2: School Event Management Platform

Build a system for planning and managing school events like dances, fundraisers, and competitions.

**Solution Overview:**

```python
# This would involve:
# 1. Event base class
# 2. Event types (Dance, Sports, Fundraiser, Competition)
# 3. Student participation tracking
# 4. Budget and resource management
# 5. Planning timeline using Observer pattern

# Key concepts used:
# - Inheritance hierarchies
# - Observer pattern
# - Polymorphism
# - Composition
# - State management
```

### Challenge 3: School Transportation System

Create a system for managing school buses, routes, and student transportation.

**Solution Overview:**

```python
# This would involve:
# 1. Vehicle hierarchy (Bus, Van, Car)
# 2. Route and Stop management
# 3. Student passenger tracking
# 4. Driver assignment
# 5. Real-time tracking and notifications

# Key concepts used:
# - Complex inheritance
# - Observer pattern
# - Data validation
# - State tracking
# - Event handling
```

### 🎓 Challenge 4: Virtual Classroom System (Bonus!)

Build a simplified online learning platform for remote education.

**Solution Overview:**

```python
# This would involve:
# 1. User hierarchy (Student, Teacher, Admin)
# 2. Course and Lesson classes
# 3. Assignment and Submission system
# 4. Virtual meeting rooms
# 5. Grade calculation and reporting

# Key concepts used:
# - Multiple inheritance
# - Abstract classes
# - Polymorphism
# - Observer pattern
# - Data persistence concepts
```

---

## 🎓 You're Now an OOP Master! What's Next?

### 🌟 What You Just Accomplished

CONGRATULATIONS! 🎉 You've just learned Python Object-Oriented Programming using examples from YOUR world - not boring bank accounts or employee databases, but REAL school stuff!

**Here's what you now know how to build:**

✅ **Student Management Systems** - Track grades, attendance, clubs
✅ **School Libraries** - Book borrowing, fines, late returns  
✅ **Teacher Tools** - Class management, grading systems
✅ **School Clubs** - Chess teams, drama clubs, science fairs
✅ **School Buildings** - Classrooms, gyms, cafeterias
✅ **Transportation** - Bus routes, student pickup
✅ **Cafeteria Systems** - Food ordering, account balances
✅ **Complex Projects** - Complete school management platforms

### 📚 Your Personal Learning Journey

**🟢 Beginner (You're Here!)**

- You understand what classes and objects are
- You can create simple student tracking systems
- You know how methods work
- You can validate data (like grade ranges)

**🟡 Intermediate (Keep Going!)**

- You master inheritance (like different types of clubs)
- You understand encapsulation (protecting student data)
- You can create property methods
- You build multi-class systems

**🔴 Advanced (Challenge Time!)**

- You use polymorphism (different schools, same interface)
- You implement abstract classes
- You work with design patterns
- You build complete applications

**⚫ Expert (You're a Pro!)**

- You combine multiple OOP concepts
- You create real-world school systems
- You can explain concepts to others
- You're ready for CS courses!

### 🛠️ Practical Next Steps for Students

**1. Make It About YOUR School 🎯**

- Change student names to your classmates
- Add subjects your school offers
- Create clubs that actually exist
- Use real school events (homecoming, prom, etc.)

**2. Build Portfolio Projects 💼**

- Your own school management system
- Attendance tracker for your class
- Club membership manager
- School event planner

**3. Teach Others 👥**

- Explain OOP to classmates using school examples
- Help friends with coding projects
- Start a school coding club
- Create tutorial videos

**4. Challenge Yourself 🏆**

- Add new features to existing projects
- Combine multiple systems (library + cafeteria)
- Create a mobile app interface
- Add database storage

### 💡 OOP Concepts in Plain English (School Edition)

| OOP Concept       | School Equivalent                    | What It Means                                     |
| ----------------- | ------------------------------------ | ------------------------------------------------- |
| **Class**         | Student registration form template   | The blueprint for creating students               |
| **Object**        | Your actual student record           | A specific student with real data                 |
| **Method**        | Student ability (introduce yourself) | What a student can do                             |
| **Inheritance**   | Special student programs             | Honor Roll Student = Student + extra features     |
| **Encapsulation** | Privacy of grades                    | Keeping student data safe                         |
| **Polymorphism**  | Different ways to get to school      | Walk, bus, car - all "go to school" but different |

### 🚀 College & Career Prep

**If you're planning to study Computer Science:**

- These concepts are the foundation of ALL programming
- You'll use them in Java, C++, and other languages too
- Interviewers love school-based examples
- Your portfolio will stand out!

**If you're going into other fields:**

- OOP teaches problem-solving skills
- You'll understand how apps and websites work
- Great for business, design, or any tech-adjacent career
- Shows logical thinking abilities

### 🎯 Fun Challenges to Try This Week

**🥉 Bronze Challenge (30 mins)**

- Add a "lunch_period" attribute to your Student class
- Create a method for students to "eat_lunch()"
- Track which students are in the same lunch period

**🥈 Silver Challenge (1 hour)**

- Create a School class that manages multiple students
- Add a method to calculate average grades by class
- Make it track which students are honor roll eligible

**🥇 Gold Challenge (2-3 hours)**

- Build a complete school event management system
- Include different event types (dance, fundraiser, competition)
- Track student participation and planning tasks

**💎 Platinum Challenge (Weekend project)**

- Create a "School Day Simulator"
- Students move between classes, eat lunch, go to clubs
- Track attendance, grades, and social interactions
- Make it visual with emojis or simple graphics!

### 📱 Real Apps You Could Build

Based on what you learned:

- **Student Portal** - Grades, schedule, announcements
- **Cafeteria App** - Menu, orders, account balance
- **Club Tracker** - Membership, events, competitions
- **Bus Tracker** - Routes, delays, pickup times
- **Event Planner** - Dances, fundraisers, school activities

### 🎉 You're Ready When...

You'll know you've mastered OOP when you can:

- ✅ Explain classes/objects to a friend using school examples
- ✅ Build a simple school system from scratch
- ✅ Debug your code when something breaks
- ✅ Read other people's code and understand it
- ✅ Feel confident starting a new coding project

### 🌈 Final Words of Encouragement

Look, learning to code can feel overwhelming sometimes. But remember:

- **Every expert was once a beginner** 👶 → 👨‍💻
- **Every pro programmer started with "Hello World"** 💻
- **Every complex app started with simple concepts** 🏗️

You've just learned programming concepts using examples from YOUR world. That makes you ahead of the game! Most people learn with abstract examples, but you learned with STUDENTS, TEACHERS, and SCHOOLS - things you actually understand.

**Keep coding, keep experimenting, and most importantly - keep having fun!**

The world needs more programmers who understand real-world problems, and you've got that foundation now. Who knows? Maybe you'll build the next great school management app that helps thousands of students! 🚀

**Your coding journey has just begun...** 💻✨

---

_Made with ❤️ for students who want to learn programming through relatable examples. Now go build something amazing!_
