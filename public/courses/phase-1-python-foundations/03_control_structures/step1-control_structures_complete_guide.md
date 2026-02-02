# 🎮 Python Control Structures & Functions Guide

_Making Smart Decisions and Super Code Reuse - Made for Students!_

## Table of Contents

1. [What Are Control Structures? 🎯](#1-what-are-control-structures-)
2. [Making Decisions with If/Else 📝](#2-making-decisions-with-ifelse-)
3. [Multiple Choices with elif 🎲](#3-multiple-choices-with-elif-)
4. [For Loops: Doing Things Over and Over 🔄](#4-for-loops-doing-things-over-and-over-)
5. [While Loops: Keep Going Until... ⏳](#5-while-loops-keep-going-until-)
6. [Functions: Your Code Superpowers 💪](#6-functions-your-code-superpowers-)
7. [Function Parameters: Feeding Your Functions 🍕](#7-function-parameters-feeding-your-functions-)
8. [Return Values: Getting Stuff Back 🎁](#8-return-values-getting-stuff-back-)
9. [Variable Scope: Where Do Variables Live? 🏠](#9-variable-scope-where-do-variables-live-)
10. [Advanced Function Magic ✨](#10-advanced-function-magic-)
11. [Real School Projects 🏫](#11-real-school-projects-)
12. [Practice Challenges 🎪](#12-practice-challenges-)

---

## 1. What Are Control Structures? 🎯

Think of **control structures** like the rules in your school! Just like how you follow different rules in different situations, Python programs need rules to make smart decisions.

### 🏫 School Examples of Control Structures

#### If/Else = Classroom Rules

```python
# Like school rules: "If it's raining, stay inside; else, go to recess!"
if it's_raining:
    stay_inside()
else:
    go_to_recess()
```

#### Loops = Roll Call

```python
# Like checking every student in attendance
students = ["Emma", "Liam", "Olivia", "Noah", "Ava"]
for student in students:
    call_name(student)
```

#### Functions = Class Subjects

```python
# Like a reusable math lesson
def solve_math_problem():
    print("2 + 2 = 4")

solve_math_problem()  # Can use this again anytime!
```

### 🚀 Why Normal Programs Are Boring

**Without control structures** - Your program is like a robot that only says the same thing:

```python
# Super boring - just repeats the same thing
print("Welcome to school!")
print("Welcome to school!")
print("Welcome to school!")
print("Welcome to school!")
print("Welcome to school!")
```

**With control structures** - Your program is like a smart student assistant:

```python
# Smart program - adapts to different students
student_name = input("Enter your name: ")

if student_name == "Principal":
    print("👑 Welcome, Principal! Please come in!")
elif student_name == "Teacher":
    print("📚 Welcome, Teacher! Ready for class?")
else:
    print(f"🎒 Welcome, {student_name}! Let's learn something awesome!")
```

---

## 2. Making Decisions with If/Else 📝

Imagine you're a **hall monitor** checking students' hall passes. You have rules to follow:

- "If student has a pass, let them through"
- "Else, send them to office"

That's exactly what `if/else` does in Python!

### 🎒 The Basic If Statement

**School Example:** "If you have your homework done, you can go to recess!"

```python
homework_done = True

if homework_done:
    print("✅ You can go to recess!")
    print("🎮 Have fun!")
else:
    print("❌ No recess for you!")
    print("📚 Finish your homework first!")
```

### 🔤 Python If Syntax (The Rules)

```python
if condition:  # The condition to check
    # This code runs ONLY if condition is True
    print("This will run")
    print("So will this")
    # All this is indented (spaced in)
```

### 🎮 Fun School Examples

#### Example 1: Hall Monitor Scanner

```python
student_name = input("Enter your name: ")
has_hall_pass = input("Do you have a hall pass? (yes/no): ").lower() == "yes"

if has_hall_pass:
    print(f"✅ {student_name}, you may pass!")
else:
    print(f"❌ {student_name}, you need a hall pass!")
```

#### Example 2: Lunch Line Priority

```python
student_grade = int(input("What grade are you in? (1-12): "))

if student_grade <= 5:
    print("🍎 You eat first - little ones need fuel!")
elif student_grade <= 8:
    print("🍕 Middle schoolers go next!")
else:
    print("🍔 High schoolers, please wait your turn!")
```

#### Example 3: School Spirit Day

```python
today_is_friday = True
has_spirit_wear = True

if today_is_friday and has_spirit_wear:
    print("🎉 Perfect! Spirit Wear Friday!")
elif today_is_friday:
    print("📢 It's Friday! Wear your school colors!")
else:
    print("📚 Regular school day - let's learn!")
```

### 🤔 Understanding True/False (The Boolean Magic)

Think of `True` as "YES!" and `False` as "NO!" in school:

```python
# These are like saying "YES!" to these conditions
if 1:              # Any non-zero number = YES!
if "homework":     # Any non-empty text = YES!
if [10, 20, 30]:   # Any list with stuff = YES!
if True:           # Obviously YES!

# These are like saying "NO!" to these conditions
if 0:              # Zero = NO!
if "":             # Empty text = NO!
if []:             # Empty list = NO!
if False:          # Obviously NO!
```

### 🏆 Example: Grades and Consequences

```python
grade = float(input("What grade did you get? "))

if grade >= 90:
    print("🎉 AMAZING! You got an A!")
    print("⭐ You're on the honor roll!")
elif grade >= 80:
    print("👍 GREAT JOB! You got a B!")
    print("👏 Keep up the good work!")
elif grade >= 70:
    print("👌 GOOD EFFORT! You got a C!")
    print("📈 You can do even better!")
elif grade >= 60:
    print("⚠️ You passed with a D")
    print("📚 Need to study more!")
else:
    print("❌ You failed with an F")
    print("🆘 Time for extra help!")
```

---

## 3. Multiple Choices with elif 🎲

Think of `elif` like a **school lunch menu**:

- First option: "If you want pizza..."
- Second option: "Else if you want salad..."
- Third option: "Else if you want soup..."
- Final option: "Else (no matter what) you get a sandwich!"

### 🍕 Using elif for Multiple Conditions

**School Scenario:** Choosing your after-school activity!

```python
free_period = input("What do you want to do? (art/music/sports/computer): ").lower()

if free_period == "art":
    print("🎨 Let's paint some masterpieces!")
elif free_period == "music":
    print("🎵 Time to make beautiful music!")
elif free_period == "sports":
    print("⚽ Let's play some games!")
elif free_period == "computer":
    print("💻 Coding time - let's build something cool!")
else:
    print("🏃 You can just relax and hang out!")
```

### ⚡ Why elif is Smarter Than Multiple ifs

```python
# ❌ WRONG - Using multiple ifs (bad idea!)
score = 95

if score >= 90:
    grade = "A"
if score >= 80:    # This ALSO runs! (oops!)
    grade = "B"     # Grade becomes "B" even with a 95!

# ✅ CORRECT - Using elif (smart!)
if score >= 90:
    grade = "A"
elif score >= 80:   # Only runs if first condition is False
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
```

### 🤝 Combining Conditions with and/or

**AND (`and`) = ALL conditions must be true**

```python
# Must be BOTH on time AND have homework
on_time = input("Are you on time? (yes/no): ").lower() == "yes"
homework_done = input("Homework complete? (yes/no): ").lower() == "yes"

if on_time and homework_done:
    print("⭐ Perfect! You're ready for class!")
else:
    print("📝 You need to meet both requirements")
```

**OR (`or`) = AT LEAST one condition must be true**

```python
# Can go to recess if it's nice weather OR if you finished work
nice_weather = input("Is it nice outside? (yes/no): ").lower() == "yes"
work_finished = input("Is your work done? (yes/no): ").lower() == "yes"

if nice_weather or work_finished:
    print("🏃 Go enjoy recess!")
else:
    print("📚 Keep working!")
```

### 🏫 Nested If Statements (Checking Inside Checks)

Think of this like **security at school**:

1. First check: "Are you a student?"
2. If yes, then check: "Do you have your ID?"
3. If yes, then check: "Are you on the visitor list?"

```python
is_student = input("Are you a student here? (yes/no): ").lower() == "yes"

if is_student:
    has_id = input("Do you have your student ID? (yes/no): ").lower() == "yes"
    if has_id:
        print("🎉 Welcome to school!")
    else:
        print("📋 Please get your ID from the office")
else:
    is_visitor = input("Are you a visitor? (yes/no): ").lower() == "yes"
    if is_visitor:
        print("👋 Welcome! Please check in at the office")
    else:
        print("🚫 Sorry, you can't enter")
```

### 🎪 Real School Examples

#### Example 1: Bus Route Selector

```python
grade_level = int(input("What grade are you in? "))
distance = float(input("How far do you live? (miles): "))

if grade_level <= 2:
    print("🚌 You get bus service no matter what!")
elif grade_level <= 5:
    if distance >= 0.5:
        print("🚌 You qualify for bus service!")
    else:
        print("🚶 You live too close - walk or get a ride!")
else:  # grades 6-12
    if distance >= 1.0:
        print("🚌 You qualify for bus service!")
    else:
        print("🚶 You live too close - walk or get a ride!")
```

#### Example 2: Lunch Payment System

```python
has_money = float(input("How much money do you have? $"))
has_lunch_ticket = input("Do you have a lunch ticket? (yes/no): ").lower() == "yes"
brought_lunch = input("Did you bring lunch from home? (yes/no): ").lower() == "yes"

if brought_lunch:
    print("🍱 Enjoy your homemade lunch!")
elif has_lunch_ticket:
    print("🎫 Use your lunch ticket!")
elif has_money >= 5:
    print("💰 Buy a school lunch!")
else:
    print("🍎 No lunch options available - talk to the school counselor!")
```

---

## 4. For Loops: Doing Things Over and Over 🔄

Think of `for` loops like **taking attendance** in your class:

- Without loops: Call each name manually (so tedious!)
- With loops: Let the computer do the repetitive work!

### 📚 Why For Loops Are Awesome

**Without loops** - The boring way (like calling each student individually):

```python
# This would take FOREVER with 30 students!
print("Emma is here")
print("Liam is here")
print("Olivia is here")
print("Noah is here")
# ... 26 more students to go!
```

**With loops** - The smart way (let the computer work):

```python
# Much better! Computer handles all the repetition!
students = ["Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia", "James"]

for student in students:
    print(f"✅ {student} is here!")
```

### 🎯 Basic For Loop Syntax

```python
for item in collection:  # Go through each thing in the collection
    # Do something with each item
    print(item)
```

### 🎲 Cool School Examples

#### Example 1: Taking Attendance

```python
class_roster = ["Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia"]

print("📋 Taking attendance...")
for student in class_roster:
    print(f"✅ {student} - Here!")

print("🎉 Attendance complete!")
```

#### Example 2: Grade Checker

```python
test_scores = [85, 92, 78, 96, 88, 91, 89]

print("📊 Checking everyone's test scores:")
for score in test_scores:
    if score >= 90:
        print(f"🎉 {score} - Excellent!")
    elif score >= 80:
        print(f"👍 {score} - Good job!")
    else:
        print(f"📚 {score} - Keep studying!")
```

### 🔢 Using range() - Counting Numbers

**range()** is like counting in math class:

```python
# Count from 1 to 5 (like days of school week + weekend!)
for day in range(1, 6):
    print(f"📅 School Day {day}")

print("🎉 Weekend! Time to rest!")
```

**Understanding range():**

- `range(5)` → 0, 1, 2, 3, 4 (starts at 0, stops before 5)
- `range(1, 6)` → 1, 2, 3, 4, 5 (starts at 1, stops before 6)
- `range(2, 11, 2)` → 2, 4, 6, 8, 10 (count by 2s!)

#### Practice: Times Tables

```python
# Print the 5 times table
for number in range(1, 11):
    result = 5 * number
    print(f"5 × {number} = {result}")
```

### 📝 Looping Through Text

**Like spelling out words letter by letter:**

```python
school_name = "ELEMENTARY"

print("🅰️ Spelling your school:")
for letter in school_name:
    print(f"   {letter}")

print("✅ Complete spelling!")
```

### 📋 Working with Lists

**Like checking your backpack items:**

```python
backpack_items = ["📚 textbook", "✏️ pencils", "🍎 apple", "💧 water bottle", "🎒 lunch box"]

print("🎒 Checking your backpack:")
for item in backpack_items:
    print(f"   ✓ {item}")

print("🎉 Ready for school!")
```

### 👥 Using enumerate() - Getting Position Numbers

When you need to know **which number** item you're on:

```python
lunch_options = ["🍕 Pizza", "🍎 Apple", "🥪 Sandwich", "🍌 Banana"]

print("🍽️ Today's lunch menu:")
for position, food in enumerate(lunch_options, 1):  # Start counting from 1
    print(f"{position}. {food}")
```

**Output:**

```
🍽️ Today's lunch menu:
1. 🍕 Pizza
2. 🍎 Apple
3. 🥪 Sandwich
4. 🍌 Banana
```

### 🎯 Working with Multiple Values

**Like keeping track of student names AND their grades:**

```python
class_data = [
    ("Emma", 92),
    ("Liam", 88),
    ("Olivia", 95),
    ("Noah", 85),
    ("Ava", 90)
]

print("📊 Class Report:")
for student_name, grade in class_data:
    if grade >= 90:
        status = "⭐ Honor Roll"
    elif grade >= 80:
        status = "👍 Good Job"
    else:
        status = "📚 Keep Trying"

    print(f"{student_name}: {grade}% - {status}")
```

### 🏆 Fun Challenges

#### Challenge 1: School Supply Counter

```python
school_supplies = ["pencils", "erasers", "notebooks", "folders", "markers"]

print("📝 Counting school supplies:")
for i, supply in enumerate(school_supplies, 1):
    print(f"{i}. {supply.capitalize()} ✅")
```

#### Challenge 2: Recess Timer

```python
print("⏰ Recess Timer:")
for minute in range(10, 0, -1):  # Count down from 10
    print(f"⏰ {minute} minutes left!")

print("🎉 Recess is over! Time to go inside!")
```

---

## 5. While Loops: Keep Going Until... ⏳

Think of `while` loops like being a **substitute teacher**:

- "Keep watching the class until the real teacher comes back"
- "Keep helping students until everyone's done"
- "Keep playing until recess is over"

### 🎮 Why While Loops Are Useful

**School Scenario:** Keep practicing math until you get 5 right!

```python
correct_answers = 0

while correct_answers < 5:  # Keep going until we get 5 right
    print(f"Current score: {correct_answers}/5")
    answer = input("What is 2 + 2? ")

    if answer == "4":
        correct_answers += 1
        print("🎉 Correct!")
    else:
        print("❌ Try again!")

print("🎊 You passed the practice test!")
```

### ⚡ Basic While Loop Syntax

```python
while condition:  # Keep doing this
    # Code that repeats
    # MUST eventually change the condition
    # or it will run FOREVER! (like a broken record!)
```

### 🏫 Fun School Examples

#### Example 1: Hall Pass Return System

```python
# Keep checking for hall passes until all are returned
passes_out = 5  # 5 hall passes given out today

while passes_out > 0:
    print(f"📋 Hall passes still out: {passes_out}")
    returned = input("Did someone return a hall pass? (yes/no): ").lower()

    if returned == "yes":
        passes_out -= 1
        print("✅ Pass returned!")
    else:
        print("⏳ Still waiting...")

print("🎉 All hall passes are back!")
```

#### Example 2: Snack Machine Money Counter

```python
# Keep accepting money until we have enough for a snack
snack_price = 2.50
money_inserted = 0

while money_inserted < snack_price:
    coin = float(input(f"Insert coin ($0.25, $0.50, $1.00): $"))
    money_inserted += coin
    print(f"💰 Total: ${money_inserted:.2f}")

    if money_inserted >= snack_price:
        change = money_inserted - snack_price
        print(f"🍎 Enjoy your snack! Change: ${change:.2f}")
```

#### Example 3: Number Guessing Game (School Edition!)

```python
import random

# Teacher picks a number between 1-10
secret_number = random.randint(1, 10)
attempts = 0

print("🎯 Math Challenge: Guess the Number!")
print("I'm thinking of a number between 1 and 10...")

while True:  # Keep guessing until correct
    guess = int(input("Your guess: "))
    attempts += 1

    if guess == secret_number:
        print(f"🎉 Excellent! You got it in {attempts} tries!")
        break  # Exit the loop when correct
    elif guess < secret_number:
        print("📈 Too low! Think bigger!")
    else:
        print("📉 Too high! Think smaller!")
```

#### Example 4: Cafeteria Line System

```python
students_in_line = 15

print("🍽️ Cafeteria is open!")
while students_in_line > 0:
    print(f"👥 Students waiting: {students_in_line}")
    served = input("How many students got served? ")

    try:
        students_served = int(served)
        students_in_line -= students_served
        if students_in_line < 0:
            students_in_line = 0
        print(f"✅ Served {students_served} students!")
    except:
        print("❌ Please enter a number!")

print("🎉 Cafeteria closed!")
```

### 🚨 Loop Control - Managing Your Loops

#### `break` = "Stop Everything NOW!"

```python
# Like stopping class when the fire alarm rings
for i in range(10):
    if i == 5:
        print("🔥 FIRE ALARM! Stop everything!")
        break  # Exit loop immediately
    print(f"Class period {i+1}")
```

#### `continue` = "Skip This One, Keep Going"

```python
# Skip students who are absent
students_present = ["Emma", "Liam", "ABSENT", "Olivia", "ABSENT", "Noah"]

for student in students_present:
    if student == "ABSENT":
        continue  # Skip this iteration, move to next student
    print(f"✅ {student} is here!")
```

#### `pass` = "Do Nothing for Now"

```python
# Planning schedule but haven't decided lunch time yet
schedule = ["Math", "Science", "LUNCH", "History", "Art"]

for activity in schedule:
    if activity == "LUNCH":
        pass  # Will decide lunch time later
    print(f"🕐 Next: {activity}")
```

### 🎪 Real School Applications

#### Application 1: Recess Timer

```python
recess_time = 15  # minutes

while recess_time > 0:
    print(f"⏰ {recess_time} minutes of recess left!")
    recess_time -= 1

print("🎒 Time to go back inside!")
```

#### Application 2: Homework Tracker

```python
homework_items = [
    "Math worksheet",
    "Read chapter 5",
    "Science experiment",
    "History timeline"
]
completed = 0

print("📚 Starting homework session...")
while completed < len(homework_items):
    print(f"\nTask {completed + 1}: {homework_items[completed]}")
    done = input("Completed? (yes/no): ").lower()

    if done == "yes":
        completed += 1
        print("✅ Great job!")
    else:
        print("📝 Keep working on it!")

print("🎉 All homework completed!")
```

#### Application 3: Library Book Return System

```python
books_to_return = 3

print("📚 Library Book Return System")
while books_to_return > 0:
    print(f"📖 Books to return: {books_to_return}")

    book_title = input("Enter book title (or 'done' to quit): ")
    if book_title.lower() == "done":
        break

    books_to_return -= 1
    print(f"✅ '{book_title}' returned!")

    if books_to_return > 0:
        remaining = "book" if books_to_return == 1 else "books"
        print(f"📚 {books_to_return} {remaining} remaining")

print("🎉 Library visit complete!")
```

    print("1. View balance")
    print("2. Deposit money")
    print("3. Withdraw money")
    print("4. Exit")

    choice = input("Choose option (1-4): ")

    if choice == "1":
        print("Your balance: $1000")
    elif choice == "2":
        amount = float(input("Enter deposit amount: $"))
        print(f"Deposited ${amount}")
    elif choice == "3":
        amount = float(input("Enter withdrawal amount: $"))
        print(f"Withdrew ${amount}")
    elif choice == "4":
        print("Goodbye!")
        break  # Exit menu loop
    else:
        print("Invalid option, try again")

````

### Loop Control Statements

#### `break` - Exit Loop Completely
```python
for i in range(10):
    if i == 5:
        break  # Exit loop when i is 5
    print(i)

# Output: 0 1 2 3 4
````

#### `continue` - Skip to Next Iteration

```python
for i in range(10):
    if i % 2 == 0:  # If even number
        continue   # Skip this iteration
    print(i)  # Only odd numbers: 1, 3, 5, 7, 9
```

#### `pass` - Do Nothing (Placeholder)

```python
for i in range(5):
    if i == 2:
        pass  # Do nothing when i is 2
    print(i)  # Still prints: 0 1 2 3 4
```

---

## 6. Functions: Your Code Superpowers 💪

Think of functions like **teaching the same lesson to different classes**:

- You write the lesson once (define the function)
- You can teach it to any class (call the function with different students)
- You don't have to rewrite the whole lesson every time!

### 🍕 School Examples of Functions

**Without functions** - Like writing the same homework problem 5 times:

```python
# Calculate grade percentage for Math
math_correct = 8
math_total = 10
math_percentage = (math_correct / math_total) * 100
print(f"Math: {math_percentage}%")

# Calculate grade percentage for Science
science_correct = 7
science_total = 10
science_percentage = (science_correct / science_total) * 100
print(f"Science: {science_percentage}%")

# Calculate grade percentage for History
history_correct = 9
history_total = 10
history_percentage = (history_correct / history_total) * 100
print(f"History: {history_percentage}%")
```

**With functions** - Like having a reusable grading calculator:

```python
def calculate_grade_percentage(correct, total):
    """Calculate and return the grade percentage"""
    percentage = (correct / total) * 100
    return percentage

# Use the function for different subjects
math_percentage = calculate_grade_percentage(8, 10)
science_percentage = calculate_grade_percentage(7, 10)
history_percentage = calculate_grade_percentage(9, 10)

print(f"Math: {math_percentage}%")
print(f"Science: {science_percentage}%")
print(f"History: {history_percentage}%")
```

### 🎯 The Function Recipe Format

```python
def function_name(parameters):  # The "recipe name" and "ingredients"
    """What this function does (like recipe description)"""
    # Steps to follow (the actual cooking)
    result = do_something_with(parameters)
    return result  # What you give back (the finished dish!)
```

### 🔧 Function Parts (Like School Supplies!)

```python
def calculate_test_average(score1, score2, score3):  # 1. Function name + ingredients
    """Calculate the average of three test scores"""  # 2. Description (what it does)
    total = score1 + score2 + score3  # 3. Work it does
    average = total / 3  # 4. Calculate the result
    return average  # 5. Give back the answer
```

**The 5 Important Parts:**

1. **`def`**: Like saying "I'm going to make something"
2. **Function name**: What you're making (be descriptive!)
3. **`()`**: Always needed (like parentheses in math)
4. **`:`**: Always needed after the parentheses
5. **`return`**: What you give back (optional, but super useful!)

### 🏫 School Function Examples

#### Example 1: Hall Monitor Checker

```python
def check_hall_pass(student_name):
    """Check if student has hall pass"""
    print(f"🔍 Checking {student_name}'s hall pass...")
    print("✅ Pass is valid!")
    print("🚶 You may proceed to your destination")

# Call the function for different students
check_hall_pass("Emma")
check_hall_pass("Liam")
check_hall_pass("Olivia")
```

#### Example 2: Grade Calculator

```python
def calculate_letter_grade(score):
    """Convert numerical score to letter grade"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

# Use it for different students
emma_grade = calculate_letter_grade(95)
liam_grade = calculate_letter_grade(87)
olivia_grade = calculate_letter_grade(73)

print(f"Emma: {emma_grade}")
print(f"Liam: {liam_grade}")
print(f"Olivia: {olivia_grade}")
```

#### Example 3: Lunch Money Calculator

```python
def calculate_lunch_total(main_dessert, drink):
    """Calculate total lunch cost"""
    prices = {
        "pizza": 3.50,
        "sandwich": 3.00,
        "salad": 2.50,
        "cookie": 1.00,
        "fruit": 0.75,
        "juice": 1.25,
        "milk": 1.00,
        "water": 0.50
    }

    total = prices[main_dessert] + prices[drink]
    return total

# Calculate different lunches
emma_lunch = calculate_lunch_total("pizza", "juice")
liam_lunch = calculate_lunch_total("salad", "milk")
olivia_lunch = calculate_lunch_total("sandwich", "water")

print(f"Emma's lunch: ${emma_lunch}")
print(f"Liam's lunch: ${liam_lunch}")
print(f"Olivia's lunch: ${olivia_lunch}")
```

### 🎯 Function Naming Rules (Like Good Homework Titles!)

```python
# ❌ Bad names (unclear what they do)
def f(x):
    return x * 2

def calc(a, b):
    return a + b

# ✅ Good names (clear what they do)
def double_homework_score(score):
    return score * 2

def calculate_classroom_temperature(celsius, fahrenheit):
    return celsius, fahrenheit

def check_if_student_passed(test_score):
    return test_score >= 60
```

    Args:
        original_price (float): The original price before discount
        discount_percent (float): The discount percentage (0-100)

    Returns:
        float: The final price after discount
    """
    discount_amount = original_price * (discount_percent / 100)
    final_price = original_price - discount_amount
    return final_price

````

---

## 7. Function Parameters: Feeding Your Functions 🍕

Think of **parameters** like the **ingredients** you give to a recipe:
- You can give specific ingredients (arguments)
- You can say which ingredient goes where
- Some ingredients have default values (like always having bread!)

### 🥪 Positional Arguments - Order Matters!

**Like following a lunch recipe exactly:**
```python
def make_school_lunch(main, side, drink):
    """Make a complete school lunch"""
    print(f"🍽️ Your lunch: {main} + {side} + {drink}")

# Arguments must be in the right order!
make_school_lunch("pizza", "salad", "juice")
# Output: 🍽️ Your lunch: pizza + salad + juice

# If you change the order, you get wrong lunch!
make_school_lunch("juice", "pizza", "salad")
# Output: 🍽️ Your lunch: juice + pizza + salad (weird!)
````

### 🏷️ Keyword Arguments - Specify Exactly!

**Like writing a detailed lunch order:**

```python
def make_school_lunch(main, side, drink):
    """Make a complete school lunch"""
    print(f"🍽️ Your lunch: {main} + {side} + {drink}")

# Tell Python exactly which ingredient goes where!
make_school_lunch(drink="milk", main="sandwich", side="fruit")
# Output: 🍽️ Your lunch: sandwich + fruit + milk
```

### 🎯 Default Parameters - Smart Defaults!

**Like having a favorite lunch that you customize:**

```python
def order_school_meal(main, side="apple", drink="milk"):
    """Order lunch with smart defaults"""
    print(f"🍽️ Order: {main} + {side} + {drink}")

# Use default side and drink
order_school_meal("pizza")
# Output: 🍽️ Order: pizza + apple + milk

# Customize only what you want
order_school_meal("salad", drink="juice")
# Output: 🍽️ Order: salad + apple + juice

# Customize everything
order_school_meal("sandwich", "cookie", "water")
# Output: 🍽️ Order: sandwich + cookie + water
```

### 🎒 School Examples with Parameters

#### Example 1: Student Information System

```python
def create_student_card(name, grade, favorite_subject="Unknown"):
    """Create a student information card"""
    print(f"📋 Student: {name}")
    print(f"📚 Grade: {grade}")
    print(f"⭐ Favorite Subject: {favorite_subject}")
    print("-" * 20)

# Different ways to call this function
create_student_card("Emma", 5)
create_student_card("Liam", 6, "Math")
create_student_card(grade=4, name="Olivia")
create_student_card(name="Noah", favorite_subject="Science", grade=5)
```

#### Example 2: Homework Grader

```python
def grade_homework(student_name, subject, score, total_points):
    """Grade homework and give feedback"""
    percentage = (score / total_points) * 100

    print(f"📝 {student_name}'s {subject} Homework")
    print(f"Score: {score}/{total_points} ({percentage:.1f}%)")

    if percentage >= 90:
        feedback = "🌟 Excellent work!"
    elif percentage >= 80:
        feedback = "👍 Great job!"
    elif percentage >= 70:
        feedback = "👌 Good effort!"
    else:
        feedback = "📚 Keep practicing!"

    print(f"Feedback: {feedback}")

# Grade different homeworks
grade_homework("Emma", "Math", 18, 20)
grade_homework("Liam", "Science", 15, 20, total_points=20)
grade_homework(subject="History", student_name="Olivia", score=19, total_points=20)
```

```python
def order_pizza(size, toppings="cheese"):
    """Order a pizza with optional toppings"""
    print(f"Ordering {size} pizza with {toppings}")

# Use default toppings
order_pizza("large")              # Large pizza with cheese
order_pizza("medium", "pepperoni") # Medium pizza with pepperoni
```

### \*args - Variable Number of Arguments

```python
def add_numbers(*numbers):
    """Add any number of arguments together"""
    total = 0
    for num in numbers:
        total += num
    return total

# Can pass any number of arguments
result1 = add_numbers(1, 2, 3)           # 6
result2 = add_numbers(10, 20, 30, 40)    # 100
result3 = add_numbers(5)                 # 5
```

### \*\*kwargs - Keyword Arguments Dictionary

```python
def create_profile(**info):
    """Create a profile with any keyword arguments"""
    for key, value in info.items():
        print(f"{key}: {value}")

create_profile(name="Alice", age=25, city="NYC")
# Output:
# name: Alice
# age: 25
# city: NYC
```

### Parameter Order Rules

```python
# Correct order: positional, *args, default, **kwargs
def function(pos1, pos2, *args, default1="value", **kwargs):
    pass

# Examples
function("a", "b", "c", "d", default1="custom", extra="info")
```

---

## 8. Return Values: Getting Results Back

### Why Return Values?

**Real Scenario:** Like a vending machine:

- You put in money (arguments)
- You get a snack back (return value)
- The machine doesn't just print "here's your snack" - it actually gives you the snack

### Functions Without Return Values

```python
def greet(name):
    """Just displays a greeting - doesn't return anything"""
    print(f"Hello, {name}!")

result = greet("Alice")  # result is None
print(result)            # Output: None
```

### Functions With Return Values

```python
def calculate_area(length, width):
    """Calculate area and return the result"""
    area = length * width
    return area  # Give back the calculated area

# Store the returned value
room_area = calculate_area(10, 8)
print(f"Room area: {room_area} square feet")  # Room area: 80 square feet

# Use the returned value directly
if calculate_area(5, 4) > 20:
    print("Large room")
else:
    print("Small room")
```

### Returning Multiple Values

```python
def get_name_and_age():
    """Return both name and age"""
    name = "Alice"
    age = 25
    return name, age  # Returns a tuple

# Unpack the returned values
person_name, person_age = get_name_and_age()
print(f"{person_name} is {person_age} years old")
```

### Common Return Patterns

#### Boolean Returns

```python
def is_even(number):
    """Check if a number is even"""
    return number % 2 == 0  # Returns True or False

if is_even(10):
    print("10 is even")
```

#### Status Returns

```python
def withdraw_money(balance, amount):
    """Try to withdraw money and return success status"""
    if amount <= balance:
        return True, balance - amount
    else:
        return False, balance

success, new_balance = withdraw_money(1000, 500)
if success:
    print(f"Withdrawal successful. New balance: ${new_balance}")
else:
    print("Insufficient funds")
```

---

## 9. Variable Scope: Where Variables Live

### Global vs Local Scope

**Real Scenario:** Like different rooms in a house:

- **Global**: Kitchen - everyone can access it
- **Local**: Your bedroom - only you can access it

```python
# Global variable
global_message = "Hello from outside!"

def show_message():
    # Local variable
    local_message = "Hello from inside!"
    print(local_message)      # This works
    print(global_message)     # This also works

show_message()
print(global_message)         # This works
# print(local_message)       # This would cause an error!
```

### Local vs Global Naming

```python
# Global variable
count = 0

def increment_counter():
    # Local variable with same name
    count = 10  # This is a NEW local variable
    count += 1
    print(f"Inside function: {count}")  # 11

increment_counter()
print(f"Outside function: {count}")  # 0 (global unchanged)
```

### Modifying Global Variables

```python
score = 0

def add_points(points):
    global score  # Declare we want to use global variable
    score += points
    print(f"Current score: {score}")

add_points(5)    # Current score: 5
add_points(10)   # Current score: 15
print(f"Final score: {score}")  # Final score: 15
```

### Nonlocal Variables (Nested Functions)

```python
def outer_function():
    outer_var = "outer"

    def inner_function():
        nonlocal outer_var  # Use variable from outer function
        outer_var = "modified"
        print(f"Inner: {outer_var}")

    print(f"Before: {outer_var}")  # Before: outer
    inner_function()
    print(f"After: {outer_var}")   # After: modified

outer_function()
```

### Scope Best Practices

1. **Prefer local over global**
2. **Use parameters instead of global variables**
3. **Avoid modifying globals inside functions**
4. **Keep functions small and focused**

```python
# GOOD - Uses parameters
def calculate_total(prices):
    return sum(prices)

# BAD - Depends on global variable
def calculate_total():
    return sum(global_price_list)
```

---

## 10. Advanced Function Concepts

### Lambda Functions (Anonymous Functions)

**When to use:** Quick, simple operations without needing a named function

```python
# Traditional function
def square(x):
    return x ** 2

# Lambda function
square_lambda = lambda x: x ** 2

# Usage
result1 = square(5)           # 25
result2 = square_lambda(5)    # 25
```

**Common lambda uses:**

```python
# Sorting with custom key
names = ["alice", "Bob", "charlie"]
names.sort(key=lambda name: name.upper())
print(names)  # ['alice', 'Bob', 'charlie'] - sorted case-insensitively

# Filter with lambda
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6]
```

### Recursive Functions

**Real Scenario:** Like Russian dolls - each doll contains a smaller version of itself

```python
def factorial(n):
    """Calculate factorial of n (n!)"""
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

# 5! = 5 × 4 × 3 × 2 × 1 = 120
print(factorial(5))  # 120
```

**How recursion works:**

```
factorial(5) = 5 × factorial(4)
factorial(4) = 4 × factorial(3)
factorial(3) = 3 × factorial(2)
factorial(2) = 2 × factorial(1)
factorial(1) = 1
```

### Function Decorators

**Purpose:** Modify function behavior without changing its code

```python
def timer_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def slow_function():
    import time
    time.sleep(1)
    return "Done!"

result = slow_function()
# Output: slow_function took 1.0000 seconds
```

---

## 11. Real-World Applications

### Application 1: Bank Account System

```python
def create_account(name, initial_balance=0):
    """Create a bank account"""
    return {
        'name': name,
        'balance': initial_balance,
        'transactions': []
    }

def deposit(account, amount):
    """Deposit money into account"""
    if amount > 0:
        account['balance'] += amount
        account['transactions'].append(f"Deposit: +${amount}")
        return True
    return False

def withdraw(account, amount):
    """Withdraw money from account"""
    if 0 < amount <= account['balance']:
        account['balance'] -= amount
        account['transactions'].append(f"Withdrawal: -${amount}")
        return True
    return False

def check_balance(account):
    """Display account balance"""
    print(f"{account['name']}'s balance: ${account['balance']}")

# Use the bank system
account1 = create_account("Alice", 1000)
deposit(account1, 500)
withdraw(account1, 200)
check_balance(account1)
```

### Application 2: Student Grade Calculator

```python
def calculate_letter_grade(score):
    """Convert numerical score to letter grade"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def calculate_gpa(letter_grade):
    """Convert letter grade to GPA points"""
    grade_points = {
        "A": 4.0,
        "B": 3.0,
        "C": 2.0,
        "D": 1.0,
        "F": 0.0
    }
    return grade_points.get(letter_grade, 0.0)

def process_student_grades():
    """Process multiple student grades"""
    students = []

    while True:
        name = input("Enter student name (or 'quit'): ")
        if name.lower() == 'quit':
            break

        score = float(input(f"Enter {name}'s score: "))
        letter_grade = calculate_letter_grade(score)
        gpa = calculate_gpa(letter_grade)

        students.append({
            'name': name,
            'score': score,
            'grade': letter_grade,
            'gpa': gpa
        })

        print(f"{name}: {letter_grade} (GPA: {gpa})")

    # Calculate class average
    if students:
        avg_gpa = sum(student['gpa'] for student in students) / len(students)
        print(f"\nClass Average GPA: {avg_gpa:.2f}")

process_student_grades()
```

### Application 3: Simple Game Framework

```python
class Game:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.game_over = False

    def display_status(self):
        """Display current game status"""
        print(f"Score: {self.score}")
        print(f"Level: {self.level}")
        print("-" * 20)

    def play_turn(self):
        """Play one turn of the game"""
        import random

        # Simulate game action
        action = random.choice(['attack', 'defend', 'heal'])

        if action == 'attack':
            points = random.randint(10, 30)
            self.score += points
            print(f"Attacked! Gained {points} points")
        elif action == 'defend':
            print("Defended - no points gained")
        elif action == 'heal':
            self.level += 1
            print("Healed! Level up!")

    def check_win_condition(self):
        """Check if game should continue"""
        if self.score >= 100:
            self.game_over = True
            print("🎉 You won! Score reached 100!")

        if self.level >= 5:
            self.game_over = True
            print("🏆 You won! Reached level 5!")

def play_game():
    """Main game loop"""
    game = Game()

    while not game.game_over:
        game.display_status()
        input("Press Enter to continue...")
        game.play_turn()
        game.check_win_condition()

    print("Game Over!")
    print(f"Final Score: {game.score}")
    print(f"Final Level: {game.level}")

# Start the game
# play_game()
```

---

## 12. Practice Challenges 🎪

Get ready to test your Python superpowers! These challenges use real school scenarios to help you practice.

### 🎯 Challenge 1: Math Test Analyzer

```python
def analyze_test_score(score):
    """Analyze a test score and give detailed feedback"""
    result = {
        'score': score,
        'is_perfect': score == 100,
        'is_excellent': 90 <= score < 100,
        'is_good': 80 <= score < 90,
        'is_passing': score >= 60,
        'needs_improvement': score < 60,
        'grade_letter': 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 60 else 'F'
    }
    return result

# Test with your classmates' scores
class_scores = [95, 87, 76, 45, 100, 68, 92, 55]

print("📊 Class Test Results:")
for score in class_scores:
    analysis = analyze_test_score(score)
    if analysis['is_perfect']:
        feedback = "🎉 PERFECT SCORE!"
    elif analysis['is_excellent']:
        feedback = "🌟 Amazing work!"
    elif analysis['is_good']:
        feedback = "👍 Great job!"
    elif analysis['is_passing']:
        feedback = "✅ You passed!"
    else:
        feedback = "📚 Time for extra help!"

    print(f"Score {score}: {feedback}")
```

### 🏆 Challenge 2: School Lunch营养 Calculator

```python
def calculate_lunch_nutrition(main_course, side, drink):
    """Calculate lunch nutrition and health score"""
    # Nutrition values (simplified)
    nutrition = {
        'pizza': {'calories': 285, 'protein': 12, 'vegetables': 1},
        'salad': {'calories': 150, 'protein': 4, 'vegetables': 3},
        'sandwich': {'calories': 320, 'protein': 18, 'vegetables': 2},
        'soup': {'calories': 120, 'protein': 6, 'vegetables': 2}
    }

    sides = {
        'fruit': {'calories': 80, 'vegetables': 1},
        'chips': {'calories': 160, 'vegetables': 0},
        'vegetables': {'calories': 50, 'vegetables': 2}
    }

    drinks = {
        'milk': {'calories': 120, 'protein': 8},
        'juice': {'calories': 110, 'protein': 0},
        'water': {'calories': 0, 'protein': 0}
    }

    # Calculate totals
    total_calories = nutrition[main_course]['calories'] + sides[side]['calories'] + drinks[drink]['calories']
    total_protein = nutrition[main_course]['protein'] + drinks[drink]['protein']
    total_vegetables = nutrition[main_course]['vegetables'] + sides[side]['vegetables']

    # Health score calculation
    health_score = 0
    if total_vegetables >= 3:
        health_score += 2
    elif total_vegetables >= 2:
        health_score += 1

    if total_protein >= 15:
        health_score += 2
    elif total_protein >= 10:
        health_score += 1

    if total_calories <= 500:
        health_score += 1
    elif total_calories > 700:
        health_score -= 1

    return {
        'calories': total_calories,
        'protein': total_protein,
        'vegetables': total_vegetables,
        'health_score': health_score,
        'rating': 'Excellent' if health_score >= 4 else 'Good' if health_score >= 2 else 'Needs Improvement'
    }

# Test different lunch combinations
lunches = [
    ('salad', 'vegetables', 'milk'),
    ('pizza', 'chips', 'juice'),
    ('sandwich', 'fruit', 'water'),
    ('soup', 'vegetables', 'milk')
]

print("🍽️ School Lunch Nutrition Analysis:")
for main, side, drink in lunches:
    nutrition = calculate_lunch_nutrition(main, side, drink)
    print(f"\n🍕 Lunch: {main} + {side} + {drink}")
    print(f"📊 Calories: {nutrition['calories']}")
    print(f"🥩 Protein: {nutrition['protein']}g")
    print(f"🥕 Vegetables: {nutrition['vegetables']}")
    print(f"🏆 Health Rating: {nutrition['rating']} ({nutrition['health_score']}/5)")
```

    if score >= 4:
        strength = "Strong"
    elif score >= 3:
        strength = "Medium"
    else:
        strength = "Weak"

    return {
        'score': score,
        'strength': strength,
        'feedback': feedback
    }

# Test the password checker

test_passwords = ["password", "Password123", "P@ssw0rd!", "123456"]
for pwd in test_passwords:
result = check_password_strength(pwd)
print(f"Password: {pwd}")
print(f"Strength: {result['strength']} (Score: {result['score']}/5)")
for feedback in result['feedback']:
print(f" - {feedback}")
print()

````

### Exercise 3: Text Analyzer
```python
def analyze_text(text):
    """Analyze text and return statistics"""
    if not text:
        return {"error": "Empty text"}

    analysis = {
        'total_chars': len(text),
        'total_words': len(text.split()),
        'total_lines': len(text.split('\n')),
        'vowels': sum(1 for char in text.lower() if char in 'aeiou'),
        'consonants': sum(1 for char in text.lower() if char.isalpha() and char not in 'aeiou'),
        'spaces': text.count(' '),
        'uppercase': sum(1 for char in text if char.isupper()),
        'lowercase': sum(1 for char in text if char.islower()),
        'digits': sum(1 for char in text if char.isdigit()),
        'sentences': text.count('.') + text.count('!') + text.count('?'),
        'average_word_length': 0
    }

    if analysis['total_words'] > 0:
        analysis['average_word_length'] = analysis['total_chars'] / analysis['total_words']

    return analysis

# Test with sample text
sample_text = """Hello World! This is a sample text.
It has multiple sentences, words, and characters.
Let's analyze it!"""

result = analyze_text(sample_text)
print("Text Analysis:")
for key, value in result.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
````

### Exercise 4: Grade Calculator with Functions

```python
def get_student_info():
    """Get student information from user"""
    name = input("Student name: ")
    age = int(input("Age: "))
    return {'name': name, 'age': age}

def get_grades():
    """Get grades for multiple subjects"""
    grades = []
    subjects = ['Math', 'English', 'Science', 'History', 'Art']

    for subject in subjects:
        grade = float(input(f"{subject} grade: "))
        grades.append({'subject': subject, 'grade': grade})

    return grades

def calculate_average(grades):
    """Calculate average of grades"""
    if not grades:
        return 0
    return sum(grades) / len(grades)

def get_letter_grade(average):
    """Convert numerical average to letter grade"""
    if average >= 90:
        return 'A'
    elif average >= 80:
        return 'B'
    elif average >= 70:
        return 'C'
    elif average >= 60:
        return 'D'
    else:
        return 'F'

def display_report_card(student, grades):
    """Display formatted report card"""
    print("\n" + "="*50)
    print(f"REPORT CARD - {student['name']}")
    print("="*50)

    total_points = 0
    for grade_info in grades:
        subject = grade_info['subject']
        grade = grade_info['grade']
        total_points += grade
        print(f"{subject:12} {grade:6.1f}")

    average = calculate_average([g['grade'] for g in grades])
    letter = get_letter_grade(average)

    print("-"*50)
    print(f"{'Average':12} {average:6.1f} ({letter})")
    print(f"{'Total Points':12} {total_points:6.1f}")

    if average >= 90:
        print("🎉 Excellent work!")
    elif average >= 80:
        print("👍 Good job!")
    elif average >= 70:
        print("👌 Keep improving!")
    else:
        print("📚 Study more!")

def main():
    """Main program function"""
    print("Student Grade Calculator")
    print("-"*30)

    while True:
        student = get_student_info()
        grades = get_grades()

        display_report_card(student, grades)

        continue_calc = input("\nCalculate another student? (y/n): ")
        if continue_calc.lower() != 'y':
            break

    print("Thank you for using Grade Calculator!")

# Run the program
# main()
```

---

## 🎉 Congratulations! You're Now a Python Programming Superhero! 🦸‍♀️🦸‍♂️

### 🏆 What You've Mastered:

#### Decision Making (If/Else) - The School Principal Powers! 📝

✅ **If/Else**: Making smart decisions like "If homework is done, you can go play!"  
✅ **Elif**: Multiple choices like choosing lunch options  
✅ **Logic Gates**: Using `and`, `or`, `not` to combine conditions  
✅ **Nested Ifs**: Checking inside checks like hall security

#### Loops - The Super-Speed Powers! ⚡

✅ **For Loops**: Doing tasks repeatedly like taking attendance for all students  
✅ **While Loops**: Continuing until a goal is reached like practicing until you get 5 right  
✅ **Loop Control**: `break` (emergency stop), `continue` (skip ahead), `pass` (placeholder)

#### Functions - The Reuse Everything Powers! 💪

✅ **Creating Functions**: Writing reusable code like teaching the same lesson to different classes  
✅ **Parameters**: Giving functions ingredients to work with  
✅ **Return Values**: Getting results back like getting your graded papers  
✅ **Variable Scope**: Understanding where variables live (like which students can access which classrooms)

#### Advanced Magic - The Expert Level! ✨

✅ **Lambda Functions**: Quick one-line functions  
✅ **Recursion**: Functions that call themselves (like Russian dolls!)  
✅ **Decorators**: Adding superpowers to your functions

### 🎯 Real School Projects You Can Now Build:

✅ **Grade Calculator**: Calculate averages and letter grades for your class  
✅ **Attendance System**: Track who's present using loops  
✅ **Lunch Menu Builder**: Create nutritious meal combinations  
✅ **Homework Tracker**: Never forget assignments again!  
✅ **Class Schedule Organizer**: Plan your perfect school day  
✅ **Quiz Game**: Make learning fun with interactive questions

### 🚀 What's Next on Your Programming Journey?

You've mastered the fundamentals! Next up:

- **Data Structures**: Lists, Dictionaries, and Sets (your new super-tools!)
- **File Handling**: Saving and loading your programs
- **Object-Oriented Programming**: Creating your own custom data types
- **Building Real Apps**: Games, websites, and mobile apps!

### 💡 Remember These Superhero Rules:

1. **Start Simple**: Every expert was once a beginner
2. **Practice Daily**: Even 15 minutes a day makes you better
3. **Break Problems Down**: Large problems become easy when you split them
4. **Debug with Patience**: Every error is a learning opportunity
5. **Be Creative**: Use your imagination to build cool stuff!

---

### 🎊 Final Challenge: Your First Solo Project!

**Build a School Report Card Generator that:**

- Asks for student name and grades in multiple subjects
- Calculates averages and letter grades
- Provides encouraging feedback based on performance
- Uses functions to organize the code
- Loops through multiple students

**You've got this! You're ready to build amazing things!** 🌟

---

_Ready for the next adventure: Data Structures Mastery!_ 🗺️⚡
