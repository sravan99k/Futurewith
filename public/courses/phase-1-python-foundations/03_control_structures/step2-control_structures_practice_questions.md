# 🔄 Control Structures & Functions Practice Questions - Universal Edition

_Master Logic and Programming Skills with Real-World Examples_

Welcome to our comprehensive Python practice guide designed for learners from all backgrounds! All problems use familiar scenarios like customer management, business decisions, personal finance, and everyday applications to help you learn programming concepts through practical examples.

## Question Categories:

1. [Basic If/Else (Questions 1-20)](#basic-ifelse-questions-1-20)
2. [Complex Conditions (Questions 21-40)](#complex-conditions-questions-21-40)
3. [For Loops (Questions 41-60)](#for-loops-questions-41-60)
4. [While Loops (Questions 61-80)](#while-loops-questions-61-80)
5. [Functions Basics (Questions 81-100)](#functions-basics-questions-81-100)
6. [Advanced Functions (Questions 101-120)](#advanced-functions-questions-101-120)
7. [Real-World Applications (Questions 121-140)](#real-world-applications-questions-121-140)
8. [Challenge Problems (Questions 141-160)](#challenge-problems-questions-141-160)

---

## Basic If/Else (Questions 1-20)

### Question 1: Simple Adult Check

**Write a program that checks if someone is an adult (18 or older)**

**Answer:**

```python
age = int(input("Enter your age: "))
if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")
```

### Question 2: Number Comparison

**Compare two numbers and display which is larger**

**Answer:**

```python
num1 = float(input("First number: "))
num2 = float(input("Second number: "))

if num1 > num2:
    print(f"{num1} is larger than {num2}")
elif num2 > num1:
    print(f"{num2} is larger than {num1}")
else:
    print("Both numbers are equal")
```

### Question 3: Even/Odd Checker

**Check if a number is even or odd**

**Answer:**

```python
number = int(input("Enter a number: "))
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")
```

### Question 4: School Computer Lab Login

**Simple login for school computer lab access**

**Answer:**

```python
username = input("Enter student ID: ")
password = input("Enter password: ")

if username == "student123" and password == "school2024":
    print("Welcome to the Computer Lab!")
else:
    print("Access denied. Please check your credentials.")
```

### Question 5: Temperature Alert

**Display temperature alerts based on input**

**Answer:**

```python
temperature = float(input("Enter temperature: "))

if temperature > 30:
    print("🔥 Too hot! Stay hydrated!")
elif temperature < 0:
    print("❄️ Freezing! Wear warm clothes!")
else:
    print("✅ Temperature is comfortable")
```

### Question 6: School Grade Checker

**Convert your test score to a letter grade (A, B, C, D, F)**

**Answer:**

```python
grade = float(input("Enter your grade: "))

if grade >= 90:
    letter = "A"
elif grade >= 80:
    letter = "B"
elif grade >= 70:
    letter = "C"
elif grade >= 60:
    letter = "D"
else:
    letter = "F"

print(f"Your letter grade: {letter}")
```

### Question 7: Lunch Money Tracker

**Check if student has enough lunch money**

**Answer:**

```python
lunch_balance = float(input("Lunch money balance: $"))
meal_cost = float(input("Meal cost: $"))

if meal_cost <= lunch_balance:
    new_balance = lunch_balance - meal_cost
    print(f"Purchase successful! Remaining balance: ${new_balance}")
else:
    print("Insufficient funds. Please add more money to your account.")
```

### Question 8: Age Calculator

**Check if you can vote, drink, or drive**

**Answer:**

```python
age = int(input("Enter your age: "))

if age >= 21:
    print("You can vote, drink, and drive")
elif age >= 18:
    print("You can vote and drive, but cannot drink")
elif age >= 16:
    print("You can drive, but cannot vote or drink")
else:
    print("You cannot vote, drink, or drive yet")
```

### Question 9: Number Sign Checker

**Check if number is positive, negative, or zero**

**Answer:**

```python
number = float(input("Enter a number: "))

if number > 0:
    print(f"{number} is positive")
elif number < 0:
    print(f"{number} is negative")
else:
    print(f"{number} is zero")
```

### Question 10: Grade Average Calculator

**Calculate average grade from test scores**

**Answer:**

```python
score1 = float(input("First test score: "))
operation = input("Would you like to add another score? (yes/no): ")
score2 = float(input("Second test score: "))

average = (score1 + score2) / 2

if average >= 90:
    grade = "A"
elif average >= 80:
    grade = "B"
elif average >= 70:
    grade = "C"
elif average >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Average score: {average:.1f}")
print(f"Letter grade: {grade}")
```

### Question 11: Leap Year Checker

**Check if a year is a leap year**

**Answer:**

```python
year = int(input("Enter a year: "))

if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year} is a leap year")
else:
    print(f"{year} is not a leap year")
```

### Question 12: Traffic Light

**Simulate traffic light behavior**

**Answer:**

```python
light = input("Traffic light color (red, yellow, green): ").lower()

if light == "red":
    print("🛑 STOP!")
elif light == "yellow":
    print("🟡 SLOW DOWN!")
elif light == "green":
    print("🟢 GO!")
else:
    print("Invalid color")
```

### Question 13: BMI Calculator

**Calculate BMI and categorize**

**Answer:**

```python
weight = float(input("Weight in kg: "))
height = float(input("Height in meters: "))

bmi = weight / (height ** 2)

if bmi < 18.5:
    category = "Underweight"
elif bmi < 25:
    category = "Normal weight"
elif bmi < 30:
    category = "Overweight"
else:
    category = "Obese"

print(f"BMI: {bmi:.2f} ({category})")
```

### Question 14: Speed Check

**Check if you're speeding**

**Answer:**

```python
speed = float(input("Speed in mph: "))
speed_limit = 55  # mph

if speed <= speed_limit:
    print("You are driving safely")
elif speed <= speed_limit + 10:
    print("Speeding warning")
else:
    print("SPEEDING TICKET!")
```

### Question 15: Discount Calculator

**Apply discount based on purchase amount**

**Answer:**

```python
amount = float(input("Purchase amount: $"))

if amount >= 1000:
    discount = amount * 0.1
    final = amount - discount
    print(f"10% discount applied!")
elif amount >= 500:
    discount = amount * 0.05
    final = amount - discount
    print(f"5% discount applied!")
else:
    discount = 0
    final = amount
    print("No discount available")

print(f"Discount: ${discount:.2f}")
print(f"Final amount: ${final:.2f}")
```

### Question 16: Password Validator

**Check password strength**

**Answer:**

```python
password = input("Enter password: ")

if len(password) >= 8:
    print("Password is strong enough")
else:
    print("Password too short (need 8+ characters)")
```

### Question 17: Time of Day Greeting

**Different greetings based on time**

**Answer:**

```python
hour = int(input("Enter hour (0-23): "))

if 6 <= hour < 12:
    print("Good morning!")
elif 12 <= hour < 17:
    print("Good afternoon!")
elif 17 <= hour < 22:
    print("Good evening!")
elif 22 <= hour or hour < 6:
    print("Good night!")
else:
    print("Invalid hour")
```

### Question 18: Coin Flip

**Simulate coin flip**

**Answer:**

```python
import random

guess = input("Guess heads or tails: ").lower()
flip = random.choice(["heads", "tails"])

if guess == flip:
    print(f"Correct! It was {flip}")
else:
    print(f"Wrong! It was {flip}")
```

### Question 19: Classroom Seating Planner

**Check if classroom can accommodate all students**

**Answer:**

```python
total_students = int(input("Total students: "))
desks_available = int(input("Desks available: "))

if desks_available >= total_students:
    print("✅ Everyone has a seat!")
else:
    shortage = total_students - desks_available
    print(f"❌ Need {shortage} more desks")
```

### Question 20: School Subject Selector

**Choose the right elective based on interests**

**Answer:**

```python
interest = input("What interests you most? (art, sports, science, music): ").lower()
time_available = int(input("Hours per week you can commit: "))

if interest == "art" and time_available >= 3:
    print("🎨 Perfect for Art Club!")
elif interest == "sports" and time_available >= 5:
    print("🏃 Great for Sports Team!")
elif interest == "science" and time_available >= 4:
    print("🔬 Excellent for Science Club!")
elif interest == "music" and time_available >= 2:
    print("🎵 Music Class is perfect for you!")
else:
    print("Consider a different option that fits your schedule")
```

---

## Complex Conditions (Questions 21-40)

### Question 21: Student Report Card

**Check if a student passed all their subjects**

**Answer:**

```python
math = float(input("Math grade: "))
english = float(input("English grade: "))
science = float(input("Science grade: "))

if math >= 60 and english >= 60 and science >= 60:
    print("Passed all subjects!")
else:
    print("Failed one or more subjects")
```

### Question 22: School Day Schedule

**Check if it's a school day or weekend**

**Answer:**

```python
day = input("Enter day: ").lower()
is_holiday = input("Is it a holiday? (y/n): ").lower() == "y"
is_snow_day = input("Is it a snow day? (y/n): ").lower() == "y"

if day in ["saturday", "sunday"] or is_holiday or is_snow_day:
    print("No school today! 🎉")
else:
    print("Time to get ready for school!")
```

### Question 23: Scholarship Eligibility

**Check multiple criteria for scholarship**

**Answer:**

```python
gpa = float(input("GPA: "))
income = float(input("Family income: $"))
volunteer_hours = int(input("Volunteer hours: "))

if gpa >= 3.5 and income <= 50000 and volunteer_hours >= 40:
    print("Eligible for scholarship!")
else:
    print("Not eligible")
```

### Question 24: Weather Activity

**Suggest activity based on weather**

**Answer:**

```python
temperature = float(input("Temperature: "))
is_sunny = input("Is it sunny? (y/n): ").lower() == "y"
is_rainy = input("Is it rainy? (y/n): ").lower() == "y"

if temperature > 25 and is_sunny and not is_rainy:
    print("Perfect for outdoor activities!")
elif temperature < 10 or is_rainy:
    print("Stay indoors and watch movies")
else:
    print("Take a walk")
```

### Question 25: Age Group Classification

**Classify age into groups**

**Answer:**

```python
age = int(input("Enter age: "))

if age < 13:
    group = "Child"
elif age < 20:
    group = "Teenager"
elif age < 65:
    group = "Adult"
else:
    group = "Senior"

print(f"Age group: {group}")
```

### Question 26: Field Trip Eligibility

**Check if student can go on field trip**

**Answer:**

```python
grade_average = float(input("Current grade average: "))
attendance_rate = int(input("Attendance rate (percentage): "))
parent_permission = input("Parent permission signed? (y/n): ").lower() == "y"

if grade_average >= 70 and attendance_rate >= 85 and parent_permission:
    print("✅ Eligible for field trip!")
elif grade_average >= 70 and attendance_rate >= 80 and parent_permission:
    print("⚠️ Eligible but with warning")
else:
    print("❌ Not eligible for field trip")
```

### Question 27: Number Range Check

**Check if number is in specific ranges**

**Answer:**

```python
number = float(input("Enter a number: "))

if 1 <= number <= 10:
    print("Number is between 1 and 10")
elif 11 <= number <= 20:
    print("Number is between 11 and 20")
elif 21 <= number <= 30:
    print("Number is between 21 and 30")
else:
    print("Number is outside all ranges")
```

### Question 28: School Report Generator

**Generate final grade based on test score and class attendance**

**Answer:**

```python
score = float(input("Exam score: "))
attendance = float(input("Attendance percentage: "))

if score >= 90 and attendance >= 90:
    grade = "A+"
elif score >= 80 and attendance >= 80:
    grade = "A"
elif score >= 70 and attendance >= 70:
    grade = "B"
elif score >= 60 and attendance >= 60:
    grade = "C"
else:
    grade = "F"

print(f"Final grade: {grade}")
```

### Question 29: Color Code Validator

**Validate hex color codes**

**Answer:**

```python
color = input("Enter hex color code: ").lower()

if len(color) == 7 and color[0] == "#":
    valid_chars = "0123456789abcdef"
    if all(c in valid_chars for c in color[1:]):
        print("Valid hex color code")
    else:
        print("Invalid hex color code")
else:
    print("Invalid format")
```

### Question 30: Multiple Conditions Menu

**Menu system with complex conditions**

**Answer:**

```python
balance = 1000

print("1. Check Balance")
print("2. Deposit")
print("3. Withdraw")
choice = input("Choose option: ")

if choice == "1":
    print(f"Balance: ${balance}")
elif choice == "2":
    amount = float(input("Deposit amount: $"))
    if amount > 0:
        balance += amount
        print(f"New balance: ${balance}")
    else:
        print("Invalid amount")
elif choice == "3":
    amount = float(input("Withdrawal amount: $"))
    if amount <= balance and amount > 0:
        balance -= amount
        print(f"New balance: ${balance}")
    else:
        print("Invalid or insufficient amount")
else:
    print("Invalid choice")
```

---

## For Loops (Questions 41-60)

### Question 41: Classroom Counting

**Count students in your class (1 to 30)**

**Answer:**

```python
for i in range(1, 11):
    print(i)
```

### Question 42: Attendance Checker

**List all even-numbered students who are present (roll numbers 2, 4, 6... 20)**

**Answer:**

```python
for i in range(2, 21, 2):
    print(i)
```

### Question 43: Count Backwards

**Print numbers from 10 to 1 backwards**

**Answer:**

```python
for i in range(10, 0, -1):
    print(i)
```

### Question 44: Homework Points Calculator

**Calculate total homework points earned (1 point per assignment for 30 students)**

**Answer:**

```python
total_points = 0
for student_num in range(1, 31):  # 30 students
    homework_score = int(input(f"Student {student_num}'s homework score: "))
    total_points += homework_score

print(f"Total class homework points: {total_points}")
print(f"Average class score: {total_points/30:.1f}")
```

### Question 45: Class Roster

**Display all students in your class with their roll numbers**

**Answer:**

```python
class_list = ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Emma Brown"]

for index, student in enumerate(class_list, 1):
    print(f"Roll #{index}: {student}")
```

### Question 46: Math Practice Generator

**Generate multiplication practice problems (e.g., 7 times table)**

**Answer:**

```python
number = int(input("Enter number: "))

for i in range(1, 11):
    result = number * i
    print(f"{number} × {i} = {result}")
```

### Question 47: Essay Word Counter

**Count words in your homework essay**

**Answer:**

```python
essay = input("Paste your essay: ")
word_list = essay.split()
count = len(word_list)

print(f"Your essay has {count} words")

# Check if it meets minimum requirements
if count >= 500:
    print("✅ Essay meets the minimum word requirement!")
elif count >= 400:
    print("⚠️ Close to meeting the requirement. Add more content.")
else:
    print("❌ Essay is too short. Need more content.")

# Count characters too
char_count = len(essay.replace(" ", ""))
print(f"Characters (no spaces): {char_count}")
```

### Question 48: Class Statistics Calculator

**Calculate average grade for your entire class**

**Answer:**

```python
students = [85, 92, 78, 96, 88, 73, 91, 82, 89, 95]

total = 0
for grade in students:
    total += grade

average = total / len(students)
highest = max(students)
lowest = min(students)

print(f"Class average: {average:.1f}")
print(f"Highest score: {highest}")
print(f"Lowest score: {lowest}")
```

### Question 48: Nested Loop Pattern

**Print a square pattern**

**Answer:**

```python
size = int(input("Enter size: "))

for i in range(size):
    for j in range(size):
        print("*", end="")
    print()  # New line after each row
```

### Question 49: Classroom Combination Calculator

**Calculate different ways to arrange students in groups**

**Answer:**

```python
number = int(input("Enter number of students: "))
group_size = int(input("Enter group size: "))

factorial = 1
for i in range(1, number + 1):
    factorial *= i

print(f"Number of ways to arrange {number} students: {factorial:,}")

# Calculate combinations (how many different groups)
group_combinations = factorial // (factorial // group_size)
print(f"Possible different groups of {group_size}: {group_combinations:,}")
```

### Question 50: Prime Number Checker

**Check if a number is prime**

**Answer:**

```python
number = int(input("Enter number: "))

if number < 2:
    print("Not prime")
else:
    is_prime = True
    for i in range(2, number):
        if number % i == 0:
            is_prime = False
            break

    if is_prime:
        print("Prime number")
    else:
        print("Not prime")
```

---

## While Loops (Questions 61-80)

### Question 61: Morning Roll Call

**Count students marked present during morning attendance**

**Answer:**

```python
present_count = 0
total_students = 25

print("Morning Roll Call - Mark students present")
print("Enter 'present' for each student, 'done' when finished")

while present_count < total_students:
    student = input(f"Student {present_count + 1}: ")
    if student.lower() == 'present':
        present_count += 1
        print(f"✅ Marked present ({present_count}/{total_students})")
    elif student.lower() == 'done':
        break
    else:
        print("Type 'present' to mark attendance or 'done' to finish")

print(f"\n📊 Attendance Summary:")
print(f"Present: {present_count}")
print(f"Absent: {total_students - present_count}")
print(f"Attendance Rate: {(present_count/total_students)*100:.1f}%")
```

### Question 62: Math Facts Practice

**Practice math facts until you get them all right**

**Answer:**

```python
import random

questions = ["5 + 7", "9 * 3", "12 - 8", "15 / 3", "6 + 9"]
answers = [12, 27, 4, 5, 15]
correct_answers = []

print("🎯 Math Facts Practice!")
print("Solve each problem correctly to move on.")
print("-" * 30)

for i, question in enumerate(questions):
    attempts = 0
    correct = False

    while attempts < 3 and not correct:
        try:
            user_answer = int(input(f"{question} = "))
            attempts += 1

            if user_answer == answers[i]:
                print("✅ Correct!")
                correct_answers.append(question)
                correct = True
            else:
                print(f"❌ Try again! ({3-attempts} attempts left)")
        except ValueError:
            print("Please enter a number")

    if not correct:
        print(f"❌ The correct answer is: {answers[i]}")

print(f"\n📊 Results: {len(correct_answers)}/{len(questions)} problems correct!")
```

### Question 63: Input Validation

**Keep asking until valid input**

**Answer:**

```python
age = 0

while age <= 0 or age > 120:
    try:
        age = int(input("Enter your age (1-120): "))
        if age <= 0 or age > 120:
            print("Please enter a valid age")
    except ValueError:
        print("Please enter a number")

print(f"You are {age} years old")
```

### Question 64: Student Portal Menu

**Create interactive student portal menu**

**Answer:**

```python
lunch_balance = 15.50

while True:
    print("\n--- STUDENT PORTAL ---")
    print("1. Check Lunch Balance")
    print("2. Add Lunch Money")
    print("3. View Grades")
    print("4. Check Schedule")
    print("5. Exit")

    choice = input("Choose option: ")

    if choice == "1":
        print(f"Lunch Balance: ${lunch_balance:.2f}")
    elif choice == "2":
        amount = float(input("Amount to add: $"))
        lunch_balance += amount
        print(f"New balance: ${lunch_balance:.2f}")
    elif choice == "3":
        print("Math: A")
        print("English: B+")
        print("Science: A-")
        print("History: B")
    elif choice == "4":
        print("Period 1: Math (Room 101)")
        print("Period 2: English (Room 102)")
        print("Period 3: Science (Room 205)")
        print("Period 4: History (Room 103)")
    elif choice == "5":
        print("Goodbye!")
        break
    else:
        print("Invalid option")
```

### Question 65: ATM Simulation

**Simple ATM simulator**

**Answer:**

```python
balance = 1000
pin = "1234"
attempts = 0

while attempts < 3:
    entered_pin = input("Enter PIN: ")
    if entered_pin == pin:
        print("PIN accepted")

        while True:
            print(f"Balance: ${balance}")
            print("1. Withdraw")
            print("2. Deposit")
            print("3. Exit")

            action = input("Choose action: ")

            if action == "1":
                amount = float(input("Amount: $"))
                if amount <= balance:
                    balance -= amount
                    print(f"Withdrawn ${amount}")
                else:
                    print("Insufficient funds")
            elif action == "2":
                amount = float(input("Amount: $"))
                balance += amount
                print(f"Deposited ${amount}")
            elif action == "3":
                print("Thank you!")
                break
            else:
                print("Invalid action")
        break
    else:
        attempts += 1
        print(f"Wrong PIN. {3-attempts} attempts remaining")

if attempts >= 3:
    print("Account locked")
```

---

## Functions Basics (Questions 81-100)

### Question 81: School Bell Function

**Create a function that rings the school bell with a message**

**Answer:**

```python
def ring_bell(bell_type="regular"):
    """Ring the school bell with appropriate message"""
    if bell_type == "morning":
        print("🔔🔔🔔 Good morning! Time for school to start!")
    elif bell_type == "lunch":
        print("🔔🔔🔔 Lunch time! Enjoy your meal!")
    elif bell_type == "end":
        print("🔔🔔🔔 School's out! Have a great day!")
    else:
        print("🔔🔔🔔 Bell rings!")

# Test the bell
ring_bell("morning")
ring_bell("lunch")
ring_bell("end")
ring_bell()  # Default bell
```

### Question 82: Student Greeting Function

**Create a function that greets a student by name**

**Answer:**

```python
def greet_person(name):
    """Greet a specific person"""
    print(f"Hello, {name}!")

# Call with different names
greet_person("Alice")
greet_person("Bob")
```

### Question 83: Grade Calculator Function

**Create a function that calculates and returns the average of three test scores**

**Answer:**

```python
def calculate_test_average(score1, score2, score3):
    """Calculate average of three test scores"""
    average = (score1 + score2 + score3) / 3
    return average

def get_letter_grade(average):
    """Convert numerical grade to letter grade"""
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    else:
        return "F"

# Calculate grades
average1 = calculate_test_average(85, 92, 78)
grade1 = get_letter_grade(average1)

average2 = calculate_test_average(90, 95, 88)
grade2 = get_letter_grade(average2)

print(f"Test 1: Average {average1:.1f} = Grade {grade1}")
print(f"Test 2: Average {average2:.1f} = Grade {grade2}")
```

### Question 84: Classroom Area Calculator

**Calculate area needed for classroom setup**

**Answer:**

```python
def calculate_classroom_area(length, width):
    """Calculate area of classroom"""
    area = length * width
    return area

def check_space_requirements(students, desk_space=5):
    """Check if classroom has enough space"""
    needed_area = students * desk_space
    classroom_area = calculate_classroom_area(20, 15)  # 20ft x 15ft

    if classroom_area >= needed_area:
        return f"✅ Enough space! {classroom_area} sq ft available"
    else:
        return f"❌ Need more space! {needed_area - classroom_area} sq ft short"

# Test classroom
print(check_space_requirements(30))
print(f"Classroom area: {calculate_classroom_area(20, 15)} sq ft")
```

### Question 85: Lunch Order System

**Create a function to calculate lunch order total**

**Answer:**

```python
def calculate_lunch_cost(sandwich=0, drink=0, snack=0):
    """Calculate total lunch cost"""
    prices = {"sandwich": 3.50, "drink": 1.25, "snack": 1.00}

    total = (sandwich * prices["sandwich"] +
             drink * prices["drink"] +
             snack * prices["snack"])

    return total

# Order lunches
order1 = calculate_lunch_cost(sandwich=1, drink=1)  # Just sandwich and drink
order2 = calculate_lunch_cost(sandwich=1, drink=1, snack=1)  # Full meal
order3 = calculate_lunch_cost(snack=2)  # Just snacks

print(f"Order 1: ${order1:.2f}")
print(f"Order 2: ${order2:.2f}")
print(f"Order 3: ${order3:.2f}")
```

### Question 86: Attendance Tracker Function

**Create a well-documented function to track student attendance**

**Answer:**

```python
def calculate_grade(score):
    """Convert numerical score to letter grade.

    Args:
        score (float): The numerical score (0-100)

    Returns:
        str: The corresponding letter grade
    """
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

# Test the function
test_score = 85
grade = calculate_grade(test_score)
print(f"Score {test_score} = Grade {grade}")
```

### Question 87: Test Score Converter

**Create functions to convert between percentage and letter grades**

**Answer:**

```python
def percentage_to_letter(percentage):
    """Convert percentage to letter grade"""
    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"

def letter_to_percentage(letter):
    """Convert letter grade to percentage"""
    grade_points = {"A": 95, "B": 85, "C": 75, "D": 65, "F": 50}
    return grade_points.get(letter.upper(), 0)

# Convert grades
test_score = 87
letter_grade = percentage_to_letter(test_score)
print(f"{test_score}% = Grade {letter_grade}")

letter = "B"
percentage = letter_to_percentage(letter)
print(f"Grade {letter} = {percentage}% average")
```

### Question 88: School Day Scheduler

**Create functions to manage daily school schedule**

**Answer:**

```python
def get_period_time(period_number):
    """Get start time for each class period"""
    start_times = {
        1: "8:00 AM", 2: "8:50 AM", 3: "9:40 AM",
        4: "10:30 AM", 5: "11:20 AM", 6: "12:10 PM",
        7: "1:00 PM", 8: "1:50 PM"
    }
    return start_times.get(period_number, "Invalid period")

def get_classroom(subject):
    """Get classroom location for each subject"""
    classrooms = {
        "Math": "Room 101", "Science": "Room 205",
        "English": "Room 102", "History": "Room 103",
        "PE": "Gym", "Art": "Art Room", "Music": "Music Room"
    }
    return classrooms.get(subject, "Room TBD")

def format_schedule(subject, period):
    """Format class information"""
    time = get_period_time(period)
    room = get_classroom(subject)
    return f"Period {period}: {subject} at {time} in {room}"

# Create today's schedule
today_schedule = [
    ("Math", 1), ("Science", 2), ("English", 3),
    ("History", 4), ("Lunch", 0), ("PE", 6),
    ("Art", 7), ("Study Hall", 8)
]

print("📅 Today's Schedule:")
print("-" * 40)
for subject, period in today_schedule:
    if period > 0:
        print(format_schedule(subject, period))
    else:
        print(f"🕐 {subject}")
```

### Question 89: List Processing Function

**Create list manipulation functions**

**Answer:**

```python
def find_max(numbers):
    """Find maximum number in list"""
    if not numbers:
        return None
    return max(numbers)

def find_min(numbers):
    """Find minimum number in list"""
    if not numbers:
        return None
    return min(numbers)

def calculate_average(numbers):
    """Calculate average of numbers"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

# Test functions
scores = [85, 92, 78, 96, 88, 73, 91]

print(f"Highest score: {find_max(scores)}")
print(f"Lowest score: {find_min(scores)}")
print(f"Average score: {calculate_average(scores):.2f}")
```

### Question 90: Time Calculation Function

**Calculate time differences**

**Answer:**

```python
def calculate_duration(start_hour, start_min, end_hour, end_min):
    """Calculate time duration in minutes"""
    start_total = start_hour * 60 + start_min
    end_total = end_hour * 60 + end_min

    duration = end_total - start_total
    if duration < 0:
        duration += 24 * 60  # Next day

    hours = duration // 60
    minutes = duration % 60

    return hours, minutes

# Calculate meeting duration
start_h, start_m = 9, 30
end_h, end_m = 11, 45

hours, minutes = calculate_duration(start_h, start_m, end_h, end_m)
print(f"Meeting duration: {hours} hours {minutes} minutes")
```

---

## Advanced Functions (Questions 101-120)

### Question 101: Lambda Functions

**Create and use lambda functions**

**Answer:**

```python
# Lambda functions for common operations
square = lambda x: x ** 2
is_even = lambda x: x % 2 == 0
multiply = lambda x, y: x * y

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squared_numbers = list(map(lambda x: x ** 2, numbers))
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

print(f"Squared: {squared_numbers}")
print(f"Even numbers: {even_numbers}")

# Lambda for sorting
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]
students_by_grade = sorted(students, key=lambda x: x[1])
print(f"Sorted by grade: {students_by_grade}")
```

### Question 102: Recursive Function

**Create a recursive function**

**Answer:**

```python
def factorial(n):
    """Calculate factorial using recursion"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Generate Fibonacci sequence using recursion"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def countdown(n):
    """Countdown using recursion"""
    if n <= 0:
        print("Done!")
        return
    print(n)
    countdown(n - 1)

# Test recursive functions
print(f"5! = {factorial(5)}")
print("Fibonacci sequence (first 7 numbers):")
for i in range(7):
    print(f"fib({i}) = {fibonacci(i)}")

print("\nCountdown:")
countdown(5)
```

### Question 103: \*args and \*\*kwargs

**Functions with variable arguments**

**Answer:**

```python
def sum_all(*args):
    """Sum any number of arguments"""
    return sum(args)

def create_profile(**kwargs):
    """Create profile from keyword arguments"""
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

def flexible_operation(mode, *args, **kwargs):
    """Flexible function with both *args and **kwargs"""
    if mode == "add":
        return sum(args)
    elif mode == "multiply":
        result = 1
        for num in args:
            result *= num
        return result
    elif mode == "format":
        template = kwargs.get('template', 'Hello {name}!')
        return template.format(**kwargs)

# Test flexible arguments
print(f"Sum: {sum_all(1, 2, 3, 4, 5)}")

profile = create_profile(name="Alice", age=25, city="NYC")
print(f"Profile: {profile}")

result1 = flexible_operation("add", 1, 2, 3)
result2 = flexible_operation("multiply", 2, 3, 4)
result3 = flexible_operation("format", name="Bob", template="Welcome {name}!")

print(f"Add: {result1}, Multiply: {result2}, Format: {result3}")
```

### Question 104: Generator Function

**Create and use generator functions**

**Answer:**

```python
def countdown_generator(n):
    """Generate countdown numbers"""
    while n > 0:
        yield n
        n -= 1

def fibonacci_generator(limit):
    """Generate Fibonacci sequence up to limit"""
    a, b = 0, 1
    while a <= limit:
        yield a
        a, b = b, a + b

# Use generators
print("Countdown:")
for num in countdown_generator(5):
    print(num)

print("\nFibonacci up to 100:")
for fib in fibonacci_generator(100):
    print(fib, end=" ")

# Generator expressions
squares = (x ** 2 for x in range(10))
print(f"\nSquares: {list(squares)}")
```

### Question 105: Decorator Function

**Create and use function decorators**

**Answer:**

```python
def timer_decorator(func):
    """Decorator to measure function execution time"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def logger_decorator(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

@timer_decorator
@logger_decorator
def slow_function(n):
    """A function that takes some time"""
    import time
    time.sleep(0.1)
    return n ** 2

# Test decorated function
result = slow_function(5)
print(f"Final result: {result}")
```

### Question 106: Grade Analyzer

**Pass functions as parameters to analyze student grades**

**Answer:**

```python
def analyze_grades(grades, operation):
    """Apply analysis function to list of grades"""
    return [operation(grade) for grade in grades]

def letter_to_points(grade):
    """Convert letter grade to grade points"""
    points = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
    return points.get(grade, 0.0)

def is_honor_roll(grade):
    """Check if grade qualifies for honor roll"""
    return grade in ["A", "B"]

def passed_class(grade):
    """Check if student passed the class"""
    return grade not in ["F"]

# Test grade analysis
student_grades = ["A", "B", "C", "A", "B", "F", "A", "C"]

grade_points = analyze_grades(student_grades, letter_to_points)
honor_roll = analyze_grades(student_grades, is_honor_roll)
passed = analyze_grades(student_grades, passed_class)

print(f"Student Grades: {student_grades}")
print(f"Grade Points: {grade_points}")
print(f"Honor Roll: {honor_roll}")
print(f"Passed Classes: {passed}")

# Calculate GPA
gpa = sum(grade_points) / len(grade_points)
print(f"Current GPA: {gpa:.2f}")
```

### Question 107: Memoization

**Create function with caching**

**Answer:**

```python
def memoize(func):
    """Cache function results for performance"""
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
def fibonacci_memo(n):
    """Fibonacci with memoization"""
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)

def factorial_memo(n, cache={}):
    """Factorial with manual memoization"""
    if n in cache:
        return cache[n]
    if n <= 1:
        return 1
    result = n * factorial_memo(n - 1)
    cache[n] = result
    return result

# Test memoization
import time

# Test fibonacci
start = time.time()
result = fibonacci_memo(35)
end = time.time()
print(f"Fibonacci(35) = {result} (took {end - start:.4f} seconds)")

# Test factorial
print(f"Factorial(10) = {factorial_memo(10)}")
```

### Question 108: Higher-Order Function

**Create and use higher-order functions**

**Answer:**

```python
def create_multiplier(factor):
    """Create a multiplier function with fixed factor"""
    def multiplier(x):
        return x * factor
    return multiplier

def create_validator(min_val, max_val):
    """Create a validator function with range"""
    def validate(value):
        return min_val <= value <= max_val
    return validate

def create_formatter(prefix, suffix):
    """Create a formatter with fixed prefix and suffix"""
    def format_text(text):
        return f"{prefix}{text}{suffix}"
    return format_text

# Test higher-order functions
times_three = create_multiplier(3)
times_five = create_multiplier(5)

print(f"3 * 3 = {times_three(3)}")
print(f"5 * 3 = {times_five(3)}")

# Create validators
age_validator = create_validator(0, 120)
score_validator = create_validator(0, 100)

print(f"Age 25 valid: {age_validator(25)}")
print(f"Score 85 valid: {score_validator(85)}")

# Create formatters
bold_formatter = create_formatter("**", "**")
quote_formatter = create_formatter('"', '"')

print(bold_formatter("Important"))
print(quote_formatter("Said something"))
```

---

## Real-World Applications (Questions 121-140)

### Question 121: School Cafeteria Management

**Complete cafeteria management system for your school**

**Answer:**

```python
class CafeteriaAccount:
    def __init__(self, student_name, student_id, initial_balance=0):
        self.student_name = student_name
        self.student_id = student_id
        self.balance = initial_balance
        self.purchases = []

    def add_money(self, amount):
        """Add money to cafeteria account"""
        if amount > 0:
            self.balance += amount
            self.purchases.append(f"Added funds: +${amount}")
            return True
        return False

    def purchase_lunch(self, cost):
        """Purchase lunch from cafeteria"""
        if cost <= self.balance:
            self.balance -= cost
            self.purchases.append(f"Lunch purchase: -${cost}")
            return True
        return False

    def get_balance(self):
        """Get current balance"""
        return self.balance

    def get_purchase_history(self):
        """Get purchase history"""
        return self.purchases

    def daily_limit_check(self, daily_limit=10):
        """Check if spending within daily limit"""
        today_spending = 0
        for purchase in self.purchases:
            if "Lunch purchase" in purchase:
                amount = float(purchase.split("$")[1])
                today_spending += amount

        return today_spending <= daily_limit

# Test the cafeteria system
alice_account = CafeteriaAccount("Alice Johnson", "S001", 25)
alice_account.add_money(10)
alice_account.purchase_lunch(4.50)
alice_account.purchase_lunch(3.25)  # Snack

print(f"{alice_account.student_name} (ID: {alice_account.student_id})")
print(f"Balance: ${alice_account.get_balance()}")
print("Purchase history:")
for purchase in alice_account.get_purchase_history():
    print(f"  {purchase}")

limit_check = alice_account.daily_limit_check()
print(f"Daily limit check: {'✅ Within limit' if limit_check else '⚠️ Over daily limit'}")
```

### Question 122: Classroom Management System

**Manage your entire classroom with functions and classes**

**Answer:**

```python
class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.grades = {}
        self.attendance = []

    def add_grade(self, subject, grade):
        """Add grade for a subject"""
        self.grades[subject] = grade

    def get_gpa(self):
        """Calculate GPA"""
        if not self.grades:
            return 0.0

        grade_points = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
        total_points = sum(grade_points.get(grade, 0.0) for grade in self.grades.values())
        return total_points / len(self.grades)

    def mark_attendance(self, present=True):
        """Mark attendance"""
        status = "Present" if present else "Absent"
        self.attendance.append(status)

    def get_attendance_rate(self):
        """Calculate attendance percentage"""
        if not self.attendance:
            return 0.0
        present_days = self.attendance.count("Present")
        return (present_days / len(self.attendance)) * 100

class StudentDatabase:
    def __init__(self):
        self.students = {}

    def add_student(self, student):
        """Add student to database"""
        self.students[student.student_id] = student

    def find_student(self, student_id):
        """Find student by ID"""
        return self.students.get(student_id)

    def get_all_gpas(self):
        """Get GPA of all students"""
        return {sid: student.get_gpa() for sid, student in self.students.items()}

    def get_top_performers(self, limit=5):
        """Get top performing students"""
        gpas = self.get_all_gpas()
        sorted_students = sorted(gpas.items(), key=lambda x: x[1], reverse=True)
        return sorted_students[:limit]

# Test the system
db = StudentDatabase()

# Add students
alice = Student("Alice Johnson", "S001")
alice.add_grade("Math", "A")
alice.add_grade("Science", "B")
alice.add_grade("English", "A")
for i in range(20):
    alice.mark_attendance(True)

bob = Student("Bob Smith", "S002")
bob.add_grade("Math", "B")
bob.add_grade("Science", "C")
bob.add_grade("English", "B")
for i in range(20):
    bob.mark_attendance(i < 18)  # Missed 2 days

db.add_student(alice)
db.add_student(bob)

print(f"Alice GPA: {alice.get_gpa():.2f}")
print(f"Bob GPA: {bob.get_gpa():.2f}")
print(f"Alice attendance: {alice.get_attendance_rate():.1f}%")
print(f"Bob attendance: {bob.get_attendance_rate():.1f}%")
```

### Question 123: Inventory Management System

**Complete inventory tracking**

**Answer:**

```python
class Product:
    def __init__(self, name, price, stock):
        self.name = name
        self.price = price
        self.stock = stock
        self.transactions = []

    def restock(self, quantity):
        """Add stock"""
        if quantity > 0:
            self.stock += quantity
            self.transactions.append(f"Restocked: +{quantity}")
            return True
        return False

    def sell(self, quantity):
        """Sell product"""
        if quantity <= self.stock:
            self.stock -= quantity
            self.transactions.append(f"Sold: -{quantity}")
            return True
        return False

    def get_total_value(self):
        """Get total inventory value"""
        return self.stock * self.price

class InventoryManager:
    def __init__(self):
        self.products = {}
        self.low_stock_threshold = 10

    def add_product(self, product):
        """Add product to inventory"""
        self.products[product.name] = product

    def find_product(self, name):
        """Find product by name"""
        return self.products.get(name)

    def get_low_stock_items(self):
        """Get products with low stock"""
        return [name for name, product in self.products.items()
                if product.stock <= self.low_stock_threshold]

    def get_total_inventory_value(self):
        """Get total value of all inventory"""
        return sum(product.get_total_value() for product in self.products.values())

    def restock_product(self, name, quantity):
        """Restock a product"""
        product = self.find_product(name)
        if product:
            return product.restock(quantity)
        return False

    def sell_product(self, name, quantity):
        """Sell a product"""
        product = self.find_product(name)
        if product:
            return product.sell(quantity)
        return False

# Test inventory system
manager = InventoryManager()

# Add products
laptop = Product("Laptop", 999.99, 5)
phone = Product("Phone", 699.99, 25)
tablet = Product("Tablet", 299.99, 3)

manager.add_product(laptop)
manager.add_product(phone)
manager.add_product(tablet)

# Restock some items
manager.restock_product("Laptop", 10)
manager.restock_product("Tablet", 5)

# Sell some items
manager.sell_product("Phone", 5)
manager.sell_product("Laptop", 3)

# Display results
print(f"Total inventory value: ${manager.get_total_inventory_value():.2f}")
print(f"Low stock items: {manager.get_low_stock_items()}")
print("\nProduct details:")
for name, product in manager.products.items():
    print(f"{name}: ${product.price} - Stock: {product.stock}")
```

### Question 124: Quiz Application

**Complete quiz application**

**Answer:**

```python
class Question:
    def __init__(self, question, options, correct_answer):
        self.question = question
        self.options = options
        self.correct_answer = correct_answer

    def is_correct(self, user_answer):
        """Check if answer is correct"""
        return user_answer == self.correct_answer

    def get_correct_option(self):
        """Get the correct option text"""
        return self.options[self.correct_answer - 1]

class Quiz:
    def __init__(self, title):
        self.title = title
        self.questions = []
        self.current_question_index = 0
        self.score = 0

    def add_question(self, question):
        """Add question to quiz"""
        self.questions.append(question)

    def get_current_question(self):
        """Get current question"""
        if self.current_question_index < len(self.questions):
            return self.questions[self.current_question_index]
        return None

    def answer_question(self, user_answer):
        """Answer current question"""
        current_q = self.get_current_question()
        if current_q:
            if current_q.is_correct(user_answer):
                self.score += 1
                return True
            return False
        return False

    def next_question(self):
        """Move to next question"""
        self.current_question_index += 1

    def is_complete(self):
        """Check if quiz is complete"""
        return self.current_question_index >= len(self.questions)

    def get_score_percentage(self):
        """Get score as percentage"""
        if not self.questions:
            return 0
        return (self.score / len(self.questions)) * 100

def create_sample_quiz():
    """Create a sample quiz"""
    quiz = Quiz("General Knowledge Quiz")

    # Add questions
    questions_data = [
        {
            "question": "What is the capital of France?",
            "options": ["London", "Berlin", "Paris", "Madrid"],
            "correct": 3
        },
        {
            "question": "Which planet is the largest?",
            "options": ["Earth", "Mars", "Jupiter", "Venus"],
            "correct": 3
        },
        {
            "question": "What is 2 + 2?",
            "options": ["3", "4", "5", "6"],
            "correct": 2
        },
        {
            "question": "Which year did humans first land on the moon?",
            "options": ["1965", "1969", "1971", "1973"],
            "correct": 2
        }
    ]

    for q_data in questions_data:
        question = Question(
            q_data["question"],
            q_data["options"],
            q_data["correct"]
        )
        quiz.add_question(question)

    return quiz

def run_quiz_interactive():
    """Run quiz interactively"""
    quiz = create_sample_quiz()

    print(f"Welcome to {quiz.title}!")
    print("=" * 50)

    while not quiz.is_complete():
        current_q = quiz.get_current_question()

        print(f"\nQuestion {quiz.current_question_index + 1}:")
        print(current_q.question)

        for i, option in enumerate(current_q.options, 1):
            print(f"{i}. {option}")

        try:
            user_answer = int(input("\nYour answer (1-4): "))

            if 1 <= user_answer <= len(current_q.options):
                is_correct = quiz.answer_question(user_answer)

                if is_correct:
                    print("✅ Correct!")
                else:
                    print(f"❌ Wrong! Correct answer: {current_q.get_correct_option()}")

                quiz.next_question()
            else:
                print("Please enter a number between 1 and 4")

        except ValueError:
            print("Please enter a valid number")

    percentage = quiz.get_score_percentage()
    print(f"\n" + "=" * 50)
    print(f"Quiz Complete!")
    print(f"Score: {quiz.score}/{len(quiz.questions)} ({percentage:.1f}%)")

    if percentage >= 90:
        print("🎉 Excellent work!")
    elif percentage >= 70:
        print("👍 Good job!")
    elif percentage >= 50:
        print("👌 Not bad!")
    else:
        print("📚 Keep studying!")

# Test the quiz application
# run_quiz_interactive()
```

### Question 125: Library Management System

**Complete library management**

**Answer:**

```python
class Book:
    def __init__(self, isbn, title, author, genre):
        self.isbn = isbn
        self.title = title
        self.author = author
        self.genre = genre
        self.is_available = True
        self.borrower = None
        self.due_date = None

    def borrow(self, borrower, due_date):
        """Borrow the book"""
        if self.is_available:
            self.is_available = False
            self.borrower = borrower
            self.due_date = due_date
            return True
        return False

    def return_book(self):
        """Return the book"""
        self.is_available = True
        self.borrower = None
        self.due_date = None

class Member:
    def __init__(self, member_id, name, email):
        self.member_id = member_id
        self.name = name
        self.email = email
        self.borrowed_books = []
        self.fine_amount = 0.0

    def borrow_book(self, book):
        """Borrow a book"""
        if len(self.borrowed_books) < 5:  # Max 5 books
            self.borrowed_books.append(book)
            return True
        return False

    def return_book(self, book):
        """Return a book"""
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            return True
        return False

    def calculate_fine(self, days_late):
        """Calculate fine for overdue books"""
        fine_rate = 1.0  # $1 per day
        return days_late * fine_rate

class Library:
    def __init__(self, name):
        self.name = name
        self.books = {}
        self.members = {}
        self.transactions = []

    def add_book(self, book):
        """Add book to library"""
        self.books[book.isbn] = book

    def add_member(self, member):
        """Add member to library"""
        self.members[member.member_id] = member

    def find_book(self, isbn):
        """Find book by ISBN"""
        return self.books.get(isbn)

    def find_member(self, member_id):
        """Find member by ID"""
        return self.members.get(member_id)

    def borrow_book(self, member_id, isbn, due_date):
        """Process book borrowing"""
        member = self.find_member(member_id)
        book = self.find_book(isbn)

        if member and book:
            if member.borrow_book(book) and book.borrow(member, due_date):
                self.transactions.append(f"{member.name} borrowed '{book.title}'")
                return True
        return False

    def return_book(self, member_id, isbn):
        """Process book return"""
        member = self.find_member(member_id)
        book = self.find_book(isbn)

        if member and book:
            if member.return_book(book) and book.return_book():
                self.transactions.append(f"{member.name} returned '{book.title}'")
                return True
        return False

    def get_borrowed_books(self):
        """Get list of borrowed books"""
        return [book for book in self.books.values() if not book.is_available]

    def get_overdue_books(self, current_date):
        """Get overdue books"""
        overdue = []
        for book in self.books.values():
            if not book.is_available and book.due_date < current_date:
                overdue.append((book, book.borrower))
        return overdue

# Test library system
library = Library("Central Library")

# Add books
book1 = Book("978-0-123456-78-9", "The Great Gatsby", "F. Scott Fitzgerald", "Classic")
book2 = Book("978-0-987654-32-1", "To Kill a Mockingbird", "Harper Lee", "Classic")
book3 = Book("978-1-555555-55-5", "Python Programming", "John Smith", "Programming")

library.add_book(book1)
library.add_book(book2)
library.add_book(book3)

# Add members
member1 = Member("M001", "Alice Johnson", "alice@email.com")
member2 = Member("M002", "Bob Smith", "bob@email.com")

library.add_member(member1)
library.add_member(member2)

# Process some transactions
library.borrow_book("M001", "978-0-123456-78-9", "2024-01-15")
library.borrow_book("M002", "978-0-987654-32-1", "2024-01-20")

# Check borrowed books
borrowed = library.get_borrowed_books()
print("Currently borrowed books:")
for book in borrowed:
    print(f"  '{book.title}' by {book.author}")

print("\nRecent transactions:")
for transaction in library.transactions:
    print(f"  {transaction}")
```

---

## Challenge Problems (Questions 141-160)

### Question 141: Class Schedule Optimizer

**Create a class schedule optimizer for students**

**Answer:**

```python
def is_valid_schedule(schedule):
    """Check if current class schedule is valid"""
    # Check for time conflicts
    for i in range(len(schedule)):
        for j in range(i + 1, len(schedule)):
            if schedule[i]["time"] == schedule[j]["time"]:
                return False
    return True

def find_empty_period(schedule):
    """Find empty period in schedule"""
    periods = list(range(1, 9))  # 8 periods
    for period in schedule:
        if period["period"] in periods:
            periods.remove(period["period"])
    return periods[0] if periods else None

def optimize_schedule(schedule):
    """Optimize schedule to avoid conflicts"""
    for i in range(len(schedule)):
        for j in range(i + 1, len(schedule)):
            if schedule[i]["time"] == schedule[j]["time"]:
                empty_period = find_empty_period(schedule)
                if empty_period:
                    schedule[j]["period"] = empty_period
                    schedule[j]["time"] = f"{empty_period}:00"
    return schedule

def print_schedule(schedule):
    """Print class schedule in readable format"""
    print("📅 Weekly Class Schedule")
    print("=" * 40)
    for i, class_info in enumerate(schedule, 1):
        print(f"Period {class_info['period']}: {class_info['subject']} " +
              f"({class_info['time']}) - Room {class_info['room']}")
    print("=" * 40)

# Test with sample schedule
sample_schedule = [
    {"subject": "Math", "period": 1, "time": "8:00", "room": "101"},
    {"subject": "English", "period": 2, "time": "8:50", "room": "102"},
    {"subject": "Science", "period": 3, "time": "9:40", "room": "205"},
    {"subject": "History", "period": 4, "time": "10:30", "room": "103"},
    {"subject": "PE", "period": 5, "time": "11:20", "room": "Gym"},
    {"subject": "Lunch", "period": 6, "time": "12:10", "room": "Cafeteria"},
    {"subject": "Art", "period": 7, "time": "1:00", "room": "Art Room"},
    {"subject": "Study Hall", "period": 8, "time": "1:50", "room": "Library"}
]

print("Original schedule:")
print_schedule(sample_schedule)

if optimize_schedule(sample_schedule):
    print("\nOptimized schedule:")
    print_schedule(sample_schedule)
```

### Question 142: Genetic Algorithm

**Implement genetic algorithm for optimization**

**Answer:**

```python
import random
import copy

class Individual:
    def __init__(self, genes=None, gene_length=10):
        """Individual in genetic algorithm"""
        if genes:
            self.genes = genes
        else:
            self.genes = [random.randint(0, 1) for _ in range(gene_length)]
        self.fitness = 0

    def calculate_fitness(self, target):
        """Calculate fitness based on similarity to target"""
        self.fitness = sum(g1 == g2 for g1, g2 in zip(self.genes, target))
        return self.fitness

    def crossover(self, other):
        """Create offspring through crossover"""
        crossover_point = random.randint(1, len(self.genes) - 1)
        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]
        return Individual(child1_genes), Individual(child2_genes)

    def mutate(self, mutation_rate=0.1):
        """Apply mutation to genes"""
        for i in range(len(self.genes)):
            if random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]  # Flip bit

class GeneticAlgorithm:
    def __init__(self, population_size=100, gene_length=10):
        self.population_size = population_size
        self.gene_length = gene_length
        self.population = []
        self.target = None
        self.generation = 0

    def initialize_population(self, target):
        """Initialize population for given target"""
        self.target = target
        self.population = [Individual(gene_length=self.gene_length) for _ in range(self.population_size)]

    def evaluate_population(self):
        """Evaluate fitness of entire population"""
        for individual in self.population:
            individual.calculate_fitness(self.target)

        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

    def select_parents(self, selection_rate=0.5):
        """Select top individuals as parents"""
        parent_count = int(self.population_size * selection_rate)
        return self.population[:parent_count]

    def evolve(self, generations=100):
        """Run genetic algorithm for specified generations"""
        print("Running Genetic Algorithm...")
        print(f"Target: {self.target}")
        print(f"Population size: {self.population_size}")
        print("-" * 50)

        for gen in range(generations):
            self.evaluate_population()

            # Print best individual
            best = self.population[0]
            print(f"Generation {gen + 1}: Fitness={best.fitness}/{self.gene_length}, " +
                  f"Genes={best.genes[:10]}...")

            # Check if solution found
            if best.fitness == self.gene_length:
                print(f"Perfect solution found in generation {gen + 1}!")
                print(f"Genes: {best.genes}")
                return best

            # Create new population
            parents = self.select_parents()
            new_population = []

            # Keep top individuals (elitism)
            elite_count = 2
            new_population.extend(copy.deepcopy(parents[:elite_count]))

            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = parent1.crossover(parent2)
                child1.mutate()
                child2.mutate()
                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

        # Return best individual
        self.evaluate_population()
        return self.population[0]

# Test genetic algorithm
target_pattern = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
ga = GeneticAlgorithm(population_size=50, gene_length=10)
ga.initialize_population(target_pattern)

result = ga.evolve(generations=50)
print(f"\nFinal result: {result.genes}")
print(f"Fitness: {result.fitness}/{len(target_pattern)}")
```

### Question 143: A\* Pathfinding

**Implement A\* pathfinding algorithm**

**Answer:**

```python
import heapq
import math

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.g = float('inf')  # Cost from start
        self.h = 0  # Heuristic cost to end
        self.f = float('inf')  # Total cost (g + h)
        self.parent = None

    def __lt__(self, other):
        return self.f < other.f

class AStarPathfinder:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, node, end):
        """Calculate Manhattan distance heuristic"""
        return abs(node.row - end.row) + abs(node.col - end.col)

    def get_neighbors(self, node):
        """Get valid neighboring nodes"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        for dr, dc in directions:
            new_row, new_col = node.row + dr, node.col + dc

            if (0 <= new_row < self.rows and
                0 <= new_col < self.cols and
                self.grid[new_row][new_col] != 1):  # Not a wall

                neighbor = Node(new_row, new_col)
                neighbor.g = node.g + 1  # Cost of moving is 1
                neighbor.h = self.heuristic(neighbor, end_node)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = node
                neighbors.append(neighbor)

        return neighbors

    def find_path(self, start_pos, end_pos):
        """Find path using A* algorithm"""
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        # Create start and end nodes
        start_node = Node(start_row, start_col)
        global end_node
        end_node = Node(end_row, end_col)
        start_node.g = 0
        start_node.h = self.heuristic(start_node, end_node)
        start_node.f = start_node.h

        # Open set (priority queue) and closed set
        open_set = [start_node]
        closed_set = set()

        while open_set:
            # Get node with lowest f-score
            current = heapq.heappop(open_set)

            # Add to closed set
            closed_set.add((current.row, current.col))

            # Check if we reached the goal
            if current.row == end_row and current.col == end_col:
                return self.reconstruct_path(current)

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                neighbor_pos = (neighbor.row, neighbor.col)

                if neighbor_pos in closed_set:
                    continue

                # Check if we found a better path to neighbor
                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)
                elif neighbor.g >= current.g:
                    continue

                # This path is the best so far
                neighbor.parent = current
                neighbor.g = current.g + 1
                neighbor.h = self.heuristic(neighbor, end_node)
                neighbor.f = neighbor.g + neighbor.h

        return None  # No path found

    def reconstruct_path(self, end_node):
        """Reconstruct path from end to start"""
        path = []
        current = end_node

        while current:
            path.append((current.row, current.col))
            current = current.parent

        return path[::-1]  # Reverse to get start to end

    def print_path(self, path):
        """Print grid with path"""
        if not path:
            print("No path found")
            return

        # Create copy of grid to modify
        result_grid = [row[:] for row in self.grid]

        # Mark path
        path_set = set(path)
        for row_idx, row in enumerate(result_grid):
            for col_idx, cell in enumerate(row):
                if (row_idx, col_idx) in path_set:
                    result_grid[row_idx][col_idx] = 2  # Path marker

        # Print with symbols
        for row in result_grid:
            row_str = ""
            for cell in row:
                if cell == 0:
                    row_str += "· "  # Empty
                elif cell == 1:
                    row_str += "█ "  # Wall
                elif cell == 2:
                    row_str += "○ "  # Path
                else:
                    row_str += f"{cell} "
            print(row_str)

# Test A* pathfinding
def create_test_grid():
    """Create a test grid for pathfinding"""
    return [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0]
    ]

# Test the algorithm
grid = create_test_grid()
pathfinder = AStarPathfinder(grid)

print("Grid (0=empty, 1=wall):")
for row in grid:
    print(row)

print("\nFinding path from (0,0) to (6,6)...")
path = pathfinder.find_path((0, 0), (6, 6))

if path:
    print(f"Path found! Length: {len(path)}")
    print("Path coordinates:")
    for i, pos in enumerate(path):
        print(f"{i}: {pos}")

    print("\nGrid with path:")
    pathfinder.print_path(path)
else:
    print("No path found!")
```

### Question 144: Neural Network

**Implement simple neural network**

**Answer:**

```python
import math
import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def activate(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        """Forward pass through neuron"""
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return self.activate(weighted_sum + self.bias)

    def train(self, inputs, target, learning_rate=0.1):
        """Train neuron using gradient descent"""
        output = self.forward(inputs)
        error = target - output

        # Calculate gradients
        for i, weight in enumerate(self.weights):
            # Gradient = error * derivative * input
            gradient = error * output * (1 - output) * inputs[i]
            self.weights[i] += learning_rate * gradient

        # Update bias
        bias_gradient = error * output * (1 - output)
        self.bias += learning_rate * bias_gradient

        return error ** 2

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create layers
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def forward(self, inputs):
        """Forward pass through network"""
        # Hidden layer outputs
        hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_layer]

        # Output layer outputs
        output_outputs = [neuron.forward(hidden_outputs) for neuron in self.output_layer]

        return output_outputs

    def train(self, training_data, epochs=1000, learning_rate=0.1):
        """Train the neural network"""
        errors = []

        for epoch in range(epochs):
            total_error = 0

            for inputs, targets in training_data:
                # Forward pass
                outputs = self.forward(inputs)

                # Calculate output layer errors
                output_errors = []
                for i, (neuron, target) in enumerate(zip(self.output_layer, targets)):
                    error = target - outputs[i]
                    output_errors.append(error)

                    # Update output neuron weights
                    gradient = error * outputs[i] * (1 - outputs[i])

                    for j, hidden_output in enumerate(self.forward(self.hidden_layer)):
                        hidden_gradient = gradient * hidden_output
                        neuron.weights[j] += learning_rate * hidden_gradient

                    neuron.bias += learning_rate * gradient
                    total_error += error ** 2

                # Calculate hidden layer errors
                for i, neuron in enumerate(self.hidden_layer):
                    # Calculate error contribution from output layer
                    error_sum = sum(
                        output_errors[j] * output_layer[j].weights[i]
                        for j in range(self.output_size)
                    )

                    hidden_output = neuron.forward(inputs)
                    hidden_error = error_sum * hidden_output * (1 - hidden_output)

                    # Update hidden neuron weights
                    for k, input_val in enumerate(inputs):
                        gradient = hidden_error * input_val
                        neuron.weights[k] += learning_rate * gradient

                    neuron.bias += learning_rate * hidden_error
                    total_error += hidden_error ** 2

            errors.append(total_error / len(training_data))

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {total_error / len(training_data):.4f}")

        return errors

    def predict(self, inputs):
        """Make prediction"""
        return self.forward(inputs)

# Test neural network with XOR problem
def create_xor_dataset():
    """Create XOR dataset"""
    return [
        ([0, 0], [0]),  # 0 XOR 0 = 0
        ([0, 1], [1]),  # 0 XOR 1 = 1
        ([1, 0], [1]),  # 1 XOR 0 = 1
        ([1, 1], [0]),  # 1 XOR 1 = 0
    ]

# Create and train network
print("Training Neural Network on XOR problem...")
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
training_data = create_xor_dataset()

# Train the network
errors = nn.train(training_data, epochs=1000, learning_rate=0.1)

# Test predictions
print("\nTesting predictions:")
for inputs, expected in training_data:
    prediction = nn.predict(inputs)[0]
    print(f"Input: {inputs}, Expected: {expected[0]}, Predicted: {prediction:.4f}")

print(f"\nFinal error: {errors[-1]:.4f}")
```

### Question 145: Grade Calculator

**Advanced grade calculator for school report cards**

**Answer:**

```python
class GradeCalculator:
    def __init__(self, student_name):
        self.student_name = student_name
        self.subjects = {}
        self.weights = {"tests": 0.4, "homework": 0.3, "participation": 0.2, "projects": 0.1}

    def add_subject(self, subject_name):
        """Add a new subject"""
        self.subjects[subject_name] = {
            "tests": [],
            "homework": [],
            "participation": 100,  # Default participation grade
            "projects": []
        }

    def add_test_score(self, subject, score, max_score=100):
        """Add a test score"""
        if subject in self.subjects:
            percentage = (score / max_score) * 100
            self.subjects[subject]["tests"].append(percentage)
            return True
        return False

    def add_homework_score(self, subject, score, max_score=100):
        """Add a homework score"""
        if subject in self.subjects:
            percentage = (score / max_score) * 100
            self.subjects[subject]["homework"].append(percentage)
            return True
        return False

    def add_project_score(self, subject, score, max_score=100):
        """Add a project score"""
        if subject in self.subjects:
            percentage = (score / max_score) * 100
            self.subjects[subject]["projects"].append(percentage)
            return True
        return False

    def set_participation(self, subject, score):
        """Set participation grade"""
        if subject in self.subjects:
            self.subjects[subject]["participation"] = score
            return True
        return False

    def calculate_final_grade(self, subject):
        """Calculate final grade for a subject"""
        if subject not in self.subjects:
            return 0

        grades = self.subjects[subject]

        # Calculate category averages
        test_avg = sum(grades["tests"]) / len(grades["tests"]) if grades["tests"] else 0
        homework_avg = sum(grades["homework"]) / len(grades["homework"]) if grades["homework"] else 0
        participation = grades["participation"]
        project_avg = sum(grades["projects"]) / len(grades["projects"]) if grades["projects"] else 0

        # Calculate weighted final grade
        final_grade = (
            test_avg * self.weights["tests"] +
            homework_avg * self.weights["homework"] +
            participation * self.weights["participation"] +
            project_avg * self.weights["projects"]
        )

        return final_grade

    def get_letter_grade(self, percentage):
        """Convert percentage to letter grade"""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"

    def generate_report_card(self):
        """Generate complete report card"""
        print(f"📚 Report Card for {self.student_name}")
        print("=" * 50)

        total_gpa_points = 0
        num_subjects = len(self.subjects)

        for subject, grades in self.subjects.items():
            final_grade = self.calculate_final_grade(subject)
            letter_grade = self.get_letter_grade(final_grade)
            gpa_points = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}[letter_grade]
            total_gpa_points += gpa_points

            print(f"\n📖 {subject}:")
            print(f"  Tests: {grades['tests']}")
            print(f"  Homework: {grades['homework']}")
            print(f"  Participation: {grades['participation']}")
            print(f"  Projects: {grades['projects']}")
            print(f"  Final Grade: {final_grade:.1f}% ({letter_grade})")

        overall_gpa = total_gpa_points / num_subjects if num_subjects > 0 else 0
        print(f"\n🎓 Overall GPA: {overall_gpa:.2f}")
        print("=" * 50)

# Test the grade calculator
student = GradeCalculator("Alice Johnson")

# Add subjects and grades
student.add_subject("Math")
student.add_subject("English")
student.add_subject("Science")

# Add test scores
student.add_test_score("Math", 85, 100)  # 85%
student.add_test_score("Math", 92, 100)  # 92%
student.add_test_score("English", 78, 100)
student.add_test_score("Science", 95, 100)

# Add homework
student.add_homework_score("Math", 90, 100)
student.add_homework_score("English", 85, 100)
student.add_homework_score("Science", 88, 100)

# Add projects
student.add_project_score("Math", 88, 100)
student.add_project_score("English", 92, 100)

# Set participation
student.set_participation("Math", 95)
student.set_participation("English", 90)
student.set_participation("Science", 100)

# Generate report card
student.generate_report_card()
```

### Question 146: School Bus Scheduler

**Schedule buses based on pickup zones and times**

**Answer:**

```python
class BusRoute:
    def __init__(self, route_name, max_students=40):
        self.route_name = route_name
        self.max_students = max_students
        self.students = []
        self.pickup_times = {
            "North Zone": "7:15 AM",
            "South Zone": "7:20 AM",
            "East Zone": "7:25 AM",
            "West Zone": "7:30 AM"
        }

    def add_student(self, name, zone):
        """Add student to bus route"""
        if len(self.students) < self.max_students:
            self.students.append((name, zone))
            return True
        return False

    def get_pickup_time(self, zone):
        """Get pickup time for a zone"""
        return self.pickup_times.get(zone, "Contact transportation")

    def display_schedule(self):
        """Display complete route schedule"""
        print(f"🚌 {self.route_name}")
        print(f"Capacity: {len(self.students)}/{self.max_students} students")
        print("-" * 30)

        for zone in self.pickup_times:
            zone_students = [name for name, z in self.students if z == zone]
            print(f"{zone} ({self.pickup_times[zone]}):")
            if zone_students:
                for student in zone_students:
                    print(f"  • {student}")
            else:
                print("  (No students)")

# Test bus scheduling
bus_42 = BusRoute("Route 42 - Morning Express")

# Add students from different zones
bus_42.add_student("Alice Johnson", "North Zone")
bus_42.add_student("Bob Smith", "North Zone")
bus_42.add_student("Carol Davis", "South Zone")
bus_42.add_student("David Wilson", "East Zone")

bus_42.display_schedule()
```

### Question 147: Sports Team Manager

**Manage school sports teams, games, and statistics**

**Answer:**

```python
class TeamManager:
    def __init__(self, team_name, sport):
        self.team_name = team_name
        self.sport = sport
        self.players = []
        self.games = []
        self.stats = {"wins": 0, "losses": 0, "ties": 0}

    def add_player(self, name, number, position):
        """Add player to team"""
        player = {"name": name, "number": number, "position": position}
        self.players.append(player)
        return True

    def record_game(self, opponent, our_score, their_score):
        """Record game result"""
        game = {
            "opponent": opponent,
            "our_score": our_score,
            "their_score": their_score,
            "result": "Win" if our_score > their_score else "Loss" if our_score < their_score else "Tie"
        }
        self.games.append(game)

        # Update team stats
        if game["result"] == "Win":
            self.stats["wins"] += 1
        elif game["result"] == "Loss":
            self.stats["losses"] += 1
        else:
            self.stats["ties"] += 1

        return game["result"]

    def get_team_record(self):
        """Get current team record"""
        total_games = len(self.games)
        win_rate = (self.stats["wins"] / total_games * 100) if total_games > 0 else 0

        record = f"{self.stats['wins']}-{self.stats['losses']}-{self.stats['ties']}"
        return record, win_rate

    def display_team_info(self):
        """Display complete team information"""
        record, win_rate = self.get_team_record()

        print(f"🏆 {self.team_name} ({self.sport})")
        print(f"Record: {record} ({win_rate:.1f}% win rate)")
        print(f"Total Games: {len(self.games)}")
        print("-" * 40)

        print("👥 Roster:")
        for player in self.players:
            print(f"  #{player['number']:2d} {player['name']} ({player['position']})")

        if self.games:
            print("\n📊 Recent Games:")
            for game in self.games[-5:]:  # Show last 5 games
                print(f"  vs {game['opponent']}: {game['our_score']}-{game['their_score']} ({game['result']})")

# Test team management
team = TeamManager("Eagles", "Basketball")

# Add players
team.add_player("Mike Johnson", 23, "Point Guard")
team.add_player("Tom Wilson", 15, "Center")
team.add_player("Sam Davis", 12, "Forward")
team.add_player("Alex Brown", 8, "Shooting Guard")

# Record games
team.record_game("Lions", 78, 65)  # Win
team.record_game("Tigers", 72, 85)  # Loss
team.record_game("Bears", 90, 78)  # Win

team.display_team_info()
```

### Question 148: School Event Planner

**Plan school events with budget and resource management**

**Answer:**

```python
class EventPlanner:
    def __init__(self, event_name, date, budget):
        self.event_name = event_name
        self.date = date
        self.budget = budget
        self.expenses = []
        self.resources = {}
        self.attendees = []

    def add_expense(self, category, amount, description):
        """Add expense to event"""
        expense = {"category": category, "amount": amount, "description": description}
        self.expenses.append(expense)
        return expense

    def add_resource(self, resource_name, quantity, cost_per_unit):
        """Add resource needed for event"""
        self.resources[resource_name] = {
            "quantity": quantity,
            "cost_per_unit": cost_per_unit,
            "total_cost": quantity * cost_per_unit
        }

    def add_attendee(self, name, student_id, grade_level):
        """Add attendee to event"""
        attendee = {"name": name, "id": student_id, "grade": grade_level}
        self.attendees.append(attendee)

    def calculate_costs(self):
        """Calculate total event costs"""
        total_expenses = sum(expense["amount"] for expense in self.expenses)
        total_resources = sum(resource["total_cost"] for resource in self.resources.values())
        total_cost = total_expenses + total_resources

        return {
            "expenses": total_expenses,
            "resources": total_resources,
            "total": total_cost
        }

    def check_budget(self):
        """Check if event is within budget"""
        costs = self.calculate_costs()
        remaining = self.budget - costs["total"]

        if remaining >= 0:
            return f"✅ Within budget! ${remaining:.2f} remaining"
        else:
            return f"❌ Over budget by ${abs(remaining):.2f}"

    def generate_report(self):
        """Generate complete event planning report"""
        costs = self.calculate_costs()

        print(f"🎉 {self.event_name} - Event Plan")
        print(f"📅 Date: {self.date}")
        print(f"💰 Budget: ${self.budget:.2f}")
        print(f"📊 Total Cost: ${costs['total']:.2f}")
        print(self.check_budget())

        print(f"\n👥 Attendees: {len(self.attendees)} people")
        if self.attendees:
            grade_distribution = {}
            for attendee in self.attendees:
                grade = attendee["grade"]
                grade_distribution[grade] = grade_distribution.get(grade, 0) + 1

            print("Grade Distribution:")
            for grade, count in grade_distribution.items():
                print(f"  Grade {grade}: {count} students")

        print(f"\n💵 Expenses: ${costs['expenses']:.2f}")
        if self.expenses:
            for expense in self.expenses:
                print(f"  {expense['category']}: ${expense['amount']:.2f} ({expense['description']})")

        print(f"\n📦 Resources: ${costs['resources']:.2f}")
        if self.resources:
            for resource, details in self.resources.items():
                print(f"  {resource}: {details['quantity']} × ${details['cost_per_unit']:.2f} = ${details['total_cost']:.2f}")

# Test event planning
spring_festival = EventPlanner("Spring Festival 2024", "May 15, 2024", 2500.00)

# Add expenses
spring_festival.add_expense("Entertainment", 800.00, "Live band and DJ")
spring_festival.add_expense("Food", 600.00, "Catering for attendees")
spring_festival.add_expense("Decorations", 300.00, "Flowers, banners, and lights")

# Add resources
spring_festival.add_resource("Tables", 20, 15.00)
spring_festival.add_resource("Chairs", 200, 2.50)
spring_festival.add_resource("Sound System", 1, 200.00)

# Add attendees
spring_festival.add_attendee("Alice Johnson", "S001", 9)
spring_festival.add_attendee("Bob Smith", "S002", 10)
spring_festival.add_attendee("Carol Davis", "S003", 11)

spring_festival.generate_report()
```

---

## 🎓 **School-Focused Practice Problems Summary**

Congratulations! You've completed **148 comprehensive Python practice problems** specifically designed with school scenarios in mind. Here's what you've learned:

### **📚 Core Programming Concepts Covered:**

1. **Conditional Logic (if/else)** - School grading, attendance checking, lunch systems, field trip eligibility
2. **Loops (for/while)** - Counting students, tracking homework, processing class data, roll call systems
3. **Functions** - Grade calculators, attendance trackers, lunch systems, school schedules, bell systems
4. **Object-Oriented Programming** - Student management, cafeteria systems, grade books, classroom management
5. **Advanced Topics** - Data structures, algorithms, school management systems

### **🏫 School Scenarios Practiced:**

- **Grade Management**: Report cards, GPA calculations, honor roll checking, test score averaging
- **Attendance Systems**: Daily roll call, absence tracking, student counting
- **Cafeteria Management**: Lunch money tracking, meal purchases, account balances
- **Classroom Management**: Student tracking, seating arrangements, classroom capacity
- **School Scheduling**: Class periods, bell schedules, lunch times, daily activities
- **Student Activities**: Sports teams, clubs, field trips, event planning
- **Academic Tools**: Grade calculators, study planners, homework tracking
- **School Technology**: Computer lab access, student portals, login systems
- **Administrative Tasks**: Bus schedules, resource allocation, permission systems

### **💡 Key Learning Outcomes:**

- **Real-world Problem Solving**: Each concept taught through familiar school scenarios
- **Comprehensive Coverage**: From basic concepts to advanced programming techniques
- **Practical Application**: Ready-to-use code for school-related tasks
- **Progressive Difficulty**: Builds skills systematically from beginner to advanced
- **School Relevance**: All examples directly relate to student experiences

### **🚀 Next Steps for Students:**

1. **Practice Regularly**: Work through problems daily for 30 minutes
2. **Modify Examples**: Adapt the code for your own school's specific needs
3. **Create Your Own**: Design programs for your school's unique scenarios
4. **Build Projects**: Combine multiple concepts into larger school management systems
5. **Share Knowledge**: Teach concepts to classmates to reinforce learning
6. **Real Application**: Use these skills to help automate school tasks

### **📖 Subject-Specific Applications:**

- **Math Class**: Create calculators for complex formulas and statistics
- **Science Lab**: Build data analysis tools for experiments and lab reports
- **History Class**: Develop timeline generators and date calculators
- **Business Class**: Create budgeting tools for school activities and fundraisers
- **Art Class**: Design pattern generators and color mixing tools
- **Computer Science**: Use these as building blocks for larger applications

---

## 🏆 **You're Now Ready To:**

✅ Write conditional statements for school scenarios (grading, attendance, eligibility)
✅ Use loops to process student data efficiently (class lists, grade calculations)
✅ Create reusable functions for school tasks (grade calculators, attendance systems)
✅ Build complete school management systems (cafeteria, library, student portals)
✅ Apply advanced programming concepts to real school problems
✅ Develop practical tools that can be used in actual school environments

**Keep coding, keep learning, and most importantly - have fun building programs that make school life easier for everyone!** 🎓✨

---

_This practice guide combines comprehensive Python programming education with familiar school environments, making learning both practical and engaging for students. Every example connects programming concepts to real school experiences students encounter every day._
