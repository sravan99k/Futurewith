# 🌍 Python Fundamentals Practice Questions - Universal Edition

_Programming Skills for Real Life - From Beginner to Professional_

**Welcome to your Python learning journey!** 🎉✨

Whether you're a student, professional, hobbyist, or career changer, this guide helps you master Python fundamentals using examples from everyday life. From managing personal finances to organizing household tasks, these 100 carefully designed questions make programming concepts accessible and practical.

**What makes this guide universally accessible:**

- 🏠 **Life-Relevant Examples** - Budget tracking, shopping lists, fitness apps, travel planning
- 📈 **Progressive Learning** - Each question builds on previous concepts (like climbing stairs!)
- 🎯 **Practical Applications** - Solve real problems you encounter daily
- 💼 **Career-Focused** - Skills that transfer to any professional environment
- 🌟 **Any Background** - No prior experience needed, just curiosity and persistence

**How to use this guide effectively:**

1. 🤔 **Try First** - Attempt each question before looking at solutions
2. 💻 **Code Along** - Run examples to see concepts in action
3. 🧪 **Experiment** - Modify examples to fit your own needs
4. 📝 **Take Notes** - Write down insights for future reference
5. 🤝 **Share & Teach** - Help others solidify your understanding

**Learning Tips:**

- **Visual Learners:** Draw diagrams, create flowcharts
- **Hands-On Learners:** Type code and experiment immediately
- **Analytical Learners:** Focus on the "why" behind each concept
- **Beginners:** Start slowly, practice regularly
- **Experienced:** Jump to challenge sections

**Remember:** Programming is a skill anyone can learn with practice. Focus on understanding concepts rather than memorizing syntax. Let's build your Python foundation! 🚀

## 🎯 Your Learning Journey - Question Categories:

1. **[🚀 Getting Started (Questions 1-20)](#getting-started)** - First steps into Python programming
2. **[📊 Working with Data (Questions 21-40)](#working-with-data)** - Storing and managing information
3. **[🔢 Math & Logic (Questions 41-60)](#math-logic)** - Operators, calculations, and decision-making
4. **[💡 Problem Solving (Questions 61-80)](#problem-solving)** - Building practical programs for daily life
5. **[🚀 Advanced Applications (Questions 81-100)](#advanced-applications)** - Complex projects and algorithms

---

## 🚀 Getting Started - Your Python Journey Begins (Questions 1-20)

### Question 1: Your First Python Moment! 🎉

**Before you start coding, you want to verify Python is installed on your computer. What command checks your Python installation?**

_Think of this as checking if you have the right tools before starting a project!_

**Answer:** `python --version` or `python -V`

**Why this matters:** Checking your Python version ensures compatibility with code examples and prevents potential issues. Version 3.x is the current standard and most widely supported.

### Question 2: Understanding Python Version Numbers 🔢

**You see Python 3.12.2 installed on your computer, but what do these numbers actually mean?**

_Think of it like software releases - major updates, new features, and bug fixes!_

**Answer:**

- **3** = Major version (significant changes that may affect compatibility)
- **12** = Minor version (new features added, backward compatible)
- **2** = Patch version (bug fixes and security updates)

**Understanding versions helps you:** Choose compatible libraries, troubleshoot issues, and stay updated with security patches.

### Question 3: The .py File Extension - Why Files Need Special Names? 🤔

**Every Python file you've seen ends with .py - like `my_project.py` or `calculator.py`. Why is this important?**

_Think of it like file formats - .pdf for documents, .jpg for images, .py for Python code!_

**Answer:**
✅ Your computer recognizes it as Python code (like knowing which app opens which file type!)
✅ Text editors provide syntax highlighting (color-coded, easier to read)
✅ Python interpreter can find and run your program
✅ You can easily organize and identify Python files

**Pro Tip:** Never rename a .py file - it would be like changing a file format and expecting it to work the same way!

### Question 4: Understanding Python's >>> Prompt 💬

**When you open Python interactively, you see this mysterious >>> symbol everywhere. What is Python trying to tell you?**

_It's like a conversation prompt - Python is waiting for your input and ready to respond!_

**Answer:** The >>> is Python's interactive prompt that means:

- "Hello! I'm listening and ready to help!"
- "Give me a command and I'll execute it immediately!"
- "I'm here to help you solve problems!"

**Real-world analogy:** Just like a customer service chat shows "Agent is typing...", Python shows >>> to show it's ready and waiting for your input.

### Question 5: Your Very First Python Program! 🌟

**Time for your coding debut! Write a program that makes Python display a message. This is like writing your first line in any new programming language!**

**Answer:**

```python
print("Hello, World!")
```

**What just happened?** 🎉

- You just created your first Python program!
- The `print()` function displays output to the screen
- Whatever is in quotes gets displayed exactly as written
- This is the foundation of all programs you'll build!

**Try this:** Change "Hello, World!" to "Hello, [Your Name]!" or "My first Python program!" - make it your own!

### Question 6: The Great Indentation Debate - Spaces vs Tabs! 🏠

**You've written some code but accidentally mixed tabs and spaces. What happens now?**

_Think of it like formatting a document where some paragraphs use tabs and others use spaces - it looks messy and confuses the reader!_

**Answer:**
❌ Python gets confused and shows an error!
❌ You'll see an `IndentationError` - Python's way of saying "I can't understand your code structure!"
❌ Your program will refuse to run until you fix the formatting

**💡 Pro Tips:**

- Pick ONE method (tabs or spaces) and stick with it throughout your file
- Most programmers use 4 spaces (recommended standard)
- Your text editor can automatically convert tabs to spaces
- Consistent indentation makes code much easier to read and debug

**Remember:** Clean formatting makes your code professional and much easier to maintain!

### Question 7: The Art of Code Comments - Leaving Helpful Notes 📝

**Write a helpful comment that explains what this code is doing. Think of it like adding notes to help yourself or others understand later!**

```python
# Your comment here
user_name = "Emma"
```

**Answer:**

```python
# Store the user's name for personalized greetings
user_name = "Emma"
```

**Why comments are valuable:**
🧠 **For Your Future Self:** In 6 months, you'll thank past-you for explaining what the code does!
🤝 **For Your Team:** Other developers working on the same project will understand your code instantly
📚 **For Code Reviews:** Shows you understand what you're doing and helps others review your work
🎯 **For Debugging:** When something breaks, comments help you find the problem faster

**Golden Rule:** Write comments like you're explaining to a colleague who has never seen this code before!

### Question 8: The Case of the Capital Letters - Same or Different? 🎭

**Look at these two variables. Are they the same person with different grades, or completely different things?**

```python
grade = 95    # Emma's math grade
Grade = 88    # Emma's science grade
```

**Answer:** They are completely different variables! (Think of them as different students - one with a capital G, one without!)

**Why this matters:**
🎯 Python treats `grade` and `Grade` as completely different people (like "Emma" vs "emma" - one could be your friend, the other could be a pet!)
⚠️ This can lead to bugs where you think you're using one variable but you're actually using another
💡 **Best Practice:** Use consistent naming - either always lowercase (`student_grade`) or always descriptive (`studentGrade`)

**Real-world analogy:** It's like having two different lockers - one with your textbooks and one with your lunch. They're both yours, but contain different things!

### Question 9: Naming Your Variables Like a Pro 🏷️

**Your school wants to name everything properly for the yearbook. Which of these variable names would the principal approve?**

**Great Variable Names:**
✅ `student_name` - Clear, descriptive, professional
✅ `_private_grade` - Shows it's private data (underscores are OK!)
✅ `class_of_2025` - Tells you exactly what it contains
✅ `total_attendance` - Action + object = crystal clear!

**Not-So-Great Names:**
❌ `2nd_homework` - Can't start with numbers (like trying to name your homework "2nd Assignment")
❌ `club-president` - Hyphens aren't allowed (Python thinks it's subtraction!)
❌ `a` - Too vague (what does 'a' mean?)
❌ `myname` - Not descriptive enough

**Pro Naming Tips:**
🎯 **Be descriptive:** `student_grade` is better than `sg`
🔤 **Use underscores:** `final_exam_score` instead of `finalexam`
📚 **Follow school rules:** If you wouldn't say it out loud in class, don't use it in code!

**Remember:** Good names = code that reads like English!

### Question 10: The Forbidden Words - Reserved Keywords 🚫

**You really want to name your variable "print" because it sounds cool. But can you? Think of it like trying to name your locker "Principal's Office"!**

**Answer:** NO WAY! 🚫

**Why "print" is off-limits:**
🏛️ **Official Use Only:** "print" is Python's built-in command (like "principal" is an official school title)
⚡ **Critical Function:** Without `print()`, you can't show results to users!
🚫 **Reserved by Python:** Just like certain words in your school handbook, Python has words it keeps for itself

**Examples of Python's 'Forbidden' Words:**

- `if`, `else`, `elif` (decision-making words)
- `for`, `while` (loop words)
- `def` (creating functions)
- `import` (bringing in code)
- `class` (making blueprints)

**Try This Instead:**
Instead of `print = "hello"`, use:

- `message_to_display`
- `print_value`
- `output_text`
- `display_message`

**Remember:** Just like you can't name your dog "Principal," you can't use reserved words for variables!

### Question 11: Writing Epic Code Documentation 📚

**You want to write a detailed explanation about your awesome program. How do you add a multi-line comment that spans several lines?**

_Think of it like writing the introduction page for your science fair project!_

**Answer:**

```python
"""
Program: Student Grade Calculator
Author: Emma Rodriguez, Grade 10
Date: November 2025
Purpose: This program calculates a student's final grade
         by averaging their test scores and displaying
         the result with encouraging feedback.

Features:
- Takes multiple test scores
- Calculates weighted average
- Provides motivational messages
- Shows letter grade equivalent
"""
```

**Why Multi-line Comments Are Amazing:**
📋 **Project Documentation:** Shows the world (and your teacher) what your program does
👥 **Team Projects:** When multiple students work together, comments keep everyone on the same page
📖 **Future Reference:** When you forget how your own code works, comments are like leaving yourself a treasure map!
🏆 **Professional Quality:** Great documentation impresses teachers and future employers

**Pro Tip:** Use triple quotes `"""` at the beginning and end for multi-line comments, and make them informative!

### Question 12: Text Magic - Making Words Dance Together! 🎭

**Watch what happens when we ask Python to combine words! Like mixing paint colors, but with letters!**

```python
print("Math" + " " + "Club")
```

**Answer:** `Math Club`

**What just happened?** ✨

- The `+` operator glued the strings together like super-strong glue
- "Math" + " " + "Club" = "Math Club"
- It's like building words with LEGO blocks!

**More Text Tricks to Try:**

```python
print("Science" + "Club")        # ScienceClub
print("Student" + " " + "ID" + ": " + "12345")  # Student ID: 12345
print("Grade: " + "A+")           # Grade: A+
```

**Real-world Use:** This is how you'd create personalized messages like "Welcome, Emma!" or display student information on screens!

### Question 13: The Text Clone Machine - Repetition Magic! 🔄

**Want to repeat text multiple times? Python can make copies faster than you can say "Copy that!"**

```python
print("Hoop!" * 3)
```

**Answer:** `Hoop!Hoop!Hoop!` (like a basketball team chanting three times!)

**How it Works:**

- The `*` operator repeats the string the number of times you specify
- It's like having a copy machine for text!
- "Hoop!" \* 3 means "give me three copies of 'Hoop!'"

**Awesome Examples to Try:**

```python
print("🎉" * 10)           # 10 party emojis
print("A+" * 5)             # A+A+A+A+A+
print("Grade: " + "=" * 10)  # Grade: ==========
print("Study! " * 4)        # Study! Study! Study! Study!
```

**School Uses:**

- Creating visual borders around your output
- Making celebration messages
- Generating patterns for school projects
- Creating ASCII art for presentations

**Fun Fact:** This works with any text, emoji, or character!

### Question 14: The Great Classroom Division Problem! 📊

**You're organizing students for a group project. You have 15 students and want to make groups of 4. How many complete groups can you make?**

**Answer:** 3 complete groups (using floor division `15 // 4`)

**What's Happening Behind the Scenes:**

- 15 ÷ 4 = 3.75 (mathematically)
- But you can't have 0.75 of a student!
- Floor division `//` gives you only the whole groups
- That means 3 groups of 4 students each
- But wait... you have 3 students left over! (We'll talk about remainders next!)

**Real School Scenarios:**

- 🚌 **Bus Planning:** 45 students ÷ 8 seats per bus = 5 buses with 5 students left over
- 🍕 **Pizza Party:** 23 students ÷ 8 slices per pizza = 2 full pizzas with 7 slices left
- 📚 **Library Groups:** 29 students ÷ 6 study rooms = 4 rooms with 5 students left over

**Try These:**

```python
print(20 // 6)    # How many complete teams of 6 from 20 students?
print(100 // 30)  # How many complete classes of 30 from 100 students?
```

**Remember:** Floor division is like dealing cards - you only count complete, even groups!

### Question 15: The Leftover Mystery - Finding Remainders! 🍕

**You have 17 students who want to form teams of 5 for a science project. Some students will be left without a team. How many?**

**Answer:** 2 students will be left out (17 % 5 = 2)

**Why Remainders Matter:**

- Sometimes things don't divide perfectly (like sharing 17 cookies among 5 friends!)
- The `%` operator tells you what's left over
- It's super useful for:
  - 🚶 **Odd/Even detection:** `number % 2` tells you if something is odd or even
  - ⏰ **Time calculations:** `minutes % 60` gives you leftover minutes
  - 🎯 **Cycling through groups:** `student_id % 4` assigns students to 4 groups

**School Examples:**

```python
students = 17
team_size = 5
leftover = students % team_size  # 2 students left over

# Check if student ID is even or odd
student_id = 12345
is_even = (student_id % 2 == 0)

# Find which class period (1-6) a student should go to
student_num = 25
class_period = (student_num % 6) + 1  # Results in 1-6
```

**Quick Tip:** If `remainder == 0`, everything divides perfectly!

### Question 16: The Great Number Type Debate - Integer vs Float! 🔢

**You see two numbers: 5 and 5.0. They look almost the same, but Python sees them as different 'people.' Why?**

**Answer:**

- **5** = Integer (int) - whole students, whole apples, whole books
- **5.0** = Float (float) - decimal points, fractions, measurements

**Real School Examples:**

**Integers (Whole Numbers):**

```python
students_in_class = 25      # You can't have 25.5 students!
books_on_shelf = 12         # Whole books only
grade_level = 10            # 10th grade, not 10.3
```

**Floats (Decimal Numbers):**

```python
gpa = 3.75                  # GPA can have decimals
average_score = 87.5        # Class average
temperature = 72.3          # Weather in degrees
```

**Why It Matters:**

- Division behavior: `7 / 2 = 3.5` (float), but `7 // 2 = 3` (integer)
- Memory usage: Floats take up more space
- Calculations: Some operations work differently
- Display: `5` shows as "5", but `5.0` shows as "5.0"

**Fun Fact:** In Python, `5` and `5.0` are different types, but they're 'compatible' - you can add them together: `5 + 5.0 = 10.0`!

### Question 17: The Yes-No Game - Understanding Boolean Values! ✅❌

**Python has two very important words that answer questions: True and False. These are called 'boolean' values. What are they and why are they so powerful?**

**Answer:** The two boolean values are `True` and `False`

**What They Mean:**

- **True** = YES, Correct, Pass, 1, On, Approved
- **False** = NO, Incorrect, Fail, 0, Off, Denied

**School Examples:**

```python
# Attendance tracking
student_present = True
homework_submitted = False

# Grade checking
passed_exam = (score >= 60)  # True if score is 60 or higher
needs_help = (grade < 70)    # True if grade is below 70

# Club eligibility
can_join_club = (age >= 13) and (has_permission == True)
```

**Why Booleans Are Amazing:**

- 💭 **Decision Making:** "Should I go to the party?" = `is_weekend and no_homework`
- 🎯 **Smart Programs:** Your code can make choices automatically
- 🔍 **Filtering:** Find students who passed, clubs that are full, etc.
- 📊 **Data Analysis:** Count how many students scored above 80%

**Quick Practice:**

```python
exam_score = 85
is_A_grade = (exam_score >= 90)    # What will this be?
has_passed = (exam_score >= 60)     # And this?
```

**Remember:** Everything in programming comes down to True or False - it's the foundation of all decisions!

### Question 18: The Data Detective - Identifying Variable Types! 🕵️

**You've got a variable, but you're not sure if it's a number, text, or something else. How do you play detective and find out?**

**Answer:** Use the `type()` function to investigate!

**Examples:**

```python
student_name = "Emma"
student_age = 16
student_gpa = 3.85
is_honor_roll = True

print(type(student_name))    # <class 'str'> - string (text)
print(type(student_age))     # <class 'int'> - integer (whole number)
print(type(student_gpa))     # <class 'float'> - float (decimal)
print(type(is_honor_roll))   # <class 'bool'> - boolean (True/False)
```

**Why Type Checking Is Useful:**

- 🐛 **Debugging:** When your program crashes, check if you're using the right data type
- 📝 **Input Validation:** Make sure users enter numbers, not letters
- 🔄 **Data Conversion:** Know when to convert strings to numbers
- 🏫 **School Management:** Sort students by different criteria correctly

**Real School Scenario:**

```python
# Getting student input (always comes as string!)
age_input = input("Enter your age: ")      # This is "16" (string)
print(type(age_input))                   # <class 'str'>

# Converting to integer for math
age_number = int(age_input)             # Now it's 16 (integer)
next_year_age = age_number + 1          # Can do math now!
```

**Pro Tip:** When debugging, always check your data types first!

### Question 19: Making Python Ask Questions - User Input Magic! 💬

**Write code that makes Python ask a student for their name and remember it. Think of it like creating an interactive school registration form!**

**Answer:**

```python
student_name = input("What's your name? ")
```

**What Just Happened?** 🎭

1. Python displays the question: "What's your name? "
2. Waits for user to type something and press Enter
3. Stores whatever was typed in the variable `student_name`
4. Program continues with the stored information

**Interactive Examples to Try:**

```python
# School Registration System
print("🎓 Welcome to Jefferson High! 🎓")
name = input("What's your full name? ")
grade = input("What grade are you in? (9-12) ")
club = input("What club interests you? ")

print(f"\nWelcome, {name}!")
print(f"Grade: {grade}")
print(f"Interested in: {club} Club")
```

**Important Notes:**
⚠️ **Input Always Returns Text:** Even if user types "16", it's stored as "16" (string), not 16 (number)
⏸️ **Program Pauses:** The program stops and waits - like a teacher waiting for student answers
📝 **The Space Matters:** The space after "? " makes input look nicer

**Try This:**

```python
age = input("How old are you? ")
next_year_age = age + 1  # This will ERROR! Why?
```

**Answer:** `age` is a string ("16"), so you can't add 1 to it mathematically!

### Question 20: The Super-Powered Formatting - F-Strings! 🌟

**Create a personalized message that welcomes a student by name and shows their score. Make it look professional and friendly!**

**Answer:**

```python
name = "Emma"
points = 95
print(f"Hi {name}, you got {points} points!")
```

**What Makes F-Strings So Amazing? 🚀**

- **Fast:** F-strings are the quickest way to format text
- **Flexible:** Mix text, variables, and calculations seamlessly
- **Beautiful:** Create polished, professional output
- **Easy:** Just put an 'f' before your quotes and use {}

**More F-String Magic:**

```python
# Grade Report Generator
student = "Alex"
math = 88
science = 92
english = 85

print(f"🎓 Grade Report for {student}")
print(f"Math: {math}/100")
print(f"Science: {science}/100")
print(f"English: {english}/100")
total = math + science + english
print(f"Total Points: {total}/300")
print(f"Average: {total/3:.1f}%")
```

**Advanced F-String Tricks:**

```python
# Formatting numbers
price = 15.50
print(f"Lunch costs ${price:.2f}")        # $15.50

# Adding calculations directly
score = 85
print(f"You scored {score}/100 = {score}%!")

# Emoji + Variables = Fun!
student = "Sam"
achievement = "Perfect Attendance!"
print(f"🏆 Congratulations {student}! {achievement}")
```

**Real-World Uses:**

- Report cards and grade summaries
- Personalized welcome messages
- Score displays for games and quizzes
- Name tags and badges
- Celebration announcements

**Remember:** F-strings make your output look professional and are much easier to read than other formatting methods!

---

## 📊 Working with Data - Managing Information Like a Pro (Questions 21-40)

### Question 21: The Text Cleanup Crew - Removing Extra Spaces! 🧹

**People sometimes type their names with accidental spaces before or after (like " Emma "). How do we clean this up?**

**Answer:** The `.strip()` method removes unwanted spaces from the beginning and end of text!

**What it does:**

- Removes spaces at the beginning of text
- Removes spaces at the end of text
- Keeps important spaces in the middle (like "Emma Johnson")
- Works on any text, not just names

**Real-World Examples:**

```python
# Input cleanup for user forms
user_name = "   Emma Rodriguez   "
clean_name = user_name.strip()  # "Emma Rodriguez"

# Cleaning form responses
feedback = "  This is my response  "
clean_feedback = feedback.strip()     # "This is my response"

# Contact information
email = "  user@example.com  "
print(f"Email: {email.strip()}")  # Email: user@example.com
```

**Why This Matters:**

- 📝 **Forms:** When users fill out digital forms, they might accidentally add spaces
- 🔍 **Search:** "Emma" and " Emma " are different to computers
- 🏆 **Data Quality:** Clean data makes your programs work better
- 👥 **User Experience:** Professional appearance and better functionality

**Try This:**

```python
greeting = "   Hello, User!   "
print(f"Before: '{greeting}'")
print(f"After: '{greeting.strip()}'")
```

### Question 22: The Text Amplifier - Going from Small to BIG! 📢

**You need to create a banner for "fitness club" but want it to look bold and exciting. How do you make it all uppercase?**

**Answer:** Use the `.upper()` method to transform text to uppercase!

**Answer:** `"fitness club".upper()` → `"FITNESS CLUB"`

**Why Uppercase is Useful:**

- 📋 **Signage & Banners:** Makes announcements more eye-catching
- 📧 **Email Headers:** Important notifications stand out
- 🏷️ **Labels & Tags:** Makes text more prominent
- 🎯 **Instructions:** Urgent directions get attention
- 📚 **Headings:** Professional document formatting

**Practical Examples:**

```python
# Event announcements
event_name = "fitness challenge"
print(f"WELCOME TO THE {event_name.upper()}!")  # WELCOME TO THE FITNESS CHALLENGE!

# Important notifications
alert = "system maintenance"
print(f"IMPORTANT: {alert.upper()}")  # IMPORTANT: SYSTEM MAINTENANCE

# Achievements
winner = "alex"
achievement = "completed marathon"
print(f"🏆 CONGRATULATIONS {winner.upper()}! YOU {achievement.upper()}!")
```

**Related Methods:**

- `.lower()` - Convert to lowercase (for consistent searching)
- `.title()` - Make first letter of each word capital
- `.capitalize()` - Make only first letter of text capital

**Professional Use:**

```python
customer_name = "sarah johnson"
print(f"Certificate of Achievement")
print(f"Awarded to: {customer_name.title()}")
print(f"For excellence in: {service.upper()}")
```

**Best Practice:** Use `.upper()` for emphasis and headings, but avoid excessive use (all caps can feel like shouting!)

### Question 23: The Type Transformer - Converting Text to Numbers! 🔄

**You get a student's grade as text ("98") from a form, but you need to do math with it. How do you transform it into a number?**

**Answer:** Use `int()` to convert strings to integers - like translating between languages!

**Answer:** `int("98")` → `98`

**Why This is Crucial for Students:**

- 📝 **Form Data:** All input() returns text, but grades need to be numbers for calculations
- 📊 **Math Operations:** You can't add two strings together like "95" + "5"
- 📈 **Comparisons:** "98" > "95" doesn't work, but 98 > 95 does!
- 🏆 **Display:** Numbers can be formatted, rounded, and calculated with

**School Scenario - Grade Calculator:**

```python
# Getting test scores from user
math_score = input("Enter math score: ")          # "92" (string)
science_score = input("Enter science score: ")   # "88" (string)

# Converting to numbers for calculations
math_num = int(math_score)        # 92 (integer)
science_num = int(science_score)  # 88 (integer)

# Now we can do math!
total = math_num + science_num
average = total / 2
print(f"Your average is {average}%")
```

**Common Student Mistakes to Avoid:**

```python
# ❌ This won't work!
score1 = "95"
score2 = "5"
total = score1 + score2  # Error! Can't add strings this way

# ✅ This works!
score1 = int("95")
score2 = int("5")
total = score1 + score2  # Returns 100
```

**What Happens with Invalid Input:**

```python
# If user types letters instead of numbers:
age = int("sixteen")  # 💥 ValueError! Program crashes
```

**Pro Tip:** Always validate user input before converting (we'll learn this soon!)

### Question 24: The Decimal Detective - Working with Precise Numbers! 🧮

**In math class, you need Pi (3.14) for circle calculations, but you have it as text. How do you convert it to a decimal number for math operations?**

**Answer:** Use `float()` to handle decimal numbers - perfect for GPA, measurements, and scientific calculations!

**Answer:** `float("3.14")` → `3.14`

**Why Floats Are Essential for Students:**

- 📊 **GPA Calculations:** 3.75 needs precision
- 📏 **Measurements:** 5.5 feet, 72.3 degrees
- 🔬 **Science Data:** Chemical concentrations, temperatures
- 💰 **Money:** $15.75 lunch costs
- 📈 **Percentages:** 87.5% attendance rate

**School Examples:**

```python
# Math Class - Circle Calculations
pi_text = "3.14159"
pi_number = float(pi_text)  # 3.14159
radius = float(input("Enter radius: "))
area = pi_number * radius ** 2
print(f"Circle area: {area}")

# Science Lab - Temperature Conversion
celsius_text = "25.5"
celsius = float(celsius_text)  # 25.5
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit}°F")

# GPA Calculator
grades = [3.2, 3.8, 3.5, 4.0]  # Already floats
average_gpa = sum(grades) / len(grades)
print(f"GPA: {average_gpa:.2f}")
```

**Float vs Integer - When to Use Which:**

```python
# Use integers for:
student_count = 25       # You can't have 0.5 students
grade_level = 10         # 10th grade, not 10.3
attendance_days = 180    # Whole days

# Use floats for:
gpa = 3.75               # Decimals are expected
temperature = 68.5       # Measurements
money = 12.99            # Currency
percentage = 87.5        # Rates and ratios
```

**Precision Matters:**

```python
# Calculating final grades
homework = 85.5
tests = 92.0
participation = 88.5
final = (homework + tests + participation) / 3
print(f"Final grade: {final:.1f}")  # Shows 88.7
```

**Pro Tip:** Use floats when precision matters, integers when whole numbers make sense!

### Question 25: The Zero Mystery - Understanding Nothing vs Something! 🎭

**You have the number 0, but what happens when Python thinks about it as True or False? Does "nothing" count as false?**

**Answer:** `bool(0)` → `False` (Zero is considered "nothing" or "false" in Python's mind!)

**Why This is Mind-Blowing for Students:**

- 🤔 **Philosophical:** In programming, zero = nothing/false, anything else = something/true
- 🎯 **Practical:** Perfect for checking if students have scores, attendance, etc.
- 🏆 **Grading:** Any score > 0 means they participated
- 👥 **Attendance:** 0 students present = class cancelled

**School Scenarios:**

```python
# Attendance Check
students_present = 0
if students_present:
    print("Class is happening!")
else:
    print("No students came - snow day! ❄️")

# Grade Submission
homework_score = 85
if homework_score:  # Any number > 0 is True
    print("Homework was submitted! ✅")
else:
    print("No homework submitted. ❌")

# Club Members
math_club_members = 5
if math_club_members:  # 5 is True
    print("Math club is active!")
else:
    print("No math club members - need to recruit!")
```

**The Boolean Conversion Rule:**

- `0` → `False` (nothing, empty, zero)
- `1` → `True` (something, presence, one)
- `-5` → `True` (any non-zero number!)
- `""` → `False` (empty text)
- `"hello"` → `True` (non-empty text)
- `[]` → `False` (empty list)
- `[1, 2]` → `True` (non-empty list)

**Real School Application:**

```python
# Checking if student submitted any assignments
assignments_submitted = 0
if assignments_submitted:
    grade = calculate_final_grade()
    print(f"Student can receive: {grade}")
else:
    print("No assignments submitted - grade: F")

# Event planning
event_attendees = 15
if event_attendees:
    print(f"Planning event for {event_attendees} people")
else:
    print("Event cancelled - no interest")
```

**Fun Fact:** This is why in many games, "0 lives" means "game over" - zero equals false/not alive!

### Question 26: The Great Unknown - Understanding None! ❓

**Sometimes you don't know a value yet, or a value doesn't exist. How does Python express "we don't have this information yet"?**

**Answer:** `None` is Python's way of saying "I don't know" or "This doesn't exist yet" - like a blank spot on a report card!

**Real School Examples:**

```python
# Student data that might not be available yet
student_name = "Emma Rodriguez"
final_grade = None  # Grade not calculated yet
club_president = None  # Election hasn't happened
lunch_money = None  # Student hasn't entered amount

# Checking for missing information
if final_grade is None:
    print("Final grade not calculated yet - pending assignments")
else:
    print(f"Final grade: {final_grade}%")
```

**When Students Use None:**

- 📝 **Assignment Status:** `submission_time = None` (not submitted yet)
- 🏆 **Awards:** `student_award = None` (not decided yet)
- 👥 **Group Assignments:** `team_lead = None` (not assigned yet)
- 📚 **Library Books:** `book_return_date = None` (still checked out)
- 🚌 **Transportation:** `bus_route = None` (not assigned yet)

**The Difference Between None and 0:**

```python
# None = Unknown/Missing
homework_score = None  # We don't know the score yet

# 0 = Known value of zero
homework_score = 0     # We know they scored zero

# Checking for missing data
if homework_score is None:
    print("No submission recorded")
elif homework_score == 0:
    print("Submitted but got zero points")
else:
    print(f"Score: {homework_score} points")
```

**Practical School Application:**

```python
# Student information system
student_record = {
    "name": "Alex Johnson",
    "grade_level": 10,
    "email": None,  # Student hasn't provided email
    "emergency_contact": "555-0123",
    "medical_notes": None  # No medical issues
}

# Fill in missing information
for field, value in student_record.items():
    if value is None:
        print(f"Need to collect: {field}")
    else:
        print(f"Have: {field} = {value}")
```

**Why None is Useful:**

- ✅ **Distinguishes "don't know" from "zero"**
- ✅ **Prevents calculations with missing data**
- ✅ **Makes code more honest about data availability**
- ✅ \*\*Helps track what's been completed vs. what's pending

**Remember:** None is like an empty box labeled "Contents Unknown" - you know there's supposed to be something, but you don't know what yet!

### Question 27: The Great Variable Swap - Trading Places Like a Pro! 🔄

**You need to swap the values of two variables - like trading lunch money between friends. What's the Python way to do this without losing either amount?**

**Answer:** Python has special "parallel assignment" that lets variables trade values instantly!

**Answer:**

```python
points_earned = 85
points_possible = 100
points_earned, points_possible = points_possible, points_earned  # Python magic!
```

**What Just Happened? ✨**

- Python sees both variables simultaneously
- "Give points_earned the value of points_possible (100)"
- "Give points_possible the value of points_earned (85)"
- It's like two people shaking hands and swapping briefcases at the same time!

**Real School Scenarios:**

```python
# Trading seats in class
student_a = "Emma"
student_b = "Alex"
print(f"Before: Emma sits in seat A, Alex sits in seat B")
student_a, student_b = student_b, student_a
print(f"After: {student_a} sits in seat A, {student_b} sits in seat B")

# Swapping test scores for grade comparison
my_score = 87
friend_score = 92
my_score, friend_score = friend_score, my_score
print(f"Mine: {my_score}, Friend's: {friend_score}")

# Club officer positions
president = "Sam"
vice_president = "Jordan"
president, vice_president = vice_president, president
print(f"New President: {president}")
print(f"New Vice President: {vice_president}")
```

**Why This is Amazing:**

- 🚀 **Fast:** No temporary variable needed
- 🎯 **Clean:** Code is easier to read
- 💪 **Reliable:** Can't accidentally lose data
- 🔥 **Pythonic:** Shows you know Python's special features

**The Old Way (Other Languages):**

```python
# This works too, but is more complicated
temp = points_earned      # Save first value
points_earned = points_possible
points_possible = temp
```

**Advanced Swapping:**

```python
# Swap multiple values at once!
a, b, c = 1, 2, 3
print(f"Before: a={a}, b={b}, c={c}")
a, b, c = c, a, b  # Rotate values
print(f"After: a={a}, b={b}, c={c}")  # a=3, b=1, c=2
```

**School Project Ideas:**

- **Team Generator:** Randomly assign students to teams by swapping names
- **Grade Ranker:** Swap scores to sort from highest to lowest
- **Lunch Line:** Rotate who goes first by swapping queue positions
- **Schedule Swapper:** Exchange class periods between students

**Pro Tip:** Use this when you want to exchange values without creating extra variables!

### Question 28: The Shape-Shifting Variables - Dynamic Typing Explained! 🎭

**Imagine a variable that could transform from a number into text and back again. Is that possible? In Python, YES! What's this superpower called?**

**Answer:** Dynamic typing! Variables can change their "personality" (data type) during the program - like a student who can be both a math whiz AND a creative writer!

**Answer:** Variables can change types during the program (like `student_count = 25` then `student_count = "twenty-five"`)

**Mind-Blowing Examples:**

```python
# Watch a variable transform!
student_data = 25        # Integer - number of students
print(type(student_data))  # <class 'int'>

student_data = "twenty-five"  # String - written form
print(type(student_data))     # <class 'str'>

student_data = 25.5      # Float - average age
print(type(student_data))     # <class 'float'>

student_data = True      # Boolean - enrollment status
print(type(student_data))     # <class 'bool'>
```

**Real School Scenarios:**

```python
# Student ID Processing
student_id = "12345"     # String for text operations
student_id = int(student_id)  # Integer for math
student_id = str(student_id)  # String for display

# Grade Analysis
grade = 85              # Number for calculations
grade = "B+"            # Letter for display
grade = 0.85            # Decimal for percentages

# Attendance Tracking
attendance = 180        # Days attended
attendance = "180/180"  # Display format
attendance = True       # Good attendance flag
```

**Why This is Both Awesome and Dangerous:**

**✅ Awesome Because:**

- 🔄 **Flexible:** Adapts to what you need
- 🚀 **Fast Development:** Less rigid rules
- 🎯 **Practical:** Matches how we think about data

**⚠️ Dangerous Because:**

- 🐛 **Bugs:** Variable might not be what you expect
- 🤔 **Confusion:** Other people reading your code might get confused
- 🔍 **Debugging:** Harder to track what type something should be

**Best Practices for Students:**

```python
# ❌ Don't do this (confusing!)
value = 25
value = "hello"
value = True

# ✅ Do this (clear intent)
student_count = 25
student_names = "hello"
has_passed = True

# Or be explicit about changes:
score = 85              # Initial number
score_display = str(score)  # Convert to text for display
```

**School Project Example:**

```python
# Building a student information system
info = "Emma Rodriguez"      # Start with name
info = 16                   # Add age
info = "10th Grade"         # Add grade level
info = [85, 92, 88]         # Add test scores

# This is flexible but can be confusing!
# Better to use different variables:
name = "Emma Rodriguez"
age = 16
grade_level = "10th Grade"
test_scores = [85, 92, 88]
```

**Remember:** Dynamic typing gives you superpowers, but use them wisely! Clear, descriptive variable names help prevent confusion.

### Question 29: The Character Counter - Measuring Text Length! 📏

**You need to check if a student's full name fits in a text field, or you want to create a name tag of the right size. How do you count all the characters?**

**Answer:** Use `len()` function - Python's built-in measuring tape for text!

**Answer:** Use `len(student_name)` (counts all characters including spaces)

**Why Length Matters in Schools:**

- 🏷️ **Name Tags:** "Emma Rodriguez" (14 characters) fits in smaller space than "Alexandria Victoria" (22 characters)
- 📝 **Form Validation:** Make sure student IDs are correct length
- 📊 **Data Analysis:** Find longest/shortest names in class
- 💻 **Database Limits:** Text fields have maximum character limits

**School Examples:**

```python
# Measuring student names
student_name = "Emma Rodriguez"
name_length = len(student_name)  # 14
print(f"{student_name} has {name_length} characters")

# Checking name length for forms
if len(student_name) > 20:
    print("Name too long for database!")
else:
    print("Name fits perfectly!")

# Finding longest name in class
names = ["Alex", "Alexandria Victoria", "Sam", "Jordan"]
longest_name = max(names, key=len)
shortest_name = min(names, key=len)
print(f"Longest: {longest_name} ({len(longest_name)} chars)")
print(f"Shortest: {shortest_name} ({len(shortest_name)} chars)")
```

**What len() Counts:**

- 👤 **Letters:** A, e, m, a (4)
- 📏 **Spaces:** " " counts as 1 character
- 🎯 **Numbers:** "123" = 3 characters
- 🎨 **Symbols:** "@#$%" - each symbol counts
- 📧 **Email addresses:** Everything before and after @

**Real Applications:**

```python
# Student ID Validation
student_id = "STU2025001"
if len(student_id) != 10:
    print("Invalid ID format!")
else:
    print("ID format OK")

# Password Strength Checker
password = input("Create password: ")
if len(password) < 8:
    print("Password too short!")
else:
    print("Password length OK")

# Text Message for Parents
message = f"Your student {student_name} had a great day!"
if len(message) > 160:
    print("Message too long for SMS!")
```

**Creative Uses:**

```python
# Creating school motto with right length
motto = "Excellence in Everything"
print("🚀 " + "=" * len(motto))
print(motto)
print("=" * len(motto))

# Output:
# 🚀 =====================
# Excellence in Everything
# =====================
```

**Fun Challenge:**

```python
# Find average name length in your class
class_names = ["Emma", "Alexander", "Sam", "Jordan", "Maria"]
average_length = sum(len(name) for name in class_names) / len(class_names)
print(f"Average name length: {average_length:.1f} characters")
```

**Pro Tip:** len() also works on lists, dictionaries, and other collections - it's Python's universal measuring tool!

### Question 30: The Word vs. Letters Showdown - Lists vs Strings! 📝

**You have the word "Math" but want to treat it as individual letters. What's the difference between keeping it as a word versus breaking it into letters?**

**Answer:**

- `list("Math")` → `['M', 'a', 't', 'h']` (breaks into individual characters)
- `"Math"` → `'Math'` (keeps as single text)

**Why This Matters in School:**

- 📝 **Name Analysis:** Count vowels in student names
- 🔤 **Word Games:** Create acrostic poems or word scrambles
- 📊 **Data Processing:** Analyze character patterns in student IDs
- 🎨 **Creative Projects:** Generate letter-based artwork

**Real School Examples:**

```python
# Student Name Analysis
subject = "Mathematics"
letter_list = list(subject)      # ['M', 'a', 't', 'h', 'e', 'm', 'a', 't', 'i', 'c', 's']
word_string = subject             # 'Mathematics'

# Counting vowels in a name
student_name = "Emma Rodriguez"
vowels = [char for char in student_name if char.lower() in 'aeiou']
print(f"Vowels in {student_name}: {len(vowels)}")

# Finding repeated letters
club_name = "Science Club"
repeated = [char for char in set(club_name) if club_name.count(char) > 1]
print(f"Letters that repeat: {repeated}")
```

**When to Use Each:**

**Use Lists When:**

- 🧩 **Manipulating Individual Characters:** Replace, remove, count specific letters
- 🔄 **Reordering:** Sort letters alphabetically
- 🏗️ **Building New Words:** Create anagrams or word combinations
- 📈 **Character Analysis:** Find patterns, frequencies, positions

**Use Strings When:**

- 📋 **Displaying Names:** Show full student names
- 🔍 **Searching:** Find if "Math" appears in class names
- 🎯 **Comparing:** Check if two words are the same
- 📝 **Output:** Present information to users

**Cool School Projects:**

```python
# Acrostic Poem Generator
name = "ALEX"
poem_words = ["Amazing", "Lively", "Enthusiastic", "eXtraordinary"]

print("Acrostic Poem:")
for letter, word in zip(name, poem_words):
    print(f"{letter}: {word}")

# Output:
# A: Amazing
# L: Lively
# E: Enthusiastic
# X: eXtraordinary

# Name Cipher
name = "SARAH"
cipher_name = ''.join(chr(ord(char) + 1) for char in name)
print(f"Coded name: {cipher_name}")  # TBSBI
```

**Advanced Fun:**

```python
# Student ID Character Analysis
student_id = "STU2025001"
id_chars = list(student_id)  # Break into characters
print(f"ID contains {len(id_chars)} characters")
print(f"First 3 characters: {id_chars[:3]}")
print(f"Last 3 characters: {id_chars[-3:]}")

# Count digit vs letter ratio
letters = sum(1 for char in id_chars if char.isalpha())
digits = sum(1 for char in id_chars if char.isdigit())
print(f"Letters: {letters}, Digits: {digits}")
```

**Pro Tip:** Convert between lists and strings easily:

- `list("hello")` → breaks into letters
- `''.join(['h', 'e', 'l', 'l', 'o'])` → combines into word

### Question 31: The Text Replacement Magic - Find and Replace! 🔄

**You wrote an essay using the word "homework" but your teacher prefers "assignment." How do you replace every instance automatically?**

**Answer:** Use `.replace()` - Python's find-and-replace tool!

**Answer:** `text.replace("homework", "assignment")`

**Why This is Super Useful for Students:**

- 📝 **Essay Writing:** Fix consistent terminology across papers
- 🔍 **Report Updates:** Change outdated information system-wide
- 🏷️ **Data Cleaning:** Standardize inconsistent labels
- 🎨 **Creative Writing:** Transform stories with word swaps

**School Examples:**

```python
# Essay Revision
paragraph = "I love my homework assignments because homework helps me learn."
revised = paragraph.replace("homework", "assignment")
print(revised)  # "I love my assignment assignments because assignment helps me learn."

# Club Announcement Update
event_text = "Please bring your lunch money to the fundraiser."
new_text = event_text.replace("lunch money", "donations")
print(new_text)  # "Please bring your donations to the fundraiser."

# Grade Report Formatting
report = "Student Grade: A Student Score: 95"
clean_report = report.replace("Student ", "")
print(clean_report)  # "Grade: A Score: 95"
```

**Advanced Replacement Techniques:**

```python
# Multiple replacements in one go
text = "Math class is awesome! Math homework is fun!"
text = text.replace("Math", "Science")  # All math becomes science
text = text.replace("homework", "projects")  # Then homework becomes projects
print(text)  # "Science class is awesome! Science projects is fun!"

# Case-insensitive replacement (advanced)
import re
text = "HOMEWORK homework Homework"
new_text = re.sub(r'homework', 'assignment', text, flags=re.IGNORECASE)
print(new_text)  # "assignment assignment assignment"
```

**Real School Applications:**

```python
# Student Information System
old_format = "Student Name: John Doe"
new_format = old_format.replace("Student Name:", "Full Name:")
print(new_format)  # "Full Name: John Doe"

# Calendar Updates
schedule = "Monday: Math Tuesday: Science Wednesday: Math"
updated = schedule.replace("Math", "Literature")
print(updated)  # "Monday: Literature Tuesday: Science Wednesday: Literature"

# Prize List Corrections
prizes = "1st place: Laptop 2nd place: Tablet 3rd place: Headphones"
corrected = prizes.replace("Tablet", "Smart Watch")
print(corrected)
```

**Creative Uses:**

```python
# Word Game Generator
sentence = "The quick brown fox jumps over the lazy dog"
sentence = sentence.replace("fox", "student")
sentence = sentence.replace("dog", "teacher")
print(sentence)  # "The quick brown student jumps over the lazy teacher"

# Secret Code
message = "MEET AT NOON"
coded = message.replace("E", "3").replace("O", "0")
print(coded)  # "M33T AT N00N"
```

**Important Notes:**

- 🔄 **Case Sensitive:** "Math" ≠ "math" (by default)
- 🔢 **Counts Replacements:** Returns number of replacements made
- 📝 **Original Unchanged:** Creates new string, original stays same

**Try This Challenge:**

```python
# Replace multiple words at once
text = "The cat sat on the mat"
words_to_replace = {"cat": "dog", "mat": "bed"}
for old, new in words_to_replace.items():
    text = text.replace(old, new)
print(text)  # "The dog sat on the bed"
```

**Pro Tip:** Use `.replace()` for simple substitutions, but for complex patterns (like replacing "Math" only when it's a whole word), use regular expressions!

### Question 32: The Great Class List Split - Turning Text into Items! 📚

**Your teacher gives you a list of classes as one long string: "Math,Science,History,Art" but you want each subject as a separate item. How do you break them apart?**

**Answer:** Use `.split(",")` - Python's magic separator that turns one string into many!

**Answer:** `"Math,Science,History,Art".split(",")` → `['Math', 'Science', 'History', 'Art']`

**Why Splitting is Essential for Students:**

- 📊 **Data Processing:** Convert CSV files into usable lists
- 📝 **Form Data:** Handle comma-separated inputs from online forms
- 🎯 **Student Lists:** Turn "Emma,Alex,Jordan" into individual names
- 📅 **Schedule Management:** Parse complex timetable strings

**Real School Scenarios:**

```python
# Processing Student Names from Form
name_list = "Emma Rodriguez,Alex Johnson,Sarah Wilson,Mike Chen"
students = name_list.split(",")
print(f"Class has {len(students)} students:")
for i, student in enumerate(students, 1):
    print(f"{i}. {student.strip()}")

# Parsing Grade Information
grade_data = "Math:95,Science:88,History:92,Art:85"
subjects = grade_data.split(",")
for subject_info in subjects:
    subject, score = subject_info.split(":")
    print(f"{subject}: {score}%")

# Event Planning with Participants
event_list = "volleyball,basketball,soccer,tennis,swimming"
sports = event_list.split(",")
print("Available sports:")
for sport in sports:
    print(f"• {sport.title()}")
```

**Different Separators for Different Needs:**

```python
# Space-separated data
sentence = "Welcome to Jefferson High School"
words = sentence.split(" ")  # ['Welcome', 'to', 'Jefferson', 'High', 'School']

# Slash-separated dates
date = "11/01/2025"
month, day, year = date.split("/")  # '11', '01', '2025'

# Period-separated sentences
text = "First sentence. Second sentence. Third sentence."
sentences = text.split(". ")  # ['First sentence', 'Second sentence', 'Third sentence']

# Multi-character separator
data = "Name|Age|Grade|Email"
info = data.split("|")  # ['Name', 'Age', 'Grade', 'Email']
```

**School Project Ideas:**

**1. Class Roster Manager:**

```python
roster = input("Enter student names (comma-separated): ")
students = [name.strip() for name in roster.split(",")]
print(f"\nClass Roster ({len(students)} students):")
for student in students:
    print(f"• {student}")
```

**2. Schedule Parser:**

```python
daily_schedule = "Math|9:00-9:50,Science|10:00-10:50,History|11:00-11:50"
periods = daily_schedule.split(",")

print("Today's Schedule:")
for period in periods:
    subject, time = period.split("|")
    print(f"{time}: {subject}")
```

**3. Grade Calculator:**

```python
scores = "85,92,78,96,88"
score_list = [int(score) for score in scores.split(",")]
average = sum(score_list) / len(score_list)
print(f"Average grade: {average:.1f}%")
```

**Advanced Splitting Tricks:**

```python
# Split with limit (only split into 2 parts)
data = "Math,Science,History,Art"
first_two, rest = data.split(",", 2)
print(first_two)  # "Math"
print(rest)       # "Science,History,Art"

# Remove empty entries
text = "Math,,Science,,History"
subjects = [s for s in text.split(",") if s]  # ['Math', 'Science', 'History']

# Case study from form input
user_input = "  Emma  ,  Alex  ,  Jordan  "
clean_names = [name.strip() for name in user_input.split(",")]
print(clean_names)  # ['Emma', 'Alex', 'Jordan']
```

**Real-World CSV Processing:**

```python
# Reading student data from CSV string
csv_data = """Name,Grade,Email
Emma,10,emma@school.edu
Alex,11,alex@school.edu
Jordan,12,jordan@school.edu"""

lines = csv_data.strip().split("\n")
header = lines[0].split(",")
print(f"Columns: {header}")

for line in lines[1:]:
    data = line.split(",")
    student_info = dict(zip(header, data))
    print(f"{student_info['Name']}: Grade {student_info['Grade']}")
```

**Pro Tips:**

- 🎯 **Strip whitespace:** Use `.strip()` after splitting to remove extra spaces
- 🔢 **Convert types:** Often need to convert split strings to numbers or other types
- 📝 **Handle errors:** What if the data doesn't have the expected separators?

**Remember:** `.split()` turns one big thing into many small things - perfect for processing lists and form data!

### Question 33: The Precision Artist - Rounding Numbers Like a Pro! 🎨

**You calculated a GPA as 3.14159, but you want to display it as 3.14 (two decimal places). How do you make numbers look neat and professional?**

**Answer:** Use format strings to control decimal precision!

**Answer:** `"{:.2f}".format(3.14159)` → `"3.14"`

**Why Precision Matters in School:**

- 📊 **GPA Display:** Show 3.75 instead of 3.74999999999
- 💰 **Money:** $12.99 instead of $12.990000
- 📏 **Measurements:** 5.67 feet instead of 5.666666667 feet
- 📈 **Percentages:** 87.50% instead of 87.5%
- 🏆 **Grade Reports:** Clean, professional formatting

**School Examples:**

```python
# GPA Calculator
grades = [3.2, 3.8, 3.5, 4.0]
gpa = sum(grades) / len(grades)
print(f"GPA: {gpa:.2f}")  # GPA: 3.62

# Money Calculation
lunch_cost = 12.99
drink_cost = 2.50
total = lunch_cost + drink_cost
print(f"Total: ${total:.2f}")  # Total: $15.49

# Test Score Average
scores = [85.5, 92.0, 78.5, 96.5]
average = sum(scores) / len(scores)
print(f"Class average: {average:.1f}%")  # Class average: 88.1%
```

**Different Precision Levels:**

```python
# 0 decimal places
pi = 3.14159
print(f"Whole number: {pi:.0f}")  # 3

# 1 decimal place
print(f"One decimal: {pi:.1f}")  # 3.1

# 2 decimal places
print(f"Two decimals: {pi:.2f}")  # 3.14

# 3 decimal places
print(f"Three decimals: {pi:.3f}")  # 3.142

# 5 decimal places
print(f"Five decimals: {pi:.5f}")  # 3.14159
```

**Real-World School Applications:**

**1. Report Card Generator:**

```python
student = "Emma Rodriguez"
subjects = ["Math", "Science", "History", "English"]
scores = [95.5, 88.0, 92.5, 87.75]

print("📊 REPORT CARD")
print("=" * 20)
for subject, score in zip(subjects, scores):
    print(f"{subject:10}: {score:6.1f}%")

print("-" * 20)
final_grade = sum(scores) / len(scores)
print(f"{'FINAL':10}: {final_grade:6.2f}%")
```

**2. Budget Calculator:**

```python
school_supplies = [15.99, 8.50, 12.25, 5.75]
total = sum(school_supplies)
print(f"Total cost: ${total:.2f}")
print(f"With tax (8%): ${total * 1.08:.2f}")
```

**3. Temperature Converter:**

```python
celsius = 25.5
fahrenheit = (celsius * 9/5) + 32
print(f"Temperature: {celsius}°C = {fahrenheit:.1f}°F")
```

**Advanced Formatting Options:**

```python
# Percentage formatting
ratio = 0.875
print(f"Progress: {ratio:.1%}")  # Progress: 87.5%

# Currency formatting
amount = 1234.56
print(f"Amount: ${amount:,.2f}")  # Amount: $1,234.56

# Scientific notation
big_number = 1234567.89
print(f"Big number: {big_number:.2e}")  # Big number: 1.23e+06

# Padding with zeros
student_id = 123
print(f"ID: {student_id:05d}")  # ID: 00123
```

**School Project - Grade Analyzer:**

```python
def analyze_grades(scores):
    average = sum(scores) / len(scores)
    highest = max(scores)
    lowest = min(scores)

    print("📈 GRADE ANALYSIS")
    print(f"Average: {average:.2f}%")
    print(f"Highest: {highest:.1f}%")
    print(f"Lowest: {lowest:.1f}%")
    print(f"Range: {highest - lowest:.1f}%")

    return {
        'average': average,
        'highest': highest,
        'lowest': lowest
    }

# Test with class data
class_scores = [85.5, 92.0, 78.5, 96.5, 88.0, 91.5]
results = analyze_grades(class_scores)
```

**Common Formatting Patterns:**

```python
# Money: $12.34
price = 12.3456
print(f"${price:.2f}")

# Percentages: 87.5%
score = 0.875
print(f"{score:.1%}")

# Scientific: 1.23e+06
population = 1234567
print(f"{population:.2e}")

# Fixed width: "  95.5"
value = 95.5
print(f"{value:6.1f}")
```

**Pro Tips:**

- 🎯 **Choose appropriate precision:** GPA needs 2 decimals, measurements might need more
- 💰 **Currency formatting:** Always show 2 decimal places for money
- 📊 **Consistency:** Use same precision throughout your program
- 🔍 **Round vs Truncate:** Formatting rounds, doesn't truncate

**Remember:** Clean formatting makes your programs look professional and user-friendly!

### Question 34: The Logic Detective - Boolean Operation Mysteries! 🕵️

**You have this complex logical statement: `True and False or True`. Can you figure out the result? Think of it as answering multiple yes/no questions at once!**

**Answer:** `True` (Python follows a specific order: `and` before `or` - like following order of operations in math!)

**Answer:** Python evaluates left-to-right with `and` before `or`: `(True and False) or True` = `False or True` = `True`

**Why Logic Matters in School:**

- 🎯 **Decision Making:** "Can I go to the party?" = `no_homework and finished_chores`
- 📊 **Filtering Data:** Find students who passed AND attended regularly
- 🏆 **Eligibility:** "Can join team?" = `skill_level >= 80 and has_permission`
- 📅 **Scheduling:** "Free period?" = `no_test_today and no_homework_due`

**Breaking Down the Logic:**

```python
# Step by step
result = True and False or True
step1 = True and False     # False (both must be True)
result = step1 or True     # False or True = True

print(f"Final result: {result}")  # True
```

**Real School Scenarios:**

**1. Permission System:**

```python
# Can student go on field trip?
has_permission = True
paid_fees = True
parent_consent = False

can_go = has_permission and paid_fees and parent_consent
print(f"Can go on trip: {can_go}")  # False (missing parent consent)
```

**2. Grade Calculation:**

```python
# Can student get A grade?
has_tests = True
has_projects = True
attended_class = False

eligible_for_A = has_tests and has_projects and attended_class
print(f"Eligible for A: {eligible_for_A}")  # False (poor attendance)
```

**3. Club Membership:**

```python
# Can join honor society?
gpa = 3.8
has_volunteer_hours = True
no_discipline_issues = False

can_join = gpa >= 3.5 and has_volunteer_hours and no_discipline_issues
print(f"Can join: {can_join}")  # False (discipline issues)
```

**Operator Priority (Remember PEMDAS for logic!):**

```python
# Python evaluates in this order:
# 1. not (highest priority)
# 2. and
# 3. or (lowest priority)

# Examples:
print(True or False and False)   # True or False = True (and evaluated first)
print((True or False) and False) # True and False = False (or in parentheses)
print(not True or False)         # False or False = False
```

**School Decision Trees:**

```python
# Complex eligibility check
def check_graduation_eligibility(gpa, credits, community_service):
    gpa_ok = gpa >= 2.0
    credits_ok = credits >= 24
    service_ok = community_service >= 40

    can_graduate = gpa_ok and credits_ok and service_ok
    return can_graduate

# Test
result = check_graduation_eligibility(2.5, 25, 45)
print(f"Can graduate: {result}")  # True
```

**Logical Shortcuts (Short-circuit evaluation):**

```python
# Python stops as soon as it knows the answer
passed_homework = True
passed_tests = False
final_exam = True

# If first condition is False, Python doesn't check the rest
has_passed = passed_homework and passed_tests and final_exam
print(f"Student passed: {has_passed}")  # False (stops at passed_tests)

# If first condition is True, Python stops at first False in 'and'
can_participate = True and True and False and True
print(f"Can participate: {can_participate}")  # False (stops at third condition)
```

**Creative Applications:**

**1. Study Schedule Planner:**

```python
has_time = True
has_materials = True
feels_ motivated = False

can_study = has_time and has_materials and feels_motivated
if can_study:
    print("Time to study! 📚")
else:
    print("Need to prepare more.")
```

**2. Event Planning:**

```python
# Can we have outdoor assembly?
weather_good = True
space_available = True
principal_approved = False

can_have_outdoor = weather_good and space_available and principal_approved
print(f"Outdoor assembly: {can_have_outdoor}")
```

**3. Grade Prediction:**

```python
# Will student get A in class?
homework_completed = True
participation_good = True
midterm_score = 95

will_get_A = homework_completed and participation_good and midterm_score >= 90
print(f"Predicted grade A: {will_get_A}")
```

**Logic Challenge Problems:**

```python
# Practice problems - try to predict the result!
print("1.", True or True and False)      # ?
print("2.", False and True or False)     # ?
print("3.", not False or True and False) # ?
print("4.", (True or False) and False)   # ?
```

**Answers:**

1. `True` (True or anything = True)
2. `False` (False and anything = False)
3. `True` (not False = True, so True or anything = True)
4. `False` (True and False = False)

**Pro Tips:**

- 🧠 **Use parentheses:** Make complex logic clearer
- 📝 **Write it out:** Break complex conditions into smaller parts
- 🎯 **Think in English:** "Student passes if they have homework AND no tests today"

**Remember:** Logic is the foundation of all programming decisions - Master this, and you'll be able to solve complex problems!

### Question 35: Grade Ranges

**What does `85 < grade < 100` check for?**

**Answer:** Checks if grade is between 85 AND 100, returns `True` (like checking if a score is in A-grade range)

### Question 36: Text Formatting

**What's the difference between `%s` and `%d` when formatting?**

**Answer:**

- `%s` for strings (student names, subjects)
- `%d` for integers (grades, counts)

### Question 37: Getting Parts of Text

**What does `subject[1:4]` return if `subject = "MathClub"`?**

**Answer:** `"ath"` (characters from index 1 to 3 - useful for getting parts of words)

### Question 38: Last Character

**What does `word[-1]` return if `word = "Club"`?**

**Answer:** `"b"` (last character - like getting the last letter of a word)

### Question 39: Variable Scope

**What will this code output?**

```python
class_grade = 85
def check_pass():
    class_grade = 95
    print(class_grade)
check_pass()
print(class_grade)
```

**Answer:**

```
95
85
```

(The `class_grade` inside the function doesn't affect the outer one)

### Question 40: Input Validation

**Write code to get a student's score and handle invalid input**

**Answer:**

```python
try:
    score = int(input("Enter student's score: "))
    print(f"Score recorded: {score}%")
except ValueError:
    print("That's not a valid number! Please enter a number.")
```

---

## ⚡ Operators & Expressions - Math for School (Questions 41-60)

### Question 41: Order of Operations

**What is the result of `2 + 3 * 4` (just like in math class)?**

**Answer:** `14` (multiplication before addition - like PEMDAS!)

### Question 42: Using Parentheses

**What is the result of `(2 + 3) * 4` (like grouping subjects)?**

**Answer:** `20` (parentheses change the order - like using brackets in math)

### Question 43: Exponent Operator

**Calculate 2³ (2 to the power of 3)**

**Answer:** `2 ** 3` → `8`

### Question 44: Division Types

**What's the difference between `100 / 30` and `100 // 30`?**

**Answer:**

- `100 / 30` → `3.333...` (exact division with decimal)
- `100 // 30` → `3` (whole groups only - like whole classes of 30 students)

### Question 45: Remainder Operator

**Why is `%` useful in school programming?**

**Answer:** Finding remainders, checking for even/odd numbers (like dividing students into pairs), cycling through values

### Question 46: Grade Comparison Chains

**Is `85 < score < 95` valid? What does it check for?**

**Answer:** Yes! Checks if the score is in the B+ to A- range

### Question 47: Alphabetical Order

**What does `"Algebra" < "Biology"` return?**

**Answer:** `True` (strings compare alphabetically - like words in the dictionary)

### Question 48: Logical Operator Priority

**What's the order of operations for logical operators?**

**Answer:** `not` → `and` → `or` (like PEMDAS but for logic!)

### Question 49: Grade Ranges

**Write a condition for: "Student is in grade range 9-12"**

**Answer:** `9 <= grade_level <= 12` or `grade_level >= 9 and grade_level <= 12`

### Question 50: Shortcut Operators

**What does `attendance += 1` do?**

**Answer:** `attendance = attendance + 1` (add 1 to attendance count)

### Question 51: Short-circuit Logic

**Why does `True or print("Present")` not print "Present"?**

**Answer:** `or` stops when it finds `True` (like once a student is marked present, you don't need to check further)

### Question 52: Default Values

**What does `0 or "Excused"` return?**

**Answer:** `"Excused"` (if attendance is 0, use the default "Excused")

### Question 53: Bitwise Operations

**What does `5 & 3` return?**

**Answer:** `1` (bitwise AND - used for advanced operations)

### Question 54: Equality vs Identity

**What's the difference between `==` and `is`?**

**Answer:**

- `==` compares values (are grades equal?)
- `is` compares identity (are they the exact same object?)

### Question 55: Checking Lists

**How do you check if "Math" is in your class list?**

**Answer:** `"Math" in class_list`

### Question 56: Text Repetition

**Can you use `*` with strings?**

**Answer:** Yes! For repetition: `"Hi!" * 3` → `"Hi!Hi!Hi!"` (like repeating a cheer!)

### Question 57: Division by Zero Error

**What happens when you divide by zero?**

**Answer:** `ZeroDivisionError` exception (like trying to split students into zero groups!)

### Question 58: Decimal Precision

**Why might `0.1 + 0.2 != 0.3`?**

**Answer:** Floating point precision issues (computers handle decimals differently - 0.1 + 0.2 = 0.30000000000000004)

### Question 59: Operator Overloading

**What does `+` do with different types?**

**Answer:**

- Numbers: addition (95 + 5 = 100)
- Strings: concatenation ("Math" + "Club" = "MathClub")
- Lists: joining lists together

### Question 60: One-line Conditions

**Write a one-line if-else for: "Pass" if score >= 60, "Fail" otherwise**

**Answer:** `result = "Pass" if score >= 60 else "Fail"`

---

## 💡 Problem Solving - Practical Programs for Daily Life (Questions 61-80)

### Question 61: Temperature Converter

**Write a program to convert temperature from Celsius to Fahrenheit**

**Answer:**

```python
celsius = float(input("Enter temperature in Celsius: "))
fahrenheit = (celsius * 9/5) + 32
print(f"{celsius}°C = {fahrenheit:.1f}°F")
```

### Question 62: Budget Calculator

**Calculate your total monthly income (amount = weekly_amount \* weeks)**

**Answer:**

```python
weekly_income = float(input("Enter weekly income: $"))
weeks = int(input("How many weeks? "))
total = weekly_income * weeks
print(f"Total income: ${total:.2f}")
```

### Question 63: Even or Odd Number Checker

**Write a program to check if a number is even or odd**

**Answer:**

```python
number = int(input("Enter a number: "))
if number % 2 == 0:
    print(f"{number} is even")
else:
    print(f"{number} is odd")
```

### Question 64: Letter Grade Calculator

**Convert numerical test score to letter grade (A: 90-100, B: 80-89, C: 70-79, D: 60-69, F: below 60)**

**Answer:**

```python
score = float(input("Enter test score: "))
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"
print(f"Letter grade: {grade}")
```

### Question 65: Leap Year Birthday Checker

**Write a program to check if your birth year is a leap year**

**Answer:**

```python
birth_year = int(input("Enter your birth year: "))
if (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0):
    print(f"{birth_year} is a leap year - you have a Feb 29 birthday every 4 years!")
else:
    print(f"{birth_year} is not a leap year")
```

### Question 66: Highest Grade Finder

**Find the highest grade among three students' test scores**

**Answer:**

```python
math_grade = float(input("Math grade: "))
science_grade = float(input("Science grade: "))
english_grade = float(input("English grade: "))

if math_grade >= science_grade and math_grade >= english_grade:
    print(f"Math is highest: {math_grade}")
elif science_grade >= math_grade and science_grade >= english_grade:
    print(f"Science is highest: {science_grade}")
else:
    print(f"English is highest: {english_grade}")
```

### Question 67: Student Calculator

**Create a basic calculator for +, -, \*, /**

**Answer:**

```python
num1 = float(input("First number: "))
operator = input("Enter operator (+, -, *, /): ")
num2 = float(input("Second number: "))

if operator == "+":
    result = num1 + num2
elif operator == "-":
    result = num1 - num2
elif operator == "*":
    result = num1 * num2
elif operator == "/":
    if num2 != 0:
        result = num1 / num2
    else:
        result = "Cannot divide by zero!"
else:
    result = "Invalid operator"

print(f"Result: {result}")
```

### Question 68: Vowel Counter

**Count vowels in a student's full name**

**Answer:**

```python
name = input("Enter student's full name: ").lower()
vowels = "aeiou"
count = sum(1 for char in name if char in vowels)
print(f"Number of vowels in the name: {count}")
```

### Question 69: Palindrome Checker

**Check if a word is the same forwards and backwards (like "level" or "madam")**

**Answer:**

```python
word = input("Enter a word: ").lower()
if word == word[::-1]:
    print(f"'{word}' is a palindrome!")
else:
    print(f"'{word}' is not a palindrome")
```

### Question 70: Class Seating Numbers

**Generate the first 10 numbers in a Fibonacci sequence for classroom seating**

**Answer:**

```python
rows = 10
a, b = 1, 1
print("Class seating numbers:")
for i in range(rows):
    print(a, end=" ")
    a, b = b, a + b
```

### Question 71: Prime Number Checker

**Check if a student ID number is prime**

**Answer:**

```python
student_id = int(input("Enter student ID to check: "))
if student_id < 2:
    print(f"{student_id} is not prime")
else:
    is_prime = True
    for i in range(2, int(student_id ** 0.5) + 1):
        if student_id % i == 0:
            is_prime = False
            break
    if is_prime:
        print(f"{student_id} is a prime number!")
    else:
        print(f"{student_id} is not prime")
```

### Question 72: Factorial Calculator

**Calculate factorial for combinations (like 5! = 5×4×3×2×1)**

**Answer:**

```python
number = int(input("Enter a number: "))
factorial = 1
for i in range(1, number + 1):
    factorial *= i
print(f"{number}! = {factorial}")
```

### Question 73: Sum of Student ID

**Find sum of all digits in a student ID number**

**Answer:**

```python
student_id = input("Enter student ID: ")
digit_sum = sum(int(digit) for digit in student_id)
print(f"Sum of digits in {student_id}: {digit_sum}")
```

### Question 74: Reverse Student ID

**Reverse the digits of a student ID number**

**Answer:**

```python
student_id = input("Enter student ID: ")
reversed_id = student_id[::-1]
print(f"Reversed ID: {reversed_id}")
```

### Question 75: Classroom Area Calculator

**Calculate area of different classroom shapes (rectangle, circle, triangle)**

**Answer:**

```python
shape = input("What shape? (rectangle, circle, triangle): ").lower()

if shape == "rectangle":
    length = float(input("Enter length: "))
    width = float(input("Enter width: "))
    area = length * width
    print(f"Rectangle area: {area} sq units")
elif shape == "circle":
    radius = float(input("Enter radius: "))
    area = 3.14159 * radius ** 2
    print(f"Circle area: {area:.2f} sq units")
elif shape == "triangle":
    base = float(input("Enter base: "))
    height = float(input("Enter height: "))
    area = 0.5 * base * height
    print(f"Triangle area: {area} sq units")
else:
    print("Invalid shape")
```

### Question 76: Password Validator

**Check if password meets criteria (8+ chars, has digit, has letter)**

**Answer:**

```python
password = input("Enter password: ")
criteria_met = []

if len(password) >= 8:
    criteria_met.append("Length OK")
else:
    print("Password too short")

if any(char.isdigit() for char in password):
    criteria_met.append("Has number")
else:
    print("Password must have a number")

if any(char.isalpha() for char in password):
    criteria_met.append("Has letter")
else:
    print("Password must have a letter")

if len(criteria_met) == 3:
    print("Password is valid!")
else:
    print("Password requirements not met")
```

### Question 77: Lunch Money Converter

**Convert your lunch money from USD to different currencies**

**Answer:**

```python
usd = float(input("Enter lunch money amount in USD: $"))
exchange_rates = {
    "EUR": 0.85,  # Euros
    "GBP": 0.73,  # British Pounds
    "JPY": 110.0  # Japanese Yen
}

print(f"Your ${usd} USD converts to:")
for currency, rate in exchange_rates.items():
    converted = usd * rate
    print(f"{currency}: {converted:.2f}")
```

### Question 78: BMI Calculator

**Calculate Body Mass Index and categorize (Underweight < 18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese 30+)**

**Answer:**

```python
weight = float(input("Enter weight in kg: "))
height = float(input("Enter height in meters: "))

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

### Question 79: Number Guessing Game

**Create a number guessing game where computer picks 1-10 and you guess**

**Answer:**

```python
import random

secret_number = random.randint(1, 10)
attempts = 0

print("I'm thinking of a number between 1 and 10...")

while True:
    guess = int(input("Your guess: "))
    attempts += 1

    if guess == secret_number:
        print(f"Correct! You guessed it in {attempts} attempts!")
        break
    elif guess < secret_number:
        print("Too low!")
    else:
        print("Too high!")
```

### Question 80: Rock Paper Scissors

**Play Rock-Paper-Scissors against the computer**

**Answer:**

```python
import random

choices = ["rock", "paper", "scissors"]
player = input("Choose rock, paper, or scissors: ").lower()
computer = random.choice(choices)

print(f"You chose: {player}")
print(f"Computer chose: {computer}")

if player == computer:
    print("It's a tie!")
elif (player == "rock" and computer == "scissors") or \
     (player == "paper" and computer == "rock") or \
     (player == "scissors" and computer == "paper"):
    print("You win!")
else:
    print("Computer wins!")
```

---

## 🚀 Advanced Applications - Complex Projects and Algorithms (Questions 81-100)

### Question 81: Perfect Test Scores

**Find all perfect scores under 1000 (where sum of divisors equals the number)**

**Answer:**

```python
def is_perfect_score(score):
    sum_divisors = 1
    for i in range(2, int(score ** 0.5) + 1):
        if score % i == 0:
            sum_divisors += i
            if i != score // i:
                sum_divisors += score // i
    return sum_divisors == score

perfect_scores = [n for n in range(2, 1000) if is_perfect_score(n)]
print("Perfect scores under 1000:", perfect_scores)
```

### Question 82: Longest Palindrome

**Find the longest palindrome word in a sentence (like "racecar" or "level")**

**Answer:**

```python
def find_longest_palindrome(sentence):
    words = sentence.split()
    palindromes = [word.lower() for word in words if word.lower() == word.lower()[::-1]]
    return max(palindromes, key=len) if palindromes else None

class_sentence = "racecar level madam radar civic"
print(f"Longest palindrome: {find_longest_palindrome(class_sentence)}")
```

### Question 83: Anagram Checker

**Check if two words are anagrams (same letters, different order like "listen" and "silent")**

**Answer:**

```python
def are_anagrams(word1, word2):
    return sorted(word1.lower()) == sorted(word2.lower())

word1 = "Listen"
word2 = "Silent"
print(f"Are '{word1}' and '{word2}' anagrams? {are_anagrams(word1, word2)}")
```

### Question 84: Roman Numerals

**Convert a year to Roman numerals (like 1994 = MCMXCIV)**

**Answer:**

```python
def decimal_to_roman(year):
    val_roman = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]

    roman = ''
    for value, symbol in val_roman:
        while year >= value:
            roman += symbol
            year -= value
    return roman

birth_year = int(input("Enter your birth year: "))
print(f"{birth_year} in Roman numerals: {decimal_to_roman(birth_year)}")
```

### Question 85: School Schedule Validator

**Check if a 9x9 class schedule is valid (no duplicate classes in rows, columns, or blocks)**

**Answer:**

```python
def is_valid_schedule(schedule):
    # Check rows
    for row in schedule:
        if len(set(row)) != len(row) and 0 not in row:
            return False

    # Check columns
    for col in range(9):
        column = [schedule[row][col] for row in range(9)]
        if len(set(column)) != len(column) and 0 not in column:
            return False

    # Check 3x3 blocks
    for block_row in range(0, 9, 3):
        for block_col in range(0, 9, 3):
            block = []
            for i in range(3):
                for j in range(3):
                    block.append(schedule[block_row + i][block_col + j])
            if len(set(block)) != len(block) and 0 not in block:
                return False

    return True
```

### Question 86: Advanced Calculator

**Calculator with precedence handling**

**Answer:**

```python
def evaluate_expression(expression):
    try:
        # Simple implementation - only basic operations
        return eval(expression)
    except:
        return "Invalid expression"

expr = "3 + 4 * 2 - (1 / 2)"
print(f"{expr} = {evaluate_expression(expr)}")
```

### Question 87: Student Seating Pattern

**Generate a diamond pattern with student names**

**Answer:**

```python
def print_diamond(size, name="★"):
    # Top half
    for i in range(1, size + 1, 2):
        spaces = (size - i) // 2
        print(' ' * spaces + name * i + ' ' * spaces)

    # Bottom half
    for i in range(size - 2, 0, -2):
        spaces = (size - i) // 2
        print(' ' * spaces + name * i + ' ' * spaces)

print_diamond(9, "Student")
```

### Question 88: Encryption/Decryption

**Simple Caesar cipher**

**Answer:**

```python
def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            start = ord('A') if char.isupper() else ord('a')
            result += chr((ord(char) - start + shift) % 26 + start)
        else:
            result += char
    return result

message = "Hello, World!"
encrypted = caesar_cipher(message, 3)
decrypted = caesar_cipher(encrypted, -3)
print(f"Original: {message}")
print(f"Encrypted: {encrypted}")
print(f"Decrypted: {decrypted}")
```

### Question 89: Data Compression

**Run-length encoding**

**Answer:**

```python
def run_length_encode(data):
    if not data:
        return ""

    result = []
    current_char = data[0]
    count = 1

    for char in data[1:]:
        if char == current_char:
            count += 1
        else:
            result.append(current_char + str(count))
            current_char = char
            count = 1
    result.append(current_char + str(count))

    return ''.join(result)

print(run_length_encode("AAABBBCCCDDDE"))
```

### Question 90: File Organizer

**Organize files by extension**

**Answer:**

```python
import os
from collections import defaultdict

def organize_files(directory):
    files_by_ext = defaultdict(list)

    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            ext = filename.split('.')[-1]
            files_by_ext[ext].append(filename)

    return dict(files_by_ext)

# Example usage
# organized = organize_files("/path/to/files")
# print(organized)
```

### Question 91: Memory Game

**Number memory game**

**Answer:**

```python
import random
import time

def memory_game():
    number_length = 3
    while True:
        number = ''.join([str(random.randint(0, 9)) for _ in range(number_length)])
        print(f"Remember this number: {number}")
        time.sleep(3)
        print("\n" * 50)  # Clear screen effect
        user_input = input("Enter the number: ")

        if user_input == number:
            print("Correct! Moving to next level")
            number_length += 1
        else:
            print(f"Game over! You reached level {number_length}")
            break

memory_game()
```

### Question 92: Data Analyzer

**Analyze data from CSV-style string**

**Answer:**

```python
def analyze_data(csv_data):
    lines = csv_data.strip().split('\n')
    headers = lines[0].split(',')
    data = [line.split(',') for line in lines[1:]]

    analysis = {}

    # Find numeric columns
    for i, header in enumerate(headers):
        try:
            values = [float(row[i]) for row in data]
            analysis[header] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }
        except ValueError:
            # Not numeric
            pass

    return analysis

data = """Name,Age,Score
John,25,85
Alice,30,92
Bob,28,78"""
print(analyze_data(data))
```

### Question 93: Maze Solver

**Simple maze solver using backtracking**

**Answer:**

```python
def solve_maze(maze, start, end):
    def dfs(x, y, path):
        if (x, y) == end:
            return path

        maze[x][y] = 'X'  # Mark as visited

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and
                maze[nx][ny] == '.'):
                result = dfs(nx, ny, path + [(nx, ny)])
                if result:
                    return result

        maze[x][y] = '.'  # Backtrack
        return None

    return dfs(start[0], start[1], [start])

# Example maze
maze = [
    ['.', '.', '.', 'X'],
    ['X', 'X', '.', 'X'],
    ['.', '.', '.', '.'],
    ['X', 'X', 'X', '.']
]
print(solve_maze(maze, (0, 0), (3, 3)))
```

### Question 94: Genetic Algorithm

**Simple genetic algorithm for finding maximum**

**Answer:**

```python
import random

def genetic_algorithm(population_size=100, generations=50):
    # Generate initial population
    population = [[random.randint(0, 31) for _ in range(5)] for _ in range(population_size)]

    for generation in range(generations):
        # Evaluate fitness (sum of values)
        fitness = [sum(individual) for individual in population]

        # Selection - keep top 50%
        sorted_pop = [x for _, x in sorted(zip(fitness, population))]
        population = sorted_pop[-population_size//2:]

        # Crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            # Crossover
            child = [random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2)]

            # Mutation
            if random.random() < 0.1:
                child[random.randint(0, 4)] = random.randint(0, 31)

            new_population.append(child)

        population = new_population

        # Best individual
        best = max(population, key=sum)
        print(f"Generation {generation}: Best = {sum(best)}")

    return max(population, key=sum)

result = genetic_algorithm()
print(f"Final result: {result} (sum = {sum(result)})")
```

### Question 95: Web Scraper

**Simple web scraper for titles**

**Answer:**

```python
import requests
from bs4 import BeautifulSoup
import re

def scrape_titles(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        titles = []

        # Get h1, h2, h3 titles
        for tag in ['h1', 'h2', 'h3']:
            for element in soup.find_all(tag):
                title = element.get_text().strip()
                if title:
                    titles.append(f"{tag.upper()}: {title}")

        return titles
    except:
        return ["Error: Could not fetch page"]

# Example (replace with actual URL)
# titles = scrape_titles("https://example.com")
# for title in titles:
#     print(title)
```

### Question 96: Multi-threaded Downloader

**Download multiple files concurrently**

**Answer:**

```python
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

def download_file(url, filename):
    try:
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def download_multiple(urls):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, url in enumerate(urls):
            filename = f"file_{i}.txt"
            future = executor.submit(download_file, url, filename)
            futures.append(future)

        for future in futures:
            future.result()

# Example URLs
urls = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/2",
    "https://httpbin.org/delay/3"
]

# download_multiple(urls)
```

### Question 97: Data Visualization

**Create a simple histogram**

**Answer:**

```python
def create_histogram(data, bins=10):
    # Calculate bin width
    min_val, max_val = min(data), max(data)
    bin_width = (max_val - min_val) / bins

    # Initialize bins
    histogram = [0] * bins
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]

    # Count values in each bin
    for value in data:
        bin_index = min(int((value - min_val) / bin_width), bins - 1)
        histogram[bin_index] += 1

    # Print histogram
    for i, count in enumerate(histogram):
        bar = '*' * count
        print(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}: {bar}")

    return histogram

# Example usage
data = [23, 45, 56, 78, 32, 45, 67, 89, 12, 34, 56, 78, 90, 23, 45]
create_histogram(data)
```

### Question 98: Network Monitor

**Monitor network connections**

**Answer:**

```python
import psutil
import time

def monitor_network():
    prev_stats = psutil.net_io_counters()
    time.sleep(1)
    current_stats = psutil.net_io_counters()

    bytes_sent = current_stats.bytes_sent - prev_stats.bytes_sent
    bytes_recv = current_stats.bytes_recv - prev_stats.bytes_recv
    packets_sent = current_stats.packets_sent - prev_stats.packets_sent
    packets_recv = current_stats.packets_recv - prev_stats.packets_recv

    print(f"Bytes Sent: {bytes_sent:,}")
    print(f"Bytes Received: {bytes_recv:,}")
    print(f"Packets Sent: {packets_sent:,}")
    print(f"Packets Received: {packets_recv:,}")

    return {
        'bytes_sent': bytes_sent,
        'bytes_recv': bytes_recv,
        'packets_sent': packets_sent,
        'packets_recv': packets_recv
    }

# Monitor for 5 seconds
for _ in range(5):
    monitor_network()
    print("-" * 30)
    time.sleep(1)
```

### Question 99: Machine Learning Basics

**Simple linear regression**

**Answer:**

```python
import numpy as np

def simple_linear_regression(x, y):
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate slope (m) and intercept (b)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

slope, intercept = simple_linear_regression(x, y)
print(f"Linear equation: y = {slope:.2f}x + {intercept:.2f}")

# Predict new values
x_new = np.array([6, 7, 8])
y_pred = slope * x_new + intercept
print(f"Predictions for x={x_new}: y={y_pred}")
```

### Question 100: AI Chatbot

**Simple rule-based chatbot**

**Answer:**

```python
import re

def simple_chatbot():
    responses = {
        r'hello|hi|hey': "Hello! How can I help you today?",
        r'how are you': "I'm doing great! Thanks for asking!",
        r'what\'s your name': "I'm a simple Python chatbot.",
        r'quit|exit|bye': "Goodbye! Have a nice day!",
        r'.*': "I'm not sure how to respond to that. Can you try asking something else?"
    }

    print("Simple Chatbot - Type 'quit' to exit")
    print("-" * 40)

    while True:
        user_input = input("You: ").lower()

        for pattern, response in responses.items():
            if re.match(pattern, user_input):
                print(f"Bot: {response}")
                break

        if user_input in ['quit', 'exit', 'bye']:
            break

# Start the chatbot
# simple_chatbot()
```

---

## 🎆 Final Assessment - Are You Ready for Advanced Python?

### Mastery Checklist:

- [ ] Can explain Python versions and installation ✅
- [ ] Understands variable naming and data types ✅
- [ ] Can use all operators correctly ✅
- [ ] Solves real-world problems with Python ✅
- [ ] Handles edge cases and errors gracefully ✅
- [ ] Writes clean, readable code ✅
- [ ] Uses appropriate data structures ✅
- [ ] Implements complex algorithms ✅

### Your Next Steps to Python Mastery:

1. **Review** all concepts you found challenging
2. **Practice** coding without looking at solutions
3. **Experiment** with variations of real-world problems
4. **Combine** concepts from multiple questions
5. **Challenge** yourself with larger projects (personal finance tracker, task manager, etc.)
6. **Teach** others - the best way to solidify your knowledge!

### Recommended Next Topics:

- Control Structures (if/else, loops)
- Functions and code organization
- File handling and data persistence
- Object-oriented programming
- Working with APIs and databases

---

_🎉 Congratulations! You've completed the Python Fundamentals Practice Questions! You're now ready to move to more advanced topics. Keep coding and have fun building practical applications! 🚀_
