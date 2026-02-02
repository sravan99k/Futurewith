# Python Comprehensive Quizzes & Assessment Guide

## 🌟 Welcome to Python Assessment!

This guide helps you **test and improve** your Python skills at any level:

- 🎓 **Students & Beginners** (age 8-100+)
- 🔄 **Career Changers** returning to programming
- 💼 **Professionals** needing skill verification
- 🎯 **Interview Candidates** preparing for technical interviews

### How to Use This Guide

#### 📊 Difficulty Levels

- ✅ **Foundation** (Questions 1-50): Absolute basics - perfect for beginners!
- 🔧 **Core Skills** (Questions 51-100): Essential programming concepts
- 🚀 **Advanced** (Questions 101-150): Complex applications and patterns
- 🏆 **Expert** (Questions 151-200): System design and optimization

#### 🎯 Assessment Formats

- **Multiple Choice**: Test your knowledge with 4 options
- **True/False**: Quick understanding checks
- **Code Prediction**: What will this code do?
- **Fill in the Blank**: Complete the code
- **Practical Challenges**: Write actual code

### 📚 Study Tips

1. **Start with Foundation** - Even experienced developers should begin here
2. **Read Explanations** - Learn WHY answers are correct
3. **Practice Coding** - Try challenges after reading concepts
4. **Don't Rush** - Take time to understand each topic
5. **Review Mistakes** - Learn from errors to improve faster

---

### Quick Start Guide

**🔰 Complete Beginner?**
→ Start with Questions 1-25
→ Focus on Code Prediction questions
→ Review cheat sheet sections 1-3

**🔄 Returning to Python?**
→ Try Questions 26-100
→ Focus on areas you don't use daily
→ Check out the cheat sheet

**💼 Preparing for Interview?**
→ Questions 51-150 + all Coding Challenges
→ Practice explaining your answers
→ Review all cheat sheet sections

**🏆 Expert Level?**
→ Questions 151-200 + Advanced Coding
→ Focus on optimization and design
→ Study Security & Best Practices sections

---

## 🎯 Assessment Structure

### Topic Coverage

- **Python Fundamentals**: 35 questions ✅ Foundation + Core
- **Control Structures**: 30 questions Decision making & loops
- **Data Structures**: 30 questions Lists, dicts, sets, and more
- **Functions & OOP**: 25 questions Reusable code & classes
- **Libraries & Modules**: 25 questions Popular Python tools
- **File & Error Handling**: 20 questions Real-world programming
- **Practical Applications**: 25 questions Problem-solving

### Question Type Distribution

- **Multiple Choice**: 40% (80 questions)
- **True/False**: 15% (30 questions)
- **Code Prediction**: 20% (40 questions)
- **Fill in the Blank**: 15% (30 questions)
- **Practical Challenges**: 10% (20 coding tasks)

### Age-Friendly Features

✅ **Simple Language**: No jargon without explanation  
📝 **Clear Examples**: Real-world, relatable scenarios  
🎨 **Visual Aids**: Code with helpful comments  
💡 **Learning Focus**: Explanations teach, not just test  
🌍 **Universal Topics**: Examples work across cultures

---

## ✅ Foundation Level Questions (1-50)

### 🎯 What This Section Covers

These questions test your understanding of:

- ✅ Basic Python syntax
- 🔢 Numbers and text (strings)
- 📦 Variables (storing information)
- 🔍 Simple operations

**💡 Don't worry if you don't know everything!**
Each answer includes an explanation to help you learn.

---

### 🏁 Python Fundamentals (Questions 1-25)

**1. What will this code show on the screen?**

```python
x = 5
y = "Hello"
z = 3.14
print(type(x), type(y), type(z))
```

a) `<class 'int'> <class 'str'> <class 'float'>`
b) `int str float`
c) `5 "Hello" 3.14`
d) Something else

✅ **Answer: a**
📝 **Explanation**: `type()` tells you what kind of data something is.

- `5` is a number (int = integer)
- `"Hello"` is text (str = string)
- `3.14` is a decimal (float = floating point number)

---

**2. Which of these is a GOOD variable name?**
a) `2nd_variable` (starts with a number - not allowed!)
b) `my_variable` (this is perfect!)
c) `_private_var` (this works but might be confusing)
d) `MYCONSTANT` (this works but is unusual)

✅ **Answer: b**
📝 **Explanation**: Good variable names:

- Start with a letter or underscore
- Describe what they store (`age` not `a`)
- Use lowercase with underscores (`user_name` not `UserName`)
- Are easy to read and understand

---

**3. What does the `//` symbol do in Python?**
a) Regular division (like 10 ÷ 3 = 3.33)
b) Integer division (drops decimals: 10 ÷ 3 = 3)
c) Gives the remainder (10 ÷ 3 = 1)
d) Makes a number negative

✅ **Answer: b**
📝 **Explanation**: `//` is floor division - it divides and removes any decimal part.

- `10 / 3 = 3.333...` (regular division)
- `10 // 3 = 3` (integer division, drops the .333...)

Think of it like dividing pizza slices and throwing away any leftovers!

---

**4. How do you make text all lowercase in Python?**
a) `string.lower()` (string is not a variable)
b) `"hello".lower()` (correct method!)
c) `lowercase("HELLO")` (no such function)
d) `convert_lower("HELLO")` (wrong method)

✅ **Answer: b**
📝 **Explanation**: String methods are called ON the string. Think of it as asking the string to do something:

```python
"HELLO".lower()  # Ask "HELLO" to become lowercase → "hello"
name = "ALICE"
name.lower()      # Ask name to become lowercase → "alice"
```

---

**5. What will be the result of `len([1, 2, 3, [4, 5]])`?**
a) 3
b) 4
c) 5
d) Error

**Answer: c**
_Explanation: The list contains 4 elements: 1, 2, 3, and [4, 5]. The inner list counts as one element._

**6. Which statement is true about Python strings?**
a) Strings are immutable
b) Strings can be modified in-place
c) Strings support only ASCII characters
d) Strings cannot contain numbers

**Answer: a**
_Explanation: Strings in Python are immutable, meaning they cannot be changed after creation._

**7. What will `bool([])` return?**
a) True
b) False
c) Error
d) None

**Answer: b**
_Explanation: An empty list is falsy in Python, so `bool([])` returns False._

**8. Which of the following creates a set in Python?**
a) `{1, 2, 3}`
b) `[1, 2, 3]`
c) `(1, 2, 3)`
d) `{1: 'one', 2: 'two'}`

**Answer: a**
_Explanation: Curly braces `{}` with comma-separated values create a set. Option d creates a dictionary._

**9. What does `range(5)` generate?**
a) Numbers from 0 to 4
b) Numbers from 1 to 5
c) Numbers from 0 to 5
d) Numbers from 1 to 4

**Answer: a**
_Explanation: `range(n)` generates numbers from 0 to n-1, so `range(5)` gives 0, 1, 2, 3, 4._

**10. Which operator has the highest precedence in Python?**
a) `+`
b) `*`
c) `**`
d) `()`

**Answer: c**
_Explanation: Exponentiation `**` has the highest precedence, followed by `_`, then `+`.\*

**11. What will be the output of `print("Hello" * 3)`?**
a) HelloHelloHello
b) Hello 3 times
c) Error
d) 3

**Answer: a**
_Explanation: The `_` operator repeats strings, so "Hello" _ 3 produces "HelloHelloHello"._

**12. Which method adds an element to the end of a list?**
a) `list.append()`
b) `list.add()`
c) `list.insert()`
d) `list.extend()`

**Answer: a**
_Explanation: `append()` adds an element to the end of a list. `insert()` adds at a specific position._

**13. What will `3 == 3.0` evaluate to?**
a) True
b) False
c) Error
d) None

**Answer: a**
_Explanation: In Python, integers and floats of the same value are considered equal._

**14. Which of the following is a valid comment in Python?**
a) `// This is a comment`
b) `/* This is a comment */`
c) `# This is a comment`
d) `-- This is a comment`

**Answer: c**
_Explanation: Python uses `#` for single-line comments. The other formats are used in different languages._

**15. What does the `del` statement do?**
a) Delete a function
b) Delete a variable or list item
c) Delete a file
d) Delete a class

**Answer: b**
_Explanation: `del` is used to delete variables, list items, or dictionary entries._

### Control Structures (Questions 16-25)

**16. What will be the output of the following code?**

```python
for i in range(3):
    print(i, end="")
```

a) 012
b) 123
c) 0 1 2
d) Error

**Answer: a**
_Explanation: `range(3)` generates 0, 1, 2, and `end=""` prevents newlines, so output is "012"._

**17. Which loop structure is best when you know the number of iterations?**
a) `while` loop
b) `for` loop
c) `do-while` loop
d) Both `for` and `while` are equally good

**Answer: b**
_Explanation: `for` loops are specifically designed for iterating over a known sequence or range._

**18. What will `break` do inside a loop?**
a) Skip the current iteration
b) Exit the loop completely
c) Restart the loop
d) Continue to the next iteration

**Answer: b**
_Explanation: `break` immediately terminates the loop and continues execution after the loop._

**19. Which statement is true about `else` in loops?**
a) `else` executes if the loop never runs
b) `else` executes if the loop completes normally (no `break`)
c) `else` executes if the loop encounters an error
d) `else` executes after every iteration

**Answer: b**
_Explanation: The `else` clause in a loop executes only if the loop completes normally without hitting `break`._

**20. What will be the result of `any([False, False, True, False])`?**
a) False
b) True
c) Error
d) None

**Answer: b**
_Explanation: `any()` returns True if any element in the iterable is True. Since there's a True, it returns True._

**21. Which of the following creates an infinite loop?**
a) `while True: break`
b) `for i in range(10):`
c) `while False:`
d) `for i in [1,2,3]:`

**Answer: c**
_Explanation: `while False:` creates a condition that's always false, but since the condition is already False, the loop body never executes, making it not infinite but zero iterations._

**22. What does `continue` do in a loop?**
a) Exit the loop
b) Skip to the next iteration
c) Restart the loop from beginning
d) Break out of nested loops

**Answer: b**
_Explanation: `continue` skips the rest of the current iteration and moves to the next one._

**23. What will be the output of this nested loop?**

```python
for i in range(2):
    for j in range(2):
        print(i, j)
```

a) 0 0\n0 1\n1 0\n1 1
b) 0 0 0 1 1 0 1 1
c) 00 01 10 11
d) Error

**Answer: a**
_Explanation: The outer loop runs twice (i=0, i=1), and for each i, the inner loop runs twice (j=0, j=1)._

**24. Which conditional statement is NOT valid in Python?**
a) `if`
b) `elseif`
c) `else`
d) `elif`

**Answer: b**
_Explanation: Python uses `elif`, not `elseif`. The other options are all valid._

**25. What will be the result of `all([True, False, True])`?**
a) True
b) False
c) Error
d) None

**Answer: b**
_Explanation: `all()` returns True only if all elements are True. Since there's a False, it returns False._

### Data Structures (Questions 26-40)

**26. What is the main difference between a list and a tuple?**
a) Lists are faster than tuples
b) Tuples are immutable, lists are mutable
c) Lists can only contain numbers
d) Tuples cannot be indexed

**Answer: b**
_Explanation: The key difference is mutability: lists can be modified after creation, tuples cannot._

**27. What will be the result of `list({1, 2, 3, 2, 1})`?**
a) [1, 2, 3, 2, 1]
b) [1, 2, 3]
c) [1, 2, 3, 3, 2, 1]
d) Error

**Answer: b**
_Explanation: Converting a set to a list removes duplicates, so the result is [1, 2, 3]._

**28. Which dictionary method returns all keys?**
a) `dict.values()`
b) `dict.keys()`
c) `dict.items()`
d) `dict.get_all()`

**Answer: b**
_Explanation: `keys()` returns a view of all keys. `values()` returns values, `items()` returns key-value pairs._

**29. What will `sorted([3, 1, 4, 1, 5, 9, 2, 6])` return?**
a) [1, 1, 2, 3, 4, 5, 6, 9]
b) [9, 6, 5, 4, 3, 2, 1, 1]
c) [3, 1, 4, 1, 5, 9, 2, 6]
d) Error

**Answer: a**
_Explanation: `sorted()` returns a new sorted list in ascending order._

**30. What will be the result of `'hello'.replace('l', 'L')`?**
a) 'hello'
b) 'heLLo'
c) 'HeLLo'
d) 'HELLO'

**Answer: b**
_Explanation: `replace()` replaces all occurrences of the substring, so 'hello' becomes 'heLLo'._

**31. Which method removes and returns the last element of a list?**
a) `pop()`
b) `remove()`
c) `delete()`
d) `extract()`

**Answer: a**
_Explanation: `pop()` without arguments removes and returns the last element. `remove()` removes the first occurrence of a value._

**32. What will `len(set([1, 2, 3, 3, 4]))` return?**
a) 4
b) 5
c) 3
d) Error

**Answer: a**
_Explanation: The set will contain {1, 2, 3, 4}, which has 4 unique elements._

**33. How do you access the second element in a list?**
a) `list[1]`
b) `list[2]`
c) `list.second`
d) `list(1)`

**Answer: a**
_Explanation: Python uses zero-based indexing, so the second element is at index 1._

**34. What will `'python'.upper().lower()` return?**
a) 'Python'
b) 'PYTHON'
c) 'python'
d) 'PYTHON'.lower()

**Answer: c**
_Explanation: String methods can be chained. `upper()` makes it 'PYTHON', then `lower()` makes it 'python'._

**35. Which of the following is a correct way to create an empty set?**
a) `{}`
b) `set()`
c) `[]`
d) Both a and b

**Answer: b**
_Explanation: `{}` creates an empty dictionary, not a set. `set()` creates an empty set._

**36. What will `list(zip([1, 2], ['a', 'b']))` return?**
a) [(1, 2), ('a', 'b')]
b) [(1, 'a'), (2, 'b')]
c) [1, 2, 'a', 'b']
d) [(1, 'a', 2, 'b')]

**Answer: b**
_Explanation: `zip()` pairs elements from each iterable, creating [(1, 'a'), (2, 'b')]._

**37. What does `dict.get('key', 'default')` return if 'key' doesn't exist?**
a) None
b) Error
c) 'default'
d) The empty string

**Answer: c**
_Explanation: The second argument to `get()` is the default value returned when the key is not found._

**38. What will be the result of `[x*2 for x in range(5)]`?**
a) [0, 1, 2, 3, 4]
b) [0, 2, 4, 6, 8]
c) [1, 2, 3, 4, 5]
d) [0, 1, 4, 9, 16]

**Answer: b**
_Explanation: This is a list comprehension that doubles each number from 0 to 4._

**39. Which method adds multiple elements to the end of a list?**
a) `append()`
b) `extend()`
c) `insert()`
d) `add()`

**Answer: b**
_Explanation: `extend()` adds all elements from an iterable to the end of the list. `append()` adds a single element._

**40. What will be the output of `{x: x**2 for x in range(3)}`?\*\*
a) {0: 0, 1: 1, 2: 4}
b) [0, 1, 4]
c) {1: 1, 2: 4, 3: 9}
d) Error

**Answer: a**
_Explanation: This creates a dictionary comprehension with keys 0, 1, 2 and values as their squares._

### Functions & Scope (Questions 41-50)

**41. What will be the output of this function call?**

```python
def func(x):
    x = x + 1
    return x

y = 5
func(y)
print(y)
```

a) 5
b) 6
c) Error
d) None

**Answer: a**
_Explanation: Since integers are immutable in Python, the function receives a copy of the value, not the variable itself._

**42. What is a default parameter?**
a) A parameter with a fixed value
b) A parameter with a predefined value that can be overridden
c) A required parameter
d) A parameter that must be last

**Answer: b**
_Explanation: Default parameters have a value that is used if no argument is provided for that parameter._

**43. What will `len(lambda x: x)` return?**
a) Error
b) 1
c) 0
d) None

**Answer: a**
_Explanation: Lambda functions don't have a length. Trying to call `len()` on a function will raise a TypeError._

**44. Which statement about \*args is true?**
a) *args is a special keyword
b) *args collects extra positional arguments into a tuple
c) *args is required in every function
d) *args must be the first parameter

**Answer: b**
*Explanation: *args allows a function to accept any number of positional arguments, collected into a tuple.\*

**45. What will be the result of `max([1, 2, 3], key=lambda x: -x)`?**
a) 1
b) 2
c) 3
d) Error

**Answer: c**
_Explanation: The key function transforms values to their negative, so max finds the original value with the smallest negative (largest original)._

**46. What does the `global` keyword do?**
a) Makes a variable accessible to all modules
b) Allows modification of a global variable inside a function
c) Creates a new global variable
d) Imports a variable from another module

**Answer: b**
_Explanation: `global` declares that a variable inside a function refers to the global variable with that name._

**47. What will be the output of this nested function?**

```python
def outer():
    x = 'outer'
    def inner():
        print(x)
    inner()

outer()
```

a) Error
b) 'outer'
c) Nothing
d) 'inner'

**Answer: b**
_Explanation: Inner functions can access variables from outer functions (closure)._

**48. Which decorator is used to define a class method?**
a) `@staticmethod`
b) `@classmethod`
c) `@property`
d) `@classmethod`

**Answer: b**
_Explanation: `@classmethod` defines a method that receives the class as the first argument instead of the instance._

**49. What will `filter(lambda x: x > 0, [-1, 0, 1, 2])` return?**
a) [-1, 0, 1, 2]
b) [1, 2]
c) [-1, 0]
d) Error

**Answer: b**
_Explanation: `filter()` returns only elements where the function returns True. Here, it filters positive numbers._

**50. What is the difference between `*args` and `**kwargs`?**
a) No difference
b) `*args`for positional,`\*\*kwargs`for keyword arguments
c)`*args`for numbers,`**kwargs`for strings
d)`\*args`for required,`**kwargs` for optional

**Answer: b**
*Explanation: *args collects positional arguments into a tuple, \*_kwargs collects keyword arguments into a dictionary._

---

## Intermediate Level Questions (51-100)

### Object-Oriented Programming (Questions 51-65)

**51. What will be the output of this class definition?**

```python
class Person:
    species = "Homo sapiens"

    def __init__(self, name):
        self.name = name

p1 = Person("Alice")
p2 = Person("Bob")
print(p1.species, p2.species, Person.species)
```

a) Homo sapiens Homo sapiens Homo sapiens
b) Alice Bob Person
c) Error
d) None of the above

**Answer: a**
_Explanation: Class attributes are shared by all instances, so all three references point to the same value._

**52. What does the `@property` decorator do?**
a) Makes a method static
b) Creates a read-only attribute that calls a method
c) Makes a method private
d) Creates a class method

**Answer: b**
_Explanation: `@property` allows you to define a method that can be accessed like an attribute._

**53. What will be the result of `issubclass(bool, int)`?**
a) True
b) False
c) Error
d) None

**Answer: a**
_Explanation: In Python, bool is a subclass of int, so this returns True._

**54. Which magic method is called when you use the `+` operator?**
a) `__add__()`
b) `__plus__()`
c) `__operator__()`
d) `__add__` and `__radd__()`

**Answer: a**
_Explanation: The `__add__()` method is called when the `+` operator is used on an object._

**55. What does `super().__init__()` do in a subclass?**
a) Creates a new super object
b) Calls the parent class's `__init__` method
c) Overrides the parent class
d) Makes the class a superclass

**Answer: b**
_Explanation: `super()` returns a proxy object that allows calling parent class methods._

**56. What will be printed by this code?**

```python
class Base:
    def method(self):
        print("Base")

class Derived(Base):
    def method(self):
        print("Derived")
        super().method()

d = Derived()
d.method()
```

a) Base
b) Derived
c) Base Derived
d) Derived Base

**Answer: d**
_Explanation: The method is overridden, so it first prints "Derived", then calls the parent's method which prints "Base"._

**57. What is the difference between `@staticmethod` and `@classmethod`?**
a) No difference
b) `@staticmethod` doesn't receive automatic first argument, `@classmethod` receives class as first argument
c) `@staticmethod` is faster
d) `@classmethod` is for private methods

**Answer: b**
_Explanation: Static methods don't receive `self` or `cls` automatically, class methods receive the class as the first argument._

**58. What will `hasattr(obj, 'method')` return for a class with an instance method?**
a) True
b) False
c) Error
d) Depends on the instance

**Answer: a**
_Explanation: `hasattr()` checks if an object has an attribute, and methods are attributes of the class._

**59. What does the MRO (Method Resolution Order) determine?**
a) The order in which methods are defined
b) The order in which base classes are searched for methods
c) The order in which arguments are processed
d) The order in which attributes are accessed

**Answer: b**
_Explanation: MRO determines the order in which base classes are searched when looking up methods._

**60. What will `isinstance(True, bool)` return?**
a) True
b) False
c) Error
d) None

**Answer: a**
_Explanation: `True` is an instance of the `bool` class, so this returns True._

**61. What is encapsulation in OOP?**
a) Hiding implementation details and showing only functionality
b) Creating multiple instances of a class
c) Inheriting from a base class
d) Polymorphism

**Answer: a**
_Explanation: Encapsulation is the principle of hiding internal details and exposing only necessary interfaces._

**62. What will be the output of this method?**

```python
class Counter:
    count = 0
    def __init__(self):
        Counter.count += 1

c1 = Counter()
c2 = Counter()
print(Counter.count)
```

a) 0
b) 1
c) 2
d) Error

**Answer: c**
_Explanation: The class attribute `count` is incremented each time a new instance is created, so it becomes 2._

**63. What does `__str__()` method control?**
a) How an object is represented as a string
b) The object's memory address
c) Whether an object can be converted to string
d) The object's length

**Answer: a**
_Explanation: `__str__()` is called by `str()` and `print()` to get the string representation of an object._

**64. What will happen with this code?**

```python
class Test:
    pass

t = Test()
t.new_attr = "value"
print(t.new_attr)
```

a) Error
b) "value"
c) None
d) AttributeError

**Answer: b**
_Explanation: Python allows adding new attributes to instances dynamically, so this prints "value"._

**65. What is the purpose of `__repr__()`?**
a) Create a readable string representation
b) Create an unambiguous representation for debugging
c) Convert object to integer
d) Compare two objects

**Answer: b**
_Explanation: `__repr__()` should return an unambiguous string that could be used to recreate the object (though it's not guaranteed)._

### File Handling (Questions 66-75)

**66. What will this code do?**

```python
with open('file.txt', 'w') as f:
    f.write('Hello')
```

a) Read from file.txt
b) Write 'Hello' to file.txt, creating it if it doesn't exist
c) Append 'Hello' to file.txt
d) Raise an error

**Answer: b**
_Explanation: 'w' mode opens for writing, creating the file if it doesn't exist, and overwrites existing content._

**67. What does the `with` statement ensure?**
a) File is created automatically
b) File is automatically closed when exiting the block
c) File is read-only
d) File is written in binary mode

**Answer: b**
_Explanation: The `with` statement ensures that files (and other context managers) are properly closed when the block exits._

**68. What will `json.load(file)` return if file contains `{"key": "value"}`?**
a) A string
b) A dictionary
c) A list
d) A set

**Answer: b**
_Explanation: `json.load()` parses JSON and returns the corresponding Python data structure (dict in this case)._

**69. What is the difference between `read()` and `readline()`?**
a) `read()` reads the entire file, `readline()` reads one line
b) `read()` reads binary, `readline()` reads text
c) `read()` is faster, `readline()` is slower
d) No difference

**Answer: a**
_Explanation: `read()` reads the entire file (or specified number of characters), `readline()` reads one line at a time._

**70. What will happen with this CSV code?**

```python
import csv
with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Age'])
```

a) Write 'Name,Age' to the file
b) Write ['Name', 'Age'] to the file
c) Create a CSV file with headers
d) Error

**Answer: c**
_Explanation: `csv.writer` creates a writer object that properly formats CSV data. `writerow()` writes one row._

**71. What does `os.path.exists('file.txt')` return?**
a) True if file exists and is a file
b) True if file exists (file or directory)
c) True if file is readable
d) True if file can be written to

**Answer: b**
_Explanation: `os.path.exists()` returns True if the path exists, whether it's a file or directory._

**72. What will `shutil.move('source.txt', 'dest.txt')` do?**
a) Copy the file
b) Move (rename) the file
c) Delete the file
d) Create a backup

**Answer: b**
_Explanation: `shutil.move()` moves or renames files and directories. If the destination is in the same filesystem, it's a rename._

**73. What does `pathlib.Path('file.txt').suffix` return for 'document.pdf'?**
a) 'document'
b) 'pdf'
c) '.pdf'
d) 'document.pdf'

**Answer: c**
_Explanation: `suffix` includes the dot, so for 'document.pdf' it returns '.pdf'._

**74. What will `glob.glob('*.txt')` return if there are text files in the directory?**
a) A list of file paths matching the pattern
b) A generator object
c) A set of file paths
d) A dictionary

**Answer: a**
_Explanation: `glob.glob()` returns a list of paths matching the pattern. For '_.txt', it returns all .txt files.\*

**75. What does pickle do?**
a) Compresses files
b) Serializes Python objects to bytes
c) Encrypts files
d) Creates backup copies

**Answer: b**
_Explanation: The `pickle` module serializes Python objects to a binary format that can be deserialized later._

### Error Handling (Questions 76-85)

**76. What will be printed by this code?**

```python
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Error caught")
except:
    print("Other error")
finally:
    print("Finally")
```

a) Error caught
b) Other error
c) Error caught Finally
d) Finally Error caught

**Answer: c**
_Explanation: The specific exception handler catches the ZeroDivisionError, prints "Error caught", then the finally block always runs._

**77. What happens if no exception is raised in a try block?**
a) The except block runs
b) The else block runs (if present)
c) Only the try block runs
d) An error is raised

**Answer: b**
_Explanation: If no exception is raised, the else block (if present) executes after the try block completes successfully._

**78. What will be the output?**

```python
def func():
    try:
        return "try"
    finally:
        return "finally"

print(func())
```

a) try
b) finally
c) try finally
d) Error

**Answer: b**
_Explanation: The finally block always executes, and its return statement overrides the try block's return._

**79. What does `raise ValueError("Invalid input")` do?**
a) Catches a ValueError
b) Creates a ValueError with the message "Invalid input"
c) Handles the ValueError
d) Ignores ValueError

**Answer: b**
_Explanation: `raise` creates and raises an exception with the specified type and message._

**80. What will this code print?**

```python
try:
    x = [1, 2, 3][10]
except IndexError as e:
    print(f"Error: {e}")
```

a) Error: list index out of range
b) Error
c) 10
d) IndexError

**Answer: a**
_Explanation: Trying to access index 10 in a list of length 3 raises an IndexError, which is caught and the message is printed._

**81. What is the difference between `except Exception` and `except:`?**
a) No difference
b) `except Exception` catches all exceptions except system exceptions, `except:` catches all exceptions
c) `except Exception` is faster
d) `except Exception` is for integers only

**Answer: b**
_Explanation: `except Exception` excludes system-exiting exceptions like SystemExit, KeyboardInterrupt, and GeneratorExit._

**82. What will be logged?**

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.debug("This is debug")
logging.info("This is info")
```

a) Nothing
b) This is debug
c) This is info
d) This is debug This is info

**Answer: c**
_Explanation: With level set to INFO, DEBUG messages are not logged, but INFO messages are._

**83. What does `assert x > 0, "x must be positive"` do?**
a) Sets x to a positive value
b) Checks if x > 0, raises AssertionError with message if false
c) Compares x to 0
d) Returns True if x > 0

**Answer: b**
_Explanation: `assert` checks the condition and raises AssertionError with the message if the condition is False._

**84. What will `isinstance(5, (int, float))` return?**
a) True
b) False
c) Error
d) None

**Answer: a**
_Explanation: `isinstance()` checks if the object is an instance of any of the specified types. 5 is an int._

**85. What is the purpose of custom exceptions?**
a) To catch system errors
b) To make code more readable and maintainable
c) To improve performance
d) To bypass normal exception handling

**Answer: b**
_Explanation: Custom exceptions make code more readable by providing meaningful exception types that describe specific error conditions._

### Advanced Libraries (Questions 86-100)

**86. What will `requests.get('https://example.com').status_code` return if successful?**
a) 200
b) 404
c) 500
d) Depends on the server

**Answer: a**
_Explanation: A successful HTTP response typically returns status code 200 (OK)._

**87. What does `pandas.read_csv('file.csv')` return?**
a) A list
b) A dictionary
c) A DataFrame
d) A Series

**Answer: c**
_Explanation: `pandas.read_csv()` returns a DataFrame, which is a 2-dimensional labeled data structure._

**88. What will `matplotlib.pyplot.plot([1, 2, 3], [4, 5, 6])` create?**
a) A bar chart
b) A pie chart
c) A line plot
d) A scatter plot

**Answer: c**
_Explanation: `plot()` with x and y data creates a line plot by default._

**89. What does `numpy.array([1, 2, 3]).sum()` return?**
a) 6
b) 3
c) [1, 2, 3]
d) Error

**Answer: a**
_Explanation: `sum()` adds all elements in the array: 1 + 2 + 3 = 6._

**90. What will `re.findall(r'\d+', 'abc123def456')` return?**
a) ['123']
b) ['456']
c) ['123', '456']
d) '123456'

**Answer: c**
_Explanation: `findall()` returns a list of all non-overlapping matches of the pattern in the string._

**91. What does `datetime.datetime.now().strftime('%Y-%m-%d')` return?**
a) The current date in ISO format
b) A datetime object
c) The current time
d) A Unix timestamp

**Answer: a**
_Explanation: `strftime()` formats the datetime object as a string. %Y-%m-%d gives YYYY-MM-DD format._

**92. What will `sqlite3.connect('database.db').execute('SELECT * FROM users').fetchall()` return?**
a) A list of tuples
b) A dictionary
c) A string
d) A database connection

**Answer: a**
_Explanation: `fetchall()` returns a list of tuples, where each tuple represents a row from the result set._

**93. What does `collections.Counter(['a', 'b', 'a']).most_common()` return?**
a) [('a', 2), ('b', 1)]
b) [('b', 1), ('a', 2)]
c) ['a', 'a', 'b']
d) {'a': 2, 'b': 1}

**Answer: a**
_Explanation: `most_common()` returns elements sorted by count in descending order._

**94. What will `random.choice(['a', 'b', 'c'])` return?**
a) Always 'a'
b) Always 'b'
c) A random element from the list
d) An error

**Answer: c**
_Explanation: `choice()` returns a random element from the given sequence._

**95. What does `json.dumps({'key': 'value'})` return?**
a) A Python dictionary
b) A JSON string
c) A list
d) An error

**Answer: b**
_Explanation: `dumps()` serializes a Python object to a JSON formatted string._

**96. What will `pathlib.Path('.').glob('*.py')` return?**
a) A list of .py files in current directory
b) A generator of .py files
c) The current directory path
d) Error

**Answer: b**
_Explanation: `glob()` returns a generator that yields paths matching the pattern._

**97. What does `functools.lru_cache(maxsize=128)` do?**
a) Caches function return values
b) Clears function cache
c) Sets function size limit
d) Times function execution

**Answer: a**
_Explanation: `lru_cache` decorates a function to cache its results, avoiding repeated calculations with same arguments._

**98. What will be printed?**

```python
from concurrent.futures import ThreadPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * 2

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(task, [1, 2, 3]))
print(results)
```

a) [2, 4, 6]
b) [1, 2, 3]
c) Nothing (takes 3 seconds)
d) [1, 2, 3, 4, 5, 6]

**Answer: a**
*Explanation: ThreadPoolExecutor runs tasks in parallel, so all return their results: [1*2, 2*2, 3*2] = [2, 4, 6].\*

**99. What does `asyncio.run()` do?**
a) Runs a synchronous function
b) Runs an async function and manages the event loop
c) Creates an async function
d) Stops the event loop

**Answer: b**
_Explanation: `asyncio.run()` is the main entry point for async programs, creating and managing the event loop._

**100. What will `this.startswith('that') and this.endswith('other')` return for this = 'thatother'?**
a) True
b) False
c) Error
d) None

**Answer: a**
_Explanation: 'thatother' starts with 'that' and ends with 'other', so the and expression evaluates to True._

---

## Advanced Level Questions (101-150)

### System Programming (Questions 101-115)

**101. What does `os.environ.get('PATH', '/default/path')` return?**
a) Always '/default/path'
b) The PATH environment variable, or '/default/path' if not set
c) An empty string
d) Error

**Answer: b**
_Explanation: `environ.get()` returns the environment variable value, or the default if the variable doesn't exist._

**102. What will `subprocess.run(['ls', '-l'], capture_output=True, text=True).stdout` contain?**
a) The error output
b) The standard output
c) The return code
d) The process ID

**Answer: b**
_Explanation: `capture_output=True` captures both stdout and stderr, and `.stdout` contains the standard output._

**103. What does `signal.signal(signal.SIGINT, handler)` do?**
a) Ignores Ctrl+C
b) Sets a custom handler for Ctrl+C
c) Exits the program immediately
d) Starts a signal handler

**Answer: b**
_Explanation: This sets a custom signal handler that will be called when Ctrl+C (SIGINT) is received._

**104. What will be the result of `threading.active_count()` in a simple script?**
a) 0
b) 1
c) 2
d) Depends on imports

**Answer: b**
_Explanation: In a simple script, there's at least the main thread, so active_count() returns 1._

**105. What does `multiprocessing.cpu_count()` return?**
a) The number of CPU cores
b) The number of running processes
c) The number of threads
d) The CPU usage percentage

**Answer: a**
_Explanation: `cpu_count()` returns the number of CPUs available to the current process._

**106. What will `queue.Queue().put('item'); queue.Queue().get()` return?**
a) An empty queue
b) 'item'
c) None
d) Error

**Answer: b**
_Explanation: `put()` adds an item to the queue, `get()` removes and returns it._

**107. What does `os.getcwd()` return?**
a) The home directory
b) The current working directory
c) The command directory
d) The script directory

**Answer: b**
_Explanation: `getcwd()` returns the current working directory as a string._

**108. What will `pathlib.Path.home()` return?**
a) The current directory
b) The user's home directory
c) The script location
d) The parent directory

**Answer: b**
_Explanation: `Path.home()` returns a Path object representing the user's home directory._

**109. What does `tempfile.mkstemp()` return?**
a) A temporary directory
b) A temporary file descriptor and path
c) A temporary file path
d) Error

**Answer: b**
_Explanation: `mkstemp()` returns a tuple containing a file descriptor and the path to the created temporary file._

**110. What will `platform.system()` return on Windows?**
a) 'Windows'
b) 'WIN'
c) 'win32'
d) 'Microsoft'

**Answer: a**
_Explanation: `platform.system()` returns the OS name, which is 'Windows' on Windows systems._

**111. What does `atexit.register(func)` do?**
a) Registers func to run at program start
b) Registers func to run when the program exits normally
c) Unregisters a function
d) Runs func immediately

**Answer: b**
_Explanation: `atexit.register()` registers functions to be called when the program exits normally._

**112. What will `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` contain?**
a) CPU usage
b) Memory usage (RSS)
c) File descriptors
d) Process creation time

**Answer: b**
_Explanation: `ru_maxrss` contains the maximum resident set size (memory usage) in kilobytes on Unix systems._

**113. What does `weakref.finalize(obj, func)` do?**
a) Immediately calls func with obj
b) Registers func to be called when obj is garbage collected
c) Prevents obj from being garbage collected
d) Creates a weak reference to obj

**Answer: b**
_Explanation: `finalize()` creates a finalizer that calls the function when the object is about to be garbage collected._

**114. What will `inspect.signature(func)` return for a function?**
a) The function name
b) A Signature object containing parameter information
c) The function source code
d) The function bytecode

**Answer: b**
_Explanation: `inspect.signature()` returns a Signature object that represents the function's parameters._

**115. What does `warnings.filterwarnings('ignore')` do?**
a) Removes all warnings
b) Ignores warnings matching the pattern
c) Converts warnings to errors
d) Saves warnings to a file

**Answer: b**
_Explanation: This filters out warnings, so they won't be displayed._

### Database Operations (Questions 116-125)

**116. What will this SQLite code return?**

```python
conn.execute('SELECT COUNT(*) FROM users WHERE age > ?', (25,)).fetchone()
```

a) The number of users older than 25
b) A list of users
c) True if users exist
d) Error

**Answer: a**
_Explanation: `COUNT(_)`returns the number of rows matching the condition, and`fetchone()` returns that single value.\*

**117. What does `conn.commit()` do in SQLite?**
a) Starts a transaction
b) Saves changes permanently to the database
c) Rolls back changes
d) Closes the connection

**Answer: b**
_Explanation: `commit()` saves all changes in the current transaction permanently to the database._

**118. What will `pd.DataFrame({'A': [1, 2], 'B': [3, 4]}).shape` return?**
a) (2, 2)
b) (4, 2)
c) [1, 2, 3, 4]
d) {'A': [1, 2], 'B': [3, 4]}

**Answer: a**
_Explanation: `shape` returns a tuple representing the dimensions: (number of rows, number of columns)._

**119. What does `df.groupby('category').mean()` do?**
a) Calculates the mean of all numeric columns for each category
b) Groups by category and returns the mean value
c) Sorts the DataFrame by category
d) Filters rows with mean values

**Answer: a**
_Explanation: `groupby('category').mean()` groups data by the 'category' column and calculates the mean for each numeric column._

**120. What will `np.array([1, 2, 3]) + np.array([4, 5, 6])` return?**
a) [5, 7, 9]
b) [1, 2, 3, 4, 5, 6]
c) [1, 4, 2, 5, 3, 6]
d) Error

**Answer: a**
_Explanation: NumPy arrays support element-wise operations, so addition is performed element by element._

**121. What does `df.merge(df2, on='key', how='left')` do?**
a) Joins df2 to df on 'key' with left join
b) Combines rows with same 'key'
c) Updates df with df2 values
d) Filters df by df2

**Answer: a**
_Explanation: `merge()` performs a SQL-style join. Left join includes all rows from df and matching rows from df2._

**122. What will `df['column'].apply(lambda x: x * 2)` return?**
a) A Series with each value doubled
b) A DataFrame
c) A scalar value
d) Error

**Answer: a**
_Explanation: `apply()` applies the function to each element in the Series, returning a new Series._

**123. What does `conn.executemany('INSERT INTO table VALUES (?)', data)` do?**
a) Inserts one row
b) Inserts multiple rows using the data sequence
c) Updates multiple rows
d) Deletes rows

**Answer: b**
_Explanation: `executemany()` executes the statement for each element in the provided sequence._

**124. What will `df.info()` print?**
a) Data types and memory usage
b) Statistical summary
c) The first few rows
d) Column names only

**Answer: a**
_Explanation: `info()` provides a concise summary of the DataFrame, including data types and memory usage._

**125. What does `np.random.seed(42)` ensure?**
a) Sets random state for reproducible results
b) Generates random numbers between 0 and 42
c) Seeds the random number generator
d) All of the above

**Answer: a**
_Explanation: Setting the seed ensures that random operations produce the same results each time the code runs._

### Web Development (Questions 126-140)

**126. What will `requests.get('https://httpbin.org/get').json()` return?**
a) The response status code
b) A dictionary from the JSON response
c) The response headers
d) The response text

**Answer: b**
_Explanation: `.json()` parses the response content as JSON and returns the corresponding Python object._

**127. What does `BeautifulSoup(html, 'html.parser').find('div', class_='content')` do?**
a) Finds the first div with class 'content'
b) Finds all divs
c) Finds div with id 'content'
d) Creates a div element

**Answer: a**
_Explanation: `find()` returns the first matching element. `class_='content'` searches for elements with that CSS class._

**128. What will `re.sub(r'\d+', 'X', 'abc123def456')` return?**
a) 'abcXdefX'
b) 'abc123def456'
c) 'XbcXefX'
d) Error

**Answer: a**
_Explanation: `sub()` replaces all matches of the pattern with the replacement string. All digit sequences are replaced with 'X'._

**129. What does `urljoin('https://example.com/', '/page')` return?**
a) 'https://example.com/page'
b) 'https://example.com//page'
c) '/page'
d) 'https://example.com/'

**Answer: a**
_Explanation: `urljoin()` properly joins URL components, handling the trailing slash correctly._

**130. What will `session = requests.Session(); session.get('https://example.com')` do?**
a) Makes a single request
b) Creates a session object
c) Makes requests with session persistence (cookies, etc.)
d) Raises an error

**Answer: c**
_Explanation: Session objects maintain cookies and other state across multiple requests._

**131. What does `bs.select('.class-name p')` return?**
a) All paragraph elements
b) All elements with class 'class-name'
c) All paragraphs inside elements with class 'class-name'
d) The first paragraph

**Answer: c**
_Explanation: CSS selectors can specify hierarchy. This selects all 'p' elements that are descendants of '.class-name'._

**132. What will be the status code for a successful POST request?**
a) 200
b) 201
c) 204
d) 400

**Answer: b**
_Explanation: 201 Created is the typical status code for successful POST requests that create a resource._

**133. What does `html.escape('<script>')` return?**
a) '&lt;script&gt;'
b) '<script>'
c) 'script'
d) Error

**Answer: a**
_Explanation: `html.escape()` converts special HTML characters to their entity equivalents for safe display._

**134. What will `requests.head('https://example.com')` return?**
a) Same as GET
b) Only response headers
c) Only response body
d) Error

**Answer: b**
_Explanation: HEAD request returns only the headers, not the body, useful for checking resource availability._

**135. What does `time.sleep(0.5)` do?**
a) Pauses execution for 0.5 milliseconds
b) Pauses execution for 0.5 seconds
c) Fastens execution
d) Creates a delay of 5 seconds

**Answer: b**
_Explanation: `sleep()` pauses execution for the specified number of seconds._

**136. What will `html.select_one('#main').get_text()` return?**
a) The inner HTML of element with id 'main'
b) The text content of element with id 'main'
c) The tag name
d) The attributes

**Answer: b**
_Explanation: `get_text()` returns all text content from the element and its descendants._

**137. What does `requests.exceptions.RequestException` catch?**
a) Only network errors
b) Only timeout errors
c) All request-related exceptions
d) No exceptions

**Answer: c**
_Explanation: `RequestException` is the base class for all requests-related exceptions._

**138. What will `pattern = re.compile(r'\d+'); pattern.findall('abc123def')` return?**
a) ['123']
b) '123'
c) The match object
d) Error

**Answer: a**
_Explanation: Compiling a pattern and using `findall()` returns all non-overlapping matches._

**139. What does `json.decoder.JSONDecodeError` indicate?**
a) Invalid JSON syntax
b) Missing JSON file
c) Wrong JSON version
d) JSON too large

**Answer: a**
_Explanation: `JSONDecodeError` is raised when parsing invalid JSON strings._

**140. What will `session.cookies['session_id']` return?**
a) The session ID cookie value
b) Creates a new cookie
c) Deletes the cookie
d) Updates the cookie

**Answer: a**
_Explanation: Accessing a cookie key returns its value. This works for cookies that have been set._

### Performance & Optimization (Questions 141-150)

**141. What will be faster: a list comprehension or a for loop with append()?**
a) List comprehension
b) For loop with append()
c) Same performance
d) Depends on the data

**Answer: a**
_Explanation: List comprehensions are generally faster because they're optimized C-level operations._

**142. What does `@lru_cache(maxsize=None)` optimize?**
a) Function execution time
b) Memory usage for repeated function calls
c) Function parameters
d) Function return types

**Answer: b**
_Explanation: `lru_cache` caches function results, making repeated calls with same arguments very fast._

**143. What will `generator = (x*x for x in range(1000000))` create?**
a) A list of one million squares
b) A generator that produces squares on demand
c) An error for large numbers
d) A set of squares

**Answer: b**
_Explanation: Using parentheses creates a generator expression, which yields values lazily._

**144. What does `__slots__` improve in a class?**
a) Method execution speed
b) Memory usage by preventing **dict**
c) Inheritance behavior
d) Method resolution order

**Answer: b**
_Explanation: `__slots__` restricts instance attributes, reducing memory usage by not creating **dict** for each instance._

**145. What will `bisect.bisect_left([1, 2, 4, 4, 5], 4)` return?**
a) 2
b) 3
c) 4
d) 5

**Answer: a**
_Explanation: `bisect_left()` returns the position where 4 should be inserted to maintain sorted order, which is index 2._

**146. What does `array.array('i', [1, 2, 3])` create?**
a) A list
b) An array of integers (more memory efficient than list)
c) A tuple
d) A set

**Answer: b**
_Explanation: `array.array()` creates a compact array of a specific type, more memory efficient than lists for numeric data._

**147. What will be the memory difference between a list and tuple of same content?**
a) List uses more memory
b) Tuple uses more memory
c) Same memory usage
d) Depends on content

**Answer: a**
_Explanation: Lists are mutable and require extra memory for the mutability overhead. Tuples are more memory efficient._

**148. What does `memoryview(obj)` provide?**
a) A copy of the object
b) A view into the object's memory buffer
c) A new object of same type
d) An error

**Answer: b**
_Explanation: `memoryview()` creates a memory view object that allows slicing without copying the underlying data._

**149. What will `itertools.islice(range(100), 10, 20)` return?**
a) First 10 elements
b) Elements from index 10 to 19
c) Elements 10 to 20
d) All elements

**Answer: b**
_Explanation: `islice(iterable, start, stop)` returns elements from start (inclusive) to stop (exclusive)._

**150. What does `__getitem__` optimize when implemented properly?**
a) Direct access to items
b) Iteration over items
c) Both direct access and iteration
d) Item assignment

**Answer: c**
_Explanation: Implementing `__getitem__` allows both direct item access (obj[key]) and iteration (for item in obj)._

---

## Expert Level Questions (151-200)

### Design Patterns & Architecture (Questions 151-165)

**151. What is the primary benefit of the Singleton pattern?**
a) Improved inheritance
b) Multiple instances sharing same state
c) Reduced memory usage
d) Faster execution

**Answer: b**
_Explanation: Singleton ensures only one instance exists and provides global access to it, useful for shared state._

**152. What will this factory pattern code output?**

```python
class Product:
    def __init__(self, name):
        self.name = name

class Factory:
    @staticmethod
    def create_product(product_type):
        return Product(product_type)

p = Factory.create_product("widget")
print(p.name)
```

a) Factory
b) Product
c) widget
d) Error

**Answer: c**
_Explanation: Factory method creates and returns a Product instance with the given name, so 'widget' is printed._

**153. What does the Observer pattern accomplish?**
a) One object watching multiple objects
b) Multiple objects watching one object
c) Objects observing time
d) Object creation pattern

**Answer: b**
_Explanation: Observer pattern defines a one-to-many dependency where multiple observers are notified when the subject changes._

**154. What is the key benefit of Dependency Injection?**
a) Reduced code coupling
b) Faster execution
c) Less memory usage
d) Better inheritance

**Answer: a**
_Explanation: Dependency Injection decouples classes from their dependencies, making code more testable and maintainable._

**155. What will this decorator pattern code print?**

```python
class Component:
    def operation(self):
        return "Component"

class Decorator(Component):
    def __init__(self, component):
        self._component = component

    def operation(self):
        return f"Decorator({self._component.operation()})"

c = Component()
d = Decorator(c)
print(d.operation())
```

a) Component
b) Decorator
c) Decorator(Component)
d) Error

**Answer: c**
_Explanation: The decorator wraps the component and adds functionality, returning "Decorator(Component)"._

**156. What does the Strategy pattern promote?**
a) Object creation
b) Runtime algorithm selection
c) Object inheritance
d) Memory optimization

**Answer: b**
_Explanation: Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable._

**157. What is the primary use of the Command pattern?**
a) Creating objects
b) Encapsulating requests as objects
c) Managing object states
d) Optimizing performance

**Answer: b**
_Explanation: Command pattern encapsulates a request as an object, allowing for queuing, logging, and undo operations._

**158. What will this state pattern code output?**

```python
class State:
    def handle(self):
        pass

class ConcreteState(State):
    def handle(self):
        return "State A"

class Context:
    def __init__(self, state):
        self._state = state

    def request(self):
        return self._state.handle()

c = Context(ConcreteState())
print(c.request())
```

a) Context
b) State
c) State A
d) Error

**Answer: c**
_Explanation: The context delegates the request to the current state, which returns "State A"._

**159. What is the key advantage of the Adapter pattern?**
a) Object creation
b) Interface compatibility between incompatible classes
c) Performance improvement
d) Memory optimization

**Answer: b**
_Explanation: Adapter pattern allows objects with incompatible interfaces to work together by wrapping the interface._

**160. What does the Facade pattern provide?**
a) Multiple interfaces
b) Simplified interface to complex subsystem
c) Object creation
d) State management

**Answer: b**
_Explanation: Facade provides a simplified interface to a complex subsystem, making it easier to use._

**161. What is the primary purpose of the Chain of Responsibility pattern?**
a) Creating object chains
b) Passing requests along a chain of handlers
c) Managing object dependencies
d) Optimizing chain operations

**Answer: b**
_Explanation: Chain of Responsibility lets multiple objects handle a request by passing it along the chain until handled._

**162. What will this prototype pattern code return?**

```python
import copy

class Prototype:
    def __init__(self):
        self.value = "original"

    def clone(self):
        return copy.deepcopy(self)

p1 = Prototype()
p2 = p1.clone()
p2.value = "modified"
print(p1.value)
```

a) original
b) modified
c) Error
d) None

**Answer: a**
_Explanation: `clone()` creates a deep copy, so modifying p2 doesn't affect p1. p1.value remains "original"._

**163. What is the main benefit of using Abstract Base Classes (ABC)?**
a) Performance improvement
b) Memory optimization
c) Interface enforcement and polymorphism
d) Object creation

**Answer: c**
_Explanation: ABCs define abstract methods that must be implemented by concrete subclasses, enforcing interface contracts._

**164. What does the Builder pattern separate?**
a) Object construction from representation
b) Object usage from construction
c) Class from instance
d) Method from class

**Answer: a**
_Explanation: Builder pattern separates the construction of a complex object from its representation._

**165. What is the key characteristic of the Mediator pattern?**
a) Objects communicate directly
b) Objects communicate through a central mediator
c) Objects communicate through inheritance
d) Objects don't communicate

**Answer: b**
_Explanation: Mediator pattern defines an object that encapsulates how objects interact, reducing direct dependencies._

### Advanced Python Concepts (Questions 166-180)

**166. What will this metaclass code output?**

```python
class Meta(type):
    def __new__(cls, name, bases, attrs):
        attrs['created'] = True
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=Meta):
    pass

print(MyClass.created)
```

a) True
b) False
c) Error
d) None

**Answer: a**
_Explanation: The metaclass's `__new__` method adds a 'created' attribute to the class, so `MyClass.created` is True._

**167. What does `__getattr__` control?**
a) Attribute access when attribute is not found through normal lookup
b) Attribute access for any attribute
c) Method calls
d) Class creation

**Answer: a**
_Explanation: `__getattr__` is called when an attribute is not found through normal attribute lookup._

**168. What will be printed by this code?**

```python
class A:
    def __getattribute__(self, name):
        if name == 'x':
            return 'attribute_x'
        return super().__getattribute__(name)

a = A()
print(a.x)
print(a.y)
```

a) attribute_x followed by AttributeError
b) attribute_x followed by None
c) attribute_x followed by a
d) Error followed by Error

**Answer: a**
_Explanation: `__getattribute__` intercepts all attribute access. For 'x', it returns 'attribute_x'. For 'y', it raises AttributeError._

**169. What does `__slots__` prevent?**
a) Method creation
b) Instance dictionary creation
c) Class inheritance
d) Object instantiation

**Answer: b**
_Explanation: `__slots__` restricts the attributes that instances can have, preventing the creation of `__dict__`._

**170. What will `type('DynamicClass', (object,), {'x': 42})().x` return?**
a) DynamicClass
b) 42
c) Error
d) None

**Answer: b**
_Explanation: `type()` creates a class dynamically with attributes. The instance's `x` attribute returns 42._

**171. What does `functools.partial(func, arg1)` create?**
a) A new function with arg1 as first argument
b) A copy of the function
c) A faster function
d) An error

**Answer: a**
_Explanation: `partial()` creates a new function with some arguments pre-filled, making it a partial application._

**172. What will this generator code output?**

```python
def generator():
    yield 1
    yield 2
    yield 3

g = generator()
print(list(g))
print(list(g))
```

a) [1, 2, 3] followed by [1, 2, 3]
b) [1, 2, 3] followed by []
c] [] followed by [1, 2, 3]
d) Error followed by Error

**Answer: b**
_Explanation: Generators are consumed after iteration. The second `list(g)` returns an empty list because the generator is exhausted._

**173. What does `contextlib.contextmanager` do?**
a) Creates context managers from generators
b) Manages Python context variables
c) Provides context for exceptions
d) Creates context menus

**Answer: a**
_Explanation: The `@contextmanager` decorator allows using a generator function as a context manager._

**174. What will `itertools.chain.from_iterable(['ab', 'cd'])` return?**
a) ['ab', 'cd']
b) ['a', 'b', 'c', 'd']
c) Iterator over ['a', 'b', 'c', 'd']
d) Error

**Answer: c**
_Explanation: `chain.from_iterable()` creates an iterator that flattens the input iterable into a single sequence._

**175. What does `weakref.WeakSet()` create?**
a) A set that doesn't prevent garbage collection
b) A read-only set
c) An empty set
d) A set of weak references

**Answer: a**
_Explanation: `WeakSet` holds weak references to objects, allowing garbage collection even if the set holds references._

**176. What will this code print?**

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(1, 2)
print(p.x, p.y)
```

a) 1 2
b) Point(1, 2)
c) x y
d) Error

**Answer: a**
_Explanation: `@dataclass` automatically generates `__init__`, `__repr__`, and other methods. p.x and p.y return the values._

**177. What does `typing.NamedTuple` provide?**
a) A tuple with named elements
b) A dictionary with tuple values
c) A list with named elements
d) A set with named elements

**Answer: a**
_Explanation: `NamedTuple` creates a tuple with named fields, providing both tuple behavior and named access._

**178. What will be the result of `int.from_bytes(b'\x01\x02', 'big')`?**
a) 258
b) 513
c) 1026
d) Error

**Answer: b**
*Explanation: Converting bytes '\x01\x02' to integer gives 1*256 + 2 = 258 in little-endian, or 513 in big-endian.\*

**179. What does `pathlib.Path.resolve()` return?**
a) A relative path
b) The absolute path with symlinks resolved
c) A temporary path
d) The parent path

**Answer: b**
_Explanation: `resolve()` returns the absolute path with all symbolic links resolved._

**180. What will this asyncio code print?**

```python
import asyncio

async def task():
    await asyncio.sleep(0.1)
    return "done"

async def main():
    result = await task()
    print(result)

asyncio.run(main())
```

a) done
b) Nothing
c) Error
d) task

**Answer: a**
_Explanation: `asyncio.run()` executes the async function. `await` pauses execution until `task()` completes, then prints "done"._

### Security & Best Practices (Questions 181-190)

**181. What is the primary security concern with `eval()`?**
a) Performance issues
b) Code injection vulnerabilities
c) Memory leaks
d) Stack overflow

**Answer: b**
_Explanation: `eval()` can execute arbitrary code, making it vulnerable to code injection attacks._

**182. What does `hashlib.sha256(b'data').hexdigest()` return?**
a) A random string
b) A 64-character hexadecimal string representing the SHA-256 hash
c) The original data
d) An error

**Answer: b**
_Explanation: SHA-256 produces a 256-bit hash, represented as 64 hexadecimal characters._

**183. What will `secrets.token_hex(16)` return?**
a) 16 random bytes in hexadecimal
b) 32 random bytes
c) 16 random characters
d) A secure random token

**Answer: a**
_Explanation: `token_hex(16)` generates 16 random bytes and converts them to a 32-character hex string._

**184. What does `ssl.create_default_context()` provide?**
a) A secure SSL context
b) An SSL certificate
c) An SSL connection
d) An SSL server

**Answer: a**
_Explanation: This creates a default SSL context with secure settings for making HTTPS connections._

**185. What is the purpose of input sanitization?**
a) Improve performance
b) Prevent security vulnerabilities like SQL injection
c) Save memory
d) Speed up processing

**Answer: b**
_Explanation: Sanitizing input removes or escapes potentially harmful content to prevent injection attacks._

**186. What will `base64.b64encode(b'Hello').decode()` return?**
a) 'Hello'
b) 'SGVsbG8='
c) A random string
d) Error

**Answer: b**
_Explanation: Base64 encoding 'Hello' produces 'SGVsbG8='. The decode() converts bytes to string._

**187. What does `HMAC.new(key, msg, digestmod=hashlib.sha256).hexdigest()` create?**
a) A random number
b) A hash-based message authentication code
c) An encrypted message
d) A digital signature

**Answer: b**
_Explanation: HMAC provides message authentication to verify both the integrity and authenticity of a message._

**188. What is the primary use of `tempfile.NamedTemporaryFile(delete=False)`?**
a) Creating a file that automatically deletes
b) Creating a file that persists after program ends
c) Creating an encrypted file
d) Creating a temporary directory

**Answer: b**
_Explanation: `delete=False` prevents automatic deletion when the file is closed, allowing the file to persist._

**189. What does `cryptography.fernet.Fernet.generate_key()` provide?**
a) A random encryption key
b) A secure hash
c) A digital certificate
d) A password

**Answer: a**
_Explanation: This generates a 32-byte key that can be used for Fernet encryption/decryption._

**190. What is the main benefit of using `configparser` for configuration?**
a) Better performance
b) Structured configuration files with sections and key-value pairs
c) Encrypted configuration
d) Binary configuration files

**Answer: b**
_Explanation: `configparser` provides a way to create structured configuration files with sections and key-value pairs._

### Performance & Optimization (Questions 191-200)

**191. What will be more memory efficient: a list of strings or a tuple of strings?**
a) List
b) Tuple
c) Same memory usage
d) Depends on the strings

**Answer: b**
_Explanation: Tuples are more memory efficient than lists because they're immutable and don't need extra overhead for mutability._

**192. What does `@functools.lru_cache(maxsize=128)` optimize?**
a) Function execution time for repeated calls
b) Memory usage by caching results
c) Both execution time and memory
d) Only execution time

**Answer: c**
_Explanation: LRU cache stores function results, improving execution time for repeated calls and managing memory with maxsize._

**193. What will `array.array('I', range(1000)).buffer_info()` return?**
a) The memory address and size of the array
b) The array contents
c) The array type
d) Error

**Answer: a**
_Explanation: `buffer_info()` returns a tuple containing the memory address where the array's data is stored and the array size._

**194. What is the benefit of using `__slots__` in a class with many instances?**
a) Faster method calls
b) Reduced memory usage by preventing instance dictionaries
c) Better inheritance
d) More features

**Answer: b**
_Explanation: `__slots__` prevents creation of instance dictionaries, significantly reducing memory usage for classes with many instances._

**195. What will be the difference in performance between `list comprehension` and `map()`?**
a) List comprehension is generally faster
b) map() is generally fasterc) Same performance
d) Depends on the operation

**Answer: a**
_Explanation: List comprehensions are generally faster because they're executed at C speed in the interpreter, while map() involves function calls._

**196. What does `bisect.insort_left(arr, value)` do?**
a) Inserts value at the beginning of the sorted array
b) Inserts value in sorted order at the correct position
c) Inserts value at the end
d) Replaces the first occurrence

**Answer: b**
_Explanation: `insort_left()` maintains sorted order by inserting the value at the correct position (leftmost for duplicates)._

**197. What will `itertools.takewhile(lambda x: x < 5, [1, 2, 3, 4, 5, 6])` return?**
a) [1, 2, 3, 4]
b) [1, 2, 3, 4, 5]
c) [5, 6]
d) All elements

**Answer: a**
_Explanation: `takewhile()` takes elements while the condition is true, stopping when it encounters 5._

**198. What does `memoryview(obj).cast('I')` do?**
a) Changes the object type
b) Creates a view with different data type interpretation
c) Copies the object
d) Deletes the view

**Answer: b**
_Explanation: `cast()` creates a new memoryview with different data type interpretation without copying data._

**199. What will be the result of `deque([1, 2, 3], maxlen=3).append(4)`?**
a) [1, 2, 3, 4]
b) [2, 3, 4]
c) [1, 2, 3]
d) Error

**Answer: b**
_Explanation: With maxlen=3, adding a 4th element removes the leftmost element, resulting in [2, 3, 4]._

**200. What is the primary benefit of using `__slots__` with descriptors?**
a) Faster descriptor access
b) Memory optimization with typed attributes
c) Better inheritance
d) More features

**Answer: b**
_Explanation: `__slots__` with descriptors allows typed, memory-efficient attribute storage with validation._

---

## Coding Challenge Section

### Beginner Coding Challenges (CC1-CC20)

**CC1. FizzBuzz Implementation**
Write a program that prints numbers 1-100, but for multiples of 3 print "Fizz", for multiples of 5 print "Buzz", and for multiples of both print "FizzBuzz".

```python
def fizzbuzz():
    for i in range(1, 101):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
```

**CC2. Palindrome Checker**
Create a function that checks if a string is a palindrome (reads the same forwards and backwards).

```python
def is_palindrome(s):
    # Remove spaces and convert to lowercase
    s = s.replace(" ", "").lower()
    return s == s[::-1]
```

**CC3. Find Maximum in List**
Write a function that finds the maximum number in a list without using built-in `max()`.

```python
def find_max(numbers):
    if not numbers:
        return None

    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
```

**CC4. Count Word Frequency**
Count the frequency of each word in a given text.

```python
def word_frequency(text):
    words = text.lower().split()
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency
```

**CC5. List Comprehension Challenge**
Given a list of numbers, create a new list with only the even numbers squared.

```python
def even_squares(numbers):
    return [x**2 for x in numbers if x % 2 == 0]
```

**CC6. String Reversal**
Reverse a string without using string slicing or reversed().

```python
def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result
```

**CC7. Prime Number Checker**
Check if a number is prime.

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

**CC8. Factorial Function**
Calculate factorial of a number (n! = n × (n-1) × ... × 1).

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

**CC9. Fibonacci Sequence**
Generate the first n numbers in the Fibonacci sequence.

```python
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence
```

**CC10. Anagram Checker**
Check if two strings are anagrams (contain the same characters in different order).

```python
def is_anagram(str1, str2):
    return sorted(str1.lower()) == sorted(str2.lower())
```

### Intermediate Coding Challenges (CC11-CC30)

**CC11. Merge Two Sorted Lists**
Merge two sorted lists into one sorted list.

```python
def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1

    # Add remaining elements
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result
```

**CC12. Binary Search**
Implement binary search algorithm.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

**CC13. Group Anagrams**
Group anagrams from a list of strings.

```python
def group_anagrams(strings):
    anagram_groups = {}
    for s in strings:
        sorted_s = ''.join(sorted(s))
        if sorted_s not in anagram_groups:
            anagram_groups[sorted_s] = []
        anagram_groups[sorted_s].append(s)
    return list(anagram_groups.values())
```

**CC14. Find Duplicate Numbers**
Find all duplicate numbers in a list.

```python
def find_duplicates(numbers):
    seen = set()
    duplicates = set()

    for num in numbers:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)

    return list(duplicates)
```

**CC15. Rotate Array**
Rotate an array to the right by k positions.

```python
def rotate_array(arr, k):
    k = k % len(arr)
    return arr[-k:] + arr[:-k]
```

**CC16. Two Sum Problem**
Find two numbers in a list that add up to a target.

```python
def two_sum(nums, target):
    num_to_index = {}

    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i

    return []
```

**CC17. Valid Parentheses**
Check if a string of parentheses is valid.

```python
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return not stack
```

**CC18. Remove Duplicates from Sorted List**
Remove duplicates from a sorted list.

```python
def remove_duplicates(sorted_list):
    if not sorted_list:
        return []

    result = [sorted_list[0]]
    for i in range(1, len(sorted_list)):
        if sorted_list[i] != sorted_list[i-1]:
            result.append(sorted_list[i])

    return result
```

**CC19. First Non-Repeating Character**
Find the first non-repeating character in a string.

```python
def first_non_repeating_char(s):
    char_count = {}

    # Count characters
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    # Find first non-repeating
    for char in s:
        if char_count[char] == 1:
            return char

    return None
```

**CC20. Implement Stack with List**
Implement a stack data structure using a list.

```python
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)
```

### Advanced Coding Challenges (CC21-CC40)

**CC21. LRU Cache Implementation**
Implement a Least Recently Used (LRU) cache.

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)
```

**CC22. Graph Traversal (BFS/DFS)**
Implement breadth-first and depth-first search.

```python
from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def bfs(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            print(node, end=" ")

            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    def dfs(self, start):
        visited = set()
        self._dfs_recursive(start, visited)

    def _dfs_recursive(self, node, visited):
        visited.add(node)
        print(node, end=" ")

        for neighbor in self.graph.get(node, []):
            if neighbor not in visited:
                self._dfs_recursive(neighbor, visited)
```

**CC23. Dynamic Programming - Longest Common Subsequence**
Find the longest common subsequence between two strings.

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

**CC24. Topological Sort**
Perform topological sorting on a directed acyclic graph.

```python
def topological_sort(graph):
    in_degree = {node: 0 for node in graph}

    # Calculate in-degrees
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Find nodes with no incoming edges
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == len(graph) else None
```

**CC25. Trie Implementation**
Implement a Trie (prefix tree) for string storage and search.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

---

## Debugging Exercises

### Common Python Errors and Fixes (DB1-DB20)

**DB1. IndexError in List Access**

```python
# Buggy Code
numbers = [1, 2, 3]
print(numbers[3])  # IndexError: list index out of range

# Fixed Code
numbers = [1, 2, 3]
if len(numbers) > 3:
    print(numbers[3])
else:
    print("Index out of range")
```

**DB2. KeyError in Dictionary Access**

```python
# Buggy Code
student = {"name": "Alice", "age": 20}
print(student["grade"])  # KeyError: 'grade'

# Fixed Code
student = {"name": "Alice", "age": 20}
print(student.get("grade", "Grade not available"))
```

**DB3. TypeError in String Concatenation**

```python
# Buggy Code
age = 25
message = "I am " + age + " years old"  # TypeError

# Fixed Code
age = 25
message = f"I am {age} years old"  # or str(age)
```

**DB4. AttributeError with None**

```python
# Buggy Code
def get_length(obj):
    return len(obj)  # AttributeError if obj is None

# Fixed Code
def get_length(obj):
    return len(obj) if obj is not None else 0
```

**DB5. Mutable Default Argument**

```python
# Buggy Code
def add_item(item, lst=[]):
    lst.append(item)
    return lst

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] - unexpected!

# Fixed Code
def add_item(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

**DB6. Integer Division vs Float Division**

```python
# Buggy Code
result = 10 / 3  # Gives 3.333... but if assigned to int:
integer_result = int(10 / 3)  # 3 (should be 3)

# Fixed Code
result = 10 / 3  # 3.333...
integer_result = 10 // 3  # 3 (floor division)
```

**DB7. Global vs Local Variable**

```python
# Buggy Code
count = 0

def increment():
    count += 1  # UnboundLocalError

# Fixed Code
count = 0

def increment():
    global count
    count += 1
```

**DB8. List Modification During Iteration**

```python
# Buggy Code
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)  # Skips some elements

# Fixed Code
numbers = [1, 2, 3, 4, 5]
numbers = [num for num in numbers if num % 2 != 0]
```

**DB9. File Not Closed**

```python
# Buggy Code
file = open("data.txt", "r")
data = file.read()
# file.close() - missing!

# Fixed Code
with open("data.txt", "r") as file:
    data = file.read()
# File automatically closed
```

**DB10. String Immutability**

```python
# Buggy Code
text = "hello"
text[0] = "H"  # TypeError: 'str' object does not support item assignment

# Fixed Code
text = "hello"
text = "H" + text[1:]  # Creates new string
```

**DB11. Comparison vs Assignment**

```python
# Buggy Code
if x = 5:  # SyntaxError
    print("x is 5")

# Fixed Code
if x == 5:
    print("x is 5")
```

**DB12. Scope Issues with Nested Functions**

```python
# Buggy Code
def outer():
    x = 10
    def inner():
        print(x)  # This works
        x = 20    # UnboundLocalError

# Fixed Code
def outer():
    x = 10
    def inner():
        nonlocal x
        print(x)
        x = 20
    inner()
```

**DB13. Missing Return Statement**

```python
# Buggy Code
def find_max(numbers):
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    # Missing return statement

# Fixed Code
def find_max(numbers):
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num
```

**DB14. Wrong Exception Handling**

```python
# Buggy Code
try:
    result = 10 / 0
except ValueError:  # Wrong exception type
    print("Value error")

# Fixed Code
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Division by zero error")
```

**DB15. Integer vs String Input**

```python
# Buggy Code
age = input("Enter your age: ")
next_year = age + 1  # TypeError

# Fixed Code
age = input("Enter your age: ")
next_year = int(age) + 1
print(f"Next year you'll be {next_year}")
```

**DB16. Dictionary vs Set Syntax**

```python
# Buggy Code
empty_dict = {}  # This creates an empty dict, not set
empty_set = {1, 2, 3}  # This creates a set

# Fixed Code
empty_dict = {}
empty_set = set()  # For empty set
```

**DB17. List vs Tuple Mutability**

```python
# Buggy Code
coordinates = (1, 2, 3)
coordinates[0] = 10  # TypeError: 'tuple' object does not support item assignment

# Fixed Code
coordinates = [1, 2, 3]  # Use list if you need to modify
coordinates[0] = 10
```

**DB18. Import Order and Scope**

```python
# Buggy Code
from module import function1  # May not be defined yet
from module import function2

# Fixed Code
import module  # Import entire module
result = module.function1()
```

**DB19. String Formatting Errors**

```python
# Buggy Code
name = "Alice"
print("Hello %s, you are % years old" % name)  # Missing second format specifier

# Fixed Code
name = "Alice"
age = 25
print(f"Hello {name}, you are {age} years old")
```

**DB20. Generator vs List Memory**

```python
# Buggy Code
def get_numbers():
    numbers = []
    for i in range(1000000):
        numbers.append(i)
    return numbers

# For large datasets, this uses lots of memory

# Fixed Code
def get_numbers():
    for i in range(1000000):
        yield i  # Use generator for memory efficiency
```

---

## Design & Architecture Questions

### System Design (DS1-DS15)

**DS1. Design a URL Shortener**
Design a system that converts long URLs into short ones.

**Key Components:**

- Database for storing URL mappings
- Hashing algorithm for generating short URLs
- API endpoints for creation and redirection
- Rate limiting and validation

**DS2. Design a Chat Application**
Design a real-time chat system supporting multiple users and rooms.

**Key Components:**

- WebSocket connections for real-time communication
- Message storage and history
- User authentication and authorization
- Room management

**DS3. Design a File Storage System**
Design a system for uploading, storing, and retrieving files.

**Key Components:**

- File storage (local/cloud)
- Metadata database
- File sharing and permissions
- File compression and optimization

**DS4. Design a Search Engine**
Design a simple search engine for indexing and querying documents.

**Key Components:**

- Web crawler for content collection
- Index building and maintenance
- Query processing and ranking
- User interface

**DS5. Design a Social Media Feed**
Design a news feed system similar to Facebook or Twitter.

**Key Components:**

- Post creation and storage
- Following/followers relationships
- Feed generation algorithms
- Notification system

**DS6. Design an E-commerce Platform**
Design an online shopping platform with cart, checkout, and payment.

**Key Components:**

- Product catalog and search
- Shopping cart management
- Order processing
- Payment gateway integration

**DS7. Design a Recommendation System**
Design a system that recommends items based on user behavior.

**Key Components:**

- User behavior tracking
- Collaborative filtering algorithms
- Content-based filtering
- Machine learning models

**DS8. Design a Load Balancer**
Design a system that distributes incoming requests across multiple servers.

**Key Components:**

- Health check mechanisms
- Load balancing algorithms
- Session persistence
- Failover handling

**DS9. Design a Caching System**
Design a distributed caching system like Redis or Memcached.

**Key Components:**

- Key-value storage
- Data eviction policies
- Distributed consistency
- Cache warming strategies

**DS10. Design a Monitoring System**
Design a system for monitoring application performance and health.

**Key Components:**

- Metrics collection
- Alerting rules
- Dashboard visualization
- Incident management

**DS11. Design a Content Management System**
Design a system for managing digital content and workflows.

**Key Components:**

- Content creation and editing
- Version control
- Workflow management
- Publishing pipeline

**DS12. Design a Payment Processing System**
Design a secure payment processing system for online transactions.

**Key Components:**

- Payment method integration
- Transaction processing
- Fraud detection
- Reconciliation and reporting

**DS13. Design a Real-time Analytics System**
Design a system for processing and analyzing streaming data.

**Key Components:**

- Stream processing engine
- Data aggregation
- Real-time dashboards
- Alert generation

**DS14. Design a Multi-tenant SaaS Platform**
Design a Software as a Service platform serving multiple customers.

**Key Components:**

- Data isolation between tenants
- Customization and configuration
- Billing and subscription management
- Scalability considerations

**DS15. Design a Video Streaming Platform**
Design a system for streaming video content to users.

**Key Components:**

- Video encoding and processing
- Content delivery network (CDN)
- User authentication
- Adaptive streaming

### Database Design (DB1-DB10)

**DB1. Design a Library Management System**
Design database schema for managing books, members, and transactions.

**Tables:** Books, Members, Transactions, Authors, Categories
**Relationships:** Many-to-many between books and authors, one-to-many between categories and books

**DB2. Design an E-commerce Database**
Design schema for online store with products, orders, and customers.

**Tables:** Customers, Products, Orders, Order_Items, Categories, Suppliers
**Key Features:** Order history, inventory tracking, customer segmentation

**DB3. Design a Social Network Database**
Design schema for social networking platform with users and connections.

**Tables:** Users, Posts, Comments, Likes, Friendships, Groups
**Key Features:** Friend recommendations, activity feeds, privacy settings

**DB4. Design a Hospital Management System**
Design database for managing patients, doctors, and appointments.

**Tables:** Patients, Doctors, Appointments, Medical_Records, Departments
**Key Features:** Scheduling, medical history, billing

**DB5. Design a University Management System**
Design schema for managing students, courses, and academic records.

**Tables:** Students, Courses, Enrollments, Grades, Professors, Departments
**Key Features:** GPA calculation, course prerequisites, academic calendar

---

## Interview-Style Questions

### Technical Questions (IT1-IT25)

**IT1. Explain the difference between threads and processes in Python.**
Processes have separate memory spaces, while threads share memory. Python's GIL limits concurrent CPU-bound threads, but threads work well for I/O operations.

**IT2. How does Python's garbage collection work?**
Python uses reference counting and a cyclic garbage collector. Objects are automatically deleted when their reference count reaches zero.

**IT3. What are decorators and how do they work?**
Decorators are functions that modify other functions. They use the @ syntax and work by wrapping the original function.

**IT4. Explain list comprehensions vs. map/filter.**
List comprehensions are more readable and generally faster. Map/filter return iterators in Python 3, while list comprehensions create lists.

**IT5. What is the Global Interpreter Lock (GIL)?**
GIL is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously.

**IT6. How would you optimize a slow Python function?**
Use profiling tools to identify bottlenecks, consider algorithmic improvements, use appropriate data structures, leverage caching, and consider using C extensions or NumPy.

**IT7. Explain the difference between shallow and deep copy.**
Shallow copy creates a new object but references the same nested objects. Deep copy creates a completely independent copy of all objects.

**IT8. What are metaclasses in Python?**
Metaclasses are classes of classes. They define how classes behave, similar to how classes define how instances behave.

**IT9. How does multiple inheritance work in Python?**
Python uses Method Resolution Order (MRO) to determine which method to call when multiple inheritance is involved.

**IT10. What is the difference between **str** and **repr**?**
**str** is for user-friendly string representation, **repr** is for unambiguous representation, ideally recreating the object.

**IT11. Explain context managers and the with statement.**
Context managers ensure proper setup and cleanup of resources. The with statement automatically handles entering and exiting the context.

**IT12. What are lambda functions?**
Lambda functions are anonymous functions created with the lambda keyword. They're useful for short, simple functions.

**IT13. How do you handle circular imports in Python?**
Avoid them when possible, use import statements inside functions, or use string-based forward references.

**IT14. What is duck typing?**
"If it walks like a duck and quacks like a duck, it's a duck." Duck typing focuses on the object's capabilities rather than its type.

**IT15. Explain the difference between synchronous and asynchronous programming.**
Synchronous code executes sequentially, blocking on I/O. Asynchronous code allows other operations while waiting for I/O operations to complete.

**IT16. What are generators and when would you use them?**
Generators are memory-efficient iterators created with yield. Use them for large datasets or infinite sequences.

**IT17. How do you handle large files in Python?**
Use file streaming, chunked reading, or generators to avoid loading entire files into memory.

**IT18. What is the purpose of **init**.py in Python packages?**
It identifies a directory as a Python package and can contain initialization code.

**IT19. Explain the difference between args and kwargs.**
\*args collects positional arguments as a tuple, \*\*kwargs collects keyword arguments as a dictionary.

**IT20. How would you design a Python package for distribution?**
Structure with setup.py/pyproject.toml, include **init**.py files, documentation, tests, and follow PEP guidelines.

**IT21. What are abstract base classes (ABC)?**
ABCs define abstract methods that must be implemented by concrete subclasses, enforcing interface contracts.

**IT22. How do you handle configuration management in Python?**
Use config files (JSON, YAML, INI), environment variables, or configuration libraries like configparser.

**IT23. Explain Python's module import system.**
Python searches sys.path for modules, executing them once and caching in sys.modules. Circular imports can cause issues.

**IT24. What are Python's built-in data structures and their time complexities?**
Lists (O(1) append, O(n) insert), dicts (O(1) average access), sets (O(1) average operations), tuples (immutable, hashable).

**IT25. How do you test Python code?**
Use unittest, pytest, doctest for unit tests, mocking for dependencies, and coverage tools for test coverage.

### Behavioral Questions (BT1-BT15)

**BT1. Describe a challenging Python project you've worked on.**
Focus on the technical challenges, your problem-solving approach, and the lessons learned.

**BT2. How do you stay updated with Python development?**
Mention following PEPs, reading blogs, attending conferences, participating in communities.

**BT3. Describe a time when you had to debug a complex issue.**
Explain your systematic approach to debugging, tools used, and how you found the solution.

**BT4. How do you approach learning a new Python library or framework?**
Start with documentation, work through tutorials, build small projects, and contribute to the community.

**BT5. Describe your experience with code reviews.**
Talk about both giving and receiving feedback, following style guides, and improving code quality.

**BT6. How do you handle technical disagreements in a team?**
Focus on data and best practices, seek input from others, and prioritize project goals.

**BT7. What Python development tools do you use regularly?**
IDEs (PyCharm, VS Code), version control (Git), testing frameworks, profilers, linters.

**BT8. How do you ensure your Python code is maintainable?**
Write clear documentation, follow PEP style guide, use type hints, write tests, and refactor regularly.

**BT9. Describe a time when you had to optimize Python code performance.**
Explain profiling, identifying bottlenecks, implementing optimizations, and measuring improvements.

**BT10. How do you handle Python version compatibility?**
Use virtual environments, check library compatibility, use compatibility shims, and test on multiple versions.

**BT11. What is your experience with Python frameworks?**
Discuss web frameworks (Django, Flask), data science libraries (pandas, NumPy), and specific use cases.

**BT12. How do you approach error handling and logging?**
Use specific exception types, log appropriately at different levels, and provide meaningful error messages.

**BT13. Describe your experience with Python deployment.**
Talk about containerization, cloud platforms, CI/CD pipelines, and monitoring.

**BT14. How do you handle data privacy and security in Python applications?**
Discuss encryption, secure coding practices, input validation, and compliance considerations.

**BT15. What are your career goals in Python development?**
Express interest in specific areas, continuous learning, and contribution to the Python community.

---

## Assessment Rubric

### Scoring Guidelines

**Beginner Level (Questions 1-100):**

- **Excellent (90-100%)**: 85+ correct answers
- **Good (80-89%)**: 75-84 correct answers
- **Satisfactory (70-79%)**: 65-74 correct answers
- **Needs Improvement (60-69%)**: 55-64 correct answers
- **Below Standard (<60%)**: <55 correct answers

**Intermediate Level (Questions 101-150):**

- **Excellent (90-100%)**: 45+ correct answers
- **Good (80-89%)**: 40-44 correct answers
- **Satisfactory (70-79%)**: 35-39 correct answers
- **Needs Improvement (60-69%)**: 30-34 correct answers
- **Below Standard (<60%)**: <30 correct answers

**Advanced Level (Questions 151-200):**

- **Excellent (90-100%)**: 45+ correct answers
- **Good (80-89%)**: 40-44 correct answers
- **Satisfactory (70-79%)**: 35-39 correct answers
- **Needs Improvement (60-69%)**: 30-34 correct answers
- **Below Standard (<60%)**: <30 correct answers

### Coding Challenge Evaluation

**Beginner Challenges (CC1-CC20):**

- **Functionality**: Does the code work correctly?
- **Readability**: Is the code clear and well-commented?
- **Efficiency**: Does it solve the problem efficiently?
- **Edge Cases**: Does it handle edge cases properly?

**Intermediate Challenges (CC11-CC30):**

- **Algorithm Design**: Is the approach logical and correct?
- **Code Quality**: Is the implementation clean and maintainable?
- **Complexity**: Does it have good time/space complexity?
- **Testing**: Does it pass various test cases?

**Advanced Challenges (CC21-CC40):**

- **System Design**: Is the architecture sound?
- **Scalability**: Does it handle large inputs efficiently?
- **Best Practices**: Does it follow Python best practices?
- **Documentation**: Is it well-documented and explained?

### Skill Assessment Categories

**Core Python Knowledge (40% of score):**

- Syntax and basic constructs
- Data structures and algorithms
- Object-oriented programming
- Standard library usage

**Problem-Solving Skills (30% of score):**

- Algorithm design and implementation
- Debugging and troubleshooting
- Code optimization and efficiency
- Creative problem-solving approaches

**Software Engineering Practices (20% of score):**

- Code organization and structure
- Testing and quality assurance
- Documentation and comments
- Version control and collaboration

**Advanced Concepts (10% of score):**

- Design patterns and architecture
- Performance optimization
- Security considerations
- Emerging technologies and trends

### Interview Preparation

**Technical Preparation:**

- Review fundamental concepts regularly
- Practice coding challenges daily
- Study system design principles
- Keep up with Python ecosystem updates

**Soft Skills:**

- Practice explaining technical concepts clearly
- Develop problem-solving methodology
- Prepare examples of past projects
- Work on communication skills

**Portfolio Development:**

- Build diverse Python projects
- Contribute to open source
- Write technical blog posts
- Participate in coding communities

This comprehensive assessment guide provides 200+ questions covering all aspects of Python programming, from basic syntax to advanced concepts, ensuring thorough evaluation of Python skills at all levels.

---

## 🎆 Congratulations on Completing Your Assessment!

You've worked through a comprehensive Python evaluation covering:

✅ **Foundation Knowledge** - Basic syntax and concepts  
🔧 **Core Skills** - Programming fundamentals  
🚀 **Advanced Applications** - Complex patterns and libraries  
🏆 **Expert Topics** - System design and optimization

### 📊 Understanding Your Results

**📍 Score Interpretation:**

- **90-100%**: 🏆 **Expert Level** - You have strong Python skills!
- **80-89%**: 🚀 **Advanced** - Solid understanding with room to grow
- **70-79%**: 🔧 **Intermediate** - Good foundation, ready for more challenges
- **60-69%**: 📚 **Foundation+** - Decent grasp of basics, keep practicing!
- **Below 60%**: ✅ **Foundation Building** - Perfect starting point!

### 🔍 Analyzing Your Performance

**🎆 What Your Scores Tell You:**

**High Scores (>85%) in Foundation (1-50):**

- You have a solid grasp of Python basics
- Ready to move to intermediate concepts
- Focus on building projects

**High Scores (>80%) in Core Skills (51-100):**

- Strong programming fundamentals
- Ready for advanced topics
- Consider contributing to open source

**High Scores (>75%) in Advanced (101-150):**

- Excellent Python proficiency
- Ready for professional development
- Consider mentoring others

**High Scores (>70%) in Expert (151-200):**

- Exceptional expertise
- Consider teaching or speaking at conferences

### 📚 Learning Path Recommendations

**🔰 If you scored <70% on Foundation (1-50):**

1. **📝 Review the Cheat Sheet** - Start with Sections 1-3
2. **👥 Find a mentor** - Ask questions and get guidance
3. **📚 Start with basics** - "Automate the Boring Stuff" (free!)
4. **💻 Practice daily** - 30 minutes minimum
5. **🎯 Build simple projects** - Calculator, to-do list, rock-paper-scissors

**🔧 If you scored 70-85% on Foundation:**

1. **⚡ Focus on weak areas** - Review specific question types
2. **📝 Study the Cheat Sheet** - Sections 4-8
3. **🎯 Practice coding challenges** - Start with CC1-CC10
4. **📚 Read Python documentation** - Focus on standard library
5. **🤝 Join Python communities** - Reddit r/learnpython, Discord servers

**🚀 If you scored >85% on Foundation:**

1. **🏁 Move to intermediate** - Questions 51-100
2. **💼 Explore real-world projects** - Web scraping, data analysis
3. **📚 Learn popular libraries** - requests, pandas, flask
4. **🎯 Contribute to open source** - Start with documentation
5. **👥 Help beginners** - Teaching reinforces learning

### 📝 Study Plan for Improvement

**📅 Daily Practice (30 minutes):**

- Week 1-2: Foundation questions + cheat sheet review
- Week 3-4: Core skills questions + coding challenges
- Week 5-6: Advanced questions + mini-projects
- Week 7-8: Expert questions + system design

**📅 Weekly Goals:**

- **Monday**: Review missed questions from previous week
- **Tuesday**: Study new cheat sheet section
- **Wednesday**: Complete 10 new questions
- **Thursday**: Work on coding challenges
- **Friday**: Build/review a small project
- **Weekend**: Teach someone else or write about what you learned

### 🎆 Building Real-World Skills

**🔧 Practice Projects by Skill Level:**

**🔰 Beginner Projects:**

- Number guessing game
- Simple calculator
- To-do list application
- Password generator
- Rock, paper, scissors

**🔧 Intermediate Projects:**

- Weather app using API
- File organizer tool
- Web scraper for news
- Personal budget tracker
- Chat bot

**🚀 Advanced Projects:**

- Personal finance dashboard
- E-commerce web app
- Data visualization dashboard
- Social media automation tool
- Machine learning model

**🏆 Expert Projects:**

- Distributed system
- Custom programming language
- Performance optimization tool
- Security analysis tool
- Open source library

### 📚 Continued Learning Resources

**🌐 Interactive Platforms:**

- **HackerRank** - Algorithm challenges
- **LeetCode** - Interview preparation
- **Codecademy** - Structured courses
- **Real Python** - In-depth tutorials
- **Python.org Tutorial** - Official documentation

**📖 Recommended Books:**

- "Automate the Boring Stuff with Python" (Al Sweigart) - FREE!
- "Python Crash Course" (Eric Matthes)
- "Effective Python" (Brett Slatkin)
- "Clean Code in Python" (Mariano Anaya)
- "Architecture Patterns with Python" (Harry Percival)

**🎥 Video Learning:**

- **Corey Schafer** (YouTube) - Excellent beginner content
- **sentdex** (YouTube) - Practical tutorials
- **Real Python** (YouTube) - Advanced topics
- **PyCon Talks** - Community insights

### 🤝 Community Engagement

**💬 Where to Get Help:**

- **Stack Overflow** - Specific technical questions
- **Reddit r/learnpython** - Beginner-friendly community
- **Python Discord** - Real-time help
- **Local Python meetups** - In-person networking
- **GitHub Discussions** - Open source project help

**🎆 Contributing to Community:**

- Answer questions on Stack Overflow
- Write blog posts about your learning
- Contribute to documentation
- Help organize local meetups
- Mentor other beginners

### 📝 Interview Preparation

**💼 If preparing for technical interviews:**

1. **📝 Master Data Structures**
   - Lists, dictionaries, sets
   - Time complexity (Big O notation)
   - Common algorithms

2. **⚙️ Practice Coding Problems**
   - Focus on easy and medium problems
   - Time yourself (20-30 minutes per problem)
   - Explain your thinking process

3. **📚 System Design Basics**
   - Scalability concepts
   - Database design
   - API design principles
   - Caching strategies

4. **💬 Behavioral Preparation**
   - Prepare project examples
   - Practice explaining technical concepts
   - Review your past work experience
   - Prepare questions to ask interviewer

### 🎆 Your Python Journey

**Remember:**

- **✅ Everyone learns at their own pace** - Don't compare yourself to others
- **🔄 Consistent practice beats intensity** - 30 minutes daily > 5 hours weekly
- **💪 Mistakes are stepping stones** - Every error teaches you something
- **🤝 Community support is invaluable** - Don't hesitate to ask for help
- **🎆 Small wins add up** - Celebrate your progress!

**🏆 Success Stories:**

- Students who started coding at 60+ and became professionals
- Career changers who transitioned to tech in 6 months
- Self-taught developers who landed their dream jobs
- Beginners who now mentor others

### 🚀 Next Steps

1. **📝 Bookmark this assessment** - Return periodically to track progress
2. **💻 Start coding today** - Pick a project and begin
3. **📚 Set learning goals** - What do you want to build?
4. **📅 Create a schedule** - Plan your learning time
5. **🤝 Find accountability** - Join or create a study group
6. **🎆 Track your progress** - Keep a learning journal

### 🎉 Final Encouragement

Whether you scored 20% or 95%, **you've taken an important step** in your Python journey. The fact that you're here, learning and improving, puts you ahead of many people who are too scared to start.

**Your future in programming starts with your next line of code.**

**Keep learning, keep building, keep growing.**

**You've got this!** 🚀💪

---

### 📜 Quick Assessment Reference

**Self-Evaluation Questions:**

✅ **Can I write basic Python programs?**  
✅ **Do I understand data types and variables?**  
✅ **Can I use lists and dictionaries effectively?**  
✅ **Do I understand control structures (if/for/while)?**  
✅ **Can I write and call functions?**  
✅ **Do I handle errors appropriately?**  
✅ **Can I read and write files?**  
✅ **Do I use popular libraries confidently?**  
✅ **Can I debug my own code?**  
✅ **Do I write clean, readable code?**

**If you answered YES to most of these, you're ready for the next level!**

### 🌟 Keep This Guide Handy

Come back to this assessment:

- **Before starting a new Python project**
- **When preparing for interviews**
- **To track your progress over time**
- **When helping other learners**
- **As a confidence booster on tough days**

**Remember: Every expert was once a beginner who refused to give up.**

**Your Python adventure continues!** 🎆
