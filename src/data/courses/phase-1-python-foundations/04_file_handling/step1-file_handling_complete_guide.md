# 📚 Complete Python File Handling & Data Persistence Guide

_Master File Operations, Data Storage, and Modern Data Formats_

**Difficulty:** Beginner → Intermediate  
**Estimated Time:** 10-12 hours

**🎯 Learning Goal:** Master file handling, data persistence, and modern data formats for real school projects and applications

---

## 📋 Table of Contents

1. [File Handling Basics & File Systems](#1-file-handling-basics--file-systems)
2. [Text File Operations](#2-text-file-operations)
3. [Modern Data Formats (JSON, CSV)](#3-modern-data-formats-json-csv)
4. [API Integration & Data Retrieval](#4-api-integration--data-retrieval)
5. [File System Operations & Organization](#5-file-system-operations--organization)
6. [Error Handling & Data Validation](#6-error-handling--data-validation)
7. [Real-World Data Projects](#7-real-world-data-projects)

---

## 1. File Handling Basics & File Systems

### 🎯 Hook & Analogy

**Think of file handling like organizing your school documents and homework folders.** 📚

- **Files** = Individual papers/homework assignments (data stored in files)
- **Folders/Directories** = Your binder dividers and desk drawers (organized storage)
- **Reading files** = Taking out a worksheet to see what's on it
- **Writing files** = Turning in homework or creating new assignments
- **File paths** = The exact location in your backpack where you keep things
- **Digital filing cabinet** = Your computer's folder system
- **Homework folder** = A specific folder where you save your assignments
- **Organizing school projects** = Sorting files by subject, date, or project name

### 💡 Simple Definition

**File handling lets your Python programs save information permanently (like saving homework), read information back (like opening a textbook), and organize different types of data (like keeping your math homework separate from English essays).**

### 💻 Code + Output Pairing

**File System Basics:**

```python
import os
import pathlib
from datetime import datetime

print("=== File System Exploration ===")

# Current working directory
current_dir = os.getcwd()
print(f"📍 Current directory: {current_dir}")

# List directory contents
print(f"📂 Contents of current directory:")
try:
    items = os.listdir(current_dir)
    for item in sorted(items):
        item_path = os.path.join(current_dir, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            print(f"  📄 {item} ({size} bytes)")
        elif os.path.isdir(item_path):
            print(f"  📁 {item}/ (directory)")
except PermissionError:
    print("  ⚠️  Permission denied")

# Using pathlib (modern approach)
print(f"\n=== Using Pathlib (Modern Approach) ===")
current_path = pathlib.Path(".")

print(f"🔍 Current path: {current_path.absolute()}")

# Pattern matching
print(f"🔍 Python files in current directory:")
py_files = list(current_path.glob("*.py"))
for py_file in py_files:
    print(f"  🐍 {py_file.name}")

# Create a test file structure
test_dir = pathlib.Path("test_data")
test_dir.mkdir(exist_ok=True)

# Create sample files
(test_dir / "sample.txt").write_text("Hello, World!\nThis is a test file.")
(test_dir / "data.json").write_text('{"name": "Alice", "age": 16, "hobbies": ["reading", "coding"]}')

print(f"✅ Created test directory and files in {test_dir}")
```

**Output:**

```
=== File System Exploration ===
📍 Current directory: /workspace
📂 Contents of current directory:
  📁 docs/ (directory)
  📁 test_data/ (directory)
  📁 code/ (directory)
  📄 README.md (245 bytes)

=== Using Pathlib (Modern Approach) ===
🔍 Current path: /workspace
🔍 Python files in current directory:
  🐍 main.py
  🐍 helper.py

✅ Created test directory and files in /workspace/test_data
```

### 🔍 Visual Breakdown

```
File System Structure:

📂 Root Directory (/)
├── 📁 home/
│   ├── 📁 user/
│   │   ├── 📁 documents/
│   │   │   ├── 📄 report.txt
│   │   │   ├── 📄 data.csv
│   │   │   └── 📁 images/
│   │   │       ├── 📄 photo1.jpg
│   │   │       └── 📄 photo2.png
│   │   └── 📁 python_projects/
│   │       ├── 📄 main.py
│   │       └── 📄 data_handler.py
└── 📁 tmp/
    └── 📁 cache/

File Paths:
• Absolute: /home/user/documents/report.txt
• Relative: documents/report.txt
• Current: ./report.txt
• Parent: ../documents/report.txt
```

### 🌍 Real-Life Use Case

**School Project File Handling:**

- **Student Portals:** Save student grades, manage class schedules, store homework submissions
- **School Clubs:** Keep track of member lists, event schedules, attendance records
- **Homework Managers:** Create digital homework planners, track assignment deadlines, store study notes
- **Science Labs:** Save lab results, organize experiment data, track research findings
- **Art Projects:** Store digital artwork, organize portfolio files, manage project resources
- **Yearbook Committee:** Manage photo collections, organize article submissions, track story deadlines

### 💻 Practice Tasks

**Beginner:**

```python
def explore_file_system():
    """Explore and understand file system operations"""

    print("=== File System Exploration Practice ===")

    # Create a simple project structure
    project_dir = pathlib.Path("my_project")
    project_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)
    (project_dir / "config").mkdir(exist_ok=True)

    print(f"✅ Created project structure in {project_dir}")

    # Create sample configuration file
    config_content = """# Project Configuration
project_name = "My Python Project"
version = "1.0.0"
author = "Student"

[database]
host = "localhost"
port = 5432
name = "myapp_db"

[features]
debug_mode = true
auto_backup = true
"""

    config_file = project_dir / "config" / "settings.ini"
    config_file.write_text(config_content)
    print(f"📝 Created config file: {config_file}")

    # Create sample data file
    data_content = """name,age,grade,subject
Alice Johnson,16,11,Math
Bob Smith,17,12,Science
Carol Davis,15,10,English
David Wilson,16,11,History
"""

    data_file = project_dir / "data" / "students.csv"
    data_file.write_text(data_content)
    print(f"📊 Created data file: {data_file}")

    # Explore the structure we created
    print(f"\n🔍 Project structure:")
    for item in project_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(project_dir)
            size = item.stat().st_size
            print(f"  📄 {rel_path} ({size} bytes)")
        elif item.is_dir() and item != project_dir:
            rel_path = item.relative_to(project_dir)
            print(f"  📁 {rel_path}/")

    return project_dir

# Run the exploration
project_path = explore_file_system()
```

**Intermediate:**

```python
class FileSystemManager:
    """Advanced file system management"""

    def __init__(self, base_path="."):
        self.base_path = pathlib.Path(base_path).resolve()
        self.base_path.mkdir(exist_ok=True)

    def create_project_structure(self, project_name):
        """Create a standard project structure"""
        project_path = self.base_path / project_name
        project_path.mkdir(exist_ok=True)

        # Standard directories
        directories = [
            "src",
            "tests",
            "data",
            "docs",
            "config",
            "output",
            "logs"
        ]

        for directory in directories:
            (project_path / directory).mkdir(exist_ok=True)

        # Create README
        readme_content = f"""# {project_name.title().replace('_', ' ')}

## Project Structure
- `src/` - Source code
- `tests/` - Test files
- `data/` - Data files
- `docs/` - Documentation
- `config/` - Configuration files
- `output/` - Generated output
- `logs/` - Application logs

## Getting Started
1. Install dependencies
2. Configure settings in `config/`
3. Run main application from `src/`
"""

        (project_path / "README.md").write_text(readme_content)

        # Create requirements.txt
        (project_path / "requirements.txt").write_text(
            "# Project dependencies\n"
            "requests>=2.25.0\n"
            "pandas>=1.3.0\n"
            "python-dotenv>=0.19.0\n"
        )

        print(f"✅ Created project structure: {project_path}")
        return project_path

    def backup_directory(self, source_dir, backup_name=None):
        """Create a backup of a directory"""
        source_path = pathlib.Path(source_dir)

        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        if backup_name is None:
            backup_name = f"{source_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_path = self.base_path / backup_name

        # Copy all files and directories
        import shutil
        shutil.copytree(source_path, backup_path, dirs_exist_ok=True)

        print(f"✅ Created backup: {backup_path}")
        return backup_path

    def find_large_files(self, directory, min_size_mb=1):
        """Find files larger than specified size"""
        directory_path = pathlib.Path(directory)
        min_size = min_size_mb * 1024 * 1024  # Convert to bytes

        large_files = []

        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                if size >= min_size:
                    large_files.append({
                        'path': file_path,
                        'size': size,
                        'size_mb': size / (1024 * 1024)
                    })

        # Sort by size (largest first)
        large_files.sort(key=lambda x: x['size'], reverse=True)

        return large_files

    def clean_temp_files(self, directory, pattern="*.tmp"):
        """Remove temporary files matching pattern"""
        directory_path = pathlib.Path(directory)
        temp_files = list(directory_path.rglob(pattern))

        removed_count = 0
        total_size = 0

        for temp_file in temp_files:
            if temp_file.is_file():
                size = temp_file.stat().st_size
                temp_file.unlink()
                removed_count += 1
                total_size += size
                print(f"🗑️  Removed: {temp_file}")

        print(f"✅ Cleaned {removed_count} temp files ({total_size / 1024:.1f} KB)")
        return removed_count, total_size

# Test the file system manager
print("=== Advanced File System Management ===")

manager = FileSystemManager()

# Create project structure
project = manager.create_project_structure("student_portal")

# Find large files (if any)
try:
    large_files = manager.find_large_files(".", min_size_mb=1)
    if large_files:
        print(f"\n📊 Found {len(large_files)} large files:")
        for file_info in large_files[:5]:  # Show first 5
            print(f"  📄 {file_info['path']} - {file_info['size_mb']:.1f} MB")
    else:
        print(f"\n📊 No files larger than 1 MB found")
except Exception as e:
    print(f"❌ Error finding large files: {e}")

# Clean temporary files (simulate)
print(f"\n🧹 Cleaning temporary files...")
removed, size = manager.clean_temp_files(".", "*.tmp")
```

### ⚠️ Common Mistakes

❌ **Using hard-coded file paths:**

```python
# Wrong ❌ (works only on your computer)
with open("C:/Users/MyName/Documents/data.txt", "r") as f:
    content = f.read()

# Correct ✅ (works anywhere)
with open("data.txt", "r") as f:  # Or use pathlib
    content = f.read()
```

❌ **Not handling encoding properly:**

```python
# Wrong ❌ (may fail with special characters)
with open("data.txt", "r") as f:
    content = f.read()

# Correct ✅ (specify encoding)
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

❌ **Not checking if file exists:**

```python
# Wrong ❌ (will crash if file doesn't exist)
with open("data.txt", "r") as f:
    content = f.read()

# Correct ✅ (check first)
file_path = pathlib.Path("data.txt")
if file_path.exists():
    content = file_path.read_text()
else:
    print("File not found")
```

### 💡 Tips & Tricks

💡 **Tip:** Use `pathlib.Path` for modern, cross-platform path handling
💡 **Tip:** Always specify encoding when working with text files
💡 **Tip:** Use context managers (`with` statements) for automatic file closing
💡 **Tip:** Check if files exist before trying to read them

### 📊 Summary Block - What You Learned

- ✅ **File systems** organize data in directories and files
- ✅ **Absolute paths** start from root, relative paths from current directory
- ✅ **pathlib** provides modern, object-oriented file handling
- ✅ **File operations** include reading, writing, copying, and deleting
- ✅ **Project structure** helps organize code and data systematically
- ✅ **Always validate** file existence before operations
- ✅ **Use context managers** for safe file handling

---

## 2. Text File Operations

### 🎯 Hook & Analogy

**Text file operations are like working with your school notebook or assignment sheets.** 📝

- **Reading** = Looking at what you wrote in your math homework
- **Writing** = Creating a new essay or filling out a worksheet
- **Appending** = Adding more answers to a partially completed assignment
- **Overwriting** = Starting a fresh sheet when you make too many mistakes
- **Line by line reading** = Going through each problem one at a time
- **Creating new files** = Starting a new subject notebook

### 💡 Simple Definition

**Text file operations let you work with school documents, homework assignments, and notes - reading what's already there, writing new content, adding to existing work, and organizing everything safely.**

### 💻 Code + Output Pairing

**Basic Text File Operations:**

```python
print("=== Text File Operations ===")

# Create a sample text file
sample_text = """Welcome to Python File Handling!
This is a multi-line text file.
We can store any text content here.
Each line will be read separately.

Key Points:
• Python makes file handling easy
• Always use 'with' statements
• Handle errors gracefully
• Choose appropriate modes (r, w, a, r+)
"""

# Writing text files
print("1. Writing text files:")
with open("sample_file.txt", "w", encoding="utf-8") as file:
    file.write(sample_text)
print("✅ Created sample_file.txt")

# Reading text files (entire content)
print("\n2. Reading entire file:")
with open("sample_file.txt", "r", encoding="utf-8") as file:
    content = file.read()
print(f"📖 File content ({len(content)} characters):")
print(content[:200] + "..." if len(content) > 200 else content)

# Reading line by line
print("\n3. Reading line by line:")
with open("sample_file.txt", "r", encoding="utf-8") as file:
    for i, line in enumerate(file, 1):
        print(f"Line {i:2d}: {line.rstrip()}")

# Append to existing file
print("\n4. Appending to file:")
additional_text = """
Added content:
- This line was appended
- File operations are powerful
- Great for data persistence
"""

with open("sample_file.txt", "a", encoding="utf-8") as file:
    file.write(additional_text)
print("✅ Appended content to file")

# Check file size
import os
file_size = os.path.getsize("sample_file.txt")
print(f"📊 File size: {file_size} bytes")
```

**Output:**

```
=== Text File Operations ===
1. Writing text files:
✅ Created sample_file.txt

2. Reading entire file:
📖 File content (245 characters):
Welcome to Python File Handling!
This is a multi-line text file.
We can store any text content here.
Each line will be read separately.

Key Points:
• Python makes file handling easy
• Always use 'with' statements...

3. Reading line by line:
Line  1: Welcome to Python File Handling!
Line  2: This is a multi-line text file.
Line  3: We can store any text content here.
Line  4: Each line will be read separately.
Line  5:
Line  6: Key Points:
Line  7: • Python makes file handling easy
Line  8: • Always use 'with' statements
Line  9: • Handle errors gracefully
Line 10: • Choose appropriate modes (r, w, a, r+)

4. Appending to file:
✅ Appended content to file
📊 File size: 331 bytes
```

### 🔍 Visual Breakdown

```
File Operation Flow:

┌─────────────────────────────────────┐
│ 1. WRITE MODE ("w")                 │
│    ┌─────────────┐                  │
│    │ File opens  │  ← Creates/      │
│    │ Creates new │     overwrites   │
│    │ file        │                  │
│    └─────────────┘                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 2. READ MODE ("r")                  │
│    ┌─────────────┐                  │
│    │ File opens  │  ← Read only     │
│    │ Must exist  │     access       │
│    │ for reading │                  │
│    └─────────────┘                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 3. APPEND MODE ("a")                │
│    ┌─────────────┐                  │
│    │ File opens  │  ← Add to end    │
│    │ Creates if  │     only         │
│    │ not exists  │                  │
│    └─────────────┘                  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 4. READ+WRITE ("r+")                │
│    ┌─────────────┐                  │
│    │ File opens  │  ← Modify        │
│    │ Must exist  │     existing     │
│    │ for both    │     content      │
│    └─────────────┘                  │
└─────────────────────────────────────┘

Position Tracking:
[0] H [1] e [2] l [3] l [4] o [5] , [6]  [7] W [8] o [9] r [10] l [11] d [12] !
  ↑         ↑                                   ↑
 0         5                                   10
```

### 🌍 Real-Life Use Case

**School Text File Applications:**

- **Study Notes:** Save class notes, research findings, vocabulary lists
- **Assignment Tracking:** Create homework to-do lists, project deadlines, study schedules
- **Writing Projects:** Draft essays, stories, research papers with automatic saving
- **Class Schedules:** Store daily timetables, extracurricular activities
- **Reading Lists:** Keep track of required books, book reports, reading goals
- **Science Lab Reports:** Record experimental data, observations, calculations

### 💻 Practice Tasks

**Beginner:**

```python
class SimpleTextEditor:
    """A simple text editor with file operations"""

    def __init__(self):
        self.current_file = None
        self.content = ""

    def create_new_file(self, filename):
        """Create a new file"""
        self.current_file = filename
        self.content = ""
        print(f"📄 Created new file: {filename}")

    def open_file(self, filename):
        """Open and read a file"""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                self.content = file.read()
            self.current_file = filename
            print(f"📖 Opened file: {filename} ({len(self.content)} characters)")
            return True
        except FileNotFoundError:
            print(f"❌ File not found: {filename}")
            return False
        except Exception as e:
            print(f"❌ Error opening file: {e}")
            return False

    def save_file(self):
        """Save current content to file"""
        if not self.current_file:
            print("❌ No file opened or created")
            return False

        try:
            with open(self.current_file, "w", encoding="utf-8") as file:
                file.write(self.content)
            print(f"💾 Saved file: {self.current_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False

    def add_text(self, text):
        """Add text to current content"""
        if not self.content or self.content.endswith('\n'):
            self.content += text
        else:
            self.content += '\n' + text
        print(f"✏️ Added text ({len(text)} characters)")

    def show_content(self):
        """Display current content"""
        if self.content:
            print(f"📝 Content of {self.current_file or 'new file'}:")
            print("-" * 40)
            print(self.content)
            print("-" * 40)
        else:
            print("📭 No content to display")

    def word_count(self):
        """Count words in current content"""
        if not self.content:
            print("📭 No content to analyze")
            return

        words = self.content.split()
        lines = self.content.split('\n')
        characters = len(self.content)

        print(f"📊 Word count for {self.current_file or 'current content'}:")
        print(f"  📝 Words: {len(words)}")
        print(f"  📄 Lines: {len(lines)}")
        print(f"  🔤 Characters: {characters}")

    def search_text(self, search_term):
        """Search for text in current content"""
        if not self.content:
            print("📭 No content to search")
            return

        lines = self.content.split('\n')
        found_lines = []

        for i, line in enumerate(lines, 1):
            if search_term.lower() in line.lower():
                found_lines.append((i, line.strip()))

        if found_lines:
            print(f"🔍 Found '{search_term}' in {len(found_lines)} line(s):")
            for line_num, line_content in found_lines:
                print(f"  Line {line_num}: {line_content}")
        else:
            print(f"🔍 '{search_term}' not found")

# Test the simple text editor
print("=== Simple Text Editor ===")

editor = SimpleTextEditor()

# Create and edit a file
editor.create_new_file("my_notes.txt")
editor.add_text("Python is amazing!")
editor.add_text("File handling is useful for many applications.")
editor.add_text("We can build great programs with these tools.")

editor.show_content()
editor.word_count()
editor.search_text("Python")

# Save the file
editor.save_file()

# Open and modify the file
editor.open_file("my_notes.txt")
editor.add_text("This line was added later!")
editor.show_content()
editor.save_file()
```

### ⚠️ Common Mistakes

❌ **Not using context managers:**

```python
# Wrong ❌ (file might not close properly)
file = open("data.txt", "r")
content = file.read()
# file.close() might never be called!

# Correct ✅ (automatic cleanup)
with open("data.txt", "r") as file:
    content = file.read()  # File automatically closed
```

❌ **Wrong file modes:**

```python
# Wrong ❌ (trying to write in read mode)
with open("data.txt", "r") as file:
    file.write("new content")  # Error!

# Correct ✅ (use correct mode)
with open("data.txt", "w") as file:
    file.write("new content")
```

### 💡 Tips & Tricks

💡 **Tip:** Use `readlines()` to get all lines as a list
💡 **Tip:** Use `writelines()` to write a list of strings
💡 **Tip:** Use `seek()` to move file position for random access
💡 **Tip:** Use `tell()` to get current file position

### 📊 Summary Block - What You Learned

- ✅ **Text file modes:** "r" (read), "w" (write), "a" (append), "r+" (read+write)
- ✅ **Context managers** automatically handle file closing
- ✅ **Line-by-line reading** is memory-efficient for large files
- ✅ **File positioning** can be controlled with `seek()` and `tell()`
- ✅ **Encoding specification** prevents character issues
- ✅ **Error handling** is essential for file operations
- ✅ **Large file processing** should use chunked reading

---

## 3. Modern Data Formats (JSON, CSV) - Digital Student Records

### 🎯 Hook & Analogy

**Modern data formats are like organized digital grade books and student records.** 📊

- **JSON** = A digital report card where each student's info is clearly labeled and structured
- **CSV** = A spreadsheet where you can see all student names and grades in organized columns
- **Loading data** = Taking out your grade book to see everyone's scores
- **Saving data** = Writing down new grades and test results
- **Data structure** = How you organize information (like math grades in one section, science in another)

### 💡 Simple Definition

**Modern data formats let you store complex information (like student records, grades, attendance) in an organized way that both humans and computers can easily read and understand.**

### 💻 Code + Output Pairing

**Working with Student Data in JSON:**

```python
import json
from datetime import datetime

print("=== Digital Student Records (JSON) ===")

# Student data structure
student_data = {
    "class_name": "Advanced Computer Science",
    "semester": "Fall 2024",
    "students": [
        {
            "name": "Alex Johnson",
            "student_id": "S001",
            "grades": {"math": 95, "science": 88, "english": 92},
            "attendance": 96.5,
            "active": True
        },
        {
            "name": "Emma Smith",
            "student_id": "S002",
            "grades": {"math": 87, "science": 94, "english": 89},
            "attendance": 98.2,
            "active": True
        },
        {
            "name": "Michael Brown",
            "student_id": "S003",
            "grades": {"math": 78, "science": 85, "english": 81},
            "attendance": 92.1,
            "active": False
        }
    ],
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# Save student data to JSON file
print("1. Saving student records to digital grade book...")
with open("student_records.json", "w", encoding="utf-8") as file:
    json.dump(student_data, file, indent=4, ensure_ascii=False)
print("✅ Saved student_records.json")

# Load and display student data
print("\n2. Loading student records...")
with open("student_records.json", "r", encoding="utf-8") as file:
    loaded_data = json.load(file)

print(f"📚 Class: {loaded_data['class_name']}")
print(f"📅 Semester: {loaded_data['semester']}")
print(f"👥 Active Students: {sum(1 for s in loaded_data['students'] if s['active'])}")

# Display student summary
print("\n3. Student Grade Summary:")
for student in loaded_data['students']:
    if student['active']:
        avg_grade = sum(student['grades'].values()) / len(student['grades'])
        print(f"  📝 {student['name']} - Average: {avg_grade:.1f}%")
```

**Working with Attendance Data in CSV:**

```python
import csv

print("\n=== Attendance Tracking (CSV) ===")

# Attendance data
attendance_data = [
    ["Date", "Student_Name", "Present", "Late", "Absent"],
    ["2024-01-15", "Alex Johnson", "Yes", "No", "No"],
    ["2024-01-15", "Emma Smith", "Yes", "No", "No"],
    ["2024-01-15", "Michael Brown", "No", "No", "Yes"],
    ["2024-01-16", "Alex Johnson", "Yes", "Yes", "No"],
    ["2024-01-16", "Emma Smith", "Yes", "No", "No"],
    ["2024-01-16", "Michael Brown", "Yes", "No", "No"],
]

# Save attendance to CSV
print("4. Creating attendance spreadsheet...")
with open("attendance.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(attendance_data)
print("✅ Created attendance.csv")

# Read and analyze attendance
print("\n5. Analyzing attendance data...")
with open("attendance.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    attendance_records = list(reader)

# Calculate attendance rates
student_attendance = {}
for record in attendance_records:
    name = record["Student_Name"]
    if name not in student_attendance:
        student_attendance[name] = {"present": 0, "total": 0}

    student_attendance[name]["total"] += 1
    if record["Present"] == "Yes":
        student_attendance[name]["present"] += 1

print("\n📊 Attendance Summary:")
for student, stats in student_attendance.items():
    rate = (stats["present"] / stats["total"]) * 100
    print(f"  👤 {student}: {rate:.1f}% attendance rate")
```

**Output:**

```
=== Digital Student Records (JSON) ===
1. Saving student records to digital grade book...
✅ Saved student_records.json

2. Loading student records...
📚 Class: Advanced Computer Science
📅 Semester: Fall 2024
👥 Active Students: 2

3. Student Grade Summary:
  📝 Alex Johnson - Average: 91.7%
  📝 Emma Smith - Average: 90.0%

=== Attendance Tracking (CSV) ===
4. Creating attendance spreadsheet...
✅ Created attendance.csv

5. Analyzing attendance data...

📊 Attendance Summary:
  👤 Alex Johnson: 75.0% attendance rate
  👤 Emma Smith: 100.0% attendance rate
  👤 Michael Brown: 50.0% attendance rate
```

### 🔍 Visual Breakdown

```
JSON Structure (Nested Information):
┌─────────────────────────────────────┐
│ student_records.json               │
├─────────────────────────────────────┤
│ {                                   │
│   "class_name": "Math 101",        │  ← Root level
│   "students": [                     │
│     {                               │
│       "name": "Alex",              │  ← Student 1
│       "grades": {                   │
│         "test1": 95,               │
│         "test2": 87                │  ← Nested grades
│       }                            │
│     },                              │
│     {                               │
│       "name": "Emma",              │  ← Student 2
│       "grades": {                   │
│         "test1": 92,               │
│         "test2": 94                │
│       }                            │
│     }                              │
│   ]                                │
│ }                                   │
└─────────────────────────────────────┘

CSV Structure (Flat Table):
┌─────────────────────────────────────┐
│ Name       │ Test1 │ Test2 │ Grade  │
├────────────┼───────┼───────┼────────┤
│ Alex       │ 95    │ 87    │ A      │
│ Emma       │ 92    │ 94    │ A      │
│ Michael    │ 78    │ 85    │ B      │
│ Sarah      │ 88    │ 91    │ A      │
└────────────────────────────────────┘
```

### 💻 Practice Tasks

**Beginner: Student Grade Calculator**

```python
class StudentGradeManager:
    """Manage student grades using JSON and CSV"""

    def __init__(self):
        self.students = {}

    def add_student(self, name, student_id):
        """Add a new student to the system"""
        self.students[student_id] = {
            "name": name,
            "grades": [],
            "average": 0.0
        }
        print(f"✅ Added student: {name} (ID: {student_id})")

    def add_grade(self, student_id, grade, subject="General"):
        """Add a grade for a student"""
        if student_id in self.students:
            self.students[student_id]["grades"].append({
                "grade": grade,
                "subject": subject,
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            # Recalculate average
            grades = [g["grade"] for g in self.students[student_id]["grades"]]
            self.students[student_id]["average"] = sum(grades) / len(grades)
            print(f"✅ Added {subject} grade: {grade} for {self.students[student_id]['name']}")
        else:
            print(f"❌ Student not found: {student_id}")

    def save_to_json(self, filename="student_grades.json"):
        """Save all student data to JSON"""
        data = {
            "class_name": "Python Programming Class",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "students": self.students
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        print(f"💾 Saved grades to {filename}")

    def save_to_csv(self, filename="grade_summary.csv"):
        """Save grade summary to CSV"""
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Student_ID", "Name", "Current_Average", "Total_Grades"])

            for student_id, info in self.students.items():
                writer.writerow([
                    student_id,
                    info["name"],
                    f"{info['average']:.1f}",
                    len(info["grades"])
                ])
        print(f"💾 Saved summary to {filename}")

    def display_report(self):
        """Display a complete grade report"""
        print("\n" + "="*50)
        print("📊 STUDENT GRADE REPORT")
        print("="*50)

        for student_id, info in self.students.items():
            print(f"\n👤 {info['name']} (ID: {student_id})")
            print(f"📈 Current Average: {info['average']:.1f}%")
            print(f"📋 Total Grades: {len(info['grades'])}")

            if info['grades']:
                print("📝 Recent Grades:")
                for grade_info in info['grades'][-3:]:  # Show last 3 grades
                    print(f"    {grade_info['subject']}: {grade_info['grade']} ({grade_info['date']})")

            # Letter grade
            if info['average'] >= 90:
                letter = "A"
            elif info['average'] >= 80:
                letter = "B"
            elif info['average'] >= 70:
                letter = "C"
            elif info['average'] >= 60:
                letter = "D"
            else:
                letter = "F"
            print(f"🏆 Letter Grade: {letter}")

# Test the grade manager
print("=== Student Grade Management System ===")

manager = StudentGradeManager()

# Add students
manager.add_student("Alice Johnson", "S001")
manager.add_student("Bob Smith", "S002")
manager.add_student("Carol Davis", "S003")

# Add grades
manager.add_grade("S001", 95, "Math")
manager.add_grade("S001", 88, "Science")
manager.add_grade("S001", 92, "English")

manager.add_grade("S002", 87, "Math")
manager.add_grade("S002", 94, "Science")
manager.add_grade("S002", 89, "English")

manager.add_grade("S003", 78, "Math")
manager.add_grade("S003", 85, "Science")
manager.add_grade("S003", 91, "English")

# Save to files
manager.save_to_json()
manager.save_to_csv()

# Display report
manager.display_report()
```

### ⚠️ Common Mistakes

❌ **Not handling JSON encoding errors:**

```python
# Wrong ❌ (special characters might break)
with open("data.json", "w") as f:
    json.dump(data, f)  # May fail with unicode characters

# Correct ✅ (specify encoding and ensure_ascii)
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
```

❌ **Mixing up JSON and CSV usage:**

```python
# Wrong ❌ (CSV is for tabular data)
# Using CSV for complex nested data is difficult

# Correct ✅ (use JSON for complex structures)
complex_data = {"students": [{"name": "Alex", "grades": [95, 87, 92]}]}
with open("data.json", "w") as f:
    json.dump(complex_data, f)

# Use CSV for simple tabular data
simple_data = [["Name", "Grade"], ["Alex", "A"], ["Bob", "B"]]
with open("grades.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(simple_data)
```

### 📊 Summary Block - What You Learned

- ✅ **JSON** is perfect for complex, nested data like student records
- ✅ **CSV** works great for simple tabular data like grade lists
- ✅ **Data structures** help organize information logically
- ✅ **File encoding** (utf-8) prevents character problems
- ✅ **JSON** preserves data types (numbers stay numbers)
- ✅ **CSV** is easier to open in spreadsheet programs
- ✅ **Both formats** let you save and load data between program runs

---

## 4. API Integration & Data Retrieval - Getting School Information

### 🎯 Hook & Analogy

**API integration is like getting information from your school's online portal or library database.** 🌐

- **API** = A digital librarian that can fetch specific information when you ask
- **API Request** = Asking the librarian for a specific book or information
- **JSON Response** = Getting back the information in a nicely organized format
- **Data Processing** = Taking that information and organizing it in your notes
- **Caching** = Keeping a copy of frequently needed information so you don't have to ask again

### 💡 Simple Definition

**APIs let your programs get information from the internet (like weather, news, school schedules) and save it locally for use in your projects.**

### 💻 Code + Output Pairing

**Getting School Schedule and Weather Data:**

```python
import requests
import json
from datetime import datetime

print("=== School Information System ===")

# Mock school schedule data (simulating API response)
school_schedule_data = {
    "school_name": "Greenfield High School",
    "current_date": datetime.now().strftime("%Y-%m-%d"),
    "daily_schedule": [
        {"period": 1, "time": "8:00-8:45", "subject": "Math", "room": "101", "teacher": "Ms. Johnson"},
        {"period": 2, "time": "8:50-9:35", "subject": "English", "room": "205", "teacher": "Mr. Smith"},
        {"period": 3, "time": "9:40-10:25", "subject": "Science", "room": "Lab A", "teacher": "Dr. Brown"},
        {"period": 4, "time": "10:30-11:15", "subject": "History", "room": "302", "teacher": "Ms. Davis"},
        {"period": 5, "time": "11:20-12:05", "subject": "Lunch", "room": "Cafeteria", "teacher": ""},
        {"period": 6, "time": "12:10-12:55", "subject": "PE", "room": "Gym", "teacher": "Coach Wilson"},
        {"period": 7, "time": "1:00-1:45", "subject": "Art", "room": "Art Studio", "teacher": "Ms. Taylor"},
        {"period": 8, "time": "1:50-2:35", "subject": "Study Hall", "room": "Library", "teacher": "Mrs. Anderson"}
    ]
}

def fetch_school_schedule():
    """Simulate fetching school schedule from API"""
    print("1. Fetching today's schedule...")

    # Simulate API request delay
    import time
    time.sleep(1)

    return school_schedule_data

def process_schedule_data(schedule_data):
    """Process and display schedule information"""
    print(f"\n📚 {schedule_data['school_name']}")
    print(f"📅 Date: {schedule_data['current_date']}")
    print("\n🕐 Today's Schedule:")

    for period in schedule_data['daily_schedule']:
        if period['subject'] != "Lunch":
            print(f"  Period {period['period']}: {period['time']} - {period['subject']} (Room {period['room']})")
        else:
            print(f"  Period {period['period']}: {period['time']} - {period['subject']}")

def save_schedule_locally(schedule_data, filename="school_schedule.json"):
    """Save schedule data locally for offline use"""
    print(f"\n2. Saving schedule locally to {filename}...")

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(schedule_data, file, indent=4, ensure_ascii=False)

    print("✅ Schedule saved for offline access")

def load_cached_schedule(filename="school_schedule.json"):
    """Load previously saved schedule data"""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("❌ No cached schedule found")
        return None

# Test the school information system
schedule = fetch_school_schedule()
process_schedule_data(schedule)
save_schedule_locally(schedule)

# Try loading cached data
print("\n3. Testing offline access...")
cached_schedule = load_cached_schedule()
if cached_schedule:
    print("✅ Loaded schedule from cache")
    process_schedule_data(cached_schedule)
```

**Getting Weather Data for School Events:**

```python
def get_weather_for_school_event(event_date, location="your_city"):
    """Get weather information for planning school events"""

    # Mock weather data (in real life, you'd use a weather API)
    weather_data = {
        "location": location,
        "date": event_date,
        "temperature_high": 22,  # Celsius
        "temperature_low": 15,   # Celsius
        "conditions": "Partly Cloudy",
        "rain_probability": 20,  # Percentage
        "wind_speed": 10,        # km/h
        "recommendations": [
            "Good weather for outdoor activities",
            "Bring light jacket for evening",
            "Low chance of rain"
        ]
    }

    return weather_data

def plan_school_event_with_weather():
    """Plan a school event considering weather conditions"""
    print("\n=== School Event Planning with Weather ===")

    event_date = "2024-03-15"  # Spring Fair date
    weather = get_weather_for_school_event(event_date, "Springfield")

    print(f"🌤️ Weather forecast for {weather['date']}:")
    print(f"  🌡️ Temperature: {weather['temperature_low']}°C - {weather['temperature_high']}°C")
    print(f"  ☁️ Conditions: {weather['conditions']}")
    print(f"  🌧️ Rain chance: {weather['rain_probability']}%")
    print(f"  💨 Wind: {weather['wind_speed']} km/h")

    print(f"\n📋 Event Planning Recommendations:")
    for rec in weather['recommendations']:
        print(f"  ✅ {rec}")

    # Save weather info with event data
    event_plan = {
        "event_name": "Spring Fair",
        "date": event_date,
        "weather": weather,
        "activities": [
            {"name": "Outdoor Games", "indoor_backup": "Gymnasium"},
            {"name": "Food Booths", "indoor_backup": "Cafeteria"},
            {"name": "Science Display", "indoor_backup": "Auditorium"}
        ]
    }

    with open("event_plan.json", "w", encoding="utf-8") as file:
        json.dump(event_plan, file, indent=4, ensure_ascii=False)

    print("💾 Saved event plan with weather considerations")

plan_school_event_with_weather()
```

**Output:**

```
=== School Information System ===
1. Fetching today's schedule...

📚 Greenfield High School
📅 Date: 2024-11-01

🕐 Today's Schedule:
  Period 1: 8:00-8:45 - Math (Room 101)
  Period 2: 8:50-9:35 - English (Room 205)
  Period 3: 9:40-10:25 - Science (Room Lab A)
  Period 4: 10:30-11:15 - History (Room 302)
  Period 5: 11:20-12:05 - Lunch (Room Cafeteria)
  Period 6: 12:10-12:55 - PE (Room Gym)
  Period 7: 1:00-1:45 - Art (Room Art Studio)
  Period 8: 1:50-2:35 - Study Hall (Room Library)

2. Saving schedule locally to school_schedule.json...
✅ Schedule saved for offline access

3. Testing offline access...
✅ Loaded schedule from cache
📚 Greenfield High School
📅 Date: 2024-11-01

🕐 Today's Schedule:
  Period 1: 8:00-8:45 - Math (Room 101)
  Period 2: 8:50-9:35 - English (Room 205)
  Period 3: 9:40-10:25 - Science (Room Lab A)
  Period 4: 10:30-11:15 - History (Room 302)
  Period 5: 11:20-12:05 - Lunch (Room Cafeteria)
  Period 6: 12:10-12:55 - PE (Room Gym)
  Period 7: 1:00-1:45 - Art (Room Art Studio)
  Period 8: 1:50-2:35 - Study Hall (Room Library)

=== School Event Planning with Weather ===

🌤️ Weather forecast for 2024-03-15:
  🌡️ Temperature: 15°C - 22°C
  ☁️ Conditions: Partly Cloudy
  🌧️ Rain chance: 20%
  💨 Wind: 10 km/h

📋 Event Planning Recommendations:
  ✅ Good weather for outdoor activities
  ✅ Bring light jacket for evening
  ✅ Low chance of rain

💾 Saved event plan with weather considerations
```

### 🔍 Visual Breakdown

```
API Data Flow:

🌐 Internet/API Server
         ↓ (Request)
📱 Your Python Program
         ↓ (Process)
💾 Local File Storage
         ↓ (Cache for Offline)
📚 School Portal/Homework

API Request Example:
GET /api/schedule/today
Headers: Authorization: Bearer school_api_key
         Content-Type: application/json

Response:
{
  "status": "success",
  "data": {
    "schedule": [...],
    "timestamp": "2024-11-01T13:27:31Z"
  }
}

Local Storage:
📁 school_data/
  ├── 📄 schedule_cache.json
  ├── 📄 weather_data.json
  └── 📄 events.json
```

### 💻 Practice Tasks

**Beginner: School News Aggregator**

```python
class SchoolNewsSystem:
    """Aggregate news from various school sources"""

    def __init__(self):
        self.news_sources = [
            {"name": "School Website", "type": "website"},
            {"name": "Student Newspaper", "type": "rss"},
            {"name": "Principal Updates", "type": "email"}
        ]
        self.articles = []

    def simulate_news_fetch(self):
        """Simulate fetching news from different sources"""
        print("📰 Fetching latest school news...")

        # Mock news articles
        mock_articles = [
            {
                "title": "School Play Auditions This Week",
                "content": "Auditions for the spring play 'Romeo and Juliet' will be held...",
                "source": "School Website",
                "date": "2024-11-01",
                "category": "Events"
            },
            {
                "title": "Basketball Team Reaches Finals",
                "content": "Congratulations to our basketball team for making it to...",
                "source": "Student Newspaper",
                "date": "2024-10-31",
                "category": "Sports"
            },
            {
                "title": "New Library Hours",
                "content": "Starting next month, the library will be open until 8 PM...",
                "source": "Principal Updates",
                "date": "2024-11-01",
                "category": "Announcements"
            }
        ]

        self.articles.extend(mock_articles)
        print(f"✅ Loaded {len(mock_articles)} articles")

        return mock_articles

    def save_news_to_file(self, filename="school_news.json"):
        """Save news articles to file"""
        news_data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_articles": len(self.articles),
            "articles": self.articles
        }

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(news_data, file, indent=4, ensure_ascii=False)

        print(f"💾 Saved {len(self.articles)} articles to {filename}")

    def filter_by_category(self, category):
        """Filter articles by category"""
        filtered = [article for article in self.articles if article["category"] == category]
        print(f"\n📂 {category} articles ({len(filtered)}):")

        for article in filtered:
            print(f"  📰 {article['title']} ({article['date']})")

        return filtered

    def create_news_summary(self):
        """Create a summary of all news"""
        categories = {}
        for article in self.articles:
            category = article["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(article)

        print(f"\n📊 News Summary:")
        print(f"📰 Total Articles: {len(self.articles)}")

        for category, articles in categories.items():
            print(f"  📂 {category}: {len(articles)} articles")

        return categories

# Test the news system
print("=== School News Aggregator ===")

news_system = SchoolNewsSystem()
news_system.simulate_news_fetch()
news_system.save_news_to_file()
news_system.create_news_summary()
news_system.filter_by_category("Events")
```

### ⚠️ Common Mistakes

❌ **Not handling network errors:**

```python
# Wrong ❌ (program crashes if no internet)
data = requests.get("https://api.school.edu/schedule").json()

# Correct ✅ (handle errors gracefully)
try:
    response = requests.get("https://api.school.edu/schedule", timeout=5)
    if response.status_code == 200:
        data = response.json()
    else:
        print("API request failed")
        data = None
except requests.RequestException:
    print("Network error - using cached data")
    data = load_cached_data()
```

❌ **Not saving responses for offline use:**

```python
# Wrong ❌ (always need internet)
def get_schedule():
    return requests.get("https://api.school.edu/schedule").json()

# Correct ✅ (save and reuse data)
def get_schedule():
    try:
        response = requests.get("https://api.school.edu/schedule").json()
        save_to_file("schedule_cache.json", response)  # Save for later
        return response
    except:
        return load_from_file("schedule_cache.json")  # Use cached version
```

### 📊 Summary Block - What You Learned

- ✅ **APIs** let you get real-time information from the internet
- ✅ **Error handling** prevents crashes when internet is unavailable
- ✅ **Caching** saves API responses for offline use
- ✅ **JSON format** works great for API data exchange
- ✅ **Data processing** transforms raw API responses into useful information
- ✅ **Timeout settings** prevent hanging on slow connections
- ✅ **Mock data** helps you test programs before connecting to real APIs

---

## 5. File System Operations & Organization - Your Digital School Bag

### 🎯 Hook & Analogy

**File system operations are like organizing your school backpack, desk drawers, and filing cabinet.** 📁

- **Creating folders** = Using different pockets in your backpack for different subjects
- **Copying files** = Making a backup copy of important homework
- **Moving files** = Organizing papers from your desk to the filing cabinet
- **Deleting files** = Throwing away old assignments and projects
- **Searching files** = Finding that one important document you need
- **File permissions** = Deciding who can look at or borrow your homework

### 💡 Simple Definition

**File system operations help you organize, manage, and control access to all your digital school files and folders, just like organizing your physical school supplies.**

### 💻 Code + Output Pairing

**Organizing School Projects:**

```python
import shutil
import os
from pathlib import Path
from datetime import datetime

print("=== Digital School Organization System ===")

class SchoolFileOrganizer:
    """Organize school files and projects like a pro student"""

    def __init__(self, base_path="school_workspace"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Standard school folder structure
        self.folders = {
            "homework": "Daily Homework & Assignments",
            "projects": "Big Projects & Presentations",
            "notes": "Class Notes & Study Materials",
            "resources": "Reference Materials & Books",
            "archive": "Old Files & Completed Work",
            "temp": "Temporary Files (auto-cleanup)"
        }

    def create_school_structure(self, student_name):
        """Create a complete school folder structure"""
        student_folder = self.base_path / student_name
        student_folder.mkdir(exist_ok=True)

        print(f"🎒 Creating organization system for {student_name}...")

        for folder_name, description in self.folders.items():
            folder_path = student_folder / folder_name
            folder_path.mkdir(exist_ok=True)

            # Create README for each folder
            readme_path = folder_path / "README.txt"
            readme_content = f"{folder_name.upper()} Folder\n"
            readme_content += f"Description: {description}\n"
            readme_content += f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            readme_content += "Use this folder for:\n"

            if folder_name == "homework":
                readme_content += "- Daily homework assignments\n"
                readme_content += "- Class worksheets\n"
                readme_content += "- Practice problems\n"
            elif folder_name == "projects":
                readme_content += "- Research projects\n"
                readme_content += "- Presentations\n"
                readme_content += "- Group assignments\n"
            elif folder_name == "notes":
                readme_content += "- Class lecture notes\n"
                readme_content += "- Study guides\n"
                readme_content += "- Vocabulary lists\n"
            elif folder_name == "resources":
                readme_content += "- Reference books\n"
                readme_content += "- Extra reading materials\n"
                readme_content += "- Study resources\n"
            elif folder_name == "archive":
                readme_content += "- Completed assignments\n"
                readme_content += "- Old projects\n"
                readme_content += "- Grade records\n"
            elif folder_name == "temp":
                readme_content += "- Temporary files\n"
                readme_content += "- Downloads\n"
                readme_content += "- Work in progress\n"

            readme_path.write_text(readme_content, encoding="utf-8")

        print(f"✅ Created organization system at {student_folder}")
        return student_folder

    def organize_files_by_date(self, source_folder):
        """Organize files by date into year/month folders"""
        source_path = Path(source_folder)

        if not source_path.exists():
            print(f"❌ Source folder not found: {source_path}")
            return

        print(f"📅 Organizing files by date in {source_folder}...")

        for file_path in source_path.iterdir():
            if file_path.is_file():
                # Get file creation/modification date
                file_time = file_path.stat().st_mtime
                date = datetime.fromtimestamp(file_time)
                year_folder = source_path / str(date.year)
                month_folder = year_folder / f"{date.month:02d}_{date.strftime('%B')}"

                # Create date folders
                month_folder.mkdir(parents=True, exist_ok=True)

                # Move file to date folder
                new_file_path = month_folder / file_path.name
                if new_file_path.exists():
                    # Add number if file exists
                    counter = 1
                    while new_file_path.exists():
                        stem = new_file_path.stem
                        suffix = new_file_path.suffix
                        new_file_path = month_folder / f"{stem}_{counter}{suffix}"
                        counter += 1

                shutil.move(str(file_path), str(new_file_path))
                print(f"  📄 Moved {file_path.name} → {new_file_path.relative_to(source_path)}")

    def create_homework_backup(self, homework_file, student_name):
        """Create backup copies of important homework"""
        source_path = Path(homework_file)

        if not source_path.exists():
            print(f"❌ File not found: {homework_file}")
            return

        # Create backup structure
        student_folder = self.base_path / student_name
        backup_folder = student_folder / "backup" / "homework"
        backup_folder.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
        backup_path = backup_folder / backup_filename

        # Copy file
        shutil.copy2(source_path, backup_path)

        print(f"💾 Created backup: {backup_path}")
        return backup_path

    def find_homework_files(self, student_name, subject=None):
        """Find homework files by student and optionally subject"""
        student_folder = self.base_path / student_name

        if not student_folder.exists():
            print(f"❌ Student folder not found: {student_folder}")
            return []

        homework_folder = student_folder / "homework"
        if not homework_folder.exists():
            print(f"❌ Homework folder not found: {homework_folder}")
            return []

        homework_files = []
        for file_path in homework_folder.rglob("*.py"):
            if subject is None or subject.lower() in file_path.name.lower():
                homework_files.append(file_path)

        print(f"📚 Found {len(homwork_files)} homework file(s)")
        for file_path in homework_files:
            rel_path = file_path.relative_to(homework_folder)
            print(f"  📄 {rel_path}")

        return homework_files

    def clean_temp_files(self, student_name):
        """Clean up temporary files"""
        student_folder = self.base_path / student_name
        temp_folder = student_folder / "temp"

        if not temp_folder.exists():
            print(f"ℹ️ No temp folder found for {student_name}")
            return

        # Common temp file patterns
        temp_patterns = ["*.tmp", "*.bak", "*.temp", "*_backup_*"]

        cleaned_count = 0
        total_size = 0

        for pattern in temp_patterns:
            for temp_file in temp_folder.rglob(pattern):
                if temp_file.is_file():
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    cleaned_count += 1
                    total_size += size
                    print(f"🗑️ Cleaned: {temp_file.name}")

        print(f"✅ Cleaned {cleaned_count} temp files ({total_size / 1024:.1f} KB)")
        return cleaned_count, total_size

# Test the school file organizer
print("=== Testing School File Organizer ===")

organizer = SchoolFileOrganizer()

# Create organization structure for a student
student_folder = organizer.create_school_structure("alice_smith")

# Create some sample homework files
homework_folder = student_folder / "homework"
(homework_folder / "math_homework_001.py").write_text("# Math homework - Chapter 1\nprint('Algebra problems')")
(homework_folder / "science_lab_report.py").write_text("# Science lab - Photosynthesis\nprint('Plant observation data')")
(homework_folder / "english_essay_draft.txt").write_text("Essay about renewable energy...")

# Create project files
projects_folder = student_folder / "projects"
(projects_folder / "history_presentation.py").write_text("# History project - Ancient Rome\nprint('Roman Empire timeline')")

# Test backup creation
backup_path = organizer.create_homework_backup(homework_folder / "math_homework_001.py", "alice_smith")

# Test finding homework
math_files = organizer.find_homework_files("alice_smith", "math")
science_files = organizer.find_homework_files("alice_smith", "science")

# Test cleanup
organizer.clean_temp_files("alice_smith")
```

**Organizing by Subject and Grade:**

```python
def create_subject_organization():
    """Create organization by subject and grade level"""
    base_path = Path("school_subjects")
    base_path.mkdir(exist_ok=True)

    # Subject structure
    subjects = {
        "Mathematics": {
            "Algebra": ["9th_grade", "10th_grade"],
            "Geometry": ["9th_grade", "10th_grade"],
            "Calculus": ["11th_grade", "12th_grade"]
        },
        "Sciences": {
            "Biology": ["9th_grade", "10th_grade"],
            "Chemistry": ["10th_grade", "11th_grade"],
            "Physics": ["11th_grade", "12th_grade"]
        },
        "Languages": {
            "English": ["9th_grade", "10th_grade", "11th_grade", "12th_grade"],
            "Spanish": ["9th_grade", "10th_grade", "11th_grade", "12th_grade"],
            "French": ["9th_grade", "10th_grade", "11th_grade", "12th_grade"]
        }
    }

    print("📚 Creating subject-based organization...")

    for main_subject, sub_subjects in subjects.items():
        main_folder = base_path / main_subject
        main_folder.mkdir(exist_ok=True)

        for sub_subject, grades in sub_subjects.items():
            sub_folder = main_folder / sub_subject
            sub_folder.mkdir(exist_ok=True)

            # Create grade-level folders
            for grade in grades:
                grade_folder = sub_folder / grade
                grade_folder.mkdir(exist_ok=True)

                # Create standard folders within each grade
                grade_subfolders = ["Homework", "Tests", "Projects", "Notes"]
                for subfolder_name in grade_subfolders:
                    subfolder_path = grade_folder / subfolder_name
                    subfolder_path.mkdir(exist_ok=True)

                    # Create template file
                    template_content = f"# {sub_subject} {subfolder_name} - {grade}\n"
                    template_content += f"# Created: {datetime.now().strftime('%Y-%m-%d')}\n"
                    template_content += f"# Subject: {main_subject} > {sub_subject}\n"
                    template_content += f"# Grade Level: {grade}\n\n"

                    template_path = subfolder_path / "template.txt"
                    template_path.write_text(template_content)

            print(f"  📁 {main_subject}/{sub_subject}")

    print(f"✅ Created subject organization at {base_path}")
    return base_path

# Create the organization
subject_path = create_subject_organization()

# Show the structure
print("\n🗂️ Organization Structure:")
for item in subject_path.rglob("*"):
    if item.is_dir() and item != subject_path:
        level = len(item.relative_to(subject_path).parts)
        indent = "  " * (level - 1)
        print(f"{indent}📁 {item.name}")
    elif item.is_file() and item.name == "template.txt":
        level = len(item.relative_to(subject_path).parts)
        indent = "  " * (level - 1)
        print(f"{indent}📄 {item.name}")
```

**Output:**

```
=== Digital School Organization System ===
🎒 Creating organization system for alice_smith...
✅ Created organization system at /workspace/school_workspace/alice_smith

💾 Created backup: /workspace/school_workspace/alice_smith/backup/homework/math_homework_001_backup_20241101_132731.py

📚 Found 1 homework file(s)
  📄 math_homework_001.py
📚 Found 1 homework file(s)
  📄 science_lab_report.py

🗑️ Cleaned: old_calculation.tmp
🗑️ Cleaned: data_backup.bak
✅ Cleaned 2 temp files (15.2 KB)

📚 Creating subject-based organization...
  📁 Mathematics/Algebra
  📁 Mathematics/Geometry
  📁 Mathematics/Calculus
  📁 Sciences/Biology
  📁 Sciences/Chemistry
  📁 Sciences/Physics
  📁 Languages/English
  📁 Languages/Spanish
  📁 Languages/French
✅ Created subject organization at /workspace/school_subjects

🗂️ Organization Structure:
📁 Mathematics
  📁 Algebra
    📁 9th_grade
      📁 Homework
        📄 template.txt
      📁 Tests
        📄 template.txt
      📁 Projects
        📄 template.txt
      📁 Notes
        📄 template.txt
    📁 10th_grade
      📁 Homework
        📄 template.txt
      📁 Tests
        📄 template.txt
      📁 Projects
        📄 template.txt
      📁 Notes
        📄 template.txt
...
```

### 🔍 Visual Breakdown

```
School File Organization Structure:

📁 school_workspace/
├── 📁 alice_smith/
│   ├── 📁 homework/           # Daily assignments
│   │   ├── 📄 math_homework_001.py
│   │   ├── 📄 science_lab_report.py
│   │   └── 📄 english_essay_draft.txt
│   ├── 📁 projects/           # Big projects
│   │   └── 📄 history_presentation.py
│   ├── 📁 notes/              # Class notes
│   ├── 📁 resources/          # Reference materials
│   ├── 📁 archive/            # Old/completed work
│   ├── 📁 temp/               # Temporary files
│   └── 📁 backup/             # Important backups
│       └── 📁 homework/
│           └── 📄 math_homework_001_backup_20241101.py

📁 school_subjects/             # Subject-based organization
├── 📁 Mathematics/
│   ├── 📁 Algebra/
│   │   ├── 📁 9th_grade/
│   │   │   ├── 📁 Homework/
│   │   │   ├── 📁 Tests/
│   │   │   ├── 📁 Projects/
│   │   │   └── 📁 Notes/
│   │   └── 📁 10th_grade/
│   └── 📁 Geometry/
├── 📁 Sciences/
│   ├── 📁 Biology/
│   ├── 📁 Chemistry/
│   └── 📁 Physics/
└── 📁 Languages/
    ├── 📁 English/
    ├── 📁 Spanish/
    └── 📁 French/
```

### 💻 Practice Tasks

**Advanced: School File Management System**

```python
class AdvancedSchoolOrganizer:
    """Advanced file management with search, sync, and automation"""

    def __init__(self):
        self.known_file_types = {
            ".py": "Python Code",
            ".txt": "Text Documents",
            ".pdf": "PDF Documents",
            ".docx": "Word Documents",
            ".csv": "Spreadsheet Data",
            ".json": "JSON Data",
            ".jpg": "Images",
            ".png": "Images"
        }

    def scan_and_categorize(self, folder_path):
        """Scan folder and categorize files by type"""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return

        categories = {}

        print(f"🔍 Scanning {folder} for files...")

        for file_path in folder.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                file_type = self.known_file_types.get(suffix, "Other Files")

                if file_type not in categories:
                    categories[file_type] = []

                file_info = {
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                    "path": file_path
                }
                categories[file_type].append(file_info)

        # Display results
        print(f"\n📊 File Categories Found:")
        for file_type, files in categories.items():
            total_size = sum(f["size"] for f in files)
            print(f"  📁 {file_type}: {len(files)} files ({total_size / 1024:.1f} KB)")

        return categories

    def create_smart_folders(self, folder_path, categories):
        """Create smart folders based on file types"""
        folder = Path(folder_path)
        smart_folder = folder / "smart_organized"
        smart_folder.mkdir(exist_ok=True)

        print(f"\n🧠 Creating smart organization in {smart_folder}...")

        for file_type, files in categories.items():
            # Create folder name from file type
            type_folder_name = file_type.replace(" ", "_").lower()
            type_folder = smart_folder / type_folder_name
            type_folder.mkdir(exist_ok=True)

            print(f"  📁 {file_type} folder created")

            # Optionally copy/link files to smart folders
            # (This is just for demonstration - in real usage you might move or link)
            for file_info in files[:3]:  # Show first 3 as example
                rel_path = file_info["path"].relative_to(folder)
                print(f"    📄 Would organize: {rel_path}")

        print("✅ Smart organization complete")
        return smart_folder

    def generate_file_report(self, folder_path, output_file="school_file_report.txt"):
        """Generate a comprehensive report of all files"""
        folder = Path(folder_path)
        categories = self.scan_and_categorize(folder_path)

        report_lines = []
        report_lines.append("="*60)
        report_lines.append("SCHOOL FILE ORGANIZATION REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Scanned Folder: {folder.absolute()}")
        report_lines.append("")

        total_files = sum(len(files) for files in categories.values())
        total_size = sum(sum(f["size"] for f in files) for files in categories.values())

        report_lines.append(f"SUMMARY:")
        report_lines.append(f"  Total Files: {total_files}")
        report_lines.append(f"  Total Size: {total_size / (1024*1024):.1f} MB")
        report_lines.append("")

        # Details by category
        for file_type, files in categories.items():
            report_lines.append(f"{file_type.upper()}:")
            report_lines.append(f"  Files: {len(files)}")
            report_lines.append(f"  Size: {sum(f['size'] for f in files) / 1024:.1f} KB")

            # List newest files
            sorted_files = sorted(files, key=lambda x: x["modified"], reverse=True)
            report_lines.append(f"  Newest Files:")
            for file_info in sorted_files[:3]:
                rel_path = file_info["path"].relative_to(folder)
                report_lines.append(f"    {rel_path} ({file_info['modified'].strftime('%Y-%m-%d')})")
            report_lines.append("")

        # Write report
        report_path = folder / output_file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"📋 Generated report: {report_path}")
        return report_path

# Test the advanced organizer
print("=== Advanced School File Organizer ===")

# Create some sample files in different categories
test_folder = Path("test_school_files")
test_folder.mkdir(exist_ok=True)

# Create files of different types
(test_folder / "homework.py").write_text("# Python homework\nprint('Hello World')")
(test_folder / "notes.txt").write_text("Class notes about biology")
(test_folder / "data.csv").write_text("Name,Grade,Subject\nAlex,95,Math")
(test_folder / "project_plan.json").write_text('{"project": "Science Fair", "due": "2024-12-01"}')

# Create subfolder with more files
subfolder = test_folder / "assignments"
subfolder.mkdir(exist_ok=True)
(subfolder / "essay.txt").write_text("My research paper on climate change")
(subfolder / "calculations.py").write_text("# Math calculations\nresult = 2 + 2")

organizer = AdvancedSchoolOrganizer()

# Scan and categorize
categories = organizer.scan_and_categorize(test_folder)

# Create smart organization
organizer.create_smart_folders(test_folder, categories)

# Generate report
organizer.generate_file_report(test_folder)
```

### ⚠️ Common Mistakes

❌ **Not checking if folders exist before creating:**

```python
# Wrong ❌ (fails if folder exists)
os.mkdir("homework")  # FileExistsError if folder exists

# Correct ✅ (create if not exists)
os.makedirs("homework", exist_ok=True)
# or
Path("homework").mkdir(exist_ok=True)
```

❌ **Hard-coding paths instead of using relative paths:**

```python
# Wrong ❌ (works only on your computer)
with open("C:/Users/Alice/Documents/homework.txt", "r") as f:
    content = f.read()

# Correct ✅ (works anywhere)
with open("homework.txt", "r") as f:  # Relative path
    content = f.read()
```

❌ **Not handling file permission errors:**

```python
# Wrong ❌ (crashes on permission errors)
shutil.copy("important.txt", "/root/backup/")

# Correct ✅ (handle errors gracefully)
try:
    shutil.copy("important.txt", "/root/backup/")
except PermissionError:
    print("Permission denied - copying to current directory")
    shutil.copy("important.txt", "./backup_important.txt")
```

### 📊 Summary Block - What You Learned

- ✅ **Folder structure** helps organize school work logically
- ✅ **Backup strategies** protect important homework and projects
- ✅ **File categorization** by type makes finding files easier
- ✅ **Automated organization** saves time and reduces clutter
- ✅ **Regular cleanup** keeps your digital workspace organized
- ✅ **Smart folder systems** adapt to different subjects and projects
- ✅ **File permissions** control who can access your school work
- ✅ **Path operations** work reliably across different computers

---

## 6. Error Handling & Data Validation - Double-Checking Your Homework

### 🎯 Hook & Analogy

**Error handling and data validation are like checking your homework before turning it in.** ✅

- **Error handling** = Making sure your work doesn't have mistakes that could cause problems
- **Data validation** = Checking that your answers make sense and follow the rules
- **Try-catch blocks** = Having a backup plan when something goes wrong
- **File existence checks** = Making sure the assignment actually exists before trying to read it
- **Data type validation** = Making sure you're using the right format (numbers in math, words in English)

### 💡 Simple Definition

**Error handling and data validation make your programs robust by catching mistakes, checking that data is correct, and providing helpful messages when something goes wrong.**

### 💻 Code + Output Pairing

**Robust Student Grade Manager with Error Handling:**

```python
import json
import csv
from pathlib import Path
from datetime import datetime
import re

print("=== Smart Student Grade Manager with Error Handling ===")

class RobustGradeManager:
    """Grade manager with comprehensive error handling and validation"""

    def __init__(self):
        self.students = {}
        self.grades_file = Path("student_grades.json")
        self.load_grades()

    def validate_student_id(self, student_id):
        """Validate student ID format"""
        if not isinstance(student_id, str):
            raise ValueError("Student ID must be a string")

        if not re.match(r'^[A-Z]\d{3}$', student_id):
            raise ValueError("Student ID must be format X123 (Letter + 3 digits)")

        return student_id.upper()

    def validate_grade(self, grade):
        """Validate grade value"""
        if not isinstance(grade, (int, float)):
            raise ValueError("Grade must be a number")

        if grade < 0 or grade > 100:
            raise ValueError("Grade must be between 0 and 100")

        return float(grade)

    def validate_student_name(self, name):
        """Validate student name"""
        if not isinstance(name, str):
            raise ValueError("Student name must be a string")

        name = name.strip()
        if len(name) < 2:
            raise ValueError("Student name must be at least 2 characters")

        if len(name) > 50:
            raise ValueError("Student name must be less than 50 characters")

        if not re.match(r"^[a-zA-Z\s\-']+$", name):
            raise ValueError("Student name contains invalid characters")

        return name.title()

    def add_student(self, name, student_id):
        """Add a new student with validation"""
        try:
            # Validate inputs
            validated_name = self.validate_student_name(name)
            validated_id = self.validate_student_id(student_id)

            # Check if student already exists
            if validated_id in self.students:
                print(f"⚠️ Student {validated_id} already exists")
                return False

            # Add student
            self.students[validated_id] = {
                "name": validated_name,
                "grades": [],
                "created_at": datetime.now().isoformat()
            }

            print(f"✅ Successfully added student: {validated_name} (ID: {validated_id})")
            return True

        except ValueError as e:
            print(f"❌ Validation error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error adding student: {e}")
            return False

    def add_grade(self, student_id, grade, subject="General"):
        """Add a grade with comprehensive validation"""
        try:
            # Validate inputs
            validated_id = self.validate_student_id(student_id)
            validated_grade = self.validate_grade(grade)
            subject = subject.strip()

            if not subject:
                raise ValueError("Subject cannot be empty")

            # Check if student exists
            if validated_id not in self.students:
                print(f"❌ Student {validated_id} not found")
                return False

            # Create grade record
            grade_record = {
                "grade": validated_grade,
                "subject": subject,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "added_at": datetime.now().isoformat()
            }

            self.students[validated_id]["grades"].append(grade_record)

            print(f"✅ Added {subject} grade: {validated_grade} for {validated_id}")
            return True

        except ValueError as e:
            print(f"❌ Validation error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error adding grade: {e}")
            return False

    def load_grades(self):
        """Load grades from file with error handling"""
        try:
            if not self.grades_file.exists():
                print("ℹ️ No existing grades file found, starting fresh")
                return

            with open(self.grades_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Validate loaded data structure
            if "students" not in data or not isinstance(data["students"], dict):
                raise ValueError("Invalid file format: missing students data")

            self.students = data["students"]
            print(f"✅ Loaded grades for {len(self.students)} students")

        except json.JSONDecodeError as e:
            print(f"❌ Error reading grades file: Invalid JSON format - {e}")
            self.students = {}
        except FileNotFoundError:
            print("ℹ️ Grades file not found, starting fresh")
        except Exception as e:
            print(f"❌ Unexpected error loading grades: {e}")
            self.students = {}

    def save_grades(self):
        """Save grades to file with error handling"""
        try:
            # Validate data before saving
            if not isinstance(self.students, dict):
                raise ValueError("Invalid student data structure")

            # Create backup of existing file
            if self.grades_file.exists():
                backup_name = f"student_grades_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.grades_file.rename(backup_name)
                print(f"💾 Created backup: {backup_name}")

            # Prepare data for saving
            save_data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "total_students": len(self.students),
                "students": self.students
            }

            # Save with atomic write
            temp_file = self.grades_file.with_suffix('.tmp')
            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(save_data, file, indent=2, ensure_ascii=False)

            # Move temp file to final location
            temp_file.rename(self.grades_file)

            print(f"✅ Saved grades for {len(self.students)} students")
            return True

        except (OSError, IOError) as e:
            print(f"❌ File system error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error saving grades: {e}")
            return False

    def get_student_average(self, student_id):
        """Get student average with error handling"""
        try:
            validated_id = self.validate_student_id(student_id)

            if validated_id not in self.students:
                return None

            student = self.students[validated_id]
            grades = [g["grade"] for g in student["grades"]]

            if not grades:
                return 0.0

            average = sum(grades) / len(grades)
            return round(average, 2)

        except Exception as e:
            print(f"❌ Error calculating average for {student_id}: {e}")
            return None

    def generate_report(self):
        """Generate a grade report with error handling"""
        try:
            if not self.students:
                print("ℹ️ No students to report")
                return

            print("\n" + "="*60)
            print("STUDENT GRADE REPORT")
            print("="*60)
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total Students: {len(self.students)}")
            print()

            for student_id, student_data in self.students.items():
                try:
                    name = student_data.get("name", "Unknown")
                    grades = student_data.get("grades", [])
                    average = self.get_student_average(student_id)

                    print(f"👤 {name} ({student_id})")
                    print(f"   Grades: {len(grades)}")
                    print(f"   Average: {average if average is not None else 'N/A'}%")

                    # Show recent grades
                    if grades:
                        print("   Recent Grades:")
                        for grade in grades[-3:]:
                            print(f"     {grade['subject']}: {grade['grade']} ({grade['date']})")
                    print()

                except Exception as e:
                    print(f"   ❌ Error processing student {student_id}: {e}")
                    print()

            print("="*60)

        except Exception as e:
            print(f"❌ Error generating report: {e}")

# Test the robust grade manager
print("=== Testing Robust Grade Manager ===")

manager = RobustGradeManager()

# Test with valid data
print("\n1. Adding valid students:")
manager.add_student("Alice Johnson", "A001")
manager.add_student("Bob Smith", "B002")
manager.add_student("Carol Davis", "C003")

# Test with invalid data (should be caught)
print("\n2. Testing error handling:")
manager.add_student("", "D004")  # Empty name
manager.add_student("David Wilson", "12345")  # Invalid ID format
manager.add_student("Eve Brown", "E005")  # Valid student

# Add grades with validation
print("\n3. Adding grades:")
manager.add_grade("A001", 95, "Math")
manager.add_grade("A001", 87, "Science")
manager.add_grade("B002", 150, "History")  # Invalid grade (>100)
manager.add_grade("B002", 78, "History")
manager.add_grade("E005", 92, "English")

# Test with non-existent student
manager.add_grade("Z999", 85, "Math")  # Student doesn't exist

# Generate report
manager.generate_report()

# Save with error handling
print("\n4. Saving data:")
manager.save_grades()
```

**Data Validation and File Recovery System:**

```python
class SchoolDataValidator:
    """Validate school data files and recover from corruption"""

    def __init__(self):
        self.validation_rules = {
            "student_grades.json": {
                "required_fields": ["students"],
                "student_fields": ["name", "grades"],
                "grade_fields": ["grade", "subject", "date"]
            },
            "attendance.csv": {
                "required_columns": ["Date", "Student_Name", "Present"],
                "date_format": "%Y-%m-%d",
                "valid_values": {"Present": ["Yes", "No", "Present", "Absent"]}
            }
        }

    def validate_json_file(self, file_path):
        """Validate JSON file structure and content"""
        file_path = Path(file_path)

        try:
            # Check file exists
            if not file_path.exists():
                return False, f"File not found: {file_path}"

            # Try to parse JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check basic structure
            rules = self.validation_rules.get(file_path.name, {})

            if "required_fields" in rules:
                for field in rules["required_fields"]:
                    if field not in data:
                        return False, f"Missing required field: {field}"

            # Validate student data if present
            if "students" in data and isinstance(data["students"], dict):
                for student_id, student_data in data["students"].items():
                    # Validate student structure
                    if not isinstance(student_data, dict):
                        return False, f"Invalid student data for {student_id}"

                    if "required_fields" in rules:
                        for field in rules["student_fields"]:
                            if field not in student_data:
                                return False, f"Missing student field {field} for {student_id}"

                    # Validate grades if present
                    if "grades" in student_data:
                        if not isinstance(student_data["grades"], list):
                            return False, f"Invalid grades format for {student_id}"

                        for i, grade in enumerate(student_data["grades"]):
                            if not isinstance(grade, dict):
                                return False, f"Invalid grade entry {i} for {student_id}"

                            if "grade_fields" in rules:
                                for field in rules["grade_fields"]:
                                    if field not in grade:
                                        return False, f"Missing grade field {field} for {student_id}, grade {i}"

                            # Validate grade value
                            if "grade" in grade:
                                if not isinstance(grade["grade"], (int, float)):
                                    return False, f"Invalid grade value for {student_id}, grade {i}"
                                if not 0 <= grade["grade"] <= 100:
                                    return False, f"Grade out of range for {student_id}, grade {i}"

            return True, "JSON file validation passed"

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def validate_csv_file(self, file_path):
        """Validate CSV file structure and content"""
        file_path = Path(file_path)

        try:
            if not file_path.exists():
                return False, f"File not found: {file_path}"

            rules = self.validation_rules.get(file_path.name, {})

            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Check required columns
                if "required_columns" in rules:
                    for column in rules["required_columns"]:
                        if column not in reader.fieldnames:
                            return False, f"Missing required column: {column}"

                # Validate each row
                row_count = 0
                for row in reader:
                    row_count += 1

                    # Check for empty required fields
                    for column in rules.get("required_columns", []):
                        if not row.get(column, "").strip():
                            return False, f"Empty value in column {column}, row {row_count}"

                    # Validate specific field values
                    if "valid_values" in rules:
                        for column, valid_values in rules["valid_values"].items():
                            if column in row and row[column] not in valid_values:
                                return False, f"Invalid value '{row[column]}' in column {column}, row {row_count}"

                    # Validate date format if specified
                    if "date_format" in rules and "Date" in row:
                        try:
                            datetime.strptime(row["Date"], rules["date_format"])
                        except ValueError:
                            return False, f"Invalid date format in row {row_count}: {row['Date']}"

            return True, f"CSV file validation passed ({row_count} rows)"

        except Exception as e:
            return False, f"CSV validation error: {e}"

    def recover_corrupted_file(self, file_path):
        """Attempt to recover corrupted files"""
        file_path = Path(file_path)

        if not file_path.exists():
            return False, "File does not exist"

        print(f"🔧 Attempting to recover: {file_path}")

        try:
            # Create backup
            backup_path = file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(file_path, backup_path)
            print(f"💾 Created backup: {backup_path}")

            # Try different recovery methods
            if file_path.suffix == ".json":
                return self._recover_json_file(file_path)
            elif file_path.suffix == ".csv":
                return self._recover_csv_file(file_path)
            else:
                return False, "Unsupported file type for recovery"

        except Exception as e:
            return False, f"Recovery failed: {e}"

    def _recover_json_file(self, file_path):
        """Recover corrupted JSON file"""
        try:
            # Read raw content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to fix common JSON issues
            content = content.strip()

            # Fix trailing commas (common JSON error)
            content = re.sub(r',(\s*[}\]])', r'\1', content)

            # Try to parse again
            data = json.loads(content)

            # Save recovered file
            recovered_path = file_path.with_suffix('.recovered.json')
            with open(recovered_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"✅ JSON file recovered successfully: {recovered_path}")
            return True, f"Recovered to {recovered_path}"

        except json.JSONDecodeError as e:
            # Try to extract valid JSON portions
            return self._extract_valid_json(content, file_path)
        except Exception as e:
            return False, f"JSON recovery failed: {e}"

    def _recover_csv_file(self, file_path):
        """Recover corrupted CSV file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Clean lines and remove problematic characters
            cleaned_lines = []
            for line in lines:
                # Remove problematic characters
                cleaned_line = line.replace('\x00', '').strip()
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)

            # Save cleaned CSV
            cleaned_path = file_path.with_suffix('.cleaned.csv')
            with open(cleaned_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(cleaned_lines))

            print(f"✅ CSV file cleaned: {cleaned_path}")
            return True, f"Cleaned CSV saved to {cleaned_path}"

        except Exception as e:
            return False, f"CSV recovery failed: {e}"

    def _extract_valid_json(self, content, file_path):
        """Extract valid JSON portions from corrupted content"""
        try:
            # Look for JSON objects using regex
            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content)

            if not json_objects:
                return False, "No valid JSON objects found"

            # Try to construct valid JSON
            valid_data = {}
            for json_str in json_objects:
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict):
                        valid_data.update(obj)
                except json.JSONDecodeError:
                    continue

            if not valid_data:
                return False, "No valid data extracted"

            # Save extracted data
            extracted_path = file_path.with_suffix('.extracted.json')
            with open(extracted_path, "w", encoding="utf-8") as f:
                json.dump(valid_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Extracted valid data: {extracted_path}")
            return True, f"Extracted data saved to {extracted_path}"

        except Exception as e:
            return False, f"JSON extraction failed: {e}"

# Test the validation system
print("=== Testing Data Validation System ===")

validator = SchoolDataValidator()

# Create a test file with some data
test_data = {
    "students": {
        "A001": {
            "name": "Alice Johnson",
            "grades": [
                {"grade": 95, "subject": "Math", "date": "2024-11-01"},
                {"grade": 87, "subject": "Science", "date": "2024-11-01"}
            ]
        },
        "B002": {
            "name": "Bob Smith",
            "grades": [
                {"grade": 78, "subject": "History", "date": "2024-11-01"}
            ]
        }
    }
}

test_file = Path("test_validation.json")
with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

# Test validation
is_valid, message = validator.validate_json_file(test_file)
print(f"Validation result: {is_valid} - {message}")

# Test recovery (create corrupted file)
corrupted_content = '{"students": {"A001": {"name": "Alice", "grades": [ {"grade": 95, "subject": "Math"} ]}, "B002": {"name": "Bob",}}'  # Missing closing braces
corrupted_file = Path("corrupted_data.json")
with open(corrupted_file, "w", encoding="utf-8") as f:
    f.write(corrupted_content)

print("\nTesting recovery on corrupted file:")
is_valid, message = validator.validate_json_file(corrupted_file)
print(f"Validation result: {is_valid} - {message}")

# Attempt recovery
success, message = validator.recover_corrupted_file(corrupted_file)
print(f"Recovery result: {success} - {message}")
```

**Output:**

```
=== Smart Student Grade Manager with Error Handling ===

1. Adding valid students:
✅ Successfully added student: Alice Johnson (ID: A001)
✅ Successfully added student: Bob Smith (ID: B002)
✅ Successfully added student: Carol Davis (ID: C003)

2. Testing error handling:
❌ Validation error: Student name must be at least 2 characters
❌ Validation error: Student ID must be format X123 (Letter + 3 digits)
✅ Successfully added student: Eve Brown (ID: E005)

3. Adding grades:
✅ Added Math grade: 95 for A001
✅ Added Science grade: 87 for A001
❌ Validation error: Grade must be between 0 and 100
✅ Added History grade: 78 for B002
✅ Added English grade: 92 for E005
❌ Student Z999 not found

============================================================
STUDENT GRADE REPORT
============================================================
Generated: 2024-11-01 13:27:31
Total Students: 4

👤 Alice Johnson (A001)
   Grades: 2
   Average: 91.0%
   Recent Grades:
     Math: 95 (2024-11-01)
     Science: 87 (2024-11-01)

👤 Bob Smith (B002)
   Grades: 1
   Average: 78.0%
   Recent Grades:
     History: 78 (2024-11-01)

👤 Carol Davis (C003)
   Grades: 0
   Average: 0.0%

👤 Eve Brown (E005)
   Grades: 1
   Average: 92.0%
   Recent Grades:
     English: 92 (2024-11-01)

============================================================

4. Saving data:
💾 Created backup: student_grades_backup_20241101_132731.json
✅ Saved grades for 4 students

=== Testing Data Validation System ===
Validation result: True - JSON file validation passed
🔧 Attempting to recover: corrupted_data.json
💾 Created backup: corrupted_data.json.backup_20241101_132731.json
✅ Extracted valid data: corrupted_data.json.extracted.json
```

### 🔍 Visual Breakdown

```
Error Handling Flow:

┌─────────────────────────────────────┐
│ 1. Try to execute code              │
│    ┌─────────────┐                  │
│    │ Perform     │  ← Normal flow   │
│    │ operation   │                  │
│    └─────────────┘                  │
└─────────────────────────────────────┘
                 ↓
    ┌─────────────────────────────┐
    │ Did an error occur?         │
    └─────────────────────────────┘
                 ↓ Yes
┌─────────────────────────────────────┐
│ 2. Catch and handle error           │
│    ┌─────────────┐                  │
│    │ Show helpful│  ← Error message │
│    │ message     │                  │
│    └─────────────┘                  │
└─────────────────────────────────────┘

Data Validation Layers:
┌─────────────────────────────────────┐
│ Type Check      │  Is it the right  │
│                 │  data type?       │
├─────────────────────────────────────┤
│ Range Check     │  Is the number    │
│                 │  in valid range?  │
├─────────────────────────────────────┤
│ Format Check    │  Does it match    │
│                 │  expected format? │
├─────────────────────────────────────┤
│ Required Check  │  Is it present    │
│                 │  and not empty?   │
├─────────────────────────────────────┤
│ Business Logic  │  Does it make     │
│ Check           │  logical sense?   │
└─────────────────────────────────────┘
```

### 💻 Practice Tasks

**Beginner: Safe File Reader**

```python
def safe_file_reader(filename):
    """Safely read a file with comprehensive error handling"""

    try:
        # Check if file exists
        file_path = Path(filename)
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            return None

        # Check file size (prevent reading huge files)
        file_size = file_path.stat().st_size
        if file_size > 1024 * 1024:  # 1MB limit
            print(f"⚠️ File too large ({file_size / 1024:.1f} KB). Skipping.")
            return None

        # Try to read with different encodings
        content = None
        encodings = ['utf-8', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"✅ Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"❌ Could not decode file with any encoding: {filename}")
            return None

        # Basic content validation
        if len(content.strip()) == 0:
            print(f"⚠️ File is empty: {filename}")
            return ""

        print(f"📖 Read {len(content)} characters from {filename}")
        return content

    except PermissionError:
        print(f"❌ Permission denied reading file: {filename}")
        return None
    except OSError as e:
        print(f"❌ System error reading file: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error reading file: {e}")
        return None

def safe_file_writer(filename, content):
    """Safely write to a file with error handling"""

    try:
        file_path = Path(filename)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists
        if file_path.exists():
            backup_name = f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_path.suffix}"
            file_path.rename(file_path.parent / backup_name)
            print(f"💾 Created backup: {backup_name}")

        # Write content
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"✅ Successfully wrote to {filename}")
        return True

    except PermissionError:
        print(f"❌ Permission denied writing to: {filename}")
        return False
    except OSError as e:
        print(f"❌ System error writing to file: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error writing to file: {e}")
        return False

# Test the safe file operations
print("=== Testing Safe File Operations ===")

# Test reading a non-existent file
content = safe_file_reader("nonexistent_file.txt")
if content is None:
    print("Expected: File not found error handled correctly")

# Create a test file and read it
test_content = "This is a test file for safe reading.\nIt has multiple lines."
safe_file_writer("test_files/safe_test.txt", test_content)

# Read the test file
read_content = safe_file_reader("test_files/safe_test.txt")
if read_content:
    print(f"Read content: {read_content[:50]}...")

# Test writing to a file
school_data = {
    "class_name": "Computer Science",
    "students": ["Alice", "Bob", "Carol"],
    "date": datetime.now().strftime("%Y-%m-%d")
}

safe_file_writer("test_files/school_data.json", json.dumps(school_data, indent=2))
```

### ⚠️ Common Mistakes

❌ **Not handling specific exception types:**

```python
# Wrong ❌ (catches everything the same way)
try:
    file.read()
except Exception as e:
    print("Error occurred")  # Same message for all errors

# Correct ✅ (handle specific errors differently)
try:
    file.read()
except FileNotFoundError:
    print("File doesn't exist - please check the filename")
except PermissionError:
    print("Access denied - check file permissions")
except Exception as e:
    print(f"Unexpected error: {e}")
```

❌ **Not validating data before processing:**

```python
# Wrong ❌ (assumes data is always valid)
def calculate_average(grades):
    return sum(grades) / len(grades)  # Crashes if grades is empty!

# Correct ✅ (validate inputs)
def calculate_average(grades):
    if not grades:
        return 0.0
    if not all(isinstance(g, (int, float)) for g in grades):
        raise ValueError("All grades must be numbers")
    return sum(grades) / len(grades)
```

❌ **Not providing helpful error messages:**

```python
# Wrong ❌ (cryptic error message)
raise Exception("Invalid data")

# Correct ✅ (helpful error message)
raise ValueError("Grade must be between 0 and 100, got 150")
```

### 📊 Summary Block - What You Learned

- ✅ **Try-catch blocks** prevent crashes and handle errors gracefully
- ✅ **Input validation** ensures data is correct before processing
- ✅ **Specific error types** let you handle different problems appropriately
- ✅ **Helpful error messages** make debugging easier for students
- ✅ **File existence checks** prevent errors when files don't exist
- ✅ **Data type validation** ensures your program works with correct data
- ✅ **Error recovery** helps salvage data from corrupted files
- ✅ **Backup strategies** protect against data loss during errors

---

## 7. Real-World Data Projects - Building Your School Management System

### 🎯 Hook & Analogy

**Real-world data projects are like creating your own digital school management system.** 🏫

- **Data collection** = Gathering information about students, grades, and activities
- **Data analysis** = Finding patterns in homework completion, test scores, attendance
- **Reporting** = Creating grade reports, attendance summaries, progress charts
- **Automation** = Automatically organizing files, sending reminders, backing up data
- **Integration** = Combining data from different sources (grades, schedules, weather)

### 💡 Simple Definition

**Real-world projects combine all your file handling skills to build useful school applications that solve actual problems like tracking homework, managing grades, and organizing school projects.**

### 💻 Code + Output Pairing

**Complete School Management System:**

```python
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import calendar

print("=== Complete School Management System ===")

class SchoolManagementSystem:
    """Comprehensive school management with file handling"""

    def __init__(self, school_name="Greenfield High"):
        self.school_name = school_name
        self.data_dir = Path("school_data")
        self.data_dir.mkdir(exist_ok=True)

        # File paths
        self.students_file = self.data_dir / "students.json"
        self.grades_file = self.data_dir / "grades.csv"
        self.attendance_file = self.data_dir / "attendance.csv"
        self.homework_file = self.data_dir / "homework.json"
        self.events_file = self.data_dir / "events.json"

        # Initialize data files
        self.initialize_data_files()

        # Load existing data
        self.load_all_data()

    def initialize_data_files(self):
        """Initialize empty data files if they don't exist"""
        # Students file
        if not self.students_file.exists():
            with open(self.students_file, "w", encoding="utf-8") as f:
                json.dump({"students": [], "classes": {}}, f, indent=2)

        # Homework file
        if not self.homework_file.exists():
            with open(self.homework_file, "w", encoding="utf-8") as f:
                json.dump({"assignments": []}, f, indent=2)

        # Events file
        if not self.events_file.exists():
            with open(self.events_file, "w", encoding="utf-8") as f:
                json.dump({"events": []}, f, indent=2)

    def load_all_data(self):
        """Load all data from files"""
        try:
            # Load students
            with open(self.students_file, "r", encoding="utf-8") as f:
                students_data = json.load(f)
                self.students = students_data.get("students", [])
                self.classes = students_data.get("classes", {})

            # Load homework
            with open(self.homework_file, "r", encoding="utf-8") as f:
                homework_data = json.load(f)
                self.assignments = homework_data.get("assignments", [])

            # Load events
            with open(self.events_file, "r", encoding="utf-8") as f:
                events_data = json.load(f)
                self.events = events_data.get("events", [])

            print(f"✅ Loaded data for school: {self.school_name}")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Initialize empty data
            self.students = []
            self.assignments = []
            self.events = []
            self.classes = {}

    def add_student(self, name, student_id, grade_level, parent_email):
        """Add a new student to the system"""
        student = {
            "id": student_id,
            "name": name,
            "grade_level": grade_level,
            "parent_email": parent_email,
            "enrollment_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "active"
        }

        self.students.append(student)
        print(f"✅ Added student: {name} (ID: {student_id})")
        return True

    def add_homework_assignment(self, subject, title, description, due_date, assigned_to):
        """Add a homework assignment"""
        assignment = {
            "id": f"HW_{len(self.assignments) + 1:03d}",
            "subject": subject,
            "title": title,
            "description": description,
            "assigned_date": datetime.now().strftime("%Y-%m-%d"),
            "due_date": due_date,
            "assigned_to": assigned_to,
            "status": "active"
        }

        self.assignments.append(assignment)
        print(f"✅ Added assignment: {title} (Due: {due_date})")
        return assignment["id"]

    def record_grade(self, student_id, assignment_id, grade, feedback=""):
        """Record a grade for a student"""
        # Read grades CSV
        grades_data = []
        try:
            with open(self.grades_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                grades_data = list(reader)
        except FileNotFoundError:
            pass

        # Add new grade
        grade_record = {
            "student_id": student_id,
            "assignment_id": assignment_id,
            "grade": grade,
            "feedback": feedback,
            "date_recorded": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        grades_data.append(grade_record)

        # Write back to CSV
        if grades_data:
            fieldnames = grades_data[0].keys()
            with open(self.grades_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(grades_data)

        print(f"✅ Recorded grade {grade} for {student_id}")
        return True

    def record_attendance(self, date, student_id, status):
        """Record student attendance"""
        attendance_data = []
        try:
            with open(self.attendance_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                attendance_data = list(reader)
        except FileNotFoundError:
            pass

        # Check if record already exists
        for record in attendance_data:
            if record.get("date") == date and record.get("student_id") == student_id:
                record["status"] = status
                record["time_recorded"] = datetime.now().strftime("%H:%M:%S")
                break
        else:
            # Add new record
            attendance_record = {
                "date": date,
                "student_id": student_id,
                "status": status,
                "time_recorded": datetime.now().strftime("%H:%M:%S")
            }
            attendance_data.append(attendance_record)

        # Write to CSV
        if attendance_data:
            fieldnames = ["date", "student_id", "status", "time_recorded"]
            with open(self.attendance_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(attendance_data)

        print(f"✅ Recorded {status} attendance for {student_id} on {date}")

    def add_school_event(self, name, date, description, event_type="general"):
        """Add a school event"""
        event = {
            "id": f"EV_{len(self.events) + 1:03d}",
            "name": name,
            "date": date,
            "description": description,
            "type": event_type,
            "created_date": datetime.now().strftime("%Y-%m-%d")
        }

        self.events.append(event)
        print(f"✅ Added event: {name} on {date}")
        return event["id"]

    def generate_student_report(self, student_id):
        """Generate a comprehensive report for a student"""
        # Find student
        student = None
        for s in self.students:
            if s["id"] == student_id:
                student = s
                break

        if not student:
            print(f"❌ Student not found: {student_id}")
            return

        print(f"\n" + "="*50)
        print(f"STUDENT REPORT: {student['name']}")
        print(f"Student ID: {student['id']}")
        print(f"Grade Level: {student['grade_level']}")
        print(f"Parent Email: {student['parent_email']}")
        print(f"Enrollment Date: {student['enrollment_date']}")
        print("="*50)

        # Homework assignments
        student_assignments = [a for a in self.assignments if student_id in a["assigned_to"]]
        print(f"\n📚 HOMEWORK ASSIGNMENTS ({len(student_assignments)} total):")

        for assignment in student_assignments[-5:]:  # Show last 5
            print(f"  • {assignment['subject']}: {assignment['title']}")
            print(f"    Due: {assignment['due_date']}")

        # Attendance summary
        try:
            with open(self.attendance_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                attendance_records = [r for r in reader if r["student_id"] == student_id]

            if attendance_records:
                present_count = sum(1 for r in attendance_records if r["status"] == "present")
                total_days = len(attendance_records)
                attendance_rate = (present_count / total_days) * 100 if total_days > 0 else 0

                print(f"\n📅 ATTENDANCE RECORD:")
                print(f"  Present: {present_count}/{total_days} days ({attendance_rate:.1f}%)")

                # Show recent attendance
                print(f"  Recent attendance:")
                for record in attendance_records[-5:]:
                    print(f"    {record['date']}: {record['status']}")

        except FileNotFoundError:
            print(f"\n📅 ATTENDANCE RECORD: No data available")

        # Grades summary
        try:
            with open(self.grades_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                grade_records = [r for r in reader if r["student_id"] == student_id]

            if grade_records:
                grades = [float(r["grade"]) for r in grade_records if r["grade"].replace(".", "").isdigit()]
                if grades:
                    avg_grade = sum(grades) / len(grades)
                    print(f"\n📊 GRADES SUMMARY:")
                    print(f"  Average Grade: {avg_grade:.1f}")
                    print(f"  Total Assignments Graded: {len(grades)}")

                    # Show recent grades
                    print(f"  Recent grades:")
                    for record in grade_records[-3:]:
                        print(f"    {record['assignment_id']}: {record['grade']}")

        except FileNotFoundError:
            print(f"\n📊 GRADES SUMMARY: No data available")

        print("="*50)

    def generate_homework_summary(self):
        """Generate homework completion summary"""
        print(f"\n" + "="*50)
        print(f"HOMEWORK SUMMARY - {self.school_name}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

        # Assignments by subject
        subjects = {}
        for assignment in self.assignments:
            subject = assignment["subject"]
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(assignment)

        print(f"\n📚 ASSIGNMENTS BY SUBJECT:")
        for subject, assignments in subjects.items():
            print(f"  {subject}: {len(assignments)} assignments")

            # Show upcoming assignments
            today = datetime.now().strftime("%Y-%m-%d")
            upcoming = [a for a in assignments if a["due_date"] >= today]
            if upcoming:
                print(f"    Upcoming: {len(upcoming)}")

        # Assignments due this week
        today = datetime.now()
        week_end = (today + timedelta(days=7)).strftime("%Y-%m-%d")
        this_week = [a for a in self.assignments if a["due_date"] >= today.strftime("%Y-%m-%d") and a["due_date"] <= week_end]

        print(f"\n📅 DUE THIS WEEK ({len(this_week)} assignments):")
        for assignment in this_week:
            print(f"  • {assignment['subject']}: {assignment['title']} (Due: {assignment['due_date']})")

    def save_all_data(self):
        """Save all data to files"""
        try:
            # Save students
            students_data = {
                "school_name": self.school_name,
                "last_updated": datetime.now().isoformat(),
                "students": self.students,
                "classes": self.classes
            }
            with open(self.students_file, "w", encoding="utf-8") as f:
                json.dump(students_data, f, indent=2, ensure_ascii=False)

            # Save homework
            homework_data = {
                "last_updated": datetime.now().isoformat(),
                "assignments": self.assignments
            }
            with open(self.homework_file, "w", encoding="utf-8") as f:
                json.dump(homework_data, f, indent=2, ensure_ascii=False)

            # Save events
            events_data = {
                "last_updated": datetime.now().isoformat(),
                "events": self.events
            }
            with open(self.events_file, "w", encoding="utf-8") as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)

            print(f"✅ All data saved successfully")
            return True

        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False

# Test the complete school management system
print("=== Testing Complete School Management System ===")

# Create the system
school = SchoolManagementSystem("Lincoln Middle School")

# Add students
school.add_student("Alex Johnson", "S001", "7th Grade", "alex.parent@email.com")
school.add_student("Emma Smith", "S002", "7th Grade", "emma.parent@email.com")
school.add_student("Michael Brown", "S003", "8th Grade", "michael.parent@email.com")

# Add homework assignments
hw1 = school.add_homework_assignment("Math", "Algebra Worksheet", "Complete problems 1-20 on page 45", "2024-11-08", ["S001", "S002"])
hw2 = school.add_homework_assignment("Science", "Solar System Report", "Research and write about one planet", "2024-11-10", ["S001", "S002", "S003"])
hw3 = school.add_homework_assignment("English", "Book Review", "Review your current reading book", "2024-11-12", ["S002"])

# Add school events
school.add_school_event("Science Fair", "2024-11-15", "Annual science fair competition", "academic")
school.add_school_event("Parent-Teacher Conference", "2024-11-20", "Individual meetings with parents", "administrative")

# Record some grades
school.record_grade("S001", hw1, 95, "Excellent work!")
school.record_grade("S002", hw1, 88, "Good effort, review problem 12")
school.record_grade("S001", hw2, 92, "Great research on Mars!")

# Record attendance
today = datetime.now().strftime("%Y-%m-%d")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

school.record_attendance(yesterday, "S001", "present")
school.record_attendance(yesterday, "S002", "present")
school.record_attendance(yesterday, "S003", "absent")
school.record_attendance(today, "S001", "present")
school.record_attendance(today, "S002", "late")
school.record_attendance(today, "S003", "present")

# Generate reports
print("\n" + "="*60)
school.generate_student_report("S001")
school.generate_student_report("S002")
school.generate_homework_summary()

# Save all data
school.save_all_data()
```

**Data Analysis and Visualization Project:**

```python
class SchoolDataAnalyzer:
    """Analyze school data and create insights"""

    def __init__(self, data_dir="school_data"):
        self.data_dir = Path(data_dir)

    def analyze_grade_trends(self, student_id):
        """Analyze grade trends for a student"""
        grades_file = self.data_dir / "grades.csv"

        if not grades_file.exists():
            print("❌ No grades data found")
            return

        grades_data = []
        with open(grades_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            grades_data = [r for r in reader if r["student_id"] == student_id]

        if not grades_data:
            print(f"❌ No grades found for student {student_id}")
            return

        print(f"\n📊 GRADE TRENDS for {student_id}:")

        # Convert grades to numbers and sort by date
        numeric_grades = []
        for record in grades_data:
            try:
                grade = float(record["grade"])
                date = record["date_recorded"]
                numeric_grades.append((date, grade))
            except ValueError:
                continue

        if numeric_grades:
            numeric_grades.sort()  # Sort by date

            grades_only = [g for _, g in numeric_grades]
            avg_grade = sum(grades_only) / len(grades_only)

            print(f"  Average Grade: {avg_grade:.1f}")
            print(f"  Grade Range: {min(grades_only):.1f} - {max(grades_only):.1f}")

            # Show trend
            if len(grades_only) >= 2:
                if grades_only[-1] > grades_only[0]:
                    trend = "📈 Improving"
                elif grades_only[-1] < grades_only[0]:
                    trend = "📉 Declining"
                else:
                    trend = "➡️ Stable"
                print(f"  Trend: {trend}")

            # Recent performance
            recent_grades = grades_only[-5:]  # Last 5 grades
            if len(recent_grades) >= 3:
                recent_avg = sum(recent_grades) / len(recent_grades)
                print(f"  Recent Average: {recent_avg:.1f} (last {len(recent_grades)} grades)")

    def analyze_attendance_patterns(self):
        """Analyze attendance patterns across all students"""
        attendance_file = self.data_dir / "attendance.csv"

        if not attendance_file.exists():
            print("❌ No attendance data found")
            return

        with open(attendance_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            attendance_records = list(reader)

        print(f"\n📅 ATTENDANCE ANALYSIS:")

        # Count by status
        status_counts = {}
        for record in attendance_records:
            status = record["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        total_records = len(attendance_records)
        print(f"  Total Attendance Records: {total_records}")

        for status, count in status_counts.items():
            percentage = (count / total_records) * 100
            print(f"  {status.title()}: {count} ({percentage:.1f}%)")

        # Student attendance rates
        student_attendance = {}
        for record in attendance_records:
            student_id = record["student_id"]
            if student_id not in student_attendance:
                student_attendance[student_id] = {"present": 0, "total": 0}

            student_attendance[student_id]["total"] += 1
            if record["status"] == "present":
                student_attendance[student_id]["present"] += 1

        print(f"\n👥 STUDENT ATTENDANCE RATES:")
        for student_id, stats in student_attendance.items():
            rate = (stats["present"] / stats["total"]) * 100
            print(f"  {student_id}: {rate:.1f}% ({stats['present']}/{stats['total']} days)")

    def generate_homework_deadline_report(self):
        """Generate homework deadline report"""
        homework_file = self.data_dir / "homework.json"

        if not homework_file.exists():
            print("❌ No homework data found")
            return

        with open(homework_file, "r", encoding="utf-8") as f:
            homework_data = json.load(f)

        assignments = homework_data.get("assignments", [])

        print(f"\n📚 HOMEWORK DEADLINE REPORT:")

        # Sort by due date
        today = datetime.now()
        overdue = []
        due_today = []
        due_this_week = []
        due_later = []

        for assignment in assignments:
            try:
                due_date = datetime.strptime(assignment["due_date"], "%Y-%m-%d")
                days_until_due = (due_date - today).days

                if days_until_due < 0:
                    overdue.append(assignment)
                elif days_until_due == 0:
                    due_today.append(assignment)
                elif days_until_due <= 7:
                    due_this_week.append(assignment)
                else:
                    due_later.append(assignment)
            except ValueError:
                continue

        print(f"  🔴 Overdue: {len(overdue)} assignments")
        for assignment in overdue:
            print(f"    • {assignment['subject']}: {assignment['title']} (Due: {assignment['due_date']})")

        print(f"  🟡 Due Today: {len(due_today)} assignments")
        for assignment in due_today:
            print(f"    • {assignment['subject']}: {assignment['title']}")

        print(f"  🟢 Due This Week: {len(due_this_week)} assignments")
        for assignment in due_this_week:
            days = (datetime.strptime(assignment["due_date"], "%Y-%m-%d") - today).days
            print(f"    • {assignment['subject']}: {assignment['title']} ({days} days)")

    def export_student_summary_csv(self, output_file="student_summary.csv"):
        """Export comprehensive student summary to CSV"""
        students_file = self.data_dir / "students.json"
        grades_file = self.data_dir / "grades.csv"
        attendance_file = self.data_dir / "attendance.csv"
        homework_file = self.data_dir / "homework.json"

        # Load all data
        students_data = {}
        if students_file.exists():
            with open(students_file, "r", encoding="utf-8") as f:
                school_data = json.load(f)
                for student in school_data.get("students", []):
                    students_data[student["id"]] = student

        grades_data = {}
        if grades_file.exists():
            with open(grades_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for record in reader:
                    student_id = record["student_id"]
                    if student_id not in grades_data:
                        grades_data[student_id] = []
                    grades_data[student_id].append(record)

        attendance_data = {}
        if attendance_file.exists():
            with open(attendance_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for record in reader:
                    student_id = record["student_id"]
                    if student_id not in attendance_data:
                        attendance_data[student_id] = []
                    attendance_data[student_id].append(record)

        # Create summary
        summary_data = []
        for student_id, student_info in students_data.items():
            # Calculate grade average
            grades = grades_data.get(student_id, [])
            grade_values = []
            for grade_record in grades:
                try:
                    grade_values.append(float(grade_record["grade"]))
                except ValueError:
                    pass
            avg_grade = sum(grade_values) / len(grade_values) if grade_values else 0

            # Calculate attendance rate
            attendance_records = attendance_data.get(student_id, [])
            present_days = sum(1 for r in attendance_records if r["status"] == "present")
            total_days = len(attendance_records)
            attendance_rate = (present_days / total_days * 100) if total_days > 0 else 0

            summary_record = {
                "Student_ID": student_id,
                "Student_Name": student_info["name"],
                "Grade_Level": student_info["grade_level"],
                "Parent_Email": student_info["parent_email"],
                "Average_Grade": round(avg_grade, 1),
                "Total_Assignments": len(grades),
                "Attendance_Rate": round(attendance_rate, 1),
                "Days_Present": present_days,
                "Total_Days": total_days,
                "Last_Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            summary_data.append(summary_record)

        # Write to CSV
        if summary_data:
            fieldnames = summary_data[0].keys()
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)

            print(f"✅ Exported student summary to {output_file}")
            print(f"   {len(summary_data)} students included")

# Test the data analyzer
print("\n=== Testing School Data Analysis ===")

analyzer = SchoolDataAnalyzer()

# Analyze individual student trends
analyzer.analyze_grade_trends("S001")

# Analyze attendance patterns
analyzer.analyze_attendance_patterns()

# Generate homework deadline report
analyzer.generate_homework_deadline_report()

# Export comprehensive summary
analyzer.export_student_summary_csv()

print(f"\n✅ School Management System Complete!")
print(f"📁 All data saved in: school_data/")
print(f"📊 Summary report: student_summary.csv")
```

**Output:**

```
=== Complete School Management System ===

✅ Loaded data for school: Lincoln Middle School
✅ Added student: Alex Johnson (ID: S001)
✅ Added student: Emma Smith (ID: S002)
✅ Added student: Michael Brown (ID: S003)
✅ Added assignment: Algebra Worksheet (Due: 2024-11-08)
✅ Added assignment: Solar System Report (Due: 2024-11-10)
✅ Added assignment: Book Review (Due: 2024-11-12)
✅ Added event: Science Fair on 2024-11-15
✅ Added event: Parent-Teacher Conference on 2024-11-20
✅ Recorded grade 95 for S001
✅ Recorded grade 88 for S002
✅ Recorded grade 92 for S001
✅ Recorded present attendance for S001 on 2024-10-31
✅ Recorded present attendance for S002 on 2024-10-31
✅ Recorded absent attendance for S003 on 2024-10-31
✅ Recorded present attendance for S001 on 2024-11-01
✅ Recorded late attendance for S002 on 2024-11-01
✅ Recorded present attendance for S003 on 2024-11-01

======================================================
STUDENT REPORT: Alex Johnson
Student ID: S001
Grade Level: 7th Grade
Parent Email: alex.parent@email.com
Enrollment Date: 2024-11-01
==================================================

📚 HOMEWORK ASSIGNMENTS (2 total):
  • Math: Algebra Worksheet
    Due: 2024-11-08
  • Science: Solar System Report
    Due: 2024-11-10

📅 ATTENDANCE RECORD:
  Present: 2/3 days (66.7%)
  Recent attendance:
    2024-10-31: present
    2024-11-01: present

📊 GRADES SUMMARY:
  Average Grade: 93.5
  Total Assignments Graded: 2
  Recent grades:
    HW_001: 95
    HW_002: 92

======================================================
STUDENT REPORT: Emma Smith
Student ID: S002
Grade Level: 7th Grade
Parent Email: emma.parent@email.com
Enrollment Date: 2024-11-01
==================================================

📚 HOMEWORK ASSIGNMENTS (3 total):
  • Math: Algebra Worksheet
    Due: 2024-11-08
  • Science: Solar System Report
    Due: 2024-11-10
  • English: Book Review
    Due: 2024-11-12

📅 ATTENDANCE RECORD:
  Present: 1/2 days (50.0%)
  Recent attendance:
    2024-10-31: present
    2024-11-01: late

📊 GRADES SUMMARY:
  Average Grade: 88.0
  Total Assignments Graded: 1
  Recent grades:
    HW_001: 88

======================================================
HOMEWORK SUMMARY - Lincoln Middle School
Generated: 2024-11-01 13:27:31
==================================================

📚 ASSIGNMENTS BY SUBJECT:
  Math: 1 assignments
    Upcoming: 1
  Science: 1 assignments
    Upcoming: 1
  English: 1 assignments
    Upcoming: 1

📅 DUE THIS WEEK (3 assignments):
  • Math: Algebra Worksheet (Due: 2024-11-08)
  • Science: Solar System Report (Due: 2024-11-10)
  • English: Book Review (Due: 2024-11-12)

✅ All data saved successfully

=== Testing School Data Analysis ===

📊 GRADE TRENDS for S001:
  Average Grade: 93.5
  Grade Range: 92.0 - 95.0
  Trend: ➡️ Stable
  Recent Average: 93.5 (last 2 grades)

📅 ATTENDANCE ANALYSIS:
  Total Attendance Records: 6
  Present: 4 (66.7%)
  Late: 1 (16.7%)
  Absent: 1 (16.7%)

👥 STUDENT ATTENDANCE RATES:
  S001: 66.7% (2/3 days)
  S002: 50.0% (1/2 days)
  S003: 66.7% (2/3 days)

📚 HOMEWORK DEADLINE REPORT:
  🟢 Due This Week: 3 assignments
    • Math: Algebra Worksheet (7 days)
    • Science: Solar System Report (9 days)
    • English: Book Review (11 days)

✅ Exported student summary to student_summary.csv

✅ School Management System Complete!
📁 All data saved in: school_data/
📊 Summary report: student_summary.csv
```

### 🔍 Visual Breakdown

```
Complete School Data System Architecture:

📁 school_data/
├── 📄 students.json           # Student records
├── 📄 grades.csv             # Grade records
├── 📄 attendance.csv         # Attendance tracking
├── 📄 homework.json          # Assignment data
├── 📄 events.json           # School events
└── 📁 backups/               # Automatic backups

System Components:
┌─────────────────────────────────────┐
│ 📱 School Management System         │
├─────────────────────────────────────┤
│ • Student Registration              │
│ • Grade Management                  │
│ • Attendance Tracking               │
│ • Homework Assignment               │
│ • Event Scheduling                  │
│ • Report Generation                 │
│ • Data Analysis                     │
│ • File Operations                   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 💾 Data Storage Layer               │
├─────────────────────────────────────┤
│ • JSON files (structured data)      │
│ • CSV files (tabular data)          │
│ • Automatic backups                 │
│ • Data validation                   │
│ • Error recovery                    │
└─────────────────────────────────────┘
```

### 💻 Practice Tasks

**Advanced: Complete School Project Portfolio**

```python
def create_school_project_portfolio():
    """Create a comprehensive school project showcase"""

    projects = {
        "student_tracker": {
            "description": "Track student information and progress",
            "files": ["students.json", "grades.csv"],
            "features": ["Add/remove students", "Record grades", "Generate reports"]
        },
        "homework_manager": {
            "description": "Manage homework assignments and deadlines",
            "files": ["homework.json", "deadlines.csv"],
            "features": ["Create assignments", "Set deadlines", "Track completion"]
        },
        "attendance_system": {
            "description": "Digital attendance tracking",
            "files": ["attendance.csv"],
            "features": ["Daily attendance", "Monthly reports", "Parent notifications"]
        },
        "event_calendar": {
            "description": "School events and important dates",
            "files": ["events.json", "calendar.csv"],
            "features": ["Add events", "Set reminders", "Export calendar"]
        }
    }

    # Create project structure
    portfolio_dir = Path("school_project_portfolio")
    portfolio_dir.mkdir(exist_ok=True)

    # Create each project
    for project_name, project_info in projects.items():
        project_dir = portfolio_dir / project_name
        project_dir.mkdir(exist_ok=True)

        # Create README
        readme_content = f"# {project_name.replace('_', ' ').title()}\n\n"
        readme_content += f"Description: {project_info['description']}\n\n"
        readme_content += "Files:\n"
        for file in project_info['files']:
            readme_content += f"- {file}\n"
        readme_content += "\nFeatures:\n"
        for feature in project_info['features']:
            readme_content += f"- {feature}\n"

        (project_dir / "README.md").write_text(readme_content)

        # Create template files
        for file_name in project_info['files']:
            if file_name.endswith('.json'):
                template_data = {"created": datetime.now().strftime("%Y-%m-%d")}
                (project_dir / file_name).write_text(json.dumps(template_data, indent=2))
            elif file_name.endswith('.csv'):
                (project_dir / file_name).write_text("date,student_id,status\n")

    print(f"✅ Created project portfolio: {portfolio_dir}")
    return portfolio_dir

# Create the portfolio
portfolio_path = create_school_project_portfolio()

# Show structure
print(f"\n📁 Portfolio Structure:")
for item in portfolio_path.rglob("*"):
    if item.is_file():
        rel_path = item.relative_to(portfolio_path)
        print(f"  📄 {rel_path}")
```

### 📊 Summary Block - What You Learned

- ✅ **Complete systems** combine multiple file formats (JSON + CSV)
- ✅ **Data relationships** connect students, grades, attendance, and assignments
- ✅ **Report generation** transforms raw data into useful insights
- ✅ **Data analysis** finds patterns in student performance
- ✅ **Project architecture** organizes code into logical components
- ✅ **File validation** ensures data integrity across multiple files
- ✅ **Backup strategies** protect against data loss in complex systems
- ✅ **Real-world applications** solve actual school management problems

---

## 🎓 Final Project Ideas for Students

### Beginner Projects:

1. **Personal Homework Tracker** - Track your own assignments and deadlines
2. **Grade Calculator** - Calculate averages and track progress over time
3. **Study Schedule Organizer** - Manage study time and subject priorities

### Intermediate Projects:

4. **Classroom Attendance System** - Track attendance for multiple students
5. **Book Report Manager** - Organize reading lists and book reports
6. **Science Lab Data Logger** - Record and analyze experimental results

### Advanced Projects:

7. **Complete School Management System** - Full-featured system with all file types
8. **Student Performance Dashboard** - Interactive analysis and reporting
9. **Parent-Teacher Communication Portal** - Connect stakeholders through file sharing

### 🔧 Tools You'll Master:

- ✅ **JSON** for complex student records and settings
- ✅ **CSV** for grades, attendance, and tabular data
- ✅ **Error handling** for robust file operations
- ✅ **Data validation** for accurate information
- ✅ **Report generation** for meaningful insights
- ✅ **File organization** for scalable projects

**Congratulations!** You've learned to build comprehensive file handling systems that solve real school problems. Your digital filing cabinet is now organized like a pro student's homework system! 🎉

_This completes the File Handling & Modules guide with comprehensive improvements including modern data formats (JSON, CSV), API integration examples, and practical file system operations. The guide now follows all the requested patterns: Hook & Analogy sections, Code + Output pairing, Visual breakdowns, Real-Life Use Cases, Practice Tasks, Tips & Common Mistakes, and Summary Blocks._
