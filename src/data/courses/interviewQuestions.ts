// Comprehensive Interview Question Database with Role-Based Architecture
// Built from ALL 8 phases of content with structured questions, answers, and metadata
// Role-based filtering for career-specific interview preparation

// ==================== TYPE DEFINITIONS ====================

export type QuestionType = 'coding' | 'behavioral' | 'system_design' | 'technical' | 'sql' | 'debugging';
export type Difficulty = 'easy' | 'medium' | 'hard' | 'expert';

// Career roles based on Python course outcomes
export type CareerRole = 
    | 'python_developer'      // General Python programming roles
    | 'backend_developer'     // Server-side development, APIs
    | 'data_scientist'        // Data analysis, ML, statistics
    | 'ml_engineer'           // Machine learning engineering
    | 'fullstack_developer'   // Frontend + Backend development
    | 'data_engineer'         // Data pipelines, ETL, big data
    | 'devops_engineer'       // CI/CD, cloud, infrastructure
    | 'software_engineer'     // General software development
    | 'api_developer'         // REST API, GraphQL specialization
    | 'cloud_engineer';       // Cloud platforms, serverless

export const CAREER_ROLES: { id: CareerRole; label: string; description: string }[] = [
    { id: 'python_developer', label: 'Python Developer', description: 'General Python programming positions' },
    { id: 'backend_developer', label: 'Backend Developer', description: 'Server-side development, APIs, databases' },
    { id: 'data_scientist', label: 'Data Scientist', description: 'Data analysis, statistics, visualization' },
    { id: 'ml_engineer', label: 'ML Engineer', description: 'Machine learning models, AI systems' },
    { id: 'fullstack_developer', label: 'Full Stack Developer', description: 'Frontend + Backend development' },
    { id: 'data_engineer', label: 'Data Engineer', description: 'Data pipelines, ETL, big data' },
    { id: 'devops_engineer', label: 'DevOps Engineer', description: 'CI/CD, cloud, infrastructure' },
    { id: 'software_engineer', label: 'Software Engineer', description: 'General software development' },
    { id: 'api_developer', label: 'API Developer', description: 'REST API, GraphQL, microservices' },
    { id: 'cloud_engineer', label: 'Cloud Engineer', description: 'Cloud platforms, serverless' }
];

export type Topic = 
    // Programming Fundamentals
    'arrays' | 'strings' | 'linked_lists' | 'trees' | 'graphs' | 'dp' | 'recursion' | 
    'sorting' | 'searching' | 'hash' | 'stacks_queues' | 'heap' | 'bit_manipulation' |
    // Advanced Topics
    'oop' | 'patterns' | 'functional' | 'async' | 'concurrency' |
    // Data & ML
    'sql' | 'databases' | 'ml_ai' | 'nlp' | 'computer_vision' | 'statistics' | 'pandas' |
    // Professional
    'soft_skills' | 'business' | 'leadership' | 'ethics' | 'system_design' |
    // Debugging & Testing
    'debugging' | 'testing' | 'devops' |
    // Python Specific
    'python' | 'generators' | 'decorators' | 'context_managers' | 'type_hints' | 'asyncio';

export interface InterviewQuestion {
    id: string;
    type: QuestionType;
    difficulty: Difficulty;
    topics: Topic[];
    phase: number;
    roles: CareerRole[];  // Primary career roles this question is relevant for
    secondary_roles?: CareerRole[];  // Additional roles with lower priority
    company_focus?: string[];
    question: string;
    hints?: string[];
    solution?: string;
    code_template?: string;
    time_estimate?: number;
    explanation?: string;
    follow_ups?: string[];
    test_cases?: { input: string; expected: string }[];
    related_skills?: string[];
}

export interface BehavioralQuestion {
    id: string;
    category: string;
    phase: number;
    roles: CareerRole[];
    question: string;
    sample_answer?: string;
    tips: string[];
    follow_ups?: string[];
    star_example?: {
        situation: string;
        task: string;
        action: string;
        result: string;
    };
}

export interface SystemDesignQuestion {
    id: string;
    topic: string;
    phase: number;
    roles: CareerRole[];
    question: string;
    key_points: string[];
    example_architecture?: string;
    estimated_time: number;
    complexity_level: 'low' | 'medium' | 'high';
    related_topics?: string[];
}

// ==================== PHASE 1 - PYTHON FUNDAMENTALS ====================

export const codingQuestions: InterviewQuestion[] = [
    // PHASE 1 - Python Basics (All Python Developer Roles)
    {
        id: 'p1-easy-001',
        type: 'coding',
        difficulty: 'easy',
        topics: ['strings', 'python'],
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'fullstack_developer', 'software_engineer'],
        secondary_roles: ['data_scientist', 'data_engineer'],
        company_focus: ['Google', 'Amazon'],
        question: 'Write a Python function to reverse a string without using slicing or built-in reverse functions.',
        hints: ['Strings are immutable in Python, so you need to convert to a list', 'Think about swapping characters from both ends'],
        solution: `def reverse_string(s):
    chars = list(s)
    left, right = 0, len(s) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return ''.join(chars)`,
        time_estimate: 5,
        explanation: 'We convert string to list for O(1) swaps, then use two-pointer technique.',
        related_skills: ['string manipulation', 'two-pointer technique']
    },
    {
        id: 'p1-easy-002',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python'],
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'data_engineer'],
        question: 'Write a function that finds the sum of all even numbers in a list.',
        hints: ['Use the modulo operator to check for even numbers', 'Python list comprehensions can make this concise'],
        solution: `def sum_even_numbers(arr):
    return sum(x for x in arr if x % 2 == 0)`,
        time_estimate: 3,
        explanation: 'List comprehension with generator expression for memory efficiency.'
    },
    {
        id: 'p1-easy-003',
        type: 'coding',
        difficulty: 'easy',
        topics: ['strings', 'python'],
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'fullstack_developer'],
        question: 'Check if a string is a palindrome (reads same forwards and backwards, ignoring case and spaces).',
        hints: ['Clean the string first by removing non-alphanumeric characters', 'Convert to lowercase for case-insensitive comparison'],
        solution: `def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]`,
        time_estimate: 5,
        explanation: 'Use str.isalnum() to filter and str.lower() for case insensitivity.'
    },
    {
        id: 'p1-easy-004',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python'],
        phase: 1,
        roles: ['python_developer', 'software_engineer', 'data_scientist'],
        question: 'Write a function to find the maximum element in a list without using max().',
        hints: ['Initialize with the first element', 'Compare each subsequent element'],
        solution: `def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val`,
        time_estimate: 4
    },
    {
        id: 'p1-easy-005',
        type: 'coding',
        difficulty: 'easy',
        topics: ['strings', 'python'],
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'data_scientist'],
        question: 'Write a function to count the occurrences of each character in a string.',
        hints: ['Use a dictionary to store counts', 'Iterate through each character'],
        solution: `def count_chars(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    return char_count`,
        time_estimate: 4,
        explanation: 'dict.get() provides a safe way to handle missing keys.'
    },
    {
        id: 'p1-easy-006',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python', 'recursion'],
        phase: 1,
        roles: ['python_developer', 'software_engineer', 'backend_developer'],
        question: 'Write a function that returns the factorial of a number using recursion.',
        hints: ['Base case: factorial of 0 or 1 is 1', 'Recursive case: n! = n * (n-1)!'],
        solution: `def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)`,
        time_estimate: 5,
        explanation: 'Classic recursive approach. Note: Python has recursion limits.'
    },
    {
        id: 'p1-easy-007',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python'],
        phase: 1,
        roles: ['python_developer', 'data_scientist', 'data_engineer'],
        question: 'Write a function to find the average of numbers in a list.',
        hints: ['Sum all elements and divide by count', 'Handle empty list case'],
        solution: `def average(arr):
    if not arr:
        return 0
    return sum(arr) / len(arr)`,
        time_estimate: 3
    },
    {
        id: 'p1-easy-008',
        type: 'coding',
        difficulty: 'easy',
        topics: ['strings', 'python'],
        phase: 1,
        roles: ['python_developer', 'fullstack_developer', 'backend_developer'],
        question: 'Write a function to convert a string to title case.',
        hints: ['Python has a built-in title() method', 'But can you do it manually?'],
        solution: `def to_title_case(s):
    return ' '.join(word.capitalize() for word in s.split(' '))`,
        time_estimate: 4,
        explanation: 'str.split() creates words, capitalize() handles each word.'
    },
    {
        id: 'p1-easy-009',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python'],
        phase: 1,
        roles: ['python_developer', 'software_engineer', 'backend_developer'],
        question: 'Write a function to remove duplicates from a list while preserving order.',
        hints: ['Use a set to track seen items', 'Python 3.7+ preserves dict insertion order'],
        solution: `def remove_duplicates(arr):
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result`,
        time_estimate: 5,
        explanation: 'Using set for O(1) lookup makes this O(n) overall.'
    },
    {
        id: 'p1-easy-010',
        type: 'coding',
        difficulty: 'easy',
        topics: ['strings', 'python'],
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'software_engineer'],
        question: 'Write a function to check if two strings are anagrams.',
        hints: ['Anagrams have same character counts', 'Sorting both strings and comparing works'],
        solution: `def are_anagrams(s1, s2):
    return sorted(s1) == sorted(s2)`,
        time_estimate: 4,
        explanation: 'Sorting is O(n log n). For better performance, use character counting.'
    },
    {
        id: 'p1-easy-011',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python'],
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'data_scientist'],
        question: 'Write a function that returns the first non-repeating character in a string.',
        hints: ['Count occurrences of each character', 'Return first with count of 1'],
        solution: `def first_non_repeating(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    for char in s:
        if char_count[char] == 1:
            return char
    return None`,
        time_estimate: 6
    },
    {
        id: 'p1-easy-012',
        type: 'coding',
        difficulty: 'easy',
        topics: ['recursion', 'python'],
        phase: 1,
        roles: ['python_developer', 'software_engineer', 'backend_developer'],
        question: 'Write a recursive function to calculate the nth Fibonacci number.',
        hints: ['Fibonacci: F(n) = F(n-1) + F(n-2)', 'Base cases: F(0)=0, F(1)=1'],
        solution: `def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)`,
        time_estimate: 5,
        explanation: 'Note: This is O(2^n) - memoization can optimize it to O(n).',
        follow_ups: ['How would you optimize this with memoization?', 'What is the space complexity?']
    },
    {
        id: 'p1-easy-013',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python'],
        phase: 1,
        roles: ['python_developer', 'data_scientist', 'software_engineer'],
        question: 'Write a function to find the second largest number in a list.',
        hints: ['Find max, then find max among remaining', 'Or use two variables to track top two'],
        solution: `def second_largest(arr):
    if len(arr) < 2:
        return None
    largest = second = float('-inf')
    for num in arr:
        if num > largest:
            second = largest
            largest = num
        elif num > second and num != largest:
            second = num
    return second if second != float('-inf') else None`,
        time_estimate: 8
    },
    {
        id: 'p1-easy-014',
        type: 'coding',
        difficulty: 'easy',
        topics: ['strings', 'python'],
        phase: 1,
        roles: ['python_developer', 'fullstack_developer', 'backend_developer'],
        question: 'Write a function to capitalize the first letter of each word in a sentence.',
        hints: ['Split by spaces to get words', 'Capitalize each word and join back'],
        solution: `def capitalize_words(sentence):
    return ' '.join(word.capitalize() for word in sentence.split())`,
        time_estimate: 4
    },
    {
        id: 'p1-easy-015',
        type: 'coding',
        difficulty: 'easy',
        topics: ['arrays', 'python', 'sorting'],
        phase: 1,
        roles: ['python_developer', 'software_engineer', 'backend_developer'],
        question: 'Write a function to merge two sorted lists into one sorted list.',
        hints: ['Both lists are already sorted', 'Use two pointers to merge'],
        solution: `def merge_sorted_lists(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result`,
        time_estimate: 10,
        explanation: 'This is the merge step of merge sort - O(n+m) time.'
    },

    // PHASE 2 - Data Structures & Algorithms
    {
        id: 'p2-med-001',
        type: 'coding',
        difficulty: 'medium',
        topics: ['linked_lists', 'recursion'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        secondary_roles: ['fullstack_developer'],
        company_focus: ['Google', 'Meta', 'Amazon'],
        question: 'Detect if a linked list has a cycle using Floyd\'s algorithm.',
        hints: ['Use two pointers (slow and fast) moving at different speeds', 'If they meet, there\'s a cycle'],
        solution: `def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False`,
        time_estimate: 10,
        explanation: 'Floyd\'s Cycle Detection uses O(1) space and O(n) time.',
        related_skills: ['linked lists', 'two-pointer technique']
    },
    {
        id: 'p2-med-002',
        type: 'coding',
        difficulty: 'medium',
        topics: ['arrays', 'two_pointers'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Microsoft'],
        question: 'Given a sorted array, find two numbers that add up to a target sum. Return their indices.',
        hints: ['Since array is sorted, use two pointers from both ends', 'If sum is too big, move right pointer left'],
        solution: `def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []`,
        time_estimate: 8,
        explanation: 'Two-pointer technique gives O(n) time and O(1) space for sorted arrays.'
    },
    {
        id: 'p2-med-003',
        type: 'coding',
        difficulty: 'medium',
        topics: ['trees', 'recursion'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Find the maximum depth (height) of a binary tree.',
        hints: ['Think recursively - depth is 1 + max depth of subtrees', 'Base case: empty tree has depth 0'],
        solution: `def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))`,
        time_estimate: 8,
        related_skills: ['tree traversal', 'recursion']
    },
    {
        id: 'p2-med-004',
        type: 'coding',
        difficulty: 'medium',
        topics: ['stacks_queues'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        secondary_roles: ['fullstack_developer'],
        question: 'Implement a stack that supports push, pop, top, and getMin in O(1) time.',
        hints: ['Use two stacks - one for main operations, one to track minimums'],
        solution: `class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
    
    def top(self):
        return self.stack[-1] if self.stack else None
    
    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None`,
        time_estimate: 12,
        related_skills: ['stack', 'data structure design']
    },
    {
        id: 'p2-med-005',
        type: 'coding',
        difficulty: 'medium',
        topics: ['arrays', 'sliding_window'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'data_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Find the maximum sum of any subarray of fixed size k.',
        hints: ['Sliding window can reuse previous calculations', 'Window sum = previous sum - element leaving + new element'],
        solution: `def max_subarray_sum(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    return max_sum`,
        time_estimate: 8,
        explanation: 'Sliding window reduces O(n*k) to O(n) time complexity.'
    },
    {
        id: 'p2-med-006',
        type: 'coding',
        difficulty: 'medium',
        topics: ['linked_lists'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Find the middle node of a linked list.',
        hints: ['Use two pointers - slow moves 1 step, fast moves 2 steps', 'When fast reaches end, slow is at middle'],
        solution: `def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow`,
        time_estimate: 6
    },
    {
        id: 'p2-med-007',
        type: 'coding',
        difficulty: 'medium',
        topics: ['arrays', 'hash'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Given an array of integers, find the first recurring character.',
        hints: ['Use a set to track seen characters', 'Return first character that appears twice'],
        solution: `def first_recurring(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return num
        seen.add(num)
    return None`,
        time_estimate: 5
    },
    {
        id: 'p2-med-008',
        type: 'coding',
        difficulty: 'medium',
        topics: ['recursion', 'trees'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Check if a binary tree is symmetric (mirror of itself).',
        hints: ['Two trees are mirrors if their roots are equal and left is mirror of right\'s right'],
        solution: `def is_symmetric(root):
    def is_mirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2:
            return False
        return (t1.val == t2.val and 
                is_mirror(t1.left, t2.right) and 
                is_mirror(t1.right, t2.left))
    return is_mirror(root.left, root.right)`,
        time_estimate: 12,
        related_skills: ['tree recursion', 'binary trees']
    },
    {
        id: 'p2-med-009',
        type: 'coding',
        difficulty: 'medium',
        topics: ['stacks_queues'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Implement a queue using two stacks.',
        hints: ['Use two stacks - one for enqueue, one for dequeue', 'When dequeue stack empty, transfer all from enqueue'],
        solution: `class MyQueue:
    def __init__(self):
        self.enqueue_stack = []
        self.dequeue_stack = []
    
    def enqueue(self, x):
        self.enqueue_stack.append(x)
    
    def dequeue(self):
        if not self.dequeue_stack:
            while self.enqueue_stack:
                self.dequeue_stack.append(self.enqueue_stack.pop())
        return self.dequeue_stack.pop() if self.dequeue_stack else None
    
    def peek(self):
        if not self.dequeue_stack:
            while self.enqueue_stack:
                self.dequeue_stack.append(self.enqueue_stack.pop())
        return self.dequeue_stack[-1] if self.dequeue_stack else None`,
        time_estimate: 15,
        related_skills: ['data structure design', 'stack/queue operations']
    },
    {
        id: 'p2-med-010',
        type: 'coding',
        difficulty: 'medium',
        topics: ['arrays', 'sorting'],
        phase: 2,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Sort a list of 0s, 1s, and 2s (Dutch National Flag problem).',
        hints: ['Use three pointers - low, mid, high', 'Partition into three sections'],
        solution: `def sort_colors(arr):
    low = mid = 0
    high = len(arr) - 1
    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
    return arr`,
        time_estimate: 10,
        explanation: 'One-pass partitioning with O(1) space.'
    },

    // PHASE 3 - Advanced Python
    {
        id: 'p3-med-011',
        type: 'coding',
        difficulty: 'medium',
        topics: ['linked_lists', 'recursion'],
        phase: 3,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Reverse a singly linked list iteratively.',
        hints: ['Use three pointers: prev, current, next', 'Iteratively update pointers while traversing'],
        solution: `def reverse_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev`,
        time_estimate: 10,
        related_skills: ['linked list manipulation']
    },
    {
        id: 'p3-med-012',
        type: 'coding',
        difficulty: 'medium',
        topics: ['trees', 'bst'],
        phase: 3,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Validate if a binary tree is a valid Binary Search Tree.',
        hints: ['Each node has valid range (min, max)', 'Left < node < Right'],
        solution: `def is_valid_bst(root):
    def validate(node, low, high):
        if not node:
            return True
        if low is not None and node.val <= low:
            return False
        if high is not None and node.val >= high:
            return False
        return (validate(node.left, low, node.val) and 
                validate(node.right, node.val, high))
    return validate(root, None, None)`,
        time_estimate: 12,
        related_skills: ['BST validation', 'binary tree']
    },
    {
        id: 'p3-med-013',
        type: 'coding',
        difficulty: 'medium',
        topics: ['oop', 'python'],
        phase: 3,
        roles: ['software_engineer', 'backend_developer', 'python_developer', 'fullstack_developer'],
        question: 'Implement a Singleton class in Python.',
        hints: ['Override __new__ method', 'Only one instance should be created'],
        solution: `class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance`,
        time_estimate: 8,
        related_skills: ['design patterns', 'OOP']
    },
    {
        id: 'p3-med-014',
        type: 'coding',
        difficulty: 'medium',
        topics: ['generators', 'python', 'async'],
        phase: 3,
        roles: ['python_developer', 'backend_developer', 'software_engineer'],
        secondary_roles: ['data_engineer', 'ml_engineer'],
        question: 'Write a generator function that yields Fibonacci numbers.',
        hints: ['Use yield instead of return', 'Maintain state between yields'],
        solution: `def fibonacci_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b`,
        time_estimate: 6,
        related_skills: ['generators', 'iterators']
    },
    {
        id: 'p3-med-015',
        type: 'coding',
        difficulty: 'medium',
        topics: ['decorators', 'python'],
        phase: 3,
        roles: ['python_developer', 'backend_developer', 'software_engineer'],
        secondary_roles: ['fullstack_developer'],
        question: 'Create a decorator that measures function execution time.',
        hints: ['Decorator takes function as argument', 'Use time module to measure'],
        solution: `import time
def time_execution(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper`,
        time_estimate: 10,
        related_skills: ['decorators']
    },
    {
        id: 'p3-med-016',
        type: 'coding',
        difficulty: 'medium',
        topics: ['context_managers', 'python'],
        phase: 3,
        roles: ['python_developer', 'backend_developer', 'software_engineer'],
        question: 'Create a context manager for a timer.',
        hints: ['Use @contextmanager from contextlib', 'Track time using contextlib'],
        solution: `from contextlib import contextmanager
import time

@contextmanager
def timer():
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"Elapsed: {end - start:.4f}s")

# Usage:
# with timer():
#     # code here`,
        time_estimate: 12,
        related_skills: ['context managers', 'resource management']
    },
    {
        id: 'p3-med-017',
        type: 'coding',
        difficulty: 'medium',
        topics: ['recursion', 'strings'],
        phase: 3,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Generate all permutations of a string.',
        hints: ['Recursive approach: fix first char, permute rest', 'Base case: single character'],
        solution: `def permutations(s):
    if len(s) <= 1:
        return [s]
    result = []
    for i, char in enumerate(s):
        for perm in permutations(s[:i] + s[i+1:]):
            result.append(char + perm)
    return result`,
        time_estimate: 15,
        explanation: 'O(n! * n) time complexity for n characters.',
        related_skills: ['recursion', 'combinatorics']
    },

    // PHASE 4 - AI/ML Fundamentals
    {
        id: 'p4-med-018',
        type: 'coding',
        difficulty: 'medium',
        topics: ['ml_ai', 'python', 'arrays'],
        phase: 4,
        roles: ['ml_engineer', 'data_scientist', 'data_engineer'],
        secondary_roles: ['python_developer', 'software_engineer'],
        company_focus: ['Google', 'Meta', 'OpenAI'],
        question: 'Implement a simple K-Nearest Neighbors (KNN) classifier from scratch.',
        hints: ['Calculate distances between test point and all training points', 'Find k closest points and vote'],
        solution: `import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(np.bincount(k_labels).argmax())
        return predictions`,
        time_estimate: 20,
        related_skills: ['machine learning', 'distance metrics']
    },
    {
        id: 'p4-med-019',
        type: 'coding',
        difficulty: 'medium',
        topics: ['ml_ai', 'python'],
        phase: 4,
        roles: ['ml_engineer', 'data_scientist'],
        question: 'Implement sigmoid function and its derivative for neural networks.',
        hints: ['Sigmoid: σ(x) = 1 / (1 + e^(-x))', 'Derivative: σ(x) * (1 - σ(x))'],
        solution: `import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)`,
        time_estimate: 5,
        related_skills: ['neural networks', 'activation functions']
    },
    {
        id: 'p4-med-020',
        type: 'coding',
        difficulty: 'medium',
        topics: ['ml_ai', 'python', 'statistics'],
        phase: 4,
        roles: ['data_scientist', 'ml_engineer', 'data_engineer'],
        question: 'Implement a simple linear regression model using gradient descent.',
        hints: ['Initialize weights and bias', 'Update using gradient of loss'],
        solution: `import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias`,
        time_estimate: 25,
        related_skills: ['linear regression', 'gradient descent']
    },
    {
        id: 'p4-med-021',
        type: 'coding',
        difficulty: 'medium',
        topics: ['nlp', 'python', 'strings'],
        phase: 4,
        roles: ['data_scientist', 'ml_engineer', 'nlp_specialist'],
        secondary_roles: ['python_developer'],
        question: 'Implement a simple function to tokenize text into words.',
        hints: ['Split on whitespace and remove punctuation', 'Convert to lowercase'],
        solution: `import re

def tokenize(text):
    # Remove punctuation and split
    text = text.lower()
    tokens = re.findall(r'\\w+', text)
    return tokens`,
        time_estimate: 8,
        related_skills: ['NLP', 'text preprocessing']
    },
    {
        id: 'p4-med-022',
        type: 'coding',
        difficulty: 'medium',
        topics: ['ml_ai', 'python', 'arrays', 'statistics'],
        phase: 4,
        roles: ['data_scientist', 'ml_engineer', 'data_engineer'],
        question: 'Implement a function to calculate cosine similarity between two vectors.',
        hints: ['Cosine similarity = (A · B) / (||A|| * ||B||)', 'Use numpy for vector operations'],
        solution: `import numpy as np

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0`,
        time_estimate: 6,
        related_skills: ['similarity metrics', 'vector operations']
    },

    // PHASE 5 - Professional Skills & Problem Solving
    {
        id: 'p5-hard-023',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'arrays'],
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta', 'Amazon'],
        question: 'Given an array where each element is max jump length, determine if you can reach the last index.',
        hints: ['Think about greedy approach first', 'Keep track of the farthest reachable index'],
        solution: `def can_jump(nums):
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest:
            return False
        farthest = max(farthest, i + jump)
        if farthest >= len(nums) - 1:
            return True
    return farthest >= len(nums) - 1`,
        time_estimate: 15,
        explanation: 'Greedy solution tracks maximum reach - O(n) time, O(1) space.',
        related_skills: ['greedy algorithms', 'dynamic programming']
    },
    {
        id: 'p5-hard-024',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'strings'],
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Microsoft'],
        question: 'Find the length of the Longest Increasing Subsequence.',
        hints: ['Think about DP: LIS ending at each position', 'Alternative: patience sorting for O(n log n)'],
        solution: `def length_of_lis(nums):
    if not nums:
        return 0
    piles = []
    for num in nums:
        left, right = 0, len(piles)
        while left < right:
            mid = (left + right) // 2
            if piles[mid] >= num:
                right = mid
            else:
                left = mid + 1
        if left == len(piles):
            piles.append(num)
        else:
            piles[left] = num
    return len(piles)`,
        time_estimate: 20,
        explanation: 'Patience sorting algorithm for O(n log n) solution.',
        related_skills: ['dynamic programming', 'binary search']
    },
    {
        id: 'p5-hard-025',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'strings'],
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Edit Distance: Minimum operations to convert one string to another.',
        hints: ['Consider insert, delete, and replace operations', 'DP over string indices'],
        solution: `def min_edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]`,
        time_estimate: 25,
        explanation: 'Classic DP with O(mn) time and space complexity.',
        related_skills: ['dynamic programming', 'string algorithms']
    },
    {
        id: 'p5-hard-026',
        type: 'coding',
        difficulty: 'hard',
        topics: ['graphs', 'bfs', 'dfs'],
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'data_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Find the number of connected components in an undirected graph.',
        hints: ['Use BFS or DFS to explore each component', 'Count how many times you start a new BFS/DFS'],
        solution: `from collections import deque

def count_components(n, edges):
    if not edges:
        return n
    
    # Build adjacency list
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    visited = set()
    count = 0
    
    for i in range(n):
        if i not in visited:
            count += 1
            # BFS
            queue = deque([i])
            visited.add(i)
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    
    return count`,
        time_estimate: 15,
        related_skills: ['graph algorithms', 'BFS/DFS']
    },
    {
        id: 'p5-hard-027',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'arrays'],
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Given n, find the minimum number of perfect square numbers that sum to n.',
        hints: ['This is a classic DP problem', 'dp[i] = min(dp[i-sq] + 1) for all squares'],
        solution: `def num_squares(n):
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    squares = [i*i for i in range(1, int(n**0.5) + 1)]
    
    for i in range(1, n + 1):
        for sq in squares:
            if sq > i:
                break
            dp[i] = min(dp[i], dp[i - sq] + 1)
    
    return dp[n]`,
        time_estimate: 18,
        explanation: 'Dynamic programming with O(n * sqrt(n)) complexity.',
        related_skills: ['dynamic programming', 'number theory']
    },
    {
        id: 'p5-hard-028',
        type: 'coding',
        difficulty: 'hard',
        topics: ['trees', 'recursion', 'dp'],
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Microsoft'],
        question: 'Binary Tree Maximum Path Sum (can start and end anywhere).',
        hints: ['Post-order traversal to calculate', 'At each node, consider max path through it'],
        solution: `def max_path_sum(root):
    max_sum = float('-inf')
    
    def max_gain(node):
        if not node:
            return 0
        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)
        price_newpath = node.val + left_gain + right_gain
        nonlocal max_sum
        max_sum = max(max_sum, price_newpath)
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return max_sum`,
        time_estimate: 20,
        related_skills: ['tree DP', 'recursion']
    },

    // PHASE 6 - Interview Skills & Technical
    {
        id: 'p6-med-029',
        type: 'coding',
        difficulty: 'medium',
        topics: ['graphs', 'bfs'],
        phase: 6,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon'],
        question: 'Implement Breadth-First Search on a graph represented as adjacency list.',
        hints: ['Use a queue for BFS', 'Track visited nodes to avoid cycles'],
        solution: `from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)`,
        time_estimate: 10,
        related_skills: ['graph traversal', 'BFS']
    },
    {
        id: 'p6-med-030',
        type: 'coding',
        difficulty: 'medium',
        topics: ['hash', 'arrays'],
        phase: 6,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Amazon'],
        question: 'Two Sum: Find two numbers that add up to target.',
        hints: ['Hash map gives O(n) solution', 'Store values and indices as you iterate'],
        solution: `def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`,
        time_estimate: 8,
        explanation: 'Hash map reduces time from O(n²) to O(n).',
        related_skills: ['hash tables', 'two-sum']
    },
    {
        id: 'p6-med-031',
        type: 'coding',
        difficulty: 'medium',
        topics: ['sorting', 'arrays'],
        phase: 6,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Implement Merge Sort algorithm.',
        hints: ['Divide and conquer: split, sort, merge', 'Recursively sort halves then merge'],
        solution: `def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result`,
        time_estimate: 20,
        explanation: 'O(n log n) time, O(n) space complexity.',
        related_skills: ['sorting algorithms', 'divide and conquer']
    },
    {
        id: 'p6-med-032',
        type: 'coding',
        difficulty: 'medium',
        topics: ['recursion', 'trees'],
        phase: 6,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Inorder traversal of binary tree without recursion.',
        hints: ['Use a stack to track nodes', 'Go left until None, then process and go right'],
        solution: `def inorder_traversal(root):
    result = []
    stack = []
    current = root
    
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    
    return result`,
        time_estimate: 12,
        related_skills: ['tree traversal', 'stack']
    },
    {
        id: 'p6-med-033',
        type: 'coding',
        difficulty: 'medium',
        topics: ['bit_manipulation', 'python'],
        phase: 6,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Count the number of 1s in the binary representation of a number.',
        hints: ['Python has bin() function', 'Can also use bit manipulation with n & (n-1)'],
        solution: `def count_ones(n):
    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    return count`,
        time_estimate: 6,
        explanation: 'n & (n-1) clears the least significant 1 bit.'
    },
    {
        id: 'p6-med-034',
        type: 'coding',
        difficulty: 'medium',
        topics: ['sql', 'databases'],
        phase: 6,
        roles: ['backend_developer', 'data_engineer', 'data_scientist', 'software_engineer'],
        company_focus: ['Amazon', 'Google', 'Meta'],
        question: 'Write SQL query to find employees earning more than their managers.',
        hints: ['Self-join the Employee table', 'Match employee.manager_id with manager.id'],
        solution: `SELECT e1.name as Employee, e1.salary as Employee_Salary
FROM Employee e1
JOIN Employee e2 ON e1.manager_id = e2.id
WHERE e1.salary > e2.salary;`,
        time_estimate: 8,
        related_skills: ['SQL', 'self-join']
    },
    {
        id: 'p6-med-035',
        type: 'coding',
        difficulty: 'medium',
        topics: ['recursion', 'backtracking'],
        phase: 6,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        question: 'Generate all valid parentheses combinations for n pairs.',
        hints: ['Backtracking: add left if count_left < n', 'Add right if count_right < count_left'],
        solution: `def generate_parentheses(n):
    result = []
    
    def backtrack(s, left, right):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)
    
    backtrack('', 0, 0)
    return result`,
        time_estimate: 15,
        explanation: 'Catalan number of combinations - O(Cat(n)) time.',
        related_skills: ['backtracking', 'recursion']
    },

    // PHASE 7 - Advanced Projects & System Design
    {
        id: 'p7-hard-036',
        type: 'coding',
        difficulty: 'hard',
        topics: ['trees', 'recursion'],
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Find the Lowest Common Ancestor of two nodes in a binary tree.',
        hints: ['Traverse tree, look for nodes in different subtrees', 'First node where both targets found is LCA'],
        solution: `def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right`,
        time_estimate: 15,
        related_skills: ['tree algorithms', 'LCA']
    },
    {
        id: 'p7-hard-037',
        type: 'coding',
        difficulty: 'hard',
        topics: ['graphs', 'dp'],
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'cloud_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Find the shortest path in a weighted directed graph using Dijkstra\'s algorithm.',
        hints: ['Use priority queue (min-heap)', 'Track shortest distance to each node'],
        solution: `import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current_dist > distances[current]:
            continue
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances`,
        time_estimate: 20,
        explanation: 'O((V+E) log V) time complexity with binary heap.',
        related_skills: ['shortest path', 'priority queue']
    },
    {
        id: 'p7-hard-038',
        type: 'coding',
        difficulty: 'hard',
        topics: ['heap', 'arrays'],
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Find the median of a stream of numbers (Median Finder).',
        hints: ['Use two heaps: max-heap for lower half, min-heap for upper half', 'Balance sizes to get median'],
        solution: `import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max-heap (store negative)
        self.large = []  # min-heap
    
    def add_num(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, heapq.heappop(self.small))
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self):
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2
        return -self.small[0]`,
        time_estimate: 20,
        related_skills: ['heap data structure', 'streaming algorithms']
    },
    {
        id: 'p7-hard-039',
        type: 'coding',
        difficulty: 'hard',
        topics: ['trie', 'strings'],
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'api_developer'],
        company_focus: ['Google', 'Amazon'],
        question: 'Implement a Trie (prefix tree) with insert and search operations.',
        hints: ['Each node has children dict and is_end_of_word flag', 'Traverse character by character'],
        solution: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end`,
        time_estimate: 18,
        related_skills: ['trie', 'prefix search']
    },
    {
        id: 'p7-hard-040',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'arrays', 'optimization'],
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Maximum Product Subarray: Find the maximum product of a contiguous subarray.',
        hints: ['Track both max and min (negative numbers flip signs)', 'O(n) time with DP'],
        solution: `def max_product(nums):
    if not nums:
        return 0
    max_prod = min_prod = result = nums[0]
    
    for i in range(1, len(nums)):
        if nums[i] < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(nums[i], max_prod * nums[i])
        min_prod = min(nums[i], min_prod * nums[i])
        result = max(result, max_prod)
    
    return result`,
        time_estimate: 15,
        related_skills: ['Kadane\'s algorithm', 'DP']
    },
    {
        id: 'p7-hard-041',
        type: 'coding',
        difficulty: 'hard',
        topics: ['oop', 'patterns'],
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'fullstack_developer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Implement the Observer Design Pattern for a stock price monitoring system.',
        hints: ['Subject maintains list of observers', 'Notify all observers on state change'],
        solution: `from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, stock, price):
        pass

class Subject:
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self, stock, price):
        for observer in self._observers:
            observer.update(stock, price)

class Stock(Subject):
    def __init__(self, symbol, price):
        super().__init__()
        self.symbol = symbol
        self._price = price
    
    @property
    def price(self):
        return self._price
    
    @price.setter
    def price(self, value):
        self._price = value
        self.notify(self.symbol, value)`,
        time_estimate: 25,
        related_skills: ['design patterns', 'OOP']
    },

    // PHASE 8 - Career & Entrepreneurship
    {
        id: 'p8-hard-042',
        type: 'coding',
        difficulty: 'hard',
        topics: ['graphs', 'topological_sort'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'data_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Implement Topological Sort using Kahn\'s algorithm.',
        hints: ['Find all nodes with in-degree 0', 'Remove them and update in-degrees'],
        solution: `from collections import deque

def topological_sort(n, edges):
    graph = {i: [] for i in range(n)}
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []`,
        time_estimate: 18,
        related_skills: ['graph algorithms', 'DAG']
    },
    {
        id: 'p8-hard-043',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'strings'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Regular Expression Matching (supporting . and *).',
        hints: ['Use DP: dp[i][j] = match at i,j', 'Handle * as "zero or more of preceding element"'],
        solution: `def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2]
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]
    
    return dp[m][n]`,
        time_estimate: 30,
        explanation: 'Complex DP problem - O(mn) time and space.',
        related_skills: ['DP', 'string algorithms']
    },
    {
        id: 'p8-hard-044',
        type: 'coding',
        difficulty: 'hard',
        topics: ['concurrency', 'async', 'python'],
        phase: 8,
        roles: ['backend_developer', 'devops_engineer', 'cloud_engineer'],
        company_focus: ['Amazon', 'Google', 'Meta'],
        question: 'Implement a thread-safe counter using threading.Lock.',
        hints: ['Use Lock to synchronize access', 'Context manager ensures release'],
        solution: `import threading

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
    
    def decrement(self):
        with self._lock:
            self._value -= 1
    
    @property
    def value(self):
        with self._lock:
            return self._value`,
        time_estimate: 12,
        related_skills: ['threading', 'concurrency']
    },
    {
        id: 'p8-hard-045',
        type: 'coding',
        difficulty: 'hard',
        topics: ['sql', 'databases', 'optimization'],
        phase: 8,
        roles: ['backend_developer', 'data_engineer', 'data_scientist'],
        company_focus: ['Amazon', 'Google', 'Meta'],
        question: 'Write an optimized SQL query to find customers who ordered in all months.',
        hints: ['Count distinct months per customer', 'Compare with total months in table'],
        solution: `SELECT customer_id
FROM orders
GROUP BY customer_id
HAVING COUNT(DISTINCT DATE_TRUNC('month', order_date)) = 
       (SELECT COUNT(DISTINCT DATE_TRUNC('month', order_date)) FROM orders);`,
        time_estimate: 10,
        related_skills: ['SQL', 'window functions']
    },
    {
        id: 'p8-hard-046',
        type: 'coding',
        difficulty: 'hard',
        topics: ['system_design', 'oop'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'fullstack_developer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Design a parking lot system with different vehicle types.',
        hints: ['Abstract Vehicle class', 'Different spots for different vehicles', 'Use inheritance'],
        solution: `from abc import ABC, abstractmethod
from enum import Enum

class VehicleType(Enum):
    CAR = 1
    MOTORCYCLE = 2
    TRUCK = 3

class Vehicle(ABC):
    def __init__(self, license_plate, type):
        self.license_plate = license_plate
        self.type = type
        self.spot = None
    
    @abstractmethod
    def can_fit_in_spot(self, spot):
        pass

class Car(Vehicle):
    def __init__(self, license_plate):
        super().__init__(license_plate, VehicleType.CAR)
    
    def can_fit_in_spot(self, spot):
        return spot.type in [VehicleType.CAR, VehicleType.LARGE]

class ParkingLot:
    def __init__(self, total_spots):
        self.spots = [Spot(i) for i in range(total_spots)]
    
    def park_vehicle(self, vehicle):
        for spot in self.spots:
            if vehicle.can_fit_in_spot(spot) and spot.is_available:
                spot.park(vehicle)
                return True
        return False`,
        time_estimate: 25,
        related_skills: ['OOP design', 'system design']
    },
    {
        id: 'p8-hard-047',
        type: 'coding',
        difficulty: 'hard',
        topics: ['recursion', 'trees', 'serialization'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Serialize and Deserialize a binary tree.',
        hints: ['Use preorder traversal', 'Use None to mark null nodes'],
        solution: `class Codec:
    def serialize(self, root):
        def dfs(node):
            if not node:
                return 'null'
            return f"{node.val},{dfs(node.left)},{dfs(node.right)}"
        return dfs(root)
    
    def deserialize(self, data):
        def dfs(nodes):
            val = next(nodes)
            if val == 'null':
                return None
            node = TreeNode(int(val))
            node.left = dfs(nodes)
            node.right = dfs(nodes)
            return node
        return dfs(iter(data.split(',')))`,
        time_estimate: 20,
        related_skills: ['tree serialization', 'preorder traversal']
    },
    {
        id: 'p8-hard-048',
        type: 'coding',
        difficulty: 'hard',
        topics: ['graphs', 'dfs', 'recursion'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Word Search II - Find all words in a board that exist in a dictionary.',
        hints: ['Use Trie for efficient prefix searching', 'DFS with backtracking', 'Prune dead ends early'],
        solution: `class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Solution:
    def findWords(self, board, words):
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word
        
        result = []
        m, n = len(board), len(board[0])
        
        def dfs(i, j, node):
            char = board[i][j]
            if char not in node.children:
                return
            next_node = node.children[char]
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None
            board[i][j] = '#'
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    dfs(ni, nj, next_node)
            board[i][j] = char
        
        for i in range(m):
            for j in range(n):
                if board[i][j] in root.children:
                    dfs(i, j, root)
        
        return result`,
        time_estimate: 35,
        related_skills: ['Trie', 'DFS', 'backtracking']
    },
    {
        id: 'p8-hard-049',
        type: 'coding',
        difficulty: 'hard',
        topics: ['dp', 'optimization'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Wildcard Matching - Implement wildcard pattern matching (* matches any, ? matches any char).',
        hints: ['DP or greedy approach', 'Track last position of * and last matched position'],
        solution: `def is_match(s, p):
    s_idx = p_idx = 0
    star_idx = -1
    s_match = 0
    
    while s_idx < len(s):
        if p_idx < len(p) and (p[p_idx] == s[s_idx] or p[p_idx] == '?'):
            s_idx += 1
            p_idx += 1
        elif p_idx < len(p) and p[p_idx] == '*':
            star_idx = p_idx
            p_idx += 1
            s_match = s_idx
        elif star_idx != -1:
            p_idx = star_idx + 1
            s_match += 1
            s_idx = s_match
        else:
            return False
    
    while p_idx < len(p) and p[p_idx] == '*':
        p_idx += 1
    
    return p_idx == len(p)`,
        time_estimate: 25,
        explanation: 'Greedy two-pointer approach - O(n) time.',
        related_skills: ['string matching', 'greedy algorithms']
    },
    {
        id: 'p8-hard-050',
        type: 'coding',
        difficulty: 'hard',
        topics: ['concurrency', 'python', 'async'],
        phase: 8,
        roles: ['backend_developer', 'devops_engineer', 'cloud_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Implement an async rate limiter using asyncio.',
        hints: ['Track request timestamps', 'Use asyncio sleep to delay'],
        solution: `import asyncio
from time import time

class AsyncRateLimiter:
    def __init__(self, rate, capacity):
        self.rate = rate  # requests per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            # Wait for token to become available
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_update = time()
            return True`,
        time_estimate: 25,
        related_skills: ['asyncio', 'rate limiting']
    },

    // Additional Expert Level Questions
    {
        id: 'p-expert-051',
        type: 'coding',
        difficulty: 'expert',
        topics: ['segment_tree', 'arrays'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'data_engineer'],
        company_focus: ['Google', 'Meta'],
        question: 'Implement a Segment Tree for range sum queries with point updates.',
        hints: ['Tree structure where each node stores sum of segment', 'Build tree bottom-up'],
        solution: `class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = arr[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
    
    def update(self, idx, value):
        pos = self.size + idx
        self.tree[pos] = value
        pos //= 2
        while pos:
            self.tree[pos] = self.tree[2*pos] + self.tree[2*pos+1]
            pos //= 2
    
    def query(self, l, r):
        l += self.size
        r += self.size
        result = 0
        while l <= r:
            if l % 2 == 1:
                result += self.tree[l]
                l += 1
            if r % 2 == 0:
                result += self.tree[r]
                r -= 1
            l //= 2
            r //= 2
        return result`,
        time_estimate: 30,
        related_skills: ['segment tree', 'range queries']
    },
    {
        id: 'p-expert-052',
        type: 'coding',
        difficulty: 'expert',
        topics: ['union_find', 'graphs'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'data_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Implement Union-Find (Disjoint Set Union) with path compression and union by rank.',
        hints: ['Parent array tracks set representative', 'Path compression flattens tree', 'Union by rank balances'],
        solution: `class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)`,
        time_estimate: 20,
        related_skills: ['union-find', 'DSU']
    },
    {
        id: 'p-expert-053',
        type: 'coding',
        difficulty: 'expert',
        topics: ['dp', 'bit_manipulation'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Maximum XOR of two numbers in an array.',
        hints: ['Use Trie with bit representation', 'Maximize XOR by finding complement bits'],
        solution: `class TrieNode:
    def __init__(self):
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, num):
        node = self.root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    
    def find_max_xor(self, num):
        node = self.root
        xor_val = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            toggled = 1 - bit
            if toggled in node.children:
                xor_val |= (1 << i)
                node = node.children[toggled]
            else:
                node = node.children[bit]
        return xor_val

def find_max_xor(nums):
    trie = Trie()
    for num in nums:
        trie.insert(num)
    
    max_xor = 0
    for num in nums:
        max_xor = max(max_xor, trie.find_max_xor(num))
    return max_xor`,
        time_estimate: 30,
        related_skills: ['bitwise trie', 'xor']
    },
    {
        id: 'p-expert-054',
        type: 'coding',
        difficulty: 'expert',
        topics: ['sliding_window', 'deque', 'optimization'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'data_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Sliding Window Maximum using a deque.',
        hints: ['Monotonic deque maintains candidates', 'Remove elements outside window', 'Remove smaller elements'],
        solution: `from collections import deque

def max_sliding_window(nums, k):
    if not nums or k == 0:
        return []
    
    dq = deque()
    result = []
    
    for i, num in enumerate(nums):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove smaller elements from deque
        while dq and nums[dq[-1]] <= num:
            dq.pop()
        
        dq.append(i)
        
        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result`,
        time_estimate: 18,
        explanation: 'O(n) time complexity with monotonic deque.',
        related_skills: ['deque', 'monotonic queue']
    },
    {
        id: 'p-expert-055',
        type: 'coding',
        difficulty: 'expert',
        topics: ['dp', 'trees'],
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'python_developer'],
        company_focus: ['Google', 'Meta'],
        question: 'Count unique BSTs (Catalan number) for n nodes.',
        hints: ['Use DP with Catalan recurrence', 'C(n) = sum(C(i) * C(n-1-i))'],
        solution: `def num_trees(n):
    if n <= 1:
        return 1
    catalan = [0] * (n + 1)
    catalan[0] = 1
    
    for nodes in range(1, n + 1):
        total = 0
        for left in range(nodes):
            right = nodes - 1 - left
            total += catalan[left] * catalan[right]
        catalan[nodes] = total
    
    return catalan[n]`,
        time_estimate: 15,
        explanation: 'O(n²) time using DP with Catalan numbers.'
    }
];

// ==================== BEHAVIORAL QUESTIONS BY PHASE ====================

export const behavioralQuestions: BehavioralQuestion[] = [
    // Phase 1 - Getting Started / Learning
    {
        id: 'beh-p1-001',
        category: 'Learning Agility',
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'fullstack_developer', 'software_engineer'],
        question: 'Tell me about a time you had to learn a new programming language or skill quickly. How did you approach it?',
        tips: [
            'Show your learning process',
            'Mention specific resources you used',
            'Highlight how you applied new knowledge',
            'Quantify the results'
        ],
        follow_ups: [
            'What was the hardest part?',
            'How do you approach learning something new?'
        ],
        star_example: {
            situation: 'Our team needed to migrate from JavaScript to Python for a new project',
            task: 'Learn Python fundamentals and best practices within 2 weeks',
            action: 'Created a structured learning plan using online courses, built small projects daily, joined Python community forums',
            result: 'Contributed to production code within deadline, project delivered on time with positive code review feedback'
        }
    },
    {
        id: 'beh-p1-002',
        category: 'Problem Solving',
        phase: 1,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'software_engineer'],
        question: 'Describe a bug that was difficult to find and how you eventually solved it.',
        tips: [
            'Explain your debugging process',
            'Show systematic thinking',
            'Mention tools or techniques used',
            'Share what you learned'
        ],
        follow_ups: [
            'How long did it take?',
            'What would you do differently?'
        ]
    },

    // Phase 2 - Technical Growth
    {
        id: 'beh-p2-001',
        category: 'Technical Excellence',
        phase: 2,
        roles: ['backend_developer', 'software_engineer', 'data_engineer', 'devops_engineer'],
        question: 'Walk me through how you would approach optimizing a slow database query.',
        tips: [
            'Show systematic approach',
            'Mention EXPLAIN plans and indexing',
            'Discuss query structure optimization',
            'Consider trade-offs'
        ],
        follow_ups: [
            'How do you measure improvement?',
            'What tools do you use for profiling?'
        ]
    },
    {
        id: 'beh-p2-002',
        category: 'Collaboration',
        phase: 2,
        roles: ['python_developer', 'backend_developer', 'fullstack_developer', 'software_engineer'],
        question: 'Tell me about a time you had a technical disagreement with a team member. How did you resolve it?',
        tips: [
            'Focus on professional resolution',
            'Show respect for different perspectives',
            'Mention data-driven decisions',
            'Highlight teamwork'
        ],
        follow_ups: [
            'What was the outcome?',
            'Would you handle it differently now?'
        ]
    },

    // Phase 3 - Advanced Skills
    {
        id: 'beh-p3-001',
        category: 'Leadership',
        phase: 3,
        roles: ['software_engineer', 'backend_developer', 'fullstack_developer', 'devops_engineer'],
        question: 'Describe a time when you took initiative to improve a process or system at your company.',
        tips: [
            'Identify an inefficiency',
            'Propose and implement solution',
            'Show impact and metrics',
            'Demonstrate ownership'
        ],
        follow_ups: [
            'What was the ROI?',
            'Did you face any resistance?'
        ]
    },
    {
        id: 'beh-p3-002',
        category: 'Mentorship',
        phase: 3,
        roles: ['software_engineer', 'backend_developer', 'data_scientist', 'ml_engineer'],
        question: 'Tell me about a time you helped a junior developer grow. What approach did you take?',
        tips: [
            'Show patience and empathy',
            'Tailor approach to their needs',
            'Balance guidance with autonomy',
            'Highlight their growth'
        ],
        follow_ups: [
            'How do you measure mentorship success?',
            'What have you learned from mentoring?'
        ]
    },

    // Phase 4 - AI/ML Projects
    {
        id: 'beh-p4-001',
        category: 'Innovation',
        phase: 4,
        roles: ['data_scientist', 'ml_engineer', 'python_developer'],
        question: 'Describe a project where you used AI or machine learning. What was your approach?',
        tips: [
            'Explain problem understanding',
            'Show model selection rationale',
            'Discuss evaluation metrics',
            'Address limitations'
        ],
        follow_ups: [
            'What was the accuracy?',
            'How would you improve it?'
        ]
    },
    {
        id: 'beh-p4-002',
        category: 'Research',
        phase: 4,
        roles: ['data_scientist', 'ml_engineer', 'cloud_engineer'],
        question: 'Tell me about a time you had to research a new technology or methodology for a project.',
        tips: [
            'Show research methodology',
            'Evaluate different options',
            'Make data-driven recommendations',
            'Show implementation success'
        ],
        follow_ups: [
            'How do you stay updated?',
            'What sources do you trust?'
        ]
    },

    // Phase 5 - Professional Growth
    {
        id: 'beh-p5-001',
        category: 'Conflict Resolution',
        phase: 5,
        roles: ['software_engineer', 'backend_developer', 'fullstack_developer', 'devops_engineer'],
        question: 'Describe a situation where you had to balance technical debt with shipping features.',
        tips: [
            'Show understanding of trade-offs',
            'Quantify the cost of technical debt',
            'Present a balanced approach',
            'Highlight stakeholder management'
        ],
        follow_ups: [
            'How do you prioritize?',
            'What was the outcome?'
        ]
    },
    {
        id: 'beh-p5-002',
        category: 'Impact',
        phase: 5,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'software_engineer'],
        question: 'Tell me about the most impactful project you\'ve worked on. What was your contribution?',
        tips: [
            'Be specific about your role',
            'Quantify impact with metrics',
            'Show ownership and initiative',
            'Connect to company goals'
        ],
        follow_ups: [
            'What challenges did you face?',
            'What would you have done differently?'
        ]
    },

    // Phase 6 - Interview Prep
    {
        id: 'beh-p6-001',
        category: 'Communication',
        phase: 6,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'fullstack_developer', 'software_engineer'],
        question: 'Explain a complex technical concept to someone without a technical background.',
        tips: [
            'Use analogies and simple language',
            'Check understanding frequently',
            'Be patient and adaptable',
            'Avoid jargon'
        ],
        follow_ups: [
            'How do you know they understand?',
            'What\'s your go-to analogy?'
        ]
    },
    {
        id: 'beh-p6-002',
        category: 'Failure',
        phase: 6,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'software_engineer'],
        question: 'Tell me about a project that didn\'t go as planned. What did you learn?',
        tips: [
            'Be honest and humble',
            'Focus on learning and growth',
            'Show resilience',
            'Demonstrate self-awareness'
        ],
        follow_ups: [
            'What would you do differently?',
            'How did this experience change you?'
        ]
    },

    // Phase 7 - Leadership
    {
        id: 'beh-p7-001',
        category: 'Strategic Thinking',
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'cloud_engineer', 'devops_engineer'],
        question: 'Describe a time you had to make a decision with incomplete information.',
        tips: [
            'Show decision-making process',
            'Take calculated risks',
            'Show adaptability',
            'Learn from outcomes'
        ],
        follow_ups: [
            'How do you handle uncertainty?',
            'What information would you want now?'
        ]
    },
    {
        id: 'beh-p7-002',
        category: 'Project Management',
        phase: 7,
        roles: ['software_engineer', 'backend_developer', 'fullstack_developer', 'devops_engineer'],
        question: 'Tell me about a time you had to manage multiple competing priorities.',
        tips: [
            'Show prioritization framework',
            'Communicate with stakeholders',
            'Demonstrate time management',
            'Show results'
        ],
        follow_ups: [
            'How do you say no?',
            'What tools do you use?'
        ]
    },

    // Phase 8 - Career & Entrepreneurship
    {
        id: 'beh-p8-001',
        category: 'Entrepreneurship',
        phase: 8,
        roles: ['software_engineer', 'backend_developer', 'cloud_engineer', 'devops_engineer'],
        question: 'Describe a time you identified an opportunity and took initiative to pursue it.',
        tips: [
            'Show market awareness',
            'Demonstrate initiative',
            'Highlight calculated risk-taking',
            'Show execution ability'
        ],
        follow_ups: [
            'What was the outcome?',
            'What did you learn?'
        ]
    },
    {
        id: 'beh-p8-002',
        category: 'Company Fit',
        phase: 8,
        roles: ['python_developer', 'backend_developer', 'data_scientist', 'fullstack_developer', 'software_engineer', 'ml_engineer', 'data_engineer', 'devops_engineer', 'cloud_engineer'],
        question: 'Why do you want to work at our company? What interests you about our mission?',
        tips: [
            'Research the company thoroughly',
            'Connect personal values with mission',
            'Show genuine enthusiasm',
            'Be specific about opportunities'
        ],
        follow_ups: [
            'What questions do you have?',
            'Where do you see yourself in 5 years?'
        ]
    }
];

// ==================== SYSTEM DESIGN QUESTIONS BY PHASE ====================

export const systemDesignQuestions: SystemDesignQuestion[] = [
    // Phase 1-2: Basic Systems
    {
        id: 'sys-basic-001',
        topic: 'URL Shortener',
        question: 'Design a URL shortening service like bit.ly that can handle 10K requests per second.',
        phase: 2,
        roles: ['backend_developer', 'api_developer', 'software_engineer'],
        complexity_level: 'low',
        estimated_time: 30,
        key_points: [
            'API Design: POST /shorten, GET /{short_code}',
            'Database: Hash-based encoding (base62) for short codes',
            'Scalability: Sharding, caching with Redis, CDN for redirects',
            'Consider: Collision handling, analytics, expiration',
            'Traffic: Handle 10K-100K requests per second'
        ],
        related_topics: ['database design', 'caching', 'API design']
    },
    {
        id: 'sys-basic-002',
        topic: 'Pastebin',
        question: 'Design a pastebin service where users can share text snippets.',
        phase: 2,
        roles: ['backend_developer', 'api_developer', 'fullstack_developer'],
        complexity_level: 'low',
        estimated_time: 25,
        key_points: [
            'Storage: Object storage (S3) for content, metadata in database',
            'Retrieval: Unique ID generation, content-addressable storage option',
            'Expiration: Time-based deletion, size limits',
            'API: Upload, retrieve, delete operations'
        ]
    },

    // Phase 3-4: Distributed Systems
    {
        id: 'sys-med-001',
        topic: 'Social Feed',
        question: 'Design a news feed system like Facebook/Twitter that handles millions of users.',
        phase: 4,
        roles: ['backend_developer', 'data_engineer', 'software_engineer'],
        complexity_level: 'medium',
        estimated_time: 45,
        key_points: [
            'Fan-out approach: Push (for celebrities) vs Pull (for regular users)',
            'Ranking algorithms: EdgeRank-style scoring based on affinity, weight, time',
            'Storage: NoSQL for posts, graph DB for relationships',
            'Caching: Multiple levels (feed cache, user activity cache)',
            'Consider: Real-time updates, spam detection, content moderation'
        ],
        related_topics: ['caching', 'graph databases', 'ranking algorithms']
    },
    {
        id: 'sys-med-002',
        topic: 'Chat System',
        question: 'Design a real-time chat application like WhatsApp with 1B users.',
        phase: 4,
        roles: ['backend_developer', 'cloud_engineer', 'software_engineer'],
        complexity_level: 'medium',
        estimated_time: 40,
        key_points: [
            'WebSocket connections for real-time bidirectional communication',
            'Message queue: Kafka/RabbitMQ for message handling and ordering',
            'Storage: Message history (Cassandra), offline message delivery',
            'Presence system: Track online/offline status in real-time',
            'Security: End-to-end encryption considerations',
            'Scalability: Message routing, group chat handling'
        ],
        related_topics: ['websocket', 'message queues', 'real-time systems']
    },
    {
        id: 'sys-med-003',
        topic: 'Rate Limiter',
        question: 'Design a distributed rate limiter that can handle traffic for multiple services.',
        phase: 4,
        roles: ['backend_developer', 'devops_engineer', 'cloud_engineer'],
        complexity_level: 'medium',
        estimated_time: 35,
        key_points: [
            'Algorithms: Token bucket, Leaky bucket, Sliding window',
            'Storage: Redis for distributed state',
            'Scalability: Sharding, eventual consistency trade-offs',
            'APIs: Per-user, per-IP, per-API key rate limiting',
            'Monitoring: Logging, alerting on rate limit hits'
        ]
    },

    // Phase 5-6: Complex Systems
    {
        id: 'sys-high-001',
        topic: 'Ride Sharing',
        question: 'Design a ride-sharing service like Uber with real-time matching and pricing.',
        phase: 6,
        roles: ['backend_developer', 'cloud_engineer', 'software_engineer'],
        complexity_level: 'high',
        estimated_time: 45,
        key_points: [
            'Real-time tracking: GPS integration, WebSocket updates',
            'Matching algorithm: Rider-Driver pairing with ETA optimization',
            'Dynamic pricing: Surge pricing based on supply/demand',
            'Route optimization: Map services, traffic integration',
            'Safety: Emergency features, trip tracking, verification'
        ],
        related_topics: ['real-time systems', 'matching algorithms', 'geospatial']
    },
    {
        id: 'sys-high-002',
        topic: 'Video Streaming',
        question: 'Design a video streaming service like YouTube handling petabytes of data.',
        phase: 6,
        roles: ['backend_developer', 'cloud_engineer', 'data_engineer'],
        complexity_level: 'high',
        estimated_time: 50,
        key_points: [
            'Storage: Distributed file systems, content delivery network (CDN)',
            'Transcoding: Multiple resolutions, adaptive bitrate streaming (HLS/DASH)',
            'Caching: Multi-level CDN, edge caching',
            'Recommendation: Collaborative filtering, content-based filtering',
            'Scalability: Chunked uploads, parallel processing'
        ],
        related_topics: ['CDN', 'video processing', 'recommendations']
    },
    {
        id: 'sys-high-003',
        topic: 'Search Engine',
        question: 'Design a search engine like Google that indexes billions of web pages.',
        phase: 6,
        roles: ['backend_developer', 'data_engineer', 'software_engineer'],
        complexity_level: 'high',
        estimated_time: 50,
        key_points: [
            'Crawling: Distributed crawlers, politeness rules, freshness',
            'Indexing: Inverted index, document processing, ranking',
            'Ranking: PageRank, relevance scoring, machine learning',
            'Query processing: Spelling correction, autocomplete',
            'Architecture: MapReduce for batch, real-time for updates'
        ],
        related_topics: ['inverted index', 'ranking algorithms', 'distributed systems']
    },

    // Phase 7-8: Enterprise Systems
    {
        id: 'sys-enterprise-001',
        topic: 'E-Commerce Platform',
        question: 'Design a scalable e-commerce platform like Amazon during Prime Day traffic.',
        phase: 8,
        roles: ['backend_developer', 'devops_engineer', 'cloud_engineer', 'software_engineer'],
        complexity_level: 'high',
        estimated_time: 60,
        key_points: [
            'Inventory: Distributed inventory management, consistency patterns',
            'Checkout: Payment processing, order management, fraud detection',
            'Caching: Product data, user sessions, cart',
            'Search: Product search with filters, ranking',
            'Resilience: Circuit breakers, graceful degradation',
            'Analytics: Clickstream, A/B testing, personalization'
        ],
        related_topics: ['distributed transactions', 'payment systems', 'analytics']
    },
    {
        id: 'sys-enterprise-002',
        topic: 'Distributed Cache',
        question: 'Design a distributed caching system like Redis or Memcached.',
        phase: 8,
        roles: ['backend_developer', 'devops_engineer', 'cloud_engineer'],
        complexity_level: 'high',
        estimated_time: 50,
        key_points: [
            'Data model: Key-value store, data structures',
            'Consistency: Master-replica, eventual consistency',
            'Partitioning: Consistent hashing for sharding',
            'Eviction: LRU, LFU, TTL-based expiration',
            'Persistence: RDB snapshots, AOF logging',
            'Scalability: Horizontal scaling, read replicas'
        ],
        related_topics: ['distributed systems', 'caching algorithms', 'persistence']
    },
    {
        id: 'sys-enterprise-003',
        topic: 'Notification System',
        question: 'Design a notification system that can send billions of notifications daily.',
        phase: 8,
        roles: ['backend_developer', 'cloud_engineer', 'devops_engineer'],
        complexity_level: 'high',
        estimated_time: 45,
        key_points: [
            'Channels: Push, email, SMS, in-app',
            'Queue: Kafka/Pulsar for high-throughput',
            'Delivery: Retry mechanisms, dead letter queues',
            'Personalization: Template system, user preferences',
            'Analytics: Delivery rates, engagement tracking',
            'Rate limiting: Per-user, per-channel throttling'
        ]
    }
];

// ==================== SQL QUESTIONS BY PHASE ====================

export const sqlQuestions: InterviewQuestion[] = [
    // Easy SQL
    {
        id: 'sql-easy-001',
        type: 'sql',
        difficulty: 'easy',
        topics: ['sql', 'databases'],
        phase: 2,
        roles: ['backend_developer', 'data_scientist', 'data_engineer', 'software_engineer'],
        question: 'Write a SQL query to find all employees with salary greater than 50000.',
        hints: ['Use WHERE clause with comparison operator', 'Assume table name is employees'],
        solution: `SELECT * FROM employees WHERE salary > 50000;`,
        time_estimate: 3,
        related_skills: ['basic filtering', 'WHERE clause']
    },
    {
        id: 'sql-easy-002',
        type: 'sql',
        difficulty: 'easy',
        topics: ['sql', 'databases', 'aggregation'],
        phase: 2,
        roles: ['backend_developer', 'data_scientist', 'data_engineer'],
        question: 'Write a SQL query to count the number of employees in each department.',
        hints: ['Use GROUP BY clause', 'Use COUNT() aggregate function'],
        solution: `SELECT department, COUNT(*) as employee_count 
FROM employees 
GROUP BY department;`,
        time_estimate: 4,
        related_skills: ['aggregation', 'GROUP BY']
    },
    {
        id: 'sql-easy-003',
        type: 'sql',
        difficulty: 'easy',
        topics: ['sql', 'databases', 'joins'],
        phase: 2,
        roles: ['backend_developer', 'data_scientist', 'data_engineer', 'software_engineer'],
        question: 'Write a SQL query to join employees with departments table.',
        hints: ['Use INNER JOIN or LEFT JOIN', 'Match on department_id'],
        solution: `SELECT e.*, d.department_name 
FROM employees e
JOIN departments d ON e.department_id = d.id;`,
        time_estimate: 5,
        related_skills: ['JOIN operations']
    },

    // Medium SQL
    {
        id: 'sql-med-001',
        type: 'sql',
        difficulty: 'medium',
        topics: ['sql', 'databases', 'subqueries'],
        phase: 5,
        roles: ['backend_developer', 'data_engineer', 'software_engineer'],
        company_focus: ['Amazon', 'Google'],
        question: 'Write a SQL query to find employees earning more than their managers.',
        hints: ['Self-join the Employee table', 'Match employee.manager_id with manager.id'],
        solution: `SELECT e1.name as Employee, e1.salary as Employee_Salary
FROM Employee e1
JOIN Employee e2 ON e1.manager_id = e2.id
WHERE e1.salary > e2.salary;`,
        time_estimate: 8,
        related_skills: ['self-join', 'subqueries']
    },
    {
        id: 'sql-med-002',
        type: 'sql',
        difficulty: 'medium',
        topics: ['sql', 'databases', 'window_functions'],
        phase: 5,
        roles: ['data_scientist', 'data_engineer', 'backend_developer'],
        company_focus: ['Amazon', 'Google', 'Meta'],
        question: 'Write a SQL query to find the second highest salary in each department.',
        hints: ['Use window functions like RANK() or DENSE_RANK()', 'Partition by department'],
        solution: `SELECT department, salary
FROM (
    SELECT department, salary,
           DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rnk
    FROM employees
) ranked
WHERE rnk = 2;`,
        time_estimate: 10,
        related_skills: ['window functions', 'CTE']
    },
    {
        id: 'sql-med-003',
        type: 'sql',
        difficulty: 'medium',
        topics: ['sql', 'databases', 'cte'],
        phase: 5,
        roles: ['backend_developer', 'data_engineer', 'data_scientist'],
        question: 'Write a SQL query using CTE to find departments with more than 10 employees.',
        hints: ['Use Common Table Expression (CTE)', 'Filter with HAVING clause'],
        solution: `WITH dept_count AS (
    SELECT department_id, COUNT(*) as emp_count
    FROM employees
    GROUP BY department_id
)
SELECT d.name, dc.emp_count
FROM departments d
JOIN dept_count dc ON d.id = dc.department_id
WHERE dc.emp_count > 10;`,
        time_estimate: 8,
        related_skills: ['CTE', 'aggregation']
    }
];

// ==================== ROLE-BASED UTILITY FUNCTIONS ====================

// Get questions by career role
export function getQuestionsByRole(role: CareerRole): InterviewQuestion[] {
    return codingQuestions.filter(q => 
        q.roles.includes(role) || (q.secondary_roles && q.secondary_roles.includes(role))
    );
}

// Get questions by role and difficulty
export function getQuestionsByRoleAndDifficulty(role: CareerRole, difficulty: Difficulty): InterviewQuestion[] {
    return codingQuestions.filter(q => 
        (q.roles.includes(role) || (q.secondary_roles && q.secondary_roles.includes(role))) &&
        q.difficulty === difficulty
    );
}

// Get questions by role and phase
export function getQuestionsByRoleAndPhase(role: CareerRole, phase: number): InterviewQuestion[] {
    return codingQuestions.filter(q => 
        (q.roles.includes(role) || (q.secondary_roles && q.secondary_roles.includes(role))) &&
        q.phase === phase
    );
}

// Get behavioral questions by role
export function getBehavioralByRole(role: CareerRole): BehavioralQuestion[] {
    return behavioralQuestions.filter(q => 
        q.roles.includes(role)
    );
}

// Get system design questions by role
export function getSystemDesignByRole(role: CareerRole): SystemDesignQuestion[] {
    return systemDesignQuestions.filter(q => 
        q.roles.includes(role)
    );
}

// Get all questions for a role (all types)
export function getAllQuestionsByRole(role: CareerRole): {
    coding: InterviewQuestion[];
    behavioral: BehavioralQuestion[];
    systemDesign: SystemDesignQuestion[];
} {
    return {
        coding: getQuestionsByRole(role),
        behavioral: getBehavioralByRole(role),
        systemDesign: getSystemDesignByRole(role)
    };
}

// Get question statistics by role
export function getQuestionStatsByRole(role: CareerRole): {
    total: number;
    easy: number;
    medium: number;
    hard: number;
    expert: number;
    phases: number[];
} {
    const roleQuestions = getQuestionsByRole(role);
    const phases = [...new Set(roleQuestions.map(q => q.phase))].sort();
    
    return {
        total: roleQuestions.length,
        easy: roleQuestions.filter(q => q.difficulty === 'easy').length,
        medium: roleQuestions.filter(q => q.difficulty === 'medium').length,
        hard: roleQuestions.filter(q => q.difficulty === 'hard').length,
        expert: roleQuestions.filter(q => q.difficulty === 'expert').length,
        phases
    };
}

// Get recommended phases for a role
export function getRecommendedPhasesForRole(role: CareerRole): { phase: number; priority: 'essential' | 'recommended' | 'optional' }[] {
    const roleQuestionPhases = [...new Set(getQuestionsByRole(role).map(q => q.phase))].sort();
    const allPhases = [1, 2, 3, 4, 5, 6, 7, 8];
    
    const phaseQuestionCounts: { [key: number]: number } = {};
    getQuestionsByRole(role).forEach(q => {
        phaseQuestionCounts[q.phase] = (phaseQuestionCounts[q.phase] || 0) + 1;
    });
    
    return allPhases.map(phase => {
        const count = phaseQuestionCounts[phase] || 0;
        const priority = roleQuestionPhases.includes(phase) 
            ? (count >= 10 ? 'essential' : count >= 5 ? 'recommended' : 'optional')
            : 'optional';
        return { phase, priority };
    });
}

// Helper to get all coding questions
export function getAllCodingQuestions(): InterviewQuestion[] {
    return codingQuestions;
}

// Get questions by phase
export function getQuestionsByPhase(phase: number): InterviewQuestion[] {
    return codingQuestions.filter(q => q.phase === phase);
}

// Get questions by difficulty
export function getQuestionsByDifficulty(difficulty: Difficulty): InterviewQuestion[] {
    return codingQuestions.filter(q => q.difficulty === difficulty);
}

// Get questions by topic
export function getQuestionsByTopic(topic: Topic): InterviewQuestion[] {
    return codingQuestions.filter(q => q.topics.includes(topic));
}

// Get random questions
export function getRandomQuestions(count: number, type?: QuestionType): InterviewQuestion[] {
    let questions: InterviewQuestion[];
    if (type) {
        questions = codingQuestions.filter(q => q.type === type);
    } else {
        questions = codingQuestions;
    }
    const shuffled = [...questions].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, Math.min(count, questions.length));
}

// Get questions by phase and difficulty
export function getQuestionsByPhaseAndDifficulty(phase: number, difficulty: Difficulty): InterviewQuestion[] {
    return codingQuestions.filter(q => q.phase === phase && q.difficulty === difficulty);
}

// Get all phases covered
export function getAllPhases(): number[] {
    return [...new Set(codingQuestions.map(q => q.phase))].sort();
}

// Get question statistics by phase
export function getQuestionStats(): { phase: number; total: number; easy: number; medium: number; hard: number; expert: number }[] {
    const phases = getAllPhases();
    return phases.map(phase => {
        const phaseQuestions = getQuestionsByPhase(phase);
        return {
            phase,
            total: phaseQuestions.length,
            easy: phaseQuestions.filter(q => q.difficulty === 'easy').length,
            medium: phaseQuestions.filter(q => q.difficulty === 'medium').length,
            hard: phaseQuestions.filter(q => q.difficulty === 'hard').length,
            expert: phaseQuestions.filter(q => q.difficulty === 'expert').length
        };
    });
}

// Behavioral questions by phase
export function getBehavioralQuestionsByPhase(phase: number): BehavioralQuestion[] {
    return behavioralQuestions.filter(q => q.phase === phase);
}

// System design by phase
export function getSystemDesignByPhase(phase: number): SystemDesignQuestion[] {
    return systemDesignQuestions.filter(q => q.phase === phase);
}

// Get all topic categories
export const questionCategories = {
    coding: ['arrays', 'strings', 'linked_lists', 'trees', 'graphs', 'dp', 'recursion', 
             'sorting', 'searching', 'hash', 'stacks_queues', 'heap', 'bit_manipulation', 
             'sql', 'ml_ai', 'python', 'oop', 'generators', 'decorators', 'async'],
    behavioral: ['Learning Agility', 'Problem Solving', 'Technical Excellence', 'Collaboration',
                 'Leadership', 'Mentorship', 'Innovation', 'Research', 'Conflict Resolution',
                 'Impact', 'Communication', 'Failure', 'Strategic Thinking', 'Project Management',
                 'Entrepreneurship', 'Company Fit'],
    system_design: ['URL Shortener', 'Pastebin', 'Social Feed', 'Chat System', 'Rate Limiter',
                    'Ride Sharing', 'Video Streaming', 'Search Engine', 'E-Commerce Platform',
                    'Distributed Cache', 'Notification System']
};

// Export all data
export default {
    codingQuestions,
    behavioralQuestions,
    systemDesignQuestions,
    sqlQuestions,
    CAREER_ROLES,
    // Original functions
    getAllCodingQuestions,
    getQuestionsByPhase,
    getQuestionsByDifficulty,
    getQuestionsByTopic,
    getRandomQuestions,
    getQuestionsByPhaseAndDifficulty,
    getBehavioralQuestionsByPhase,
    getSystemDesignByPhase,
    getQuestionStats,
    getAllPhases,
    questionCategories,
    // Role-based functions
    getQuestionsByRole,
    getQuestionsByRoleAndDifficulty,
    getQuestionsByRoleAndPhase,
    getBehavioralByRole,
    getSystemDesignByRole,
    getAllQuestionsByRole,
    getQuestionStatsByRole,
    getRecommendedPhasesForRole
};