import { Phase, TopicFolder } from '../types';

// Phase 1 - Python Foundations (Folders 01-14)
// Note: Modules & Libraries moved to Phase 3 for better topic grouping

const folders: TopicFolder[] = [
    {
        id: 'folder-1',
        name: '01. Fundamentals',
        description: 'Master Python syntax, variables, data types, and basic operations',
        topics: [
            {
                id: '1-1-1',
                title: 'Python Fundamentals',
                description: 'Learn Python syntax, variables, data types, and basic operations',
                duration: '3-4 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/01_fundamentals/step1-python-fundamentals.md'
            },
            {
                id: '1-1-2',
                title: 'Python Fundamentals Practice',
                description: 'Hands-on exercises to reinforce Python basics',
                duration: '2-3 hours',
                type: 'practice',
                markdownPath: '/courses/phase-1-python-foundations/01_fundamentals/step2-python-fundamentals-pratice.md'
            },
            {
                id: '1-1-3',
                title: 'Python Fundamentals Cheat Codes',
                description: 'Quick reference guide for Python essentials',
                duration: '1 hour',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/01_fundamentals/step3-python-fundamentals-cheatcodes.md'
            }
        ]
    },
    {
        id: 'folder-2',
        name: '02. Data Structures',
        description: 'Lists, tuples, dictionaries, and sets in Python',
        topics: [
            {
                id: '1-2-1',
                title: 'Data Structures Guide',
                description: 'Comprehensive guide to Python data structures',
                duration: '3 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/02_data/step1-data-structures-guide.md'
            },
            {
                id: '1-2-2',
                title: 'Data Structures Practice',
                description: 'Practice problems for mastering data structures',
                duration: '2 hours',
                type: 'practice',
                markdownPath: '/courses/phase-1-python-foundations/02_data/step2-data-structures-practice_questions.md'
            }
        ]
    },
    {
        id: 'folder-3',
        name: '03. Control Structures',
        description: 'If statements, loops, and control flow in Python',
        topics: [
            {
                id: '1-3-1',
                title: 'Control Structures Guide',
                description: 'Complete guide to if statements, loops, and control flow',
                duration: '2 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/03_control_structures/step1-control_structures_complete_guide.md'
            },
            {
                id: '1-3-2',
                title: 'Control Structures Practice',
                description: 'Practice problems for loops and conditionals',
                duration: '2 hours',
                type: 'practice',
                markdownPath: '/courses/phase-1-python-foundations/03_control_structures/step2-control_structures_practice_questions.md'
            }
        ]
    },
    {
        id: 'folder-4',
        name: '04. File Handling',
        description: 'Read and write files, work with different file formats',
        topics: [
            {
                id: '1-4-1',
                title: 'File Handling Guide',
                description: 'Learn file I/O operations in Python',
                duration: '2 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/04_file_handling/step1-file_handling_complete_guide.md'
            },
            {
                id: '1-4-2',
                title: 'File Handling Practice',
                description: 'Practice file operations',
                duration: '2 hours',
                type: 'practice',
                markdownPath: '/courses/phase-1-python-foundations/04_file_handling/step2-file_handling_practice_questions.md'
            }
        ]
    },
    {
        id: 'folder-5',
        name: '05. Object-Oriented Programming',
        description: 'Classes, objects, inheritance, and polymorphism',
        topics: [
            {
                id: '1-5-1',
                title: 'OOP Complete Guide',
                description: 'Master classes, objects, and OOP principles',
                duration: '4 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/05_oops/step1-oop_complete_guide.md'
            },
            {
                id: '1-5-2',
                title: 'OOP Practice',
                description: 'Build classes and practice OOP concepts',
                duration: '3 hours',
                type: 'practice',
                markdownPath: '/courses/phase-1-python-foundations/05_oops/step2-oop_practice_questions.md'
            }
        ]
    },
    {
        id: 'folder-6',
        name: '06. Error Handling & Debugging',
        description: 'Try-except blocks, debugging techniques',
        topics: [
            {
                id: '1-6-1',
                title: 'Error Handling Guide',
                description: 'Master exception handling and debugging',
                duration: '2 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/06_error_handling/step1-error_handling_debugging_complete_guide.md'
            },
            {
                id: '1-6-2',
                title: 'Error Handling Practice',
                description: 'Practice writing robust error-handling code',
                duration: '2 hours',
                type: 'practice',
                markdownPath: '/courses/phase-1-python-foundations/06_error_handling/step2-error_handling_debugging_practice_questions.md'
            }
        ]
    },
    {
        id: 'folder-7',
        name: '07. Practical Projects',
        description: 'Build real-world Python projects',
        topics: [
            {
                id: '1-7-1',
                title: 'Practical Projects Guide',
                description: 'Step-by-step real-world project tutorials',
                duration: '5 hours',
                type: 'project',
                markdownPath: '/courses/phase-1-python-foundations/07_practical_projects/step1-practical_projects_complete_guide.md'
            },
            {
                id: '1-7-2',
                title: 'Project Practice Questions',
                description: 'Additional project challenges',
                duration: '3 hours',
                type: 'project',
                markdownPath: '/courses/phase-1-python-foundations/07_practical_projects/step2-practical_projects_practice_questions.md'
            }
        ]
    },
    {
        id: 'folder-8',
        name: '08. AI Tools Integration',
        description: 'Integrate AI tools and APIs into Python projects',
        topics: [
            {
                id: '1-8-1',
                title: 'AI Tools Integration',
                description: 'Connect Python to AI services and APIs',
                duration: '3 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/08_ai_tools_integration/step1-ai_tools_integration_complete_guide.md'
            }
        ]
    },
    {
        id: 'folder-9',
        name: '09. Automation',
        description: 'Automate tasks with Python scripts',
        topics: [
            {
                id: '1-9-1',
                title: 'Automation Projects',
                description: 'Build automation scripts and bots',
                duration: '3 hours',
                type: 'project',
                markdownPath: '/courses/phase-1-python-foundations/09_automation/step1-automation_projects_complete_guide.md'
            }
        ]
    },
    {
        id: 'folder-10',
        name: '10. Industry Applications',
        description: 'Real-world Python applications in different industries',
        topics: [
            {
                id: '1-10-1',
                title: 'Industry Applications',
                description: 'Explore Python use cases across industries',
                duration: '2 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/10_industry_application/step1-industry_applications_complete_guide.md'
            }
        ]
    },
    {
        id: 'folder-11',
        name: '11. Sustainability Use Cases',
        description: 'Python for environmental and sustainability projects',
        topics: [
            {
                id: '1-11-1',
                title: 'Sustainability Projects',
                description: 'Use Python for environmental impact',
                duration: '2 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/11_sustainability_use_cases/step1-sustainability_use_cases_complete_guide.md'
            }
        ]
    },
    {
        id: 'folder-12',
        name: '12. Problem Solving Mindset',
        description: 'Develop algorithmic thinking and problem-solving skills',
        topics: [
            {
                id: '1-12-1',
                title: 'Problem Solving',
                description: 'Build critical thinking and problem-solving skills',
                duration: '2 hours',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/12_problem_solving_mindset/step1-problem_solving_mindset_complete_guide.md'
            }
        ]
    },
    {
        id: 'folder-13',
        name: '13. Comprehensive Quizzes',
        description: 'Test your knowledge with comprehensive assessments',
        topics: [
            {
                id: '1-13-1',
                title: 'Comprehensive Assessment',
                description: 'Full assessment covering all Python fundamentals',
                duration: '1 hour',
                type: 'assessment',
                markdownPath: '/courses/phase-1-python-foundations/13_comprehensive_quizzes/step1-comprehensive_quizzes_assessment.md'
            }
        ]
    },
    {
        id: 'folder-14',
        name: '14. Cheat Codes',
        description: 'Quick reference for all Python concepts',
        topics: [
            {
                id: '1-14-1',
                title: 'Python Cheat Sheet',
                description: 'Complete Python reference guide',
                duration: '30 min',
                type: 'theory',
                markdownPath: '/courses/phase-1-python-foundations/14_cheat_codes/python_cheat_sheets_quick_reference.md'
            }
        ]
    }
];

// Flatten all topics for backward compatibility
const allTopics = folders.flatMap(folder => folder.topics);

export const phase1: Phase = {
    id: 'phase-1',
    number: 1,
    title: 'Python Foundations',
    description: 'Master Python fundamentals from scratch - variables, data types, control flow, functions, and OOP basics. Build your first CLI application.',
    duration: '4-6 weeks',
    skills: ['Python Syntax', 'Variables & Types', 'Control Flow', 'Functions', 'OOP Basics', 'File I/O'],
    topics: allTopics,
    folders: folders
};
