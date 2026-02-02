import { Phase, TopicFolder } from '../types';

// Phase 3: Advanced Python Mastery
// Comprehensive coverage of advanced Python concepts, professional development practices, 
// libraries, DevOps, testing, MLOps, and production-ready programming

const folders: TopicFolder[] = [
    {
        "id": "folder-3-01-modules-libraries",
        "name": "01. Modules & Libraries",
        "description": "Master Python modules, packages, and standard library essentials for professional development",
        "topics": [
            {
                "id": "3-01-0",
                "title": "Modules & Packages Complete Guide",
                "description": "Deep dive into Python modules, packages, import systems, and creating reusable code libraries",
                "duration": "2-3 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/01-modules-libraries/step1-modules_packages_complete_guide.md"
            },
            {
                "id": "3-01-1",
                "title": "Modules & Packages Practice",
                "description": "Hands-on exercises for creating, organizing, and distributing Python packages",
                "duration": "2 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/01-modules-libraries/step2-modules_packages_practice_questions.md"
            },
            {
                "id": "3-01-2",
                "title": "Modules & Libraries Cheat Sheet",
                "description": "Quick reference for module patterns, import tricks, and package best practices",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/01-modules-libraries/cheatsheet.md"
            },
            {
                "id": "3-01-3",
                "title": "Modules & Libraries Interview Prep",
                "description": "Common interview questions about Python module system, packaging, and distribution",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/01-modules-libraries/interview-prep.md"
            }
        ]
    },
    {
        "id": "folder-3-02-async-programming",
        "name": "02. Async Programming",
        "description": "Master asynchronous programming with asyncio, concurrent.futures, and modern Python async patterns",
        "topics": [
            {
                "id": "3-02-0",
                "title": "Async Programming Complete Guide",
                "description": "Comprehensive guide to asyncio, coroutines, tasks, and async/await patterns in Python",
                "duration": "3-4 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/22-async_programming/step1-async_programming_theory.md"
            },
            {
                "id": "3-02-1",
                "title": "Async Programming Practice",
                "description": "Build real-world async applications including web scrapers, API clients, and concurrent processors",
                "duration": "3 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/22-async_programming/step2-async_programming_practice.md"
            },
            {
                "id": "3-02-2",
                "title": "Async Programming Cheat Sheet",
                "description": "Quick reference for async patterns, common pitfalls, and performance optimization",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/22-async_programming/step3-production_python_cheatsheet.md"
            },
            {
                "id": "3-02-3",
                "title": "Async Programming Interview Prep",
                "description": "Interview questions covering event loops, coroutines, and concurrent programming concepts",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/22-async_programming/step4-python_testing_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-3-03-modern-features",
        "name": "03. Modern Python Features",
        "description": "Explore type hints, structural pattern matching, walrus operator, and Python 3.10+ features",
        "topics": [
            {
                "id": "3-03-0",
                "title": "Modern Features Complete Guide",
                "description": "Master type hints, pattern matching, assignment expressions, and modern Python syntactic constructs",
                "duration": "2-3 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/03-modern-features/step1-modern_features_complete_guide.md"
            },
            {
                "id": "3-03-1",
                "title": "Modern Features Practice",
                "description": "Refactor legacy code using modern Python features and write type-safe applications",
                "duration": "2 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/03-modern-features/practice.md"
            },
            {
                "id": "3-03-2",
                "title": "Modern Features Cheat Sheet",
                "description": "Quick reference for type hint syntax, pattern matching, and modern idioms",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/03-modern-features/cheatsheet.md"
            },
            {
                "id": "3-03-3",
                "title": "Modern Features Interview Prep",
                "description": "Interview questions about type systems, modern syntax, and Python evolution",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/03-modern-features/interview-prep.md"
            }
        ]
    },
    {
        "id": "folder-3-04-performance",
        "name": "04. Performance & Optimization",
        "description": "Learn profiling, optimization techniques, memory management, and writing high-performance Python",
        "topics": [
            {
                "id": "3-04-0",
                "title": "Performance Optimization Guide",
                "description": "Comprehensive guide to profiling, optimizing, and writing efficient Python code",
                "duration": "3-4 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/04-performance/step1-performance_optimization_complete_guide.md"
            },
            {
                "id": "3-04-1",
                "title": "Performance Practice Exercises",
                "description": "Profile and optimize real code, reduce latency, and improve throughput",
                "duration": "3 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/04-performance/step2-performance_practice_exercises.md"
            },
            {
                "id": "3-04-2",
                "title": "Performance Cheat Sheet",
                "description": "Quick reference for profiling tools, optimization patterns, and best practices",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/04-performance/cheatsheet.md"
            },
            {
                "id": "3-04-3",
                "title": "Performance Interview Prep",
                "description": "Interview questions about Big-O, profiling, and performance optimization strategies",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/04-performance/interview-prep.md"
            }
        ]
    },
    {
        "id": "folder-3-05-backend-development",
        "name": "05. Backend Development",
        "description": "Build robust REST APIs and microservices with FastAPI, Flask, and professional backend patterns",
        "topics": [
            {
                "id": "3-05-0",
                "title": "Backend Development Fundamentals",
                "description": "Master FastAPI, Flask, RESTful API design, authentication, and backend architecture patterns",
                "duration": "4-5 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/14-backend_development/step1-backend_development_fundamentals_theory.md"
            },
            {
                "id": "3-05-1",
                "title": "Backend Development Practice",
                "description": "Build a complete REST API with database integration, authentication, and testing",
                "duration": "4 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/14-backend_development/step2-backend_development_practice_exercises.md"
            },
            {
                "id": "3-05-2",
                "title": "Backend Development Cheat Sheet",
                "description": "Quick reference for API patterns, middleware, and backend best practices",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/14-backend_development/step3-backend_development_cheatsheet.md"
            },
            {
                "id": "3-05-3",
                "title": "Backend Interview Prep",
                "description": "Interview questions about API design, scalability, and backend architecture",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/14-backend_development/step4-backend_development_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-3-06-system-design",
        "name": "06. System Design",
        "description": "Learn distributed systems architecture, scalability patterns, and designing for millions of users",
        "topics": [
            {
                "id": "3-06-0",
                "title": "System Design Fundamentals",
                "description": "Master scalability, reliability, and designing distributed systems for production",
                "duration": "4-5 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/15-system_design/step1-system_design_fundamentals_theory.md"
            },
            {
                "id": "3-06-1",
                "title": "System Design Practice",
                "description": "Design real-world systems like URL shortener, social feed, and file storage service",
                "duration": "4 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/15-system_design/step2-system_design_practice_exercises.md"
            },
            {
                "id": "3-06-2",
                "title": "System Design Cheat Sheet",
                "description": "Quick reference for design patterns, data stores, and scalability strategies",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/15-system_design/step3-system_design_cheatsheet.md"
            },
            {
                "id": "3-06-3",
                "title": "System Design Interview Prep",
                "description": "Common system design interview questions with detailed solutions",
                "duration": "2 hours",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/15-system_design/step4-system_design_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-3-07-devops",
        "name": "07. DevOps & Containerization",
        "description": "Master Docker, CI/CD pipelines, Kubernetes fundamentals, and production deployment strategies",
        "topics": [
            {
                "id": "3-07-0",
                "title": "DevOps Containerization Fundamentals",
                "description": "Learn Docker, container orchestration, CI/CD, and infrastructure-as-code practices",
                "duration": "4-5 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/16-devops_containerization/step1-devops_containerization_fundamentals_theory.md"
            },
            {
                "id": "3-07-1",
                "title": "DevOps Containerization Practice",
                "description": "Set up Docker, write CI/CD pipelines, and deploy applications to cloud platforms",
                "duration": "4 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/16-devops_containerization/step2-devops_containerization_practice_exercises.md"
            },
            {
                "id": "3-07-2",
                "title": "DevOps Cheat Sheet",
                "description": "Quick reference for Docker commands, CI/CD patterns, and deployment strategies",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/16-devops_containerization/step3-devops_containerization_cheatsheet.md"
            },
            {
                "id": "3-07-3",
                "title": "DevOps Interview Prep",
                "description": "Interview questions about containers, CI/CD, and DevOps practices",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/16-devops_containerization/step4-devops_containerization_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-3-08-testing",
        "name": "08. Testing & QA Practices",
        "description": "Master pytest, unittest, TDD, test-driven development, and writing maintainable test suites",
        "topics": [
            {
                "id": "3-08-0",
                "title": "Testing QA Fundamentals",
                "description": "Comprehensive guide to pytest, unittest, mocking, and test automation strategies",
                "duration": "3-4 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/17-testing_qa_practices/step1-testing_qa_fundamentals_theory.md"
            },
            {
                "id": "3-08-1",
                "title": "Testing QA Practice",
                "description": "Write comprehensive test suites, implement TDD, and set up test automation",
                "duration": "3 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/17-testing_qa_practices/step2-testing_qa_practice_exercises.md"
            },
            {
                "id": "3-08-2",
                "title": "Testing QA Cheat Sheet",
                "description": "Quick reference for pytest fixtures, assertions, and testing patterns",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/17-testing_qa_practices/step3-testing_qa_cheatsheet.md"
            },
            {
                "id": "3-08-3",
                "title": "Testing QA Interview Prep",
                "description": "Interview questions about testing strategies, coverage, and test-driven development",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/17-testing_qa_practices/step4-testing_qa_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-3-09-mlops",
        "name": "09. MLOps & ML Engineering",
        "description": "Master machine learning operations, model versioning, deployment pipelines, and ML infrastructure",
        "topics": [
            {
                "id": "3-09-0",
                "title": "MLOps Fundamentals",
                "description": "Learn ML pipelines, model versioning, experiment tracking, and production ML practices",
                "duration": "4-5 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/18-modern_mlops/step1-mlops_fundamentals_theory.md"
            },
            {
                "id": "3-09-1",
                "title": "MLOps CI/CD Practice",
                "description": "Build ML pipelines with DVC, MLflow, and set up automated model training and deployment",
                "duration": "4 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/18-modern_mlops/step2-cicd_ml_practice.md"
            },
            {
                "id": "3-09-2",
                "title": "MLOps Cheat Sheet",
                "description": "Quick reference for ML tools, pipeline patterns, and deployment strategies",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/18-modern_mlops/step3-mlops_cheatsheet.md"
            },
            {
                "id": "3-09-3",
                "title": "MLOps Interview Prep",
                "description": "Interview questions about ML pipelines, model deployment, and ML infrastructure",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/18-modern_mlops/step4-mlops_interview_prep.md"
            }
        ]
    },
    {
        "id": "folder-3-10-data-engineering",
        "name": "10. Data Engineering",
        "description": "Build data pipelines, work with databases, and implement ETL processes at scale",
        "topics": [
            {
                "id": "3-10-0",
                "title": "Data Engineering Fundamentals",
                "description": "Master data pipelines, ETL processes, database design, and data warehouse concepts",
                "duration": "4-5 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/19-data_engineering/step1-data_engineering_theory.md"
            },
            {
                "id": "3-10-1",
                "title": "Data Pipeline Practice",
                "description": "Build real-time and batch data pipelines with Apache Airflow, pandas, and cloud services",
                "duration": "4 hours",
                "type": "practice",
                "markdownPath": "/courses/phase-3-advanced-python/19-data_engineering/step2-pipeline_practice.md"
            },
            {
                "id": "3-10-2",
                "title": "Data Engineering Cheat Sheet",
                "description": "Quick reference for SQL patterns, pipeline design, and data processing patterns",
                "duration": "30 min",
                "type": "theory",
                "markdownPath": "/courses/phase-3-advanced-python/19-data_engineering/step3-data_engineering_cheatsheet.md"
            },
            {
                "id": "3-10-3",
                "title": "Data Engineering Interview Prep",
                "description": "Interview questions about data pipelines, databases, and distributed systems",
                "duration": "1 hour",
                "type": "interview",
                "markdownPath": "/courses/phase-3-advanced-python/19-data_engineering/step4-data_engineering_interview_prep.md"
            }
        ]
    }
];

const allTopics = folders.flatMap(folder => folder.topics);

export const phase3: Phase = {
    id: 'phase-3',
    number: 3,
    title: 'Advanced Python Mastery',
    description: 'Master advanced Python concepts including async programming, modern features, performance optimization, backend development, DevOps, and professional engineering practices. Build production-ready applications and systems.',
    duration: '10-12 weeks',
    skills: [
        "Advanced Python Syntax",
        "Async Programming",
        "Type Hints & Pattern Matching",
        "Performance Optimization",
        "FastAPI & Flask",
        "System Design",
        "Docker & Kubernetes",
        "Testing & TDD",
        "MLOps",
        "Data Engineering",
        "CI/CD Pipelines",
        "Production Deployment"
    ],
    topics: allTopics,
    folders: folders
};
