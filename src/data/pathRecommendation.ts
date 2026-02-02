// Path Recommendation System - Find Your Path Quiz
// Helps students decide between Vibe Coding and Python Logic courses

export type PathRecommendation = 'VIBE_CODING' | 'PYTHON_LOGIC' | 'BOTH' | 'NEITHER';

export interface QuizQuestion {
    id: number;
    section: 'PROFILE' | 'GOALS' | 'LEARNING' | 'CAREER';
    question: string;
    options: QuizOption[];
}

export interface QuizOption {
    text: string;
    value: {
        vibeScore: number;
        pythonScore: number;
        flags?: string[];
    };
}

export interface QuizResult {
    recommendation: PathRecommendation;
    title: string;
    description: string;
    reasons: string[];
    nextSteps: string[];
    score: {
        vibe: number;
        python: number;
    };
}

// ============================================================
// THE PATH RECOMMENDATION QUIZ
// ============================================================

export const PATH_QUIZ_QUESTIONS: QuizQuestion[] = [
    // SECTION 1: PROFILE
    {
        id: 1,
        section: 'PROFILE',
        question: 'Which best describes your current situation?',
        options: [
            { text: 'Founder / Entrepreneur with an idea to launch', value: { vibeScore: 3, pythonScore: 0 } },
            { text: 'Student or fresh graduate seeking a software job', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'Non-technical professional (Marketing, PM, Sales) wanting to build tools', value: { vibeScore: 2, pythonScore: 0 } },
            { text: 'Experienced developer looking to speed up workflow', value: { vibeScore: 2, pythonScore: 1 } },
            { text: 'Hobbyist curious about how computers think', value: { vibeScore: 0, pythonScore: 2 } }
        ]
    },
    {
        id: 2,
        section: 'PROFILE',
        question: 'How comfortable are you with computers and technology?',
        options: [
            { text: 'I struggle with basic file management and folders', value: { vibeScore: -2, pythonScore: -3, flags: ['NEED_BASICS'] } },
            { text: 'I can use apps and browsers but never coded', value: { vibeScore: 1, pythonScore: 1 } },
            { text: 'I have tried coding tutorials before', value: { vibeScore: 1, pythonScore: 2 } },
            { text: 'I can read code but struggle to write from scratch', value: { vibeScore: 2, pythonScore: 1 } },
            { text: 'I code professionally', value: { vibeScore: 2, pythonScore: 2 } }
        ]
    },
    {
        id: 3,
        section: 'PROFILE',
        question: 'What is your age group?',
        options: [
            { text: 'Under 18 (Student)', value: { vibeScore: 1, pythonScore: 2 } },
            { text: '18-24 (College/Fresh Grad)', value: { vibeScore: 1, pythonScore: 2 } },
            { text: '25-35 (Early Career)', value: { vibeScore: 2, pythonScore: 2 } },
            { text: '35-45 (Mid Career)', value: { vibeScore: 3, pythonScore: 1 } },
            { text: '45+ (Senior Professional)', value: { vibeScore: 3, pythonScore: 1 } }
        ]
    },
    // SECTION 2: GOALS
    {
        id: 4,
        section: 'GOALS',
        question: 'What is your primary goal right now?',
        options: [
            { text: 'Launch a functioning app/website as fast as possible', value: { vibeScore: 4, pythonScore: 0 } },
            { text: 'Pass technical interviews at tech companies', value: { vibeScore: 0, pythonScore: 5 } },
            { text: 'Automate boring tasks at my current job', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Start a career in Data Science or AI', value: { vibeScore: 0, pythonScore: 5, flags: ['PYTHON_ONLY'] } },
            { text: 'Build a startup MVP to test an idea', value: { vibeScore: 4, pythonScore: 0 } }
        ]
    },
    {
        id: 5,
        section: 'GOALS',
        question: 'When do you need to see results?',
        options: [
            { text: 'This weekend - I need a prototype NOW', value: { vibeScore: 3, pythonScore: -2, flags: ['VIBE_ONLY'] } },
            { text: 'In 1-3 months - I can study seriously', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'In 3-6 months - I want career-ready skills', value: { vibeScore: 1, pythonScore: 3 } },
            { text: 'In 6-12 months - I want deep expertise', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'No rush - I enjoy learning for its own sake', value: { vibeScore: 1, pythonScore: 2 } }
        ]
    },
    {
        id: 6,
        section: 'GOALS',
        question: 'What kind of project excites you most?',
        options: [
            { text: 'A SaaS platform, mobile app, or landing page', value: { vibeScore: 3, pythonScore: 0 } },
            { text: 'A data analysis script or dashboard', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'A machine learning model or AI tool', value: { vibeScore: 0, pythonScore: 4 } },
            { text: 'An automation bot or web scraper', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'A backend API or system architecture', value: { vibeScore: 1, pythonScore: 3 } }
        ]
    },
    {
        id: 7,
        section: 'GOALS',
        question: 'Why do you want to learn coding?',
        options: [
            { text: 'To build and sell my own products', value: { vibeScore: 4, pythonScore: 0 } },
            { text: 'To get a high-paying job at a tech company', value: { vibeScore: 1, pythonScore: 4 } },
            { text: 'To understand how technology works', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'To be more efficient at my current job', value: { vibeScore: 3, pythonScore: 1 } },
            { text: 'For fun and curiosity', value: { vibeScore: 1, pythonScore: 2 } }
        ]
    },
    // SECTION 3: LEARNING STYLE
    {
        id: 8,
        section: 'LEARNING',
        question: 'How do you feel about AI writing code for you?',
        options: [
            { text: 'Love it! If it works, I do not care how it was written', value: { vibeScore: 3, pythonScore: -2 } },
            { text: 'Interested but I want to fix it when AI makes mistakes', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Skeptical - I need to understand every line to trust it', value: { vibeScore: -1, pythonScore: 3 } },
            { text: 'I want to learn the traditional way first, then use AI', value: { vibeScore: 1, pythonScore: 3 } }
        ]
    },
    {
        id: 9,
        section: 'LEARNING',
        question: 'When you encounter a difficult error, what do you do?',
        options: [
            { text: 'Frustrated - I just want the error gone', value: { vibeScore: 2, pythonScore: -2 } },
            { text: 'Copy the error message and ask AI to fix it', value: { vibeScore: 2, pythonScore: 0 } },
            { text: 'Curious - I want to understand why it happened', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'Research the error online and study the documentation', value: { vibeScore: 0, pythonScore: 2 } }
        ]
    },
    {
        id: 10,
        section: 'LEARNING',
        question: 'Which learning approach appeals to you more?',
        options: [
            { text: 'Learn by building real projects immediately', value: { vibeScore: 3, pythonScore: 0 } },
            { text: 'Learn fundamentals first, then apply them', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'Mix of theory and hands-on practice', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Follow a structured curriculum with exercises', value: { vibeScore: 1, pythonScore: 2 } }
        ]
    },
    {
        id: 11,
        section: 'LEARNING',
        question: 'Which analogy fits you best?',
        options: [
            { text: 'The Architect - Design the building, let robots build it', value: { vibeScore: 3, pythonScore: 0 } },
            { text: 'The Blacksmith - Forge tools, understand the metal', value: { vibeScore: 0, pythonScore: 3 } },
            { text: 'The Driver - Just need to get from A to B efficiently', value: { vibeScore: 3, pythonScore: 0 } },
            { text: 'The Mechanic - Want to know how the engine works', value: { vibeScore: 0, pythonScore: 3 } }
        ]
    },
    // SECTION 4: CAREER
    {
        id: 12,
        section: 'CAREER',
        question: 'Where do you see yourself in 5 years?',
        options: [
            { text: 'Running my own business or startup', value: { vibeScore: 4, pythonScore: 0 } },
            { text: 'Working as a Senior Software Engineer', value: { vibeScore: 1, pythonScore: 3 } },
            { text: 'Working as a Data Scientist or ML Engineer', value: { vibeScore: 0, pythonScore: 5, flags: ['PYTHON_ONLY'] } },
            { text: 'Working as a CTO or Technical Lead', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Being tech-literate in my current non-tech role', value: { vibeScore: 3, pythonScore: 0 } }
        ]
    },
    {
        id: 13,
        section: 'CAREER',
        question: 'How much time can you commit weekly to learning?',
        options: [
            { text: '1-3 hours (very casual)', value: { vibeScore: 2, pythonScore: -1 } },
            { text: '3-6 hours (part-time student)', value: { vibeScore: 2, pythonScore: 1 } },
            { text: '6-10 hours (serious learner)', value: { vibeScore: 2, pythonScore: 2 } },
            { text: '10-20 hours (intensive study)', value: { vibeScore: 1, pythonScore: 3 } },
            { text: '20+ hours (immersive)', value: { vibeScore: 2, pythonScore: 2 } }
        ]
    },
    {
        id: 14,
        section: 'CAREER',
        question: 'What is your budget for this course?',
        options: [
            { text: 'Looking for free resources only', value: { vibeScore: 0, pythonScore: 0, flags: ['FREE_ONLY'] } },
            { text: 'Willing to pay if it provides real value', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Investment is not an issue for the right course', value: { vibeScore: 2, pythonScore: 2 } }
        ]
    },
    {
        id: 15,
        section: 'CAREER',
        question: 'Have you tried learning coding before?',
        options: [
            { text: 'No, this is my first attempt', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Yes, but I got stuck and gave up', value: { vibeScore: 2, pythonScore: 0, flags: ['NEED_SUPPORT'] } },
            { text: 'Yes, I know basics but want to level up', value: { vibeScore: 2, pythonScore: 2 } },
            { text: 'Yes, I am comfortable with one language already', value: { vibeScore: 2, pythonScore: 1 } }
        ]
    }
];

// ============================================================
// SCORING AND RESULTS LOGIC
// ============================================================

export function calculateRecommendation(answers: Record<number, number>): QuizResult {
    let vibeScore = 0;
    let pythonScore = 0;
    const flags: string[] = [];
    const reasons: string[] = [];
    
    // Calculate scores
    Object.entries(answers).forEach(([questionId, optionIndex]) => {
        const question = PATH_QUIZ_QUESTIONS.find(q => q.id === parseInt(questionId));
        if (question && question.options[optionIndex]) {
            const option = question.options[optionIndex];
            vibeScore += option.value.vibeScore;
            pythonScore += option.value.pythonScore;
            if (option.value.flags) {
                flags.push(...option.value.flags);
            }
        }
    });
    
    // Handle edge cases with flags
    if (flags.includes('NEED_BASICS')) {
        return {
            recommendation: 'NEITHER',
            title: 'Start with Basics First',
            description: 'Before diving into either course, we recommend building fundamental computer literacy skills.',
            reasons: [
                'Both courses assume basic comfort with computers and file management',
                'Understanding how to navigate folders, create files, and use a browser is essential',
                'Consider spending a week on basic digital skills before starting'
            ],
            nextSteps: [
                'Learn basic computer skills (file management, keyboard shortcuts)',
                'Practice using a web browser and Google search effectively',
                'Then return to take this quiz again'
            ],
            score: { vibe: vibeScore, python: pythonScore }
        };
    }
    
    if (flags.includes('PYTHON_ONLY')) {
        return {
            recommendation: 'PYTHON_LOGIC',
            title: 'Python Logic is Essential for Your Goals',
            description: 'Your goals in Data Science, AI, or ML require strong Python fundamentals that Vibe Coding cannot provide.',
            reasons: [
                'Data Science and ML require understanding statistics and algorithms',
                'Technical interviews at top companies test fundamental programming skills',
                'Vibe Coding accelerates development but does not replace core knowledge'
            ],
            nextSteps: [
                'Start with Python Logic to build your foundation',
                'After completing Python, you can enhance with Vibe Coding techniques',
                'This combination will make you a very strong candidate'
            ],
            score: { vibe: vibeScore, python: pythonScore }
        };
    }
    
    if (flags.includes('VIBE_ONLY')) {
        return {
            recommendation: 'VIBE_CODING',
            title: 'Vibe Coding is Your Best Choice',
            description: 'With your tight timeline, Python fundamentals will slow you down. Vibe Coding gets you results fast.',
            reasons: [
                'Learning syntax and debugging takes weeks - you need results now',
                'Vibe Coding uses AI to bridge the gap between idea and implementation',
                'You can always learn fundamentals later if needed'
            ],
            nextSteps: [
                'Jump into Vibe Coding and build your prototype this weekend',
                'Launch your MVP and get user feedback',
                'Consider Python Logic later when you have more time'
            ],
            score: { vibe: vibeScore, python: pythonScore }
        };
    }
    
    if (flags.includes('FREE_ONLY')) {
        return {
            recommendation: 'NEITHER',
            title: 'Explore Free Resources First',
            description: 'Both our courses are paid. Consider free alternatives before committing.',
            reasons: [
                'FreeCodeCodecademy offers free Python tutorials',
                'YouTube has thousands of free coding tutorials',
                'Our courses provide structured learning with support'
            ],
            nextSteps: [
                'Try free resources first to see if you enjoy coding',
                'If you decide to invest, our courses offer better structure',
                'Return when you are ready to commit'
            ],
            score: { vibe: vibeScore, python: pythonScore }
        };
    }
    
    // Normalize scores (scale to 100)
    const maxPossibleVibe = 60; // Approximate max score
    const maxPossiblePython = 60;
    const normalizedVibe = Math.min(100, Math.round((vibeScore / maxPossibleVibe) * 100));
    const normalizedPython = Math.min(100, Math.round((pythonScore / maxPossiblePython) * 100));
    
    // Determine recommendation based on scores
    let recommendation: PathRecommendation;
    let title: string;
    let description: string;
    
    if (normalizedVibe >= 70 && normalizedPython < 60) {
        recommendation = 'VIBE_CODING';
        title = 'You are a Vibe Coder';
        description = 'You value speed and results over deep technical knowledge. Vibe Coding is perfect for your goals.';
        reasons.push(...[
            'You want to build and launch quickly',
            'You are comfortable letting AI handle the details',
            'Your goals align with rapid prototyping and entrepreneurship'
        ]);
    } else if (normalizedPython >= 70 && normalizedVibe < 60) {
        recommendation = 'PYTHON_LOGIC';
        title = 'You are a Logic Engineer';
        description = 'You want to build a strong foundation for a career in software engineering or data science.';
        reasons.push(...[
            'You value understanding how things work under the hood',
            'Your career goals require strong fundamentals',
            'You prefer thorough learning over quick results'
        ]);
    } else if (normalizedVibe >= 55 && normalizedPython >= 55) {
        recommendation = 'BOTH';
        title = 'You are a Full Stack AI Architect';
        description = 'You have the ambition of a founder and the rigor of an engineer. Both courses together will make you unstoppable.';
        reasons.push(...[
            'You want speed AND understanding',
            'Your goals span both entrepreneurship and technical depth',
            'You are willing to invest time for maximum results'
        ]);
    } else if (normalizedVibe >= 40 && normalizedPython >= 40) {
        recommendation = 'BOTH';
        title = 'You Benefit from Both Paths';
        description = 'A combination of both courses will give you the best of both worlds.';
        reasons.push(...[
            'You have diverse goals that benefit from both approaches',
            'Starting with Python then adding Vibe Coding creates a strong foundation',
            'This combination makes you more valuable than single-skill peers'
        ]);
    } else if (normalizedVibe >= 45 && normalizedPython < 40) {
        recommendation = 'VIBE_CODING';
        title = 'Vibe Coding Will Work Well for You';
        description = 'While you have some interest in fundamentals, your primary goals align with Vibe Coding.';
        reasons.push(...[
            'Your immediate goals require speed over depth',
            'You can always add Python knowledge later',
            'Vibe Coding provides immediate value for your situation'
        ]);
    } else if (normalizedPython >= 45 && normalizedVibe < 40) {
        recommendation = 'PYTHON_LOGIC';
        title = 'Python Logic is Right for You';
        description = 'Your focus on fundamentals and career growth makes Python Logic the ideal starting point.';
        reasons.push(...[
            'Building a strong foundation will serve you long-term',
            'Your goals require deep technical understanding',
            'This path sets you up for career success'
        ]);
    } else {
        recommendation = 'NEITHER';
        title = 'Consider Your Goals More Carefully';
        description = 'Your answers suggest you may need to clarify your objectives before choosing a course.';
        reasons.push(...[
            'Both scores are relatively low - consider if coding is the right fit',
            'You might benefit from exploring different career paths',
            'Take time to reflect on what you really want to achieve'
        ]);
    }
    
    // Generate next steps based on recommendation
    let nextSteps: string[];
    switch (recommendation) {
        case 'VIBE_CODING':
            nextSteps = [
                'Start with Vibe Coding course',
                'Build your first project within the first week',
                'Launch your MVP and get feedback',
                'Consider adding Python Logic later for deeper skills'
            ];
            break;
        case 'PYTHON_LOGIC':
            nextSteps = [
                'Start with Python Logic fundamentals',
                'Complete all exercises and projects',
                'Build a portfolio of Python projects',
                'Consider Vibe Coding later to accelerate your workflow'
            ];
            break;
        case 'BOTH':
            nextSteps = [
                'Bundle: Get both courses together',
                'Start with Python Logic (weeks 1-8)',
                'Follow with Vibe Coding (weeks 9-12)',
                'Build advanced projects combining both skills'
            ];
            break;
        default:
            nextSteps = [
                'Spend time exploring different career options',
                'Try free coding tutorials to see if you enjoy it',
                'Return to this quiz when your goals are clearer'
            ];
    }
    
    return {
        recommendation,
        title,
        description,
        reasons,
        nextSteps,
        score: { vibe: normalizedVibe, python: normalizedPython }
    };
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

export function getRecommendationColor(recommendation: PathRecommendation): string {
    switch (recommendation) {
        case 'VIBE_CODING': return 'text-vibe';
        case 'PYTHON_LOGIC': return 'text-python';
        case 'BOTH': return 'text-green-500';
        default: return 'text-muted-foreground';
    }
}

export function getRecommendationBg(recommendation: PathRecommendation): string {
    switch (recommendation) {
        case 'VIBE_CODING': return 'bg-vibe/10 border-vibe/20';
        case 'PYTHON_LOGIC': return 'bg-python/10 border-python/20';
        case 'BOTH': return 'bg-green-500/10 border-green-500/20';
        default: return 'bg-muted/10 border-muted/20';
    }
}

export function getRecommendationButton(recommendation: PathRecommendation): { text: string; link: string } {
    switch (recommendation) {
        case 'VIBE_CODING': return { text: 'Start Vibe Coding', link: '/vibe-coding' };
        case 'PYTHON_LOGIC': return { text: 'Start Python Logic', link: '/python-course' };
        case 'BOTH': return { text: 'Get Both Courses', link: '/bundle' };
        default: return { text: 'Explore More', link: '/' };
    }
}
