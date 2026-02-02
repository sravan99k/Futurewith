export interface CounselingQuestion {
    id: string;
    question: string;
    options: {
        id: string;
        label: string;
        value: string;
        description: string;
    }[];
}

export const counselingQuestions: CounselingQuestion[] = [
    {
        id: 'goal',
        question: 'What is your ultimate career goal?',
        options: [
            { id: 'job', label: 'Get a Job', value: 'job', description: 'I want to get hired at a tech company' },
            { id: 'startup', label: 'Build a Startup', value: 'startup', description: 'I want to create my own product/company' },
            { id: 'freelance', label: 'Freelance/Consulting', value: 'freelance', description: 'I want to work independently with clients' },
            { id: 'vibe', label: 'Vibe Coding', value: 'vibe', description: 'I want to build apps quickly with AI assistance' },
        ]
    },
    {
        id: 'time',
        question: 'How much time can you dedicate weekly?',
        options: [
            { id: 'part', label: '5-10 hours/week', value: 'part-time', description: 'I have a job/studies alongside' },
            { id: 'half', label: '15-20 hours/week', value: 'half-time', description: 'I can dedicate significant time' },
            { id: 'full', label: '30+ hours/week', value: 'full-time', description: 'I can focus full-time on learning' },
        ]
    },
    {
        id: 'experience',
        question: 'What is your current programming experience?',
        options: [
            { id: 'none', label: 'Complete Beginner', value: 'beginner', description: 'Never written code before' },
            { id: 'basic', label: 'Basic Knowledge', value: 'basic', description: 'Know some programming basics' },
            { id: 'intermediate', label: 'Intermediate', value: 'intermediate', description: 'Built small projects before' },
            { id: 'advanced', label: 'Advanced', value: 'advanced', description: 'Professional experience' },
        ]
    },
    {
        id: 'interest',
        question: 'What interests you most?',
        options: [
            { id: 'data', label: 'Data & Analytics', value: 'data', description: 'Working with data, insights, dashboards' },
            { id: 'ml', label: 'Machine Learning', value: 'ml', description: 'Building predictive models, AI systems' },
            { id: 'apps', label: 'App Development', value: 'apps', description: 'Creating web/mobile applications' },
            { id: 'automation', label: 'Automation', value: 'automation', description: 'Automating workflows and tasks' },
        ]
    },
];

// Recommended paths based on answers
export interface LearningPath {
    id: string;
    title: string;
    description: string;
    duration: string;
    phases: string[];
    bestFor: string[];
    track: 'python' | 'vibe' | 'both';
}

export const learningPaths: LearningPath[] = [
    {
        id: 'python-ml',
        title: 'Python to ML Engineer',
        description: 'Complete journey from Python basics to deploying ML models in production',
        duration: '6-12 months',
        phases: ['phase-1', 'phase-2', 'phase-3', 'phase-4', 'phase-5', 'phase-7'],
        bestFor: ['job', 'ml', 'full-time', 'intermediate'],
        track: 'python',
    },
    {
        id: 'data-analyst',
        title: 'Data Analyst Track',
        description: 'Focus on data analysis, visualization, and business insights',
        duration: '3-6 months',
        phases: ['phase-1', 'phase-3', 'phase-4'],
        bestFor: ['job', 'data', 'part-time', 'basic'],
        track: 'python',
    },
    {
        id: 'vibe-builder',
        title: 'AI-Powered Builder',
        description: 'Learn to build fast with AI tools and modern frameworks',
        duration: '2-4 months',
        phases: ['phase-1', 'phase-8'],
        bestFor: ['startup', 'vibe', 'apps', 'automation'],
        track: 'vibe',
    },
    {
        id: 'deep-learning',
        title: 'Deep Learning Specialist',
        description: 'Advanced AI with neural networks, NLP, and computer vision',
        duration: '8-12 months',
        phases: ['phase-1', 'phase-2', 'phase-4', 'phase-5', 'phase-6', 'phase-7'],
        bestFor: ['job', 'ml', 'full-time', 'advanced'],
        track: 'python',
    },
    {
        id: 'genai-expert',
        title: 'Generative AI Expert',
        description: 'Master LLMs, prompt engineering, and AI application development',
        duration: '4-6 months',
        phases: ['phase-1', 'phase-8', 'phase-9'],
        bestFor: ['startup', 'vibe', 'apps'],
        track: 'both',
    },
];
