import { Phase } from '../types';

export const vibePhases: Phase[] = [
    {
        id: 'vibe-phase-1',
        number: 1,
        title: 'Problem Definition',
        description: 'Identify a specific high-value pain point that can be solved with AI logic. Master the art of defining clear project scopes.',
        duration: '2-4 hours',
        skills: ['Market Gap Analysis', 'Pain Point Identification', 'Project Scoping'],
        role: 'Product Manager',
        requirements: ['Pain Point Map', 'User Persona'],
        techStack: ['Product Documentation'],
        aiTools: ['Perplexity', 'ChatGPT', 'Claude'],
        topics: [
            {
                id: 'vibe-1-1',
                title: 'Identifying The Problem',
                description: 'Learn how to spot inefficiencies that AI can solve.',
                duration: '1 hour',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase1/problem-definition.md'
            },
            {
                id: 'vibe-1-2',
                title: 'Prompt Engineering Foundations',
                description: 'Master CO-STAR and other frameworks to define your project vision.',
                duration: '1.5 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase1/prompt-foundations.md'
            }
        ]
    },
    {
        id: 'vibe-phase-2',
        number: 2,
        title: 'Research Phase & Web Scraping',
        description: 'Analyze competitors and extract the data you need using AI-powered web scraping tools.',
        duration: '4-6 hours',
        skills: ['Web Scraping', 'Competitor Analysis', 'Data Extraction'],
        role: 'Research Engineer',
        requirements: ['Target URL List', 'Data Schema'],
        techStack: ['Python', 'HTTP APIs', 'Markdown'],
        aiTools: ['Firecrawl', 'Jina AI', 'Bright Data'],
        topics: [
            {
                id: 'vibe-2-1',
                title: 'Web Scraping Strategy',
                description: 'When and how to use web scraping in your AI project workflow.',
                duration: '1 hour',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase2/scraping-strategy.md'
            },
            {
                id: 'vibe-2-2',
                title: 'AI Scraping Tools',
                description: 'Best free and paid tools for extracting data without writing complex code.',
                duration: '2 hours',
                type: 'practice',
                markdownPath: 'vibe-ai-engineering/phase2/scraping-tools.md'
            },
            {
                id: 'vibe-2-3',
                title: 'Documenting Research',
                description: 'How to organize and use scraped data for your AI models.',
                duration: '1 hour',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase2/documenting-research.md'
            }
        ]
    },
    {
        id: 'vibe-phase-3',
        number: 3,
        title: 'Solution Design & Data Logic',
        description: 'Architect your AI system. Map out the flow between APIs, databases, and generative UI.',
        duration: '3-5 hours',
        skills: ['Architecture Design', 'Data Flow Mapping', 'API Orchestration'],
        role: 'System Architect',
        requirements: ['User Journey Map', 'Data Flow Diagram'],
        techStack: ['Excalidraw', 'ER Diagrams', 'API Design'],
        aiTools: ['Claude (Reasoning)', 'Whimsical', 'Eraser.io'],
        topics: [
            {
                id: 'vibe-3-1',
                title: 'Technical Architecture',
                description: 'Designing the 3-tier system: Ingestion, Logic, and UI.',
                duration: '2 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase3/architecture.md'
            }
        ]
    },
    {
        id: 'vibe-phase-4',
        number: 4,
        title: 'Model & Core Build',
        description: 'Build the engine of your application. Master agentic workflows and core business logic.',
        duration: '8-12 hours',
        skills: ['AI Orchestration', 'Core Build', 'Logic Implementation'],
        role: 'Logic Engineer',
        requirements: ['Logic Flowchart', 'API Keys'],
        techStack: ['Python', 'TypeScript', 'Vector DBs'],
        aiTools: ['Cursor', 'Claude 3.5 Sonnet', 'Replit Agent'],
        topics: [
            {
                id: 'vibe-4-1',
                title: 'Agentic Workflows',
                description: 'Building multi-agent systems that solve complex tasks.',
                duration: '4 hours',
                type: 'practice',
                markdownPath: 'vibe-ai-engineering/phase4/agent-logic.md'
            },
            {
                id: 'vibe-4-2',
                title: 'The AI Engineering Workflow',
                description: 'Step-by-step through the core build process.',
                duration: '4 hours',
                type: 'practice',
                markdownPath: 'vibe-ai-engineering/phase4/core-build.md'
            }
        ]
    },
    {
        id: 'vibe-phase-5',
        number: 5,
        title: 'Integration',
        description: 'Connect your core AI logic to the frontend and backend infrastructure.',
        duration: '4-6 hours',
        skills: ['API Integration', 'Full Stack Connection', 'Frontend/Backend Glue'],
        role: 'Full Stack Engineer',
        requirements: ['UI Mockups', 'Backend Endpoint List'],
        techStack: ['Next.js', 'React', 'Supabase'],
        aiTools: ['v0.dev', 'Bolt.new', 'Cursor'],
        topics: [
            {
                id: 'vibe-5-1',
                title: 'Connecting the Pieces',
                description: 'How to stitch together generative UI, backend, and LLM APIs.',
                duration: '3 hours',
                type: 'practice',
                markdownPath: 'vibe-ai-engineering/phase5/integration-guide.md'
            }
        ]
    },
    {
        id: 'vibe-phase-6',
        number: 6,
        title: 'Testing & Validation',
        description: 'Ensure your AI doesn\'t make mistakes. Master evaluation frameworks and security.',
        duration: '4-6 hours',
        skills: ['AI Evaluation', 'Safety & Security', 'Edge Case Handling'],
        role: 'QA & Security Engineer',
        requirements: ['Test Dataset', 'Edge Case Scenarios'],
        techStack: ['Vitest', 'Pytest', 'Postman'],
        aiTools: ['Giskard', 'Lakera Guard', 'Agent Ops'],
        topics: [
            {
                id: 'vibe-6-1',
                title: 'Common AI Mistakes',
                description: 'Identifying and fixing hallucinations and logic errors.',
                duration: '2 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase6/ai-mistakes.md'
            },
            {
                id: 'vibe-6-2',
                title: 'AI Security & Jailbreaking',
                description: 'Protecting your application from prompt injection and data leaks.',
                duration: '2 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase6/security.md'
            }
        ]
    },
    {
        id: 'vibe-phase-7',
        number: 7,
        title: 'Documentation',
        description: 'Create professional documentation for your code and AI systems.',
        duration: '2-4 hours',
        skills: ['Technical Writing', 'API Documentation', 'System Manuals'],
        role: 'Technical Writer',
        requirements: ['Project Summary', 'Usage Guide'],
        techStack: ['Markdown', 'Swagger/OpenAPI', 'JSDoc'],
        aiTools: ['Cursor (Doc Gen)', 'Mintlify', 'ChatGPT'],
        topics: [
            {
                id: 'vibe-7-1',
                title: 'Industrial Documentation',
                description: 'How to document AI-native projects for professional use.',
                duration: '2 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase7/documentation.md'
            }
        ]
    },
    {
        id: 'vibe-phase-8',
        number: 8,
        title: 'Deployment',
        description: 'Launch your application to the world on industrial infrastructure.',
        duration: '3-5 hours',
        skills: ['CI/CD', 'Cloud Deployment', 'Performance Monitoring'],
        role: 'DevOps Engineer',
        requirements: ['API Keys', 'Environment Variables'],
        techStack: ['Vercel', 'Railway', 'GitHub Actions'],
        aiTools: ['Terraform AI', 'Pulumi AI', 'Vercel Ship'],
        topics: [
            {
                id: 'vibe-8-1',
                title: 'Scaling & Performance',
                description: 'Moving from local build to global production.',
                duration: '2 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase8/deployment-scaling.md'
            }
        ]
    },
    {
        id: 'vibe-phase-9',
        number: 9,
        title: 'Monetization',
        description: 'Turn your AI project into a business. Pricing strategies and market reach.',
        duration: '2-4 hours',
        skills: ['Business Strategy', 'Pricing Models', 'Marketing'],
        role: 'Business Strategist',
        requirements: ['Revenue Model', 'Marketing Plan'],
        techStack: ['Stripe', 'LemonSqueezy', 'Loops.so'],
        aiTools: ['Market Analysis AI', 'Copy.ai', 'Jasper'],
        topics: [
            {
                id: 'vibe-9-1',
                title: 'Monetizing AI Products',
                description: 'How to charge for AI services and build sustainable revenue.',
                duration: '2 hours',
                type: 'theory',
                markdownPath: 'vibe-ai-engineering/phase9/monetization.md'
            }
        ]
    }
];
