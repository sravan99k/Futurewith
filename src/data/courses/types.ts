// TypeScript interfaces for course data structure

export interface Topic {
    id: string;
    title: string;
    description: string;
    duration: string;
    type: 'theory' | 'practice' | 'project' | 'exercise' | 'assessment' | 'interview';
    markdownPath: string;
}

// Folder grouping for topics
export interface TopicFolder {
    id: string;
    name: string;
    description: string;
    topics: Topic[];
}

export interface Phase {
    id: string;
    number: number;
    title: string;
    description: string;
    duration: string;
    skills: string[];
    topics: Topic[];
    folders?: TopicFolder[];  // Optional folder grouping
    readmePath?: string;
    // Industrial Context
    role?: string;
    requirements?: string[];
    techStack?: string[];
    aiTools?: string[];
}

export interface Course {
    id: string;
    title: string;
    description: string;
    totalPhases: number;
    phases: Phase[];
}

export interface LearningPath {
    id: string;
    title: string;
    description: string;
    track: 'python' | 'vibe' | 'both';
    duration: string;
    phases: string[];
    bestFor: string[];
}

export interface CounselingOption {
    id: string;
    label: string;
    value: string;
    description: string;
}

export interface CounselingQuestion {
    id: string;
    question: string;
    options: CounselingOption[];
}
