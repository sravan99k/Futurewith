import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = [
    {
        id: 'folder-handbook-01-learning-pathways',
        name: 'Learning Pathways',
        description: 'Choose the optimal learning path based on your goals, experience, and available time',
        topics: [
            {
                id: 'handbook-pathways-0',
                title: 'Learning Pathways Guide',
                description: 'Complete guide to choosing your optimal learning track',
                duration: '30-45 minutes',
                type: 'theory',
                markdownPath: '/courses/learning-handbook/01-learning-pathways/01-learning-pathways-guide.md'
            }
        ]
    },
    {
        id: 'folder-handbook-02-quick-reference',
        name: 'Quick Reference',
        description: 'Fast navigation guide and emergency access to all course sections',
        topics: [
            {
                id: 'handbook-quickref-0',
                title: 'Quick Access Guide',
                description: 'Quick reference for navigation and emergency access',
                duration: '15-20 minutes',
                type: 'reference',
                markdownPath: '/courses/learning-handbook/02-quick-reference/01-quick-access-guide.md'
            }
        ]
    },
    {
        id: 'folder-handbook-03-phase-overview',
        name: 'Phase Overview',
        description: 'Summary of all course phases with time estimates and prerequisites',
        topics: [
            {
                id: 'handbook-overview-0',
                title: 'Phase Overview Dashboard',
                description: 'Complete course summary and phase-by-phase breakdown',
                duration: '20-30 minutes',
                type: 'reference',
                markdownPath: '/courses/learning-handbook/03-phase-overview/01-phase-overview.md'
            }
        ]
    }
];

const allTopics = folders.flatMap(folder => folder.topics);

export const learningHandbook: Phase = {
    id: 'learning-handbook',
    number: 0,
    title: 'Learning Handbook',
    description: 'Your guide to navigating the course, choosing learning paths, and maximizing your learning experience',
    duration: 'Reference - 1-2 hours total',
    skills: ['Learning Strategy', 'Pathway Selection', 'Time Management', 'Progress Optimization'],
    topics: allTopics,
    folders: folders
};
