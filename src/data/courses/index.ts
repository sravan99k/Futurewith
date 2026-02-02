import { phase1 } from './phase-1-python-foundations';
import { phase2 } from './phase-2-data-structures-algorithms';
import { phase3 } from './phase-3-advanced-python';
import { phase4 } from './phase-4-ai-ml-fundamentals';
import { phase5 } from './phase-5-professional-skills';
import { phase6 } from './phase-6-interview-skills';
import { phase7 } from './phase-7-advanced-projects';
import { phase8 } from './phase-8-career-entrepreneurship';
import { learningHandbook } from './learning-handbook';
import { vibePhases } from './vibe-ai-engineering';
import type { Course, Phase } from './types';

// Export all phases (main learning journey)
export const allPhases: Phase[] = [
    phase1,
    phase2,
    phase3,
    phase4,
    phase5,
    phase6,
    phase7,
    phase8
];

// Export reference materials (always accessible, doesn't block progression)
export const referenceMaterials: Phase[] = [
    learningHandbook
];

// All course resources
export const allResources: Phase[] = [
    ...allPhases,
    ...referenceMaterials,
    ...vibePhases
];

// Main course exports
export const pythonAICourse: Course = {
    id: 'python-ai-ml-complete',
    title: 'Complete Python, AI & ML Mastery',
    description: 'From zero to AI engineer. Master Python, data structures, machine learning, deep learning, and deploy production-ready models. Land your dream job at top tech companies.',
    totalPhases: 8,
    phases: allPhases
};

export const aiEngineeringCourse: Course = {
    id: 'ai-engineering-vibe',
    title: 'Industrial AI Engineering (Vibe Coding)',
    description: 'Master the 9-phase industrial protocol for building production-ready AI applications.',
    totalPhases: 9,
    phases: vibePhases
};

// Individual phase exports
export {
    phase1,
    phase2,
    phase3,
    phase4,
    phase5,
    phase6,
    phase7,
    phase8,
    learningHandbook,
    vibePhases
};

// Type exports
export * from './types';

// Helper functions
export function getPhaseById(phaseId: string): Phase | undefined {
    return allResources.find(phase => phase.id === phaseId);
}

export function getPhaseByNumber(phaseNumber: number): Phase | undefined {
    return allPhases.find(phase => phase.number === phaseNumber);
}

export function getTopicById(phaseId: string, topicId: string) {
    const phase = getPhaseById(phaseId);
    return phase?.topics.find(topic => topic.id === topicId);
}

// Get all topics across all phases (learning journey only)
export function getAllTopics() {
    return allPhases.flatMap(phase => {
        const directTopics = phase.topics.map(topic => ({
            ...topic,
            phaseId: phase.id,
            phaseNumber: phase.number,
            phaseTitle: phase.title
        }));

        const folderTopics = (phase.folders || []).flatMap(folder =>
            folder.topics.map(topic => ({
                ...topic,
                phaseId: phase.id,
                phaseNumber: phase.number,
                phaseTitle: phase.title,
                folderName: folder.name
            }))
        );

        return [...directTopics, ...folderTopics];
    });
}

// Search functionality (learning journey only)
export function searchTopics(query: string) {
    const lowercaseQuery = query.toLowerCase();
    return getAllTopics().filter(topic =>
        topic.title.toLowerCase().includes(lowercaseQuery) ||
        topic.description.toLowerCase().includes(lowercaseQuery)
    );
}

// Get all reference topics (handbook, guides, etc.)
export function getAllReferenceTopics() {
    return referenceMaterials.flatMap(phase => {
        const directTopics = phase.topics.map(topic => ({
            ...topic,
            phaseId: phase.id,
            phaseNumber: phase.number,
            phaseTitle: phase.title
        }));

        const folderTopics = (phase.folders || []).flatMap(folder =>
            folder.topics.map(topic => ({
                ...topic,
                phaseId: phase.id,
                phaseNumber: phase.number,
                phaseTitle: phase.title,
                folderName: folder.name
            }))
        );

        return [...directTopics, ...folderTopics];
    });
}
