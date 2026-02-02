import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const phaseDir = 'phase-7-navigation-skills';
const fullPath = path.join(process.cwd(), 'src', 'data', 'courses', phaseDir);

function getRootFiles() {
    const items = fs.readdirSync(fullPath, { withFileTypes: true });
    const mdFiles = items
        .filter(item => item.isFile() && item.name.endsWith('.md'))
        .filter(item => item.name !== 'index.ts')
        .map(item => item.name)
        .sort();

    return mdFiles.map((file, index) => {
        const fileWithoutExt = file.replace('.md', '');
        // Convert filename to title case
        const titleName = fileWithoutExt.replace(/-/g, ' ').replace(/_/g, ' ');
        const titleCased = titleName.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

        return {
            id: `7-root-${index}`,
            title: titleCased,
            description: `Learning about ${fileWithoutExt}`,
            duration: '1-2 hours',
            type: 'theory',
            markdownPath: `/src/data/courses/${phaseDir}/${file}`
        };
    });
}

const topics = getRootFiles();

const output = `import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = [
    {
        id: 'folder-7-navigation',
        name: 'Navigation & Reference',
        description: 'Cross-phase navigation guides and reference materials',
        topics: ${JSON.stringify(topics, null, 8)}
    }
];

const allTopics = folders.flatMap(folder => folder.topics);

export const phase7: Phase = {
    id: 'phase-7',
    number: 7,
    title: 'Navigation Skills & Cross-Phase Integration',
    description: 'Master the learning pathway, understand cross-phase connections, and access quick reference guides for all your learning needs.',
    duration: 'Ongoing reference',
    skills: ['Learning Pathways', 'Cross-Phase Integration', 'Quick Reference', 'Progress Tracking'],
    topics: allTopics,
    folders: folders
};
`;

const outputPath = path.join(fullPath, 'index.ts');
fs.writeFileSync(outputPath, output);
console.log(`Updated ${phaseDir}/index.ts with ${topics.length} navigation topics!`);
