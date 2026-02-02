import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = path.join(process.cwd(), 'src', 'data', 'courses', 'phase-4-ai-ml-fundamentals');

function getFolders() {
    const items = fs.readdirSync(rootDir, { withFileTypes: true });
    const folders = items.filter(item => item.isDirectory());

    return folders.map(folder => {
        const folderPath = path.join(rootDir, folder.name);
        const mdFiles = fs.readdirSync(folderPath)
            .filter(file => file.endsWith('.md'))
            .sort();

        return {
            id: `folder-4-${folder.name}`,
            name: folder.name.replace(/^\d+-/, '').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            description: `Topics for ${folder.name}`,
            topics: mdFiles.map((file, index) => ({
                id: `4-${folder.name.split('-')[0] || '0'}-${index}`,
                title: file.replace(/^\d+_/, '').replace('.md', '').split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
                description: `Learning about ${file.replace('.md', '')}`,
                duration: '1-2 hours',
                type: 'theory',
                markdownPath: `/src/data/courses/phase-4-ai-ml-fundamentals/${folder.name}/${file}`
            }))
        };
    }).filter(f => f.topics.length > 0);
}

const foldersData = getFolders();
const output = `import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = ${JSON.stringify(foldersData, null, 2)};

const allTopics = folders.flatMap(folder => folder.topics);

export const phase4: Phase = {
  id: 'phase-4',
  number: 4,
  title: 'AI & Machine Learning Complete',
  description: 'Comprehensive journey from ML fundamentals through deep learning, NLP, computer vision, transformers, and production deployment. Build industry-grade AI applications.',
  duration: '12-16 weeks',
  skills: ['Machine Learning', 'Deep Learning', 'Neural Networks', 'NLP & LLMs', 'Computer Vision', 'Generative AI', 'MLOps'],
  topics: allTopics,
  folders: folders
};
`;

fs.writeFileSync(path.join(rootDir, 'index.ts'), output);
console.log('Phase 4 index.ts updated with all files!');
