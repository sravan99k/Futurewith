import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const rootDir = process.argv[2];
const phaseNum = process.argv[3] || 'unknown';
const phaseTitle = process.argv[4] || 'Course Phase';
const phaseDescription = process.argv[5] || 'Learn skills for this phase';
const phaseDuration = process.argv[6] || '4-6 weeks';
const phaseSkills = process.argv[7] ? process.argv[7].split(',') : ['Skills'];

if (!rootDir) {
    console.log('Usage: node generate_index.js <phase-dir> <phase-num> "<phase-title>" "<phase-description>" "<phase-duration>" "<skills CSV>"');
    console.log('Example: node generate_index.js phase-1-python-foundations 1 "Python Foundations" "Learn Python programming" "8-10 weeks" "Python,Programming,Logic"');
    process.exit(1);
}

const fullPath = path.join(process.cwd(), 'src', 'data', 'courses', rootDir);

function getFolders() {
    if (!fs.existsSync(fullPath)) {
        console.error(`Directory not found: ${fullPath}`);
        return [];
    }

    const items = fs.readdirSync(fullPath, { withFileTypes: true });
    const folders = items.filter(item => item.isDirectory());

    return folders.map(folder => {
        const folderPath = path.join(fullPath, folder.name);
        const mdFiles = fs.readdirSync(folderPath)
            .filter(file => file.endsWith('.md'))
            .sort();

        // Extract clean name (remove number prefix and convert to title case)
        const cleanName = folder.name.replace(/^\d+[-_]/, '').replace(/-/g, ' ').replace(/_/g, ' ');
        const titleName = cleanName.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

        return {
            id: `folder-${phaseNum}-${folder.name}`,
            name: folder.name, // Keep original name for consistency
            description: `Topics for ${folder.name}`,
            topics: mdFiles.map((file, index) => {
                const fileWithoutExt = file.replace('.md', '');
                // Extract topic name from filename
                const topicName = fileWithoutExt.replace(/^\d+[-_]/, '').replace(/-/g, ' ').replace(/_/g, ' ');
                const titleCased = topicName.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                
                return {
                    id: `${phaseNum}-${folder.name.split('-')[0] || '0'}-${index}`,
                    title: titleCased,
                    description: `Learning about ${fileWithoutExt}`,
                    duration: '1-2 hours',
                    type: 'theory',
                    markdownPath: `/src/data/courses/${rootDir}/${folder.name}/${file}`
                };
            })
        };
    }).filter(f => f.topics.length > 0);
}

const foldersData = getFolders();

if (foldersData.length === 0) {
    console.error('No folders with content found!');
    process.exit(1);
}

const output = `import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = ${JSON.stringify(foldersData, null, 4)};

const allTopics = folders.flatMap(folder => folder.topics);

export const phase${phaseNum}: Phase = {
  id: 'phase-${phaseNum}',
  number: ${phaseNum},
  title: '${phaseTitle}',
  description: '${phaseDescription}',
  duration: '${phaseDuration}',
  skills: ${JSON.stringify(phaseSkills)},
  topics: allTopics,
  folders: folders
};
`;

const outputPath = path.join(fullPath, 'index.ts');
fs.writeFileSync(outputPath, output);
console.log(`Updated ${rootDir}/index.ts with ${foldersData.length} folders and ${foldersData.reduce((acc, f) => acc + f.topics.length, 0)} total topics!`);
