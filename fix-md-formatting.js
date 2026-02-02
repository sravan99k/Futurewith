import fs from 'fs';
import path from 'path';
import { glob } from 'glob';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Correctly resolving the start directory
const startDir = path.resolve(process.cwd(), 'src/data/courses');

console.log(`Starting scan in: ${startDir}`);

// Pattern to find markdown files
const pattern = "**/*.md";

// glob regex specific to version 10+ needs explicit call or slightly different syntax if using named export
const files = await glob(pattern, { cwd: startDir, absolute: true });

console.log(`Found ${files.length} markdown files.`);

files.forEach(file => {
    try {
        let content = fs.readFileSync(file, 'utf8');
        let originalContent = content;

        // Regex strategies to fix the formatting issues

        // 1. Fix list items with newlines (e.g., "- \n value")
        // Captures: 1=indent, 2=bullet, 3=text
        content = content.replace(/^(\s*)([-*+])\s*\n\s+(\S)/gm, '$1$2 $3');

        // 2. Fix numbered lists with newlines (e.g., "1. \n value")
        // Captures: 1=indent, 2=number+dot, 3=text
        content = content.replace(/^(\s*)(\d+\.)\s*\n\s+(\S)/gm, '$1$2 $3');

        // 3. Fix "key = value" lines that might have been broken like:
        // "key"
        // = "value"
        content = content.replace(/^(\s*`[^`]+`)\s*\n\s*(=)/gm, '$1 $2');
        content = content.replace(/^(\s*[a-zA-Z_]+)\s*\n\s*(=)/gm, '$1 $2');

        if (content !== originalContent) {
            fs.writeFileSync(file, content, 'utf8');
            console.log(`Fixed formatting in: ${path.relative(startDir, file)}`);
        }
    } catch (readErr) {
        console.error(`Error processing ${file}:`, readErr);
    }
});

console.log("Formatting fix complete.");
