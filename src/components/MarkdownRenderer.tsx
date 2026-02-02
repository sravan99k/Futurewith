import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github-dark.css';
import { InteractiveCodeBlock } from './InteractiveCodeBlock';
import { LogicGate } from './LogicGate';
import { ParsonsBuilder } from './ParsonsBuilder';

interface MarkdownRendererProps {
    markdownPath?: string;
    content?: string;
    className?: string;
}

export const MarkdownRenderer = ({ markdownPath, content: initialContent, className = '' }: MarkdownRendererProps) => {
    const [markdownContent, setMarkdownContent] = useState<string>(initialContent || '');
    const [loading, setLoading] = useState(!initialContent);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (initialContent) {
            setMarkdownContent(initialContent);
            setLoading(false);
            return;
        }

        const loadMarkdown = async () => {
            try {
                setLoading(true);
                setError(null);

                if (!markdownPath) return;

                // Fetch the markdown file
                const response = await fetch(markdownPath);

                if (!response.ok) {
                    throw new Error(`Failed to load content: ${response.status} ${response.statusText}`);
                }

                const text = await response.text();
                setMarkdownContent(text);
            } catch (err) {
                console.error('Error loading markdown:', err);
                setError(err instanceof Error ? err.message : 'Failed to load content');
                setMarkdownContent('# Content Not Available\n\nThe requested content could not be loaded. Please try again later or contact support.');
            } finally {
                setLoading(false);
            }
        };

        if (markdownPath) {
            loadMarkdown();
        }
    }, [markdownPath, initialContent]);

    if (loading) {
        return (
            <div className="flex items-center justify-center p-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-python"></div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-8 bg-destructive/10 border border-destructive/20 rounded-lg">
                <h3 className="text-lg font-semibold text-destructive mb-2">Error Loading Content</h3>
                <p className="text-muted-foreground">{error}</p>
            </div>
        );
    }

    return (
        <div className={`markdown-content prose prose-lg max-w-none ${className}`}>
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeRaw]}
                components={{
                    // Custom styling for markdown elements
                    h1: ({ node, ...props }) => (
                        <h1 className="font-heading text-4xl font-bold mb-6 mt-8 text-foreground" {...props} />
                    ),
                    h2: ({ node, ...props }) => (
                        <h2 className="font-heading text-3xl font-bold mb-4 mt-6 text-foreground" {...props} />
                    ),
                    h3: ({ node, ...props }) => (
                        <h3 className="font-heading text-2xl font-semibold mb-3 mt-5 text-foreground" {...props} />
                    ),
                    h4: ({ node, ...props }) => (
                        <h4 className="font-heading text-xl font-semibold mb-2 mt-4 text-foreground" {...props} />
                    ),
                    p: ({ node, ...props }) => (
                        <p className="mb-4 leading-relaxed text-foreground/90" {...props} />
                    ),
                    ul: ({ node, ...props }) => (
                        <ul className="list-disc list-inside mb-4 space-y-1" {...props} />
                    ),
                    ol: ({ node, ...props }) => (
                        <ol className="list-decimal list-inside mb-4 space-y-1" {...props} />
                    ),
                    li: ({ node, children, ...props }) => {
                        // Check if children contain block elements like code/pre blocks
                        const hasBlockChild = React.Children.toArray(children).some(
                            (child: any) => 
                                child?.props?.className?.includes('language-') ||
                                child?.type === 'pre' ||
                                child?.type === 'code'
                        );
                        
                        return (
                            <li className={`text-foreground/90 ${hasBlockChild ? 'my-4' : 'leading-relaxed'}`} {...props}>
                                {children}
                            </li>
                        );
                    },
                    code: ({ node, inline, className, children, ...props }: any) => {
                        const match = /language-(\w+)/.exec(className || '');
                        const lang = match ? match[1] : '';

                        if (inline) {
                            return (
                                <code
                                    className="px-1.5 py-0.5 rounded bg-muted text-sm font-mono text-python"
                                    {...props}
                                >
                                    {children}
                                </code>
                            );
                        }

                        // If it's a Python code block, check for Logic Gate or render Interactive IDE
                        if (lang === 'python') {
                            const codeContent = String(children);
                            const lines = codeContent.split('\n');
                            const expectedTrigger = '# EXPECTED: ';

                            const expectedLine = lines.find(l => l.trim().startsWith(expectedTrigger));

                            if (expectedLine) {
                                const expectedOutput = expectedLine.trim().substring(expectedTrigger.length);
                                const descriptionLine = lines.find(l => l.trim().startsWith('# MISSION: '));
                                const description = descriptionLine ? descriptionLine.trim().substring('# MISSION: '.length) : undefined;

                                // Remove meta lines from code
                                const cleanCode = lines
                                    .filter(l => !l.trim().startsWith(expectedTrigger) && !l.trim().startsWith('# MISSION: '))
                                    .join('\n')
                                    .trim();

                                return (
                                    <LogicGate
                                        initialCode={cleanCode}
                                        expectedOutput={expectedOutput}
                                        description={description}
                                    />
                                );
                            }

                            // CHECK FOR PARSONS PROBLEM
                            const parsonsTrigger = '# PARSONS:';
                            const parsonsLine = lines.find(l => l.trim().startsWith(parsonsTrigger));

                            if (parsonsLine) {
                                // Extract Title
                                const title = parsonsLine.trim().substring(parsonsTrigger.length).trim();

                                // Extract Solution (Everything else is considered the solution lines, indented as they should be)
                                // We filter out the trigger line
                                const solutionLines = lines
                                    .filter(l => !l.trim().startsWith(parsonsTrigger))
                                    .filter(l => l.trim() !== ''); // Remove empty lines for cleaner logic

                                // Create scrambled initial lines (remove indentation and shuffle - simple shuffle for now or just random sort)
                                // actually, let's just sort them alphabetically or by length to "scramble" them deterministically for now
                                // or just use a simple randomizer if we accept re-renders reshuffling. 
                                // Better: Just provide them sorted by length to disorient.
                                const initialLines = [...solutionLines]
                                    .map(l => l.trim()) // Remove indent
                                    .sort((a, b) => a.length - b.length); // Scramble by length

                                return (
                                    <ParsonsBuilder
                                        initialLines={initialLines}
                                        solutionLines={solutionLines}
                                        title={title}
                                        description="Reorder the lines and fix the indentation to reconstruct the logic."
                                    />
                                );
                            }

                            return <InteractiveCodeBlock code={codeContent} />;
                        }

                        return (
                            <code
                                className={`block p-4 rounded-lg overflow-x-auto font-mono text-sm ${className || ''}`}
                                {...props}
                            >
                                {children}
                            </code>
                        );
                    },
                    pre: ({ node, ...props }) => (
                        <pre className="mb-6 rounded-lg overflow-hidden bg-foreground/5" {...props} />
                    ),
                    blockquote: ({ node, ...props }) => (
                        <blockquote
                            className="border-l-4 border-python pl-4 italic my-4 text-muted-foreground"
                            {...props}
                        />
                    ),
                    a: ({ node, ...props }) => (
                        <a
                            className="text-python hover:text-python/80 underline font-medium"
                            target="_blank"
                            rel="noopener noreferrer"
                            {...props}
                        />
                    ),
                    table: ({ node, ...props }) => (
                        <div className="overflow-x-auto my-6">
                            <table className="min-w-full border border-border rounded-lg" {...props} />
                        </div>
                    ),
                    th: ({ node, ...props }) => (
                        <th className="border border-border px-4 py-2 bg-muted font-semibold text-left" {...props} />
                    ),
                    td: ({ node, ...props }) => (
                        <td className="border border-border px-4 py-2" {...props} />
                    ),
                    hr: ({ node, ...props }) => (
                        <hr className="my-8 border-border" {...props} />
                    ),
                    img: ({ node, src, ...props }) => {
                        let finalSrc = src;
                        if (src && src.startsWith('./') && markdownPath) {
                            const basePath = markdownPath.substring(0, markdownPath.lastIndexOf('/'));
                            finalSrc = `${basePath}/${src.substring(2)}`;
                        }
                        return (
                            <img
                                className="rounded-lg shadow-md my-6 max-w-full"
                                src={finalSrc}
                                {...props}
                            />
                        );
                    }
                }}
            >
                {markdownContent}
            </ReactMarkdown>
        </div>
    );
};
