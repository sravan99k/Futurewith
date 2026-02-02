import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, RotateCcw, Check, Copy, Trash2, Loader2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getPyodide } from '@/services/pyodideService';

interface InteractiveCodeBlockProps {
    code: string;
    language?: string;
    readOnly?: boolean;
    compact?: boolean;
}

export const InteractiveCodeBlock = ({ code: initialCode, readOnly = false, compact = false }: InteractiveCodeBlockProps) => {
    const [code, setCode] = useState(initialCode.trim());
    const [output, setOutput] = useState('');
    const [isRunning, setIsRunning] = useState(false);
    const [isInitializing, setIsInitializing] = useState(false);
    const [copied, setCopied] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const runCode = async () => {
        setIsRunning(true);
        setError(null);
        let consoleOutput = '';

        try {
            setIsInitializing(true);
            const pyodide = await getPyodide();
            setIsInitializing(false);

            // Set up stdout redirection
            pyodide.setStdout({
                batched: (str: string) => {
                    consoleOutput += str + '\n';
                }
            });

            await pyodide.runPythonAsync(code);
            setOutput(consoleOutput || 'Code executed successfully (no output).');
        } catch (err: any) {
            setError(err.message);
            setIsInitializing(false);
        } finally {
            setIsRunning(false);
        }
    };

    const resetCode = () => {
        setCode(initialCode.trim());
        setOutput('');
        setError(null);
    };

    const copyCode = async () => {
        await navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={`${compact ? 'my-2' : 'my-8'} group relative rounded-2xl border border-border bg-[#1e1e1e] overflow-hidden shadow-lg transition-all hover:shadow-xl hover:border-python/30`}>
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2 bg-[#252525] border-b border-white/5">
                <div className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f56]" />
                    <div className="w-2.5 h-2.5 rounded-full bg-[#ffbd2e]" />
                    <div className="w-2.5 h-2.5 rounded-full bg-[#27c93f]" />
                    <span className="ml-2 text-[10px] font-mono text-muted-foreground/60 uppercase tracking-widest">Python 3.12</span>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={copyCode}
                        className="p-1.5 rounded hover:bg-white/10 text-muted-foreground/60 transition-colors"
                        title="Copy Code"
                    >
                        {copied ? <Check className="w-3.5 h-3.5 text-success" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                    <button
                        onClick={resetCode}
                        className="p-1.5 rounded hover:bg-white/10 text-muted-foreground/60 transition-colors"
                        title="Reset Example"
                    >
                        <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                </div>
            </div>

            {/* Code Editor Area */}
            <div className="relative">
                <textarea
                    value={code}
                    onChange={(e) => !readOnly && setCode(e.target.value)}
                    readOnly={readOnly}
                    className={`w-full min-h-[120px] p-5 bg-transparent text-[#d4d4d4] font-mono text-sm leading-relaxed focus:outline-none resize-y caret-python ${readOnly ? 'cursor-default' : ''}`}
                    spellCheck={false}
                    rows={code.split('\n').length}
                />

                {/* Play Button Overlay (bottom right of editor) */}
                {!isRunning && !output && !error && (
                    <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button
                            onClick={runCode}
                            size="sm"
                            className="bg-python hover:bg-python/90 text-python-foreground h-8 px-4 rounded-full font-bold shadow-lg"
                        >
                            <Play className="w-3 h-3 mr-1.5 fill-current" />
                            Try it
                        </Button>
                    </div>
                )}
            </div>

            {/* Action Bar (when not running or after run) */}
            {(isRunning || output || error) && (
                <div className="px-5 pb-4">
                    <Button
                        onClick={runCode}
                        disabled={isRunning}
                        size="sm"
                        className={`${output || error ? 'bg-muted hover:bg-muted/80 text-foreground' : 'bg-python hover:bg-python/90 text-python-foreground'} h-8 px-4 rounded-full font-bold transition-all`}
                    >
                        {isRunning ? (
                            <>
                                <Loader2 className="w-3 h-3 mr-1.5 animate-spin" />
                                {isInitializing ? 'Waking Python...' : 'Running...'}
                            </>
                        ) : (
                            <>
                                <Play className="w-3 h-3 mr-1.5 fill-current" />
                                {output || error ? 'Run Again' : 'Run Code'}
                            </>
                        )}
                    </Button>
                </div>
            )}

            {/* Output & Error Section */}
            <AnimatePresence>
                {(output || error) && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-t border-white/5 bg-[#151515]"
                    >
                        <div className="p-5 font-mono text-sm">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-[10px] text-muted-foreground/40 uppercase tracking-tighter">Console Output</span>
                                <button onClick={() => { setOutput(''); setError(null); }} className="text-muted-foreground/40 hover:text-white transition-colors">
                                    <Trash2 className="w-3 h-3" />
                                </button>
                            </div>

                            {error ? (
                                <div className="text-red-400 bg-red-400/5 p-3 rounded-lg border border-red-400/10 flex items-start gap-2">
                                    <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                                    <pre className="whitespace-pre-wrap text-xs leading-relaxed">{error}</pre>
                                </div>
                            ) : (
                                <pre className="text-python-muted whitespace-pre-wrap leading-relaxed">
                                    {output}
                                </pre>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
