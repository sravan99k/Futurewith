import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Lock, Unlock, AlertTriangle, CheckCircle2, Terminal, Loader2, RotateCcw } from 'lucide-react';
import { getPyodide } from '@/services/pyodideService';
import { Button } from '@/components/ui/button';

interface LogicGateProps {
    initialCode: string;
    expectedOutput: string;
    description?: string;
    onPass?: () => void;
}

export const LogicGate = ({ initialCode, expectedOutput, description, onPass }: LogicGateProps) => {
    const [code, setCode] = useState(initialCode);
    const [output, setOutput] = useState('');
    const [status, setStatus] = useState<'IDLE' | 'RUNNING' | 'LOCKED' | 'UNLOCKED'>('LOCKED');
    const [error, setError] = useState<string | null>(null);

    const runCode = async () => {
        setStatus('RUNNING');
        setError(null);
        setOutput('');
        let consoleOutput = '';

        try {
            const pyodide = await getPyodide();
            pyodide.setStdout({
                batched: (str: string) => { consoleOutput += str + '\n'; }
            });

            await pyodide.runPythonAsync(code);
            const cleanOutput = consoleOutput.trim();
            const cleanExpected = expectedOutput.trim();

            setOutput(consoleOutput);

            if (cleanOutput === cleanExpected) {
                setStatus('UNLOCKED');
                if (onPass) onPass();
            } else {
                setStatus('LOCKED');
                setError(`Logic mismatch. \nExpected: "${cleanExpected}"\nReceived: "${cleanOutput}"`);
            }
        } catch (err: any) {
            setError(err.message);
            setStatus('LOCKED');
        }
    };

    return (
        <div className={`my-8 rounded-xl border-2 overflow-hidden transition-all duration-500 ${status === 'UNLOCKED' ? 'border-vibe bg-vibe/5 shadow-[0_0_30px_rgba(var(--vibe-rgb),0.3)]' :
                status === 'RUNNING' ? 'border-yellow-500/50' : 'border-zinc-800 bg-zinc-950'
            }`}>
            {/* Header / Control Panel */}
            <div className="flex items-center justify-between p-4 bg-zinc-900 border-b border-zinc-800">
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg transition-colors ${status === 'UNLOCKED' ? 'bg-vibe text-white' : 'bg-red-500/10 text-red-500'
                        }`}>
                        {status === 'UNLOCKED' ? <Unlock className="w-5 h-5" /> : <Lock className="w-5 h-5" />}
                    </div>
                    <div>
                        <div className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Logic Gate</div>
                        <div className={`text-xs font-bold ${status === 'UNLOCKED' ? 'text-vibe' : 'text-white'}`}>
                            {status === 'UNLOCKED' ? 'ACCESS GRANTED' : 'VERIFICATION REQUIRED'}
                        </div>
                    </div>
                </div>

                <div className="flex gap-2">
                    <Button
                        onClick={() => setCode(initialCode)}
                        variant="ghost"
                        size="sm"
                        className="text-zinc-500 hover:text-white"
                    >
                        <RotateCcw className="w-4 h-4" />
                    </Button>
                </div>
            </div>

            {/* Description Overlay */}
            {description && (
                <div className="px-5 py-3 bg-zinc-900/50 border-b border-zinc-800 text-xs text-zinc-400 font-mono">
                    <span className="text-vibe mr-2">MISSION:</span>
                    {description}
                </div>
            )}

            {/* Code Editor */}
            <div className="relative group">
                <textarea
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    className="w-full min-h-[150px] p-5 bg-zinc-950 text-emerald-400 font-mono text-sm focus:outline-none resize-y leading-relaxed"
                    spellCheck={false}
                />

                <div className="absolute bottom-4 right-4 sticky float-right z-10">
                    <Button
                        onClick={runCode}
                        disabled={status === 'RUNNING' || status === 'UNLOCKED'}
                        className={`font-black uppercase tracking-wider transition-all ${status === 'UNLOCKED' ? 'bg-vibe text-white' : 'bg-white text-black hover:scale-105'
                            }`}
                    >
                        {status === 'RUNNING' ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                        ) : status === 'UNLOCKED' ? (
                            <>Protocol Verified <CheckCircle2 className="ml-2 w-4 h-4" /></>
                        ) : (
                            <>Execute Logic <Play className="ml-2 w-4 h-4 fill-current" /></>
                        )}
                    </Button>
                </div>
            </div>

            {/* Console Output */}
            <AnimatePresence>
                {(output || error) && (
                    <motion.div
                        initial={{ height: 0 }}
                        animate={{ height: 'auto' }}
                        className={`border-t ${status === 'UNLOCKED' ? 'border-vibe/30 bg-vibe/10' : 'border-zinc-800 bg-black'}`}
                    >
                        <div className="p-4 font-mono text-xs">
                            <div className="flex items-center gap-2 mb-2 opacity-50">
                                <Terminal className="w-3 h-3" />
                                <span className="uppercase tracking-wider">System Output</span>
                            </div>

                            {output && <div className="text-zinc-300 whitespace-pre-wrap mb-2">{output}</div>}

                            {error && (
                                <div className="text-red-400 bg-red-950/30 p-3 rounded border border-red-500/20 flex gap-3 items-start">
                                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                                    <div className="whitespace-pre-wrap">{error}</div>
                                </div>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
