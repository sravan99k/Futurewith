import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowUp, ArrowDown, ChevronRight, ChevronLeft, CheckCircle2, AlertCircle, GripVertical, Shuffle } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ParsonsBuilderProps {
    initialLines: string[]; // Scrambled lines
    solutionLines: string[]; // Correct order with Correct Indentation
    title?: string;
    description?: string;
}

interface LineItem {
    id: string;
    text: string;
    indent: number;
}

export const ParsonsBuilder = ({ initialLines, solutionLines, title, description }: ParsonsBuilderProps) => {
    const [lines, setLines] = useState<LineItem[]>([]);
    const [status, setStatus] = useState<'IDLE' | 'SOLVED' | 'FAILED'>('IDLE');

    useEffect(() => {
        // Initialize lines with 0 indent and unique IDs
        setLines(initialLines.map((text, idx) => ({
            id: `line-${idx}`,
            text: text.trim(), // Remove initial indentation if any, user must fix it
            indent: 0
        })));
    }, [initialLines]);

    const moveLine = (index: number, direction: 'up' | 'down') => {
        if (status === 'SOLVED') return;
        const newLines = [...lines];
        if (direction === 'up' && index > 0) {
            [newLines[index - 1], newLines[index]] = [newLines[index], newLines[index - 1]];
        } else if (direction === 'down' && index < newLines.length - 1) {
            [newLines[index + 1], newLines[index]] = [newLines[index], newLines[index + 1]];
        }
        setLines(newLines);
        setStatus('IDLE');
    };

    const changeIndent = (index: number, direction: 'in' | 'out') => {
        if (status === 'SOLVED') return;
        const newLines = [...lines];
        const currentIndent = newLines[index].indent;
        if (direction === 'in' && currentIndent < 4) {
            newLines[index].indent += 1;
        } else if (direction === 'out' && currentIndent > 0) {
            newLines[index].indent -= 1;
        }
        setLines(newLines);
        setStatus('IDLE');
    };

    const verifyLogic = () => {
        // Construct the user's code block based on order and indentation
        const userCode = lines.map(l => '    '.repeat(l.indent) + l.text);

        // Normalize solution (assuming solutionLines are passed with correct indentation)
        // We compare line by line.
        const isCorrect = userCode.every((line, idx) => line === solutionLines[idx]);

        if (isCorrect && userCode.length === solutionLines.length) {
            setStatus('SOLVED');
        } else {
            setStatus('FAILED');
        }
    };

    return (
        <div className={`my-8 rounded-xl border-2 overflow-hidden transition-all duration-300 ${status === 'SOLVED' ? 'border-vibe bg-vibe/5' :
                status === 'FAILED' ? 'border-red-500/50 bg-red-500/5' : 'border-zinc-800 bg-zinc-950'
            }`}>
            {/* Header */}
            <div className="flex items-center justify-between p-4 bg-zinc-900 border-b border-zinc-800">
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-lg ${status === 'SOLVED' ? 'bg-vibe text-white' : 'bg-blue-500/10 text-blue-400'}`}>
                        <Shuffle className="w-5 h-5" />
                    </div>
                    <div>
                        <div className="text-[10px] font-black uppercase tracking-widest text-zinc-500">Logic Reconstructor</div>
                        <div className="text-xs font-bold text-white">{title || 'Algorithmic Sequencing'}</div>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="p-6">
                {description && (
                    <p className="text-sm text-zinc-400 mb-6 font-medium leading-relaxed bg-zinc-900/50 p-3 rounded-lg border border-zinc-800">
                        {description}
                    </p>
                )}

                <div className="space-y-2 font-mono text-sm max-w-2xl mx-auto">
                    <AnimatePresence>
                        {lines.map((line, idx) => (
                            <motion.div
                                key={line.id}
                                layout
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                className={`group flex items-center gap-2 p-2 rounded-lg border transition-all ${status === 'SOLVED' ? 'border-vibe/30 bg-vibe/10' : 'border-zinc-800 bg-zinc-900 hover:border-zinc-700'
                                    }`}
                                style={{ marginLeft: `${line.indent * 24}px` }} // Visual indentation
                            >
                                <div className="text-zinc-600 cursor-grab active:cursor-grabbing p-1">
                                    <GripVertical className="w-4 h-4" />
                                </div>

                                <span className={status === 'SOLVED' ? 'text-vibe-foreground' : 'text-zinc-300'}>{line.text}</span>

                                <div className="ml-auto flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button onClick={() => changeIndent(idx, 'out')} className="p-1 hover:bg-zinc-800 rounded text-zinc-500 hover:text-white" disabled={line.indent === 0}><ChevronLeft className="w-3 h-3" /></button>
                                    <button onClick={() => changeIndent(idx, 'in')} className="p-1 hover:bg-zinc-800 rounded text-zinc-500 hover:text-white" disabled={line.indent >= 4}><ChevronRight className="w-3 h-3" /></button>
                                    <div className="w-px h-3 bg-zinc-800 mx-1" />
                                    <button onClick={() => moveLine(idx, 'up')} className="p-1 hover:bg-zinc-800 rounded text-zinc-500 hover:text-white" disabled={idx === 0}><ArrowUp className="w-3 h-3" /></button>
                                    <button onClick={() => moveLine(idx, 'down')} className="p-1 hover:bg-zinc-800 rounded text-zinc-500 hover:text-white" disabled={idx === lines.length - 1}><ArrowDown className="w-3 h-3" /></button>
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>

                <div className="mt-8 flex justify-center">
                    <Button
                        onClick={verifyLogic}
                        disabled={status === 'SOLVED'}
                        className={`px-8 h-12 font-black uppercase tracking-widest rounded-xl transition-all ${status === 'SOLVED' ? 'bg-vibe text-white' :
                                status === 'FAILED' ? 'bg-red-500 text-white animate-shake' :
                                    'bg-white text-black hover:scale-105'
                            }`}
                    >
                        {status === 'SOLVED' ? (
                            <>Logic Verified <CheckCircle2 className="ml-2 w-5 h-5" /></>
                        ) : status === 'FAILED' ? (
                            <>Sequence Invalid <AlertCircle className="ml-2 w-5 h-5" /></>
                        ) : (
                            'Verify Sequence'
                        )}
                    </Button>
                </div>
            </div>
        </div>
    );
};
