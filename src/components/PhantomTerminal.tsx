import { useState, useEffect, useRef } from 'react';
import { Terminal, CheckCircle2, Loader2, AlertTriangle, ShieldCheck } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface PhantomTerminalProps {
    mode: 'BUILD' | 'DEPLOY' | 'RESEARCH' | 'LOGIC';
    onComplete: () => void;
}

const LOG_TEMPLATES = {
    BUILD: [
        { text: "> initializing build environment...", delay: 500 },
        { text: "> resolving dependencies...", delay: 800 },
        { text: "[npm] installing @ai-sdk/react...", delay: 1200 },
        { text: "[npm] installing framer-motion...", delay: 400 },
        { text: "> compiling source code...", delay: 1500 },
        { text: "√ types verification passed", delay: 600 },
        { text: "> optimizing static assets...", delay: 1000 },
        { text: "√ build successful (842ms)", delay: 500 }
    ],
    DEPLOY: [
        { text: "> verifying auth tokens...", delay: 600 },
        { text: "> connecting to edge network...", delay: 900 },
        { text: "> pushing build artifacts to [us-east-1]...", delay: 1500 },
        { text: "> engaging cold start prevention...", delay: 800 },
        { text: "√ database migrations applied", delay: 700 },
        { text: "> propagating DNS...", delay: 1200 },
        { text: "√ deployment active: https://vibe.app/prod", delay: 500 }
    ],
    RESEARCH: [
        { text: "> initializing brave search api...", delay: 600 },
        { text: "> crawling source: [arxiv.org]...", delay: 1000 },
        { text: "> crawling source: [github.com]...", delay: 800 },
        { text: "> synthesizing vectors...", delay: 1500 },
        { text: "√ identified 14 citation matches", delay: 500 },
        { text: "> generating summary matrix...", delay: 900 },
        { text: "√ research manifest compiled", delay: 400 }
    ],
    LOGIC: [
        { text: "> analyzing prompt structure...", delay: 700 },
        { text: "> tokenizing context window...", delay: 600 },
        { text: "> verifying logical coherence...", delay: 1200 },
        { text: "> detecting hallucinations...", delay: 1000 },
        { text: "√ logic gates passed (confidence: 98%)", delay: 500 },
        { text: "> optimizing chain-of-thought...", delay: 800 },
        { text: "√ blueprint validated", delay: 400 }
    ]
};

export const PhantomTerminal = ({ mode, onComplete }: PhantomTerminalProps) => {
    const [lines, setLines] = useState<string[]>([]);
    const [isFinished, setIsFinished] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let currentIndex = 0;
        let mounted = true;
        const template = LOG_TEMPLATES[mode] || LOG_TEMPLATES.BUILD;

        const processSequence = async () => {
            while (currentIndex < template.length && mounted) {
                const item = template[currentIndex];
                await new Promise(r => setTimeout(r, item.delay));

                if (mounted) {
                    setLines(prev => [...prev, item.text]);
                    if (scrollRef.current) {
                        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
                    }
                    currentIndex++;
                }
            }
            if (mounted) {
                setIsFinished(true);
                setTimeout(onComplete, 1500); // Wait a bit before auto-closing/completing
            }
        };

        processSequence();

        return () => { mounted = false; };
    }, [mode, onComplete]);

    return (
        <div className="w-full bg-black border border-zinc-800 rounded-xl overflow-hidden font-mono text-xs shadow-2xl">
            <div className="flex items-center justify-between px-4 py-2 bg-zinc-900 border-b border-zinc-800 text-zinc-500">
                <div className="flex items-center gap-2">
                    <Terminal className="w-3.5 h-3.5" />
                    <span className="text-[10px] uppercase tracking-widest">Vibe Execution Environment</span>
                </div>
                <div className="flex gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                    <div className="w-2 h-2 rounded-full bg-zinc-700" />
                </div>
            </div>

            <div
                ref={scrollRef}
                className="p-4 h-[200px] overflow-y-auto space-y-2 text-zinc-400"
            >
                <AnimatePresence>
                    {lines.map((line, idx) => (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className={`${line.startsWith('√') ? 'text-emerald-400 font-bold' : line.startsWith('[npm]') ? 'text-zinc-500' : ''}`}
                        >
                            {line}
                        </motion.div>
                    ))}
                </AnimatePresence>
                {!isFinished && (
                    <motion.div
                        animate={{ opacity: [0, 1, 0] }}
                        transition={{ repeat: Infinity, duration: 0.8 }}
                        className="w-1.5 h-4 bg-vibe inline-block align-middle ml-1"
                    />
                )}
            </div>

            {isFinished && (
                <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: 'auto' }}
                    className="border-t border-zinc-800 p-2 bg-emerald-500/10 text-emerald-400 flex items-center justify-center gap-2 text-[10px] uppercase font-bold tracking-widest"
                >
                    <ShieldCheck className="w-3.5 h-3.5" />
                    Process Verified
                </motion.div>
            )}
        </div>
    );
};
