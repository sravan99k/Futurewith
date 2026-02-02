import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, ArrowRight, CheckCircle2, ShieldCheck, Search, Brain, Cpu, Rocket, Lock, Zap, Link as LinkIcon, FileText, FileCheck, Linkedin, Github } from 'lucide-react';
import { VibeProject, vibeProtocolSteps } from '@/data/courses/vibe';
import { Button } from '@/components/ui/button';
import { useCourseProgress } from '@/hooks/useCourseProgress';
import { PhantomTerminal } from '@/components/PhantomTerminal';

interface VibeProjectRendererProps {
    project: VibeProject;
    onComplete: () => void;
    onExit: () => void;
}

export const VibeProjectRenderer = ({ project, onComplete, onExit }: VibeProjectRendererProps) => {
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [verificationValue, setVerificationValue] = useState('');
    const [completedSteps, setCompletedSteps] = useState<number[]>([]);
    const [showCertification, setShowCertification] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const { updateStrength, stats } = useCourseProgress();

    const currentProtocolStep = vibeProtocolSteps[currentStepIndex];
    const projectPhase = project.phases[Math.min(currentStepIndex, project.phases.length - 1)];

    const handleVerify = () => {
        if (!verificationValue.trim()) return;

        // For Build/Deploy/Research phases, we simulate a terminal process
        if (['BUILD', 'DEPLOY', 'RESEARCH', 'LOGIC'].includes(currentProtocolStep.category)) {
            setIsProcessing(true);
        } else {
            completeStep();
        }
    };

    const completeStep = () => {
        setIsProcessing(false);
        if (!completedSteps.includes(currentStepIndex)) {
            setCompletedSteps(prev => [...prev, currentStepIndex]);

            // Map category to strength profile
            const catMap: any = { RESEARCH: 'research', LOGIC: 'logic', BUILD: 'build', DEPLOY: 'integrity' };
            updateStrength(catMap[currentProtocolStep.category] || 'build', 15);
        }

        if (currentStepIndex < vibeProtocolSteps.length - 1) {
            setCurrentStepIndex(prev => prev + 1);
            setVerificationValue('');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            setShowCertification(true);
        }
    };

    const protocolIcons: Record<string, any> = {
        RESEARCH: Search,
        LOGIC: Brain,
        BUILD: Cpu,
        DEPLOY: Rocket
    };

    const Icon = protocolIcons[currentProtocolStep.category];

    if (showCertification) {
        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="max-w-4xl mx-auto p-12 lg:p-16 bg-zinc-950 rounded-[3rem] border border-vibe/30 shadow-2xl relative overflow-hidden text-white"
            >
                <div className="absolute top-0 left-0 w-full h-2 bg-vibe animate-pulse" />
                <div className="absolute top-0 right-0 p-8">
                    <CheckCircle2 className="w-16 h-16 text-vibe opacity-20" />
                </div>

                <div className="relative z-10 text-center space-y-8">
                    <div className="flex justify-center">
                        <div className="w-24 h-24 rounded-3xl bg-vibe/10 flex items-center justify-center border border-vibe/30 shadow-xl shadow-vibe/10">
                            <ShieldCheck className="w-12 h-12 text-vibe" />
                        </div>
                    </div>

                    <div>
                        <div className="text-[10px] font-black text-vibe uppercase tracking-[0.4em] mb-4">Industrial Artifact Verified</div>
                        <h2 className="text-4xl lg:text-6xl font-black italic uppercase tracking-tighter mb-4">
                            Protocol <span className="text-vibe">Verified</span>
                        </h2>
                        <p className="text-zinc-400 text-lg font-medium leading-relaxed max-w-2xl mx-auto">
                            You have successfully architected and deployed a professional-grade AI system:
                            <span className="text-white block mt-2 text-2xl font-black italic">"{project.title}"</span>
                        </p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-6 text-left">
                        <div className="p-8 rounded-3xl bg-white/5 border border-white/10 space-y-6">
                            <h4 className="text-[10px] font-black uppercase tracking-widest text-vibe">Architecture Multiplier</h4>
                            <div className="space-y-4">
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-zinc-500 font-bold uppercase">Logic Fidelity</span>
                                    <span className="text-vibe font-black italic">ELITE</span>
                                </div>
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-zinc-500 font-bold uppercase">Build velocity</span>
                                    <span className="text-vibe font-black italic">10X</span>
                                </div>
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-zinc-500 font-bold uppercase">Artifact Integrity</span>
                                    <span className="text-vibe font-black italic">99.8%</span>
                                </div>
                            </div>
                        </div>

                        <div className="p-8 rounded-3xl bg-white/5 border border-white/10 flex flex-col justify-center gap-4">
                            <Button className="w-full bg-vibe hover:bg-vibe/90 text-vibe-foreground font-black h-12 rounded-xl group">
                                <Linkedin className="w-4 h-4 mr-2" />
                                Post Verification
                            </Button>
                            <Button variant="outline" className="w-full border-white/10 hover:bg-white/5 font-black h-12 rounded-xl text-white">
                                <Github className="w-4 h-4 mr-2" />
                                Sync Project DNA
                            </Button>
                        </div>
                    </div>

                    <div className="pt-8 border-t border-white/5">
                        <Button
                            onClick={onComplete}
                            size="lg"
                            className="bg-white text-black hover:bg-zinc-200 h-16 px-12 rounded-2xl font-black text-xl shadow-xl transition-all hover:scale-105"
                        >
                            Return to Registry
                            <ArrowRight className="ml-3 w-6 h-6" />
                        </Button>
                    </div>
                </div>
            </motion.div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-12">
                <Button variant="ghost" onClick={onExit} className="text-[10px] font-black uppercase tracking-widest text-muted-foreground hover:text-vibe">
                    Abort Protocol
                </Button>
                <div className="flex gap-2">
                    {vibeProtocolSteps.map((_, idx) => (
                        <div
                            key={idx}
                            className={`h-1.5 rounded-full transition-all ${idx === currentStepIndex ? 'w-8 bg-vibe' :
                                completedSteps.includes(idx) ? 'w-4 bg-success' : 'w-4 bg-muted'
                                }`}
                        />
                    ))}
                </div>
            </div>

            <div className="grid lg:grid-cols-12 gap-12">
                {/* Protocol Sidebar */}
                <div className="lg:col-span-4 space-y-6">
                    <div className="p-8 rounded-[2.5rem] bg-zinc-900 border border-white/10 text-white relative overflow-hidden">
                        <div className="absolute top-0 right-0 w-32 h-32 bg-vibe/10 rounded-full blur-3xl -mr-16 -mt-16" />
                        <div className="relative z-10">
                            <div className="text-[10px] font-black text-vibe uppercase tracking-[0.3em] mb-4">Current Phase</div>
                            <div className="w-12 h-12 rounded-2xl bg-vibe/20 flex items-center justify-center text-vibe mb-6 border border-vibe/30">
                                <Icon className="w-6 h-6" />
                            </div>
                            <h3 className="text-2xl font-black uppercase tracking-tight mb-2">{currentProtocolStep.title}</h3>
                            <div className="text-[10px] font-black uppercase text-zinc-500 tracking-widest">{currentProtocolStep.category} Excellence</div>
                        </div>
                    </div>

                    <div className="p-6 rounded-3xl bg-muted/50 border border-border space-y-4">
                        <h4 className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Required Tools</h4>
                        <div className="flex flex-wrap gap-2">
                            {currentProtocolStep.tools.map(t => (
                                <span key={t} className="px-3 py-1 bg-white dark:bg-zinc-800 rounded-lg text-[10px] font-bold border border-border shadow-sm">{t}</span>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Main Workspace */}
                <div className="lg:col-span-8">
                    <motion.div
                        key={currentStepIndex}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="space-y-8"
                    >
                        <div className="p-10 lg:p-12 bg-card border border-border rounded-[3rem] shadow-xl relative">
                            <div className="flex items-center gap-3 mb-8">
                                <Sparkles className="w-5 h-5 text-vibe" />
                                <span className="text-xs font-black uppercase tracking-widest italic">Industrial SOP (Standard Operating Procedure)</span>
                            </div>

                            <div className="space-y-6 mb-12">
                                <h2 className="text-3xl font-black tracking-tight">{projectPhase?.title || project.title}</h2>
                                <p className="text-lg text-muted-foreground leading-relaxed font-medium italic">
                                    "{projectPhase?.task || "Proceed with the industrial protocol implementation."}"
                                </p>
                            </div>

                            <div className="p-8 rounded-[2rem] bg-vibe/5 border border-vibe/20 space-y-6">
                                <div className="flex items-center gap-3">
                                    <ShieldCheck className="w-5 h-5 text-vibe" />
                                    <span className="text-[10px] font-black uppercase tracking-widest">Verification Required</span>
                                </div>

                                <div className="space-y-4 text-sm font-medium text-muted-foreground italic leading-relaxed">
                                    <p>{currentProtocolStep.description}</p>
                                    <div className="p-4 bg-zinc-950 rounded-xl border border-white/5 font-mono text-[11px] text-vibe/80">
                                        Suggested Prompt: {currentProtocolStep.promptRef}
                                    </div>
                                </div>

                                <h4 className="font-heading font-black uppercase text-sm mb-2">Proof of Work</h4>
                                <p className="text-xs text-muted-foreground mb-6 font-medium">
                                    Paste your {currentProtocolStep.verificationType} {currentProtocolStep.verificationType === 'URL' ? 'link' : currentProtocolStep.verificationType === 'PROMPT' ? 'logic' : 'snippet'} to verify this phase.
                                </p>

                                {isProcessing ? (
                                    <PhantomTerminal
                                        mode={currentProtocolStep.category as any}
                                        onComplete={completeStep}
                                    />
                                ) : (
                                    <div className="flex gap-2">
                                        <input
                                            type="text"
                                            value={verificationValue}
                                            onChange={(e) => setVerificationValue(e.target.value)}
                                            placeholder={currentProtocolStep.verificationType === 'URL' ? "https://..." : currentProtocolStep.verificationType === 'PROMPT' ? "Paste refined prompt..." : "Paste verification artifact here..."}
                                            className="flex-1 bg-zinc-950 border border-white/10 rounded-xl px-4 py-3 text-sm font-mono text-white focus:outline-none focus:border-vibe/50 transition-colors"
                                        />
                                        <Button
                                            onClick={handleVerify}
                                            className="bg-vibe hover:bg-vibe/90 text-vibe-foreground font-black uppercase tracking-wide rounded-xl px-6"
                                        >
                                            Verify
                                        </Button>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="flex justify-center gap-8">
                            <div className="flex items-center gap-2 opacity-40">
                                <Lock className="w-3.5 h-3.5" />
                                <span className="text-[10px] font-bold uppercase tracking-widest">End-to-End Encryption Enabled</span>
                            </div>
                            <div className="flex items-center gap-2 opacity-40">
                                <FileCheck className="w-3.5 h-3.5 text-success" />
                                <span className="text-[10px] font-bold uppercase tracking-widest">Artifact Permanently Logged</span>
                            </div>
                        </div>
                    </motion.div>
                </div>
            </div>
        </div>
    );
};
