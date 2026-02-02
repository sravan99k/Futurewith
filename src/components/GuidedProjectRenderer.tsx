import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Rocket, Box, Database, ArrowRight, CheckCircle2, Code, Terminal, Lightbulb, Trophy, Star, Github, Linkedin, FileText, ClipboardList, Copy, Share2, Search, Microscope, Layers, ShieldCheck, Send, Coins, Target, Cpu, Binary } from 'lucide-react';
import { useCourseProgress } from '@/hooks/useCourseProgress';
import { Button } from '@/components/ui/button';
import { MarkdownRenderer } from './MarkdownRenderer';
import { InteractiveCodeBlock } from './InteractiveCodeBlock';

interface Milestone {
    id: string;
    title: string;
    description: string;
    content: string;
    code?: string;
    icon: any;
}

interface Project {
    title: string;
    milestones: Milestone[];
}

interface GuidedProjectRendererProps {
    markdownPath: string;
    onComplete?: () => void;
}

export const GuidedProjectRenderer = ({ markdownPath, onComplete }: GuidedProjectRendererProps) => {
    const [allProjects, setAllProjects] = useState<Project[]>([]);
    const [selectedProjectIndex, setSelectedProjectIndex] = useState<number | null>(null);
    const [currentMilestoneIndex, setCurrentMilestoneIndex] = useState(0);
    const [completedMilestones, setCompletedMilestones] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [showPortfolioScreen, setShowPortfolioScreen] = useState(false);
    const { updateStrength } = useCourseProgress();

    useEffect(() => {
        const fetchProject = async () => {
            try {
                setLoading(true);
                const response = await fetch(markdownPath);
                const text = await response.text();

                // Advanced Parsing: Identify all projects and their components
                const projectSections = text.split(/\n## \d+\./g);
                const titles = text.match(/## \d+\.(.*)/g) || [];

                const projects = projectSections.slice(1).map((projectContent, idx) => {
                    const title = titles[idx]?.replace(/## \d+\./, '').trim() || `Major Project ${idx + 1}`;
                    const milestones: Milestone[] = [];

                    // Phase 1: Problem Definition
                    milestones.push({
                        id: 'problem', title: 'Problem Definition', description: 'Pain Point Analysis',
                        content: projectContent.match(/### Purpose[\s\S]*?###/)?.[0] || 'Define the core problem being solved.',
                        icon: Target
                    });

                    // Phase 2: Research Phase
                    milestones.push({
                        id: 'research', title: 'Research Phase', description: 'AI & Data Discovery',
                        content: projectContent.match(/### Libraries[\s\S]*?###/)?.[0] || 'Identify required AI tools and datasets.',
                        icon: Microscope
                    });

                    // Phase 3: Solution Design
                    milestones.push({
                        id: 'design', title: 'Solution Design', description: 'Logic Architecture',
                        content: projectContent.match(/### Step-by-Step[\s\S]*?###/)?.[0] || 'Design the logic flow and data preparation.',
                        icon: Cpu
                    });

                    // Phase 4: Core Build
                    milestones.push({
                        id: 'build', title: 'Model/Core Build', description: 'Primary Implementation',
                        content: 'Building the core engine of the solution.',
                        code: projectContent.match(/```python([\s\S]*?)```/)?.[1] || '# Core logic goes here',
                        icon: Binary
                    });

                    // Phase 5: Integration
                    milestones.push({
                        id: 'integration', title: 'Integration', description: 'System Synergy',
                        content: 'Connecting the core build with external modules and APIs.',
                        icon: Layers
                    });

                    // Phase 6: Testing & Validation
                    milestones.push({
                        id: 'testing', title: 'Testing & Validation', description: 'Edge Case Protocol',
                        content: 'Ensuring logic reliability and validating against real-world inputs.',
                        icon: ShieldCheck
                    });

                    // Phase 7: Documentation
                    milestones.push({
                        id: 'documentation', title: 'Documentation', description: 'Technical Mapping',
                        content: 'Creating a technical blueprint for future scaling.',
                        icon: FileText
                    });

                    // Phase 8: Deployment
                    milestones.push({
                        id: 'deployment', title: 'Deployment', description: 'Production Release',
                        content: 'Deploying the solution to an active runtime environment.',
                        icon: Send
                    });

                    // Phase 9: Monetization
                    milestones.push({
                        id: 'monetization', title: 'Monetization', description: 'Revenue Logic',
                        content: 'Implementing subscription, API usage, or value-add revenue streams.',
                        icon: Coins
                    });

                    return { title, milestones };
                }).filter(p => p.milestones.length > 0);

                setAllProjects(projects);
                if (projects.length === 1) setSelectedProjectIndex(0);

            } catch (err) {
                console.error("Failed to parse projects:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchProject();
    }, [markdownPath]);

    const activeProject = selectedProjectIndex !== null ? allProjects[selectedProjectIndex] : null;
    const currentMilestone = activeProject?.milestones[currentMilestoneIndex];

    const handleNext = () => {
        if (!currentMilestone || !activeProject) return;

        const milestoneId = `${selectedProjectIndex}-${currentMilestone.id}`;
        if (!completedMilestones.includes(milestoneId)) {
            setCompletedMilestones(prev => [...prev, milestoneId]);

            // Record strength points based on phase
            if (currentMilestone.id === 'problem' || currentMilestone.id === 'research') updateStrength('research', 10);
            if (currentMilestone.id === 'design') updateStrength('logic', 15);
            if (currentMilestone.id === 'build' || currentMilestone.id === 'integration') updateStrength('build', 20);
            if (currentMilestone.id === 'testing' || currentMilestone.id === 'documentation' || currentMilestone.id === 'deployment') updateStrength('integrity', 10);
            if (currentMilestone.id === 'monetization') updateStrength('business', 25);
        }

        if (currentMilestoneIndex < activeProject.milestones.length - 1) {
            setCurrentMilestoneIndex(prev => prev + 1);
            window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
            setShowPortfolioScreen(true);
            if (onComplete) onComplete();
        }
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center p-20 gap-4">
                <div className="w-16 h-16 border-4 border-python border-t-transparent rounded-full animate-spin" />
                <p className="text-muted-foreground font-black tracking-widest uppercase text-xs animate-pulse text-center">
                    Loading Project Logic<span className="block text-[10px] font-normal opacity-60 italic mt-1">Initializing Production Environment</span>
                </p>
            </div>
        );
    }

    // Project Selection Board (Mission Hub)
    if (selectedProjectIndex === null && allProjects.length > 0) {
        return (
            <div className="space-y-12">
                <div className="text-center space-y-4">
                    <h2 className="text-4xl font-black tracking-tight uppercase italic">Production <span className="text-python">Registry</span></h2>
                    <p className="text-muted-foreground max-w-2xl mx-auto font-medium">Select a protocol to begin your professional development. Each project builds verifiable engineering authority.</p>
                </div>

                <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {allProjects.map((project, idx) => (
                        <motion.div
                            key={idx}
                            whileHover={{ y: -10, scale: 1.02 }}
                            className="bg-card border border-border/50 rounded-3xl p-6 shadow-xl hover:shadow-python/10 transition-all cursor-pointer group"
                            onClick={() => setSelectedProjectIndex(idx)}
                        >
                            <div className="flex justify-between items-start mb-6">
                                <div className="p-3 rounded-2xl bg-python/10 text-python group-hover:bg-python group-hover:text-white transition-colors shadow-inner">
                                    <ShieldCheck className="w-8 h-8" />
                                </div>
                                <div className="px-3 py-1 rounded-full bg-muted text-[10px] font-black uppercase tracking-widest">Protocol {idx + 1}</div>
                            </div>
                            <h3 className="text-xl font-bold mb-4 line-clamp-2 leading-tight">{project.title}</h3>
                            <div className="space-y-3 mb-6">
                                <div className="flex items-center gap-2 text-xs text-muted-foreground font-medium">
                                    <Database className="w-4 h-4 text-python/60" />
                                    <span>{project.milestones.length} Strategic Milestones</span>
                                </div>
                                <div className="flex items-center gap-2 text-xs text-muted-foreground font-medium">
                                    <Star className="w-4 h-4 text-amber-500/60" />
                                    <span>Beginner ➔ Pro Difficulty</span>
                                </div>
                            </div>
                            <Button className="w-full bg-python hover:bg-python/90 text-python-foreground flex items-center justify-center gap-2 rounded-xl font-bold">
                                Accept Mission
                                <ArrowRight className="w-4 h-4" />
                            </Button>
                        </motion.div>
                    ))}
                </div>
            </div>
        );
    }

    if (!activeProject || !currentMilestone) return null;

    return (
        <div className="grid lg:grid-cols-12 gap-8">
            {/* Project Navigator (Sidebar) */}
            <div className="lg:col-span-3 space-y-4">
                <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => { setSelectedProjectIndex(null); setCurrentMilestoneIndex(0); }}
                    className="mb-4 text-xs font-bold uppercase tracking-widest text-muted-foreground hover:text-python"
                >
                    <ArrowRight className="w-3 h-3 mr-2 rotate-180" />
                    Back to Hub
                </Button>

                <div className="bg-card border border-border/50 rounded-2xl p-4 sticky top-24 shadow-2xl overflow-hidden group">
                    <div className="absolute top-0 right-0 w-24 h-24 bg-python/5 rounded-full blur-2xl -mr-12 -mt-12" />

                    <h4 className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground/40 mb-5 px-1">Mission Progress</h4>

                    <div className="space-y-2">
                        {activeProject.milestones.map((m, idx) => {
                            const Icon = m.icon;
                            const isActive = currentMilestoneIndex === idx;
                            const isCompleted = completedMilestones.includes(`${selectedProjectIndex}-${m.id}`);

                            return (
                                <button
                                    key={m.id}
                                    onClick={() => setCurrentMilestoneIndex(idx)}
                                    className={`
                                        w-full flex items-center gap-3 p-3 rounded-xl transition-all text-left group
                                        ${isActive ? 'bg-python/10 text-python shadow-sm' : 'hover:bg-muted/50 text-muted-foreground'}
                                    `}
                                >
                                    <div className={`p-2 rounded-lg transition-colors ${isActive ? 'bg-python text-white' : 'bg-muted'}`}>
                                        <Icon className="w-4 h-4" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[11px] font-extrabold truncate uppercase tracking-tight">{m.title}</div>
                                        <div className="text-[9px] opacity-40 font-bold truncate">STEP 0{idx + 1}</div>
                                    </div>
                                    {isCompleted && <CheckCircle2 className="w-3.5 h-3.5 text-success animate-in fade-in zoom-in duration-500" />}
                                </button>
                            );
                        })}
                    </div>

                    <div className="mt-6 pt-6 border-t border-border/50">
                        <div className="p-3 rounded-2xl bg-python/5 border border-python/10 flex items-center gap-3">
                            <Trophy className="w-5 h-5 text-python" />
                            <div>
                                <div className="text-[10px] font-black text-python uppercase tracking-widest leading-none mb-1">XP Bonus</div>
                                <div className="text-xs font-bold">+150 Energy</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Implementation Workspace */}
            <div className="lg:col-span-9">
                <AnimatePresence mode="wait">
                    {showPortfolioScreen ? (
                        <motion.div
                            key="portfolio-screen"
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="p-8 lg:p-12 bg-white dark:bg-zinc-900 rounded-[2.5rem] border border-border shadow-2xl relative overflow-hidden"
                        >
                            <div className="absolute top-0 left-0 w-full h-3 bg-python" />

                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
                                <div>
                                    <h2 className="text-4xl font-black italic mb-2 tracking-tight uppercase">Engineering Authority Synthesis</h2>
                                    <p className="text-muted-foreground font-medium">Synthesize your public technical identity. Focus on production impact and logic ownership.</p>
                                </div>
                                <div className="flex items-center gap-2 px-4 py-2 bg-python/10 rounded-2xl border border-python/20">
                                    <Share2 className="w-5 h-5 text-python" />
                                    <span className="text-xs font-black text-python uppercase tracking-widest">Logic Deployment</span>
                                </div>
                            </div>

                            <div className="grid gap-6">
                                {/* LinkedIn Tool */}
                                <div className="group bg-card border border-border rounded-3xl p-6 hover:border-python/40 transition-all">
                                    <div className="flex items-center gap-4 mb-4">
                                        <div className="p-3 rounded-2xl bg-[#0077b5] text-white">
                                            <Linkedin className="w-6 h-6" />
                                        </div>
                                        <div>
                                            <h4 className="font-black text-sm uppercase tracking-widest">LinkedIn Authority Strategy</h4>
                                            <p className="text-xs text-muted-foreground">High-impact technical articulation</p>
                                        </div>
                                        <Button variant="ghost" size="sm" className="ml-auto rounded-xl">
                                            <Copy className="w-4 h-4 mr-2" />
                                            Copy Strategy
                                        </Button>
                                    </div>
                                    <div className="p-4 bg-muted/30 rounded-2xl text-xs font-mono leading-relaxed border border-dashed border-border group-hover:bg-muted/50 transition-colors italic">
                                        "Architected and deployed a specialized production logic for {activeProject.title}. By integrating advanced Python structural patterns and AI-native workflows, I've implemented [Real World Impact] at scale. View the implementation logic below. #ProductionEngineering #PythonArchitecture #ProofOfWork"
                                    </div>
                                </div>

                                {/* GitHub README Generator */}
                                <div className="group bg-card border border-border rounded-3xl p-6 hover:border-python/40 transition-all">
                                    <div className="flex items-center gap-4 mb-4">
                                        <div className="p-3 rounded-2xl bg-zinc-900 text-white">
                                            <Github className="w-6 h-6" />
                                        </div>
                                        <div>
                                            <h4 className="font-black text-sm uppercase tracking-widest">GitHub Repository README</h4>
                                            <p className="text-xs text-muted-foreground">Standardized engineering documentation</p>
                                        </div>
                                        <Button variant="ghost" size="sm" className="ml-auto rounded-xl">
                                            <Copy className="w-4 h-4 mr-2" />
                                            Copy Template
                                        </Button>
                                    </div>
                                    <div className="p-4 bg-muted/30 rounded-2xl text-[10px] font-mono leading-normal border border-dashed border-border">
                                        # {activeProject.title} <br />
                                        ## Architecture Overview <br />
                                        Explain how the logic flows... [Logic] <br />
                                        ## Tech Stack <br />
                                        - Python 3.10+ <br />
                                        - [Modules Used]
                                    </div>
                                </div>

                                {/* Resume Bullet Points */}
                                <div className="group bg-card border border-border rounded-3xl p-6 hover:border-amber-500/40 transition-all">
                                    <div className="flex items-center gap-4 mb-4">
                                        <div className="p-3 rounded-2xl bg-amber-500/10 text-amber-500">
                                            <FileText className="w-6 h-6" />
                                        </div>
                                        <div>
                                            <h4 className="font-black text-sm uppercase tracking-widest">Resume Bullet Points</h4>
                                            <p className="text-xs text-muted-foreground">Experience-first wording for CVs</p>
                                        </div>
                                        <Button variant="ghost" size="sm" className="ml-auto rounded-xl">
                                            <Copy className="w-4 h-4 mr-2" />
                                            Copy Bullet
                                        </Button>
                                    </div>
                                    <ul className="space-y-2 pl-4">
                                        <li className="text-xs font-bold list-disc">• Architected a {activeProject.title} resulting in 0 downtime during logic simulation.</li>
                                        <li className="text-xs font-bold list-disc">• Optimized Python workflows with AI-integrated prompt engineering patterns.</li>
                                    </ul>
                                </div>
                            </div>

                            <div className="mt-12 flex justify-center gap-4">
                                <Button
                                    onClick={() => { setSelectedProjectIndex(null); setShowPortfolioScreen(false); setCurrentMilestoneIndex(0); }}
                                    variant="outline"
                                    className="h-14 px-10 rounded-2xl font-black text-lg border-white/10 hover:bg-white/5"
                                >
                                    Return to Production Registry
                                </Button>
                                <Button
                                    className="bg-vibe hover:bg-vibe/90 text-vibe-foreground h-14 px-10 rounded-2xl font-black text-lg shadow-xl shadow-vibe/20"
                                    onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
                                >
                                    Next Protocol
                                    <ArrowRight className="w-5 h-5 ml-2" />
                                </Button>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key={`${selectedProjectIndex}-${currentMilestoneIndex}`}
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.3, ease: 'easeOut' }}
                            className="space-y-8"
                        >
                            <div className="p-8 lg:p-12 bg-card border border-border/50 rounded-[2.5rem] shadow-2xl relative overflow-hidden min-h-[700px]">
                                {/* Visual Accents */}
                                <div className="absolute top-0 right-0 w-64 h-64 bg-gradient-to-br from-python/5 to-transparent rounded-full blur-3xl" />
                                <div className="absolute bottom-0 left-0 w-48 h-48 bg-gradient-to-tr from-blue-500/5 to-transparent rounded-full blur-3xl" />

                                {/* Milestone Header */}
                                <div className="flex flex-col sm:flex-row sm:items-center gap-6 mb-12 relative">
                                    <div className="w-16 h-16 rounded-[1.25rem] bg-gradient-to-br from-python to-python/80 text-white flex items-center justify-center shadow-lg shadow-python/20 rotate-3">
                                        {<currentMilestone.icon className="w-8 h-8" />}
                                    </div>
                                    <div>
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="text-[10px] font-black text-python uppercase tracking-[0.3em]">Architecture Milestone</span>
                                            <div className="h-px w-8 bg-python/30" />
                                            <span className="text-[10px] font-bold text-muted-foreground/60">0{currentMilestoneIndex + 1} / 0{activeProject.milestones.length}</span>
                                        </div>
                                        <h2 className="text-4xl font-black tracking-tight leading-none">{currentMilestone.title}</h2>
                                    </div>
                                </div>

                                {/* Milestone Content */}
                                <div className="relative mb-12">
                                    <MarkdownRenderer content={currentMilestone.content} />
                                </div>

                                {/* Interactive Code Blueprint */}
                                {currentMilestone.code && (
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: 0.2 }}
                                        className="space-y-5"
                                    >
                                        <div className="flex items-center justify-between border-b border-border pb-3">
                                            <div className="flex items-center gap-2.5">
                                                <div className="w-8 h-8 rounded-lg bg-zinc-900 border border-white/5 flex items-center justify-center">
                                                    <Terminal className="w-4 h-4 text-python" />
                                                </div>
                                                <span className="text-sm font-black tracking-tight uppercase italic">Production Blueprint</span>
                                            </div>
                                            <div className="flex items-center gap-2 px-3 py-1 bg-python/10 rounded-full">
                                                <div className="w-1.5 h-1.5 rounded-full bg-python animate-pulse" />
                                                <span className="text-[10px] font-bold text-python uppercase">Live Reference</span>
                                            </div>
                                        </div>

                                        <div className="rounded-3xl overflow-hidden border border-border shadow-2xl transition-all hover:border-python/30">
                                            <InteractiveCodeBlock code={currentMilestone.code} readOnly={true} />
                                        </div>

                                        <div className="p-5 rounded-2xl bg-amber-500/5 border border-amber-500/10 flex gap-4 items-start group hover:bg-amber-500/10 transition-colors">
                                            <div className="p-2 rounded-xl bg-amber-500/20">
                                                <Lightbulb className="w-5 h-5 text-amber-500" />
                                            </div>
                                            <div>
                                                <div className="text-[10px] font-black text-amber-600 dark:text-amber-400 uppercase tracking-widest mb-1">Architecture Insight</div>
                                                <p className="text-xs text-muted-foreground font-medium leading-relaxed italic">
                                                    Analyze the class structure and how state is preserved. Use the "Blueprint" as a reference while implementing your own solution locally.
                                                </p>
                                            </div>
                                        </div>
                                    </motion.div>
                                )}

                                {/* Footer Navigation */}
                                <div className="mt-20 flex justify-between items-center pt-8 border-t border-border relative">
                                    <Button
                                        variant="ghost"
                                        onClick={() => setCurrentMilestoneIndex(prev => Math.max(0, prev - 1))}
                                        disabled={currentMilestoneIndex === 0}
                                        className="rounded-2xl h-14 px-8 font-bold text-muted-foreground hover:bg-muted"
                                    >
                                        Previous Stage
                                    </Button>

                                    <Button
                                        onClick={handleNext}
                                        className="bg-python hover:bg-python/90 text-python-foreground h-14 px-10 rounded-2xl font-black text-lg shadow-[0_10px_30px_rgba(55,118,171,0.3)] hover:shadow-python/40 transition-all hover:scale-[1.02] active:scale-[0.98]"
                                    >
                                        {currentMilestoneIndex === activeProject.milestones.length - 1 ? 'Verify Production Artifact' : 'Deploy Progress'}
                                        <ArrowRight className="w-5 h-5 ml-2" />
                                    </Button>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};
