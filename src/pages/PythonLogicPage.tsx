import { motion } from 'framer-motion';
import { ArrowLeft, Code2, CheckCircle2, ArrowRight, Lock, Terminal, Database, Brain, Network, Bot, Layout, Award } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Button } from '@/components/ui/button';
import { useCourseProgress } from '@/hooks/useCourseProgress';
import { FloatingPassport } from '@/components/FloatingPassport';
import { PublicProofMarquee } from '@/components/PublicProofMarquee';
import { useState } from 'react';
import { pythonAICourse, allPhases } from '@/data/courses';

const PythonLogicPage = () => {
    const navigate = useNavigate();
    const { stats } = useCourseProgress();
    const [isUnlocked, setIsUnlocked] = useState(true);

    const phaseIcons = [
        Terminal, Database, Layout, Brain, Network, Bot, Award, Code2, Award
    ];

    return (
        <div className="min-h-screen bg-background">
            <Header />

            <main className="pt-20">
                {/* Hero */}
                <section className="bg-python-muted py-12 relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-[500px] h-[500px] bg-python/5 rounded-full blur-[120px] -ml-64 -mt-64" />
                    <div className="container mx-auto px-4 relative z-10">
                        <Link
                            to="/"
                            className="inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-muted-foreground hover:text-python mb-6 smooth-transition"
                        >
                            <ArrowLeft className="w-3.5 h-3.5" />
                            Return to Hub
                        </Link>

                        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-12">
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="max-w-3xl"
                            >
                                <div className="flex items-center gap-4 mb-6">
                                    <div className="w-16 h-16 rounded-2xl bg-python flex items-center justify-center shadow-xl shadow-python/20">
                                        <Code2 className="w-9 h-9 text-python-foreground" />
                                    </div>
                                    <div>
                                        <div className="text-[10px] font-black text-python uppercase tracking-[.3em] mb-1">The Logic Protocol</div>
                                        <h1 className="font-heading text-4xl sm:text-6xl font-black italic uppercase tracking-tighter leading-none">
                                            Master The <span className="text-python">Engine</span>
                                        </h1>
                                    </div>
                                </div>

                                <p className="text-xl text-muted-foreground mb-8 leading-relaxed font-medium">
                                    Before you generate, you must understand. Master the <span className="text-python font-bold italic">Deep Logic</span> of code, algorithms, and AI systems.
                                    Build the foundation that makes you irreplaceable.
                                </p>

                                <div className="flex flex-col sm:flex-row gap-4">
                                    <Button
                                        onClick={() => setIsUnlocked(true)}
                                        size="lg"
                                        className="bg-python hover:bg-python/90 text-python-foreground h-16 px-12 rounded-2xl font-black text-lg shadow-2xl shadow-python/30 transition-all hover:scale-105"
                                    >
                                        {isUnlocked ? 'Access Granted' : 'Activate Identity Protocol (₹365)'}
                                        {isUnlocked ? <CheckCircle2 className="ml-3 w-5 h-5" /> : <ArrowRight className="ml-3 w-5 h-5" />}
                                    </Button>
                                    <div className="flex items-center px-6 text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 border-l border-border ml-2">
                                        Core Engineering Track
                                    </div>
                                </div>
                            </motion.div>

                            <div className="lg:w-80 space-y-4">
                                <div className="p-6 rounded-3xl bg-background border border-border shadow-2xl">
                                    <div className="text-[10px] font-black text-python uppercase tracking-widest mb-4">Protocol Metrics</div>
                                    <div className="space-y-4">
                                        {[
                                            { label: 'Phases', val: '9 Core' },
                                            { label: 'Duration', val: 'Self-Paced' },
                                            { label: 'Outcome', val: 'AI Engineer' }
                                        ].map(m => (
                                            <div key={m.label} className="flex justify-between items-center border-b border-border pb-3 last:border-0 last:pb-0">
                                                <span className="text-[10px] font-bold text-muted-foreground uppercase">{m.label}</span>
                                                <span className="text-xs font-black text-foreground italic">{m.val}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
                <PublicProofMarquee />

                {/* Curriculum / Phases */}
                <section className="py-24 bg-background relative">
                    <div className="container mx-auto px-4">
                        <div className="text-center mb-16">
                            <h2 className="font-heading text-3xl lg:text-4xl font-black mb-4 uppercase tracking-tighter">
                                The <span className="text-python">Curriculum</span>
                            </h2>
                            <p className="text-muted-foreground font-medium max-w-xl mx-auto">
                                A rigorous journey from syntax to neural networks.
                            </p>
                        </div>

                        <div className="grid gap-6 max-w-5xl mx-auto">
                            {allPhases.map((phase, idx) => {
                                const Icon = phaseIcons[idx % phaseIcons.length];
                                return (
                                    <motion.div
                                        key={phase.id}
                                        initial={{ opacity: 0, y: 20 }}
                                        whileInView={{ opacity: 1, y: 0 }}
                                        viewport={{ once: true }}
                                        transition={{ delay: idx * 0.05 }}
                                        className="group relative"
                                    >
                                        <div className={`p-8 rounded-3xl border border-border bg-card hover:border-python smooth-transition ${!isUnlocked && idx > 0 ? 'opacity-70 grayscale' : ''}`}>
                                            <div className="flex flex-col md:flex-row gap-8 items-start md:items-center">
                                                <div className="w-16 h-16 rounded-2xl bg-python/10 text-python flex items-center justify-center shrink-0">
                                                    <Icon className="w-8 h-8" />
                                                </div>

                                                <div className="flex-1">
                                                    <div className="text-[10px] font-black text-python uppercase tracking-widest mb-2">Phase {phase.number}</div>
                                                    <h3 className="text-2xl font-heading font-bold mb-3">{phase.title}</h3>
                                                    <p className="text-muted-foreground text-sm font-medium leading-relaxed mb-6 max-w-2xl">
                                                        {phase.description}
                                                    </p>

                                                    <div className="flex flex-wrap gap-2">
                                                        {phase.topics.slice(0, 4).map(topic => (
                                                            <span key={topic.id} className="text-[10px] font-bold px-3 py-1 rounded-full bg-muted border border-border-foreground/10 text-muted-foreground uppercase tracking-wider">
                                                                {topic.title}
                                                            </span>
                                                        ))}
                                                        {phase.topics.length > 4 && (
                                                            <span className="text-[10px] font-bold px-3 py-1 rounded-full bg-muted border border-border-foreground/10 text-muted-foreground uppercase tracking-wider">
                                                                +{phase.topics.length - 4} more
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>

                                                <div className="shrink-0 flex flex-col gap-3 w-full md:w-auto">
                                                    {!isUnlocked && idx > 0 ? (
                                                        <Button variant="outline" className="w-full justify-start gap-2 cursor-pointer" onClick={() => setIsUnlocked(true)}>
                                                            <Lock className="w-4 h-4" /> Unlock Phase
                                                        </Button>
                                                    ) : (
                                                        <Button
                                                            className="w-full bg-python hover:bg-python/90 text-python-foreground font-bold justify-between group-hover:pl-6 transition-all"
                                                            onClick={() => navigate(`/phase/${phase.id}`)}
                                                        >
                                                            Start Phase <ArrowRight className="w-4 h-4 ml-2" />
                                                        </Button>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </div>
                    </div>
                </section>

                {/* Footer CTA */}
                <section className="py-32 relative overflow-hidden bg-background border-t border-border">
                    <div className="absolute inset-0 bg-python opacity-5 blur-[100px] rounded-full scale-150" />
                    <div className="container mx-auto px-4 relative z-10">
                        <div className="max-w-4xl mx-auto text-center">
                            <h2 className="text-4xl lg:text-7xl font-black italic uppercase tracking-tighter mb-8 leading-none">
                                Build Your <span className="text-python">Foundation</span>
                            </h2>
                            <p className="text-muted-foreground text-xl font-medium mb-12 max-w-3xl mx-auto leading-relaxed">
                                Don't just be a user of AI. Be the engineer who builds it.
                            </p>
                            <div className="flex flex-col sm:flex-row gap-6 justify-center">
                                <Button
                                    onClick={() => setIsUnlocked(true)}
                                    size="lg"
                                    className="bg-python text-python-foreground hover:bg-python/90 h-16 px-12 rounded-2xl font-black text-xl shadow-2xl transition-all hover:scale-105 active:scale-95"
                                >
                                    Activate Protocol (₹365)
                                    <ArrowRight className="ml-3 w-6 h-6" />
                                </Button>
                            </div>
                        </div>
                    </div>
                </section>
            </main>

            <FloatingPassport />
            <Footer />
        </div>
    );
};

export default PythonLogicPage;
