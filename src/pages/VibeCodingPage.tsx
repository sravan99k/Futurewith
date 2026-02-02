import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Webhook, Globe2, Smartphone, Code2, Zap, Clock, Shield, Search, Terminal, Brain, Database, Bot, Rocket, Coins, ArrowRight } from 'lucide-react';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { FloatingPassport } from '@/components/FloatingPassport';
import { VibeProjectRenderer } from '@/components/VibeProjectRenderer';
import { PublicProofMarquee } from '@/components/PublicProofMarquee';
import { useCourseProgress } from '@/hooks/useCourseProgress';
import type { VibeProject, VibeTool } from '@/data/courses/vibe';
import {
  vibeProjects,
  webScrapingTools,
  websiteTools,
  appDevTools,
  getFreeTools,
  getAllEngineeringCategories,
  getEngineeringConceptsByCategory,
  engineeringConcepts,
} from '@/data/courses/vibe';

import { vibePhases } from '@/data/courses/vibe-ai-engineering';
import { Button } from '@/components/ui/button';

// Import separated components
import {
  VibeHero,
  VibeEngineeringLayer,
  VibeSkillsPassport,
  VibeCTASection
} from '@/components/vibe';

type ToolCategory = 'all' | 'ide' | 'scraping' | 'website' | 'appdev' | 'automation';

const VibeCodingPage = () => {
  const navigate = useNavigate();
  const [selectedProject, setSelectedProject] = useState<VibeProject | null>(null);
  const [activeToolCategory, setActiveToolCategory] = useState<ToolCategory>('all');

  const toolCategories = [
    { id: 'all', label: 'All Tools' },
    { id: 'ide', label: 'AI IDEs' },
    { id: 'website', label: 'Web Dev' },
    { id: 'appdev', label: 'App Dev' },
    { id: 'scraping', label: 'Scraping' },
    { id: 'automation', label: 'Automation' }
  ];

  const getFilteredTools = (): VibeTool[] => {
    switch (activeToolCategory) {
      case 'scraping':
        return webScrapingTools;
      case 'website':
        return websiteTools;
      case 'appdev':
        return appDevTools;
      case 'ide':
        return getFreeTools().filter(t => t.category === 'IDE');
      case 'automation':
        return getFreeTools().filter(t => t.category === 'Automation' || t.category === 'Database');
      default:
        return getFreeTools();
    }
  };

  const getCategoryInfo = () => {
    switch (activeToolCategory) {
      case 'scraping':
        return { title: 'Web Scraping Tools', description: 'Free AI-powered tools for extracting data from websites', icon: Webhook };
      case 'website':
        return { title: 'Website Development', description: 'Build stunning websites with AI-powered tools', icon: Globe2 };
      case 'appdev':
        return { title: 'App Development', description: 'Create mobile and desktop applications', icon: Smartphone };
      case 'ide':
        return { title: 'AI Code Editors', description: 'Next-generation IDEs with AI assistance', icon: Code2 };
      case 'automation':
        return { title: 'Automation & Backend', description: 'Connect apps and build backend services', icon: Zap };
      default:
        return { title: 'Complete Tool Stack', description: 'All AI tools for modern development', icon: Code2 };
    }
  };

  const phaseIcons = [
    Terminal, Search, Database, Brain, Bot, Shield, Code2, Rocket, Coins
  ];

  if (selectedProject) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="pt-32 pb-20 container mx-auto px-4">
          <VibeProjectRenderer
            project={selectedProject}
            onComplete={() => {
              setSelectedProject(null);
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            onExit={() => setSelectedProject(null)}
          />
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="pt-20">
        {/* Hero Section */}
        <VibeHero />

        {/* Industrial Pipeline (9 Phases) */}
        <section className="py-24 bg-background">
          <div className="container mx-auto px-4">
            <div className="text-center mb-16">
              <h2 className="font-heading text-3xl lg:text-5xl font-black mb-4 uppercase tracking-tighter italic">
                The 9-Phase <span className="text-vibe">Industrial Pipeline</span>
              </h2>
              <p className="text-muted-foreground font-medium max-w-xl mx-auto">
                Master the exact steps used to build and monetize AI applications in the modern industry.
              </p>
            </div>

            <div className="grid gap-6 max-w-5xl mx-auto">
              {vibePhases.map((phase, idx) => {
                const Icon = phaseIcons[idx % phaseIcons.length];
                return (
                  <motion.div
                    key={phase.id}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: idx * 0.05 }}
                    className="group"
                  >
                    <div className="p-8 rounded-[2.5rem] bg-card border border-border/50 hover:border-vibe/50 hover:shadow-2xl hover:shadow-vibe/5 smooth-transition relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-vibe/5 rounded-full blur-3xl -mr-16 -mt-16 group-hover:bg-vibe/10 transition-colors" />

                      <div className="flex flex-col md:flex-row gap-8 items-start md:items-center">
                        <div className="w-16 h-16 rounded-2xl bg-vibe/10 text-vibe flex items-center justify-center shrink-0 group-hover:bg-vibe group-hover:text-vibe-foreground transition-all duration-500">
                          <Icon className="w-8 h-8" />
                        </div>

                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <span className="text-[10px] font-black text-vibe uppercase tracking-[0.2em]">Phase {phase.number}</span>
                            <div className="flex items-center gap-1.5 text-[10px] font-bold text-muted-foreground uppercase">
                              <Clock className="w-3 h-3" />
                              {phase.duration}
                            </div>
                          </div>
                          <h3 className="text-2xl font-black uppercase tracking-tight mb-3 italic">{phase.title}</h3>
                          <p className="text-muted-foreground text-sm font-medium leading-relaxed mb-6 max-w-2xl">
                            {phase.description}
                          </p>

                          <div className="flex flex-wrap gap-2 mb-6">
                            {phase.skills.map(skill => (
                              <span key={skill} className="text-[9px] font-black px-3 py-1 rounded-lg bg-muted border border-border text-muted-foreground uppercase tracking-wider">
                                {skill}
                              </span>
                            ))}
                          </div>

                          {/* Industrial Context Grid */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-5 rounded-2xl bg-zinc-950/50 border border-white/5">
                            <div>
                              <div className="text-[9px] font-black text-vibe uppercase tracking-widest mb-1">Industrial Role</div>
                              <div className="text-xs font-bold text-foreground italic">{phase.role}</div>
                            </div>
                            <div>
                              <div className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-1">Requirements</div>
                              <div className="flex flex-wrap gap-1">
                                {phase.requirements?.map(req => (
                                  <div key={req} className="text-[9px] font-bold text-zinc-300">
                                    • {req}
                                  </div>
                                ))}
                              </div>
                            </div>
                            <div>
                              <div className="text-[9px] font-black text-zinc-500 uppercase tracking-widest mb-1">Tech Stack</div>
                              <div className="flex flex-wrap gap-1">
                                {phase.techStack?.map(tech => (
                                  <span key={tech} className="text-[9px] font-bold text-zinc-400">{tech}</span>
                                ))}
                              </div>
                            </div>
                            <div>
                              <div className="text-[9px] font-black text-python uppercase tracking-widest mb-1">AI Tools</div>
                              <div className="flex flex-wrap gap-1">
                                {phase.aiTools?.map(tool => (
                                  <span key={tool} className="text-[9px] font-bold text-python/80">{tool}</span>
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="shrink-0 w-full md:w-auto mt-4 md:mt-0">
                          <Button
                            onClick={() => navigate(`/phase/${phase.id}`)}
                            className="w-full md:w-auto bg-vibe hover:bg-vibe/90 text-vibe-foreground font-black px-8 py-6 rounded-2xl flex items-center justify-between gap-4 group-hover:shadow-xl group-hover:shadow-vibe/20 transition-all active:scale-95"
                          >
                            Enter Phase
                            <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </section>



        {/* Supporting Modules */}
        <section className="py-20 bg-vibe-muted border-y border-border/50">
          <div className="container mx-auto px-4 text-center mb-12">
            <h3 className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.5em] mb-4 text-center">Industrial Tooling & Security</h3>
          </div>

          <VibeEngineeringLayer
            categories={getAllEngineeringCategories()}
            concepts={engineeringConcepts}
            frameworks={[]}
            getAllCategories={getAllEngineeringCategories}
            filterByCategory={(catId) => getEngineeringConceptsByCategory(catId as any)}
          />
        </section>

        {/* Projects Section */}


        {/* CTA Section */}
        <VibeCTASection />
      </main>

      <FloatingPassport />
      <PublicProofMarquee />
      <Footer />
    </div>
  );
};

export default VibeCodingPage;
