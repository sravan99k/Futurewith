import { motion } from 'framer-motion';
import { Brain, Sparkles, CheckCircle2, ArrowLeft, ArrowRight } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { useCourseProgress } from '@/hooks/useCourseProgress';

export const VibeHero = () => {
  const navigate = useNavigate();
  const { isCourseUnlocked } = useCourseProgress();

  return (
    <section className="bg-vibe-muted py-16 lg:py-24 relative overflow-hidden">
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-vibe/5 rounded-full blur-[120px] -mr-64 -mt-64" />
      <div className="container mx-auto px-4 relative z-10">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-muted-foreground hover:text-vibe mb-12 smooth-transition"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Return to Protocol Hub
        </Link>

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-3xl"
          >
            <div className="flex items-center gap-4 mb-8">
              <div className="w-16 h-16 rounded-2xl bg-vibe flex items-center justify-center shadow-xl shadow-vibe/20">
                <Brain className="w-9 h-9 text-vibe-foreground" />
              </div>
              <div>
                <div className="text-[10px] font-black text-vibe uppercase tracking-[.3em] mb-1">The Industrial Journey</div>
                <h1 className="font-heading text-4xl sm:text-6xl font-black italic uppercase tracking-tighter leading-none">
                  AI Engineering <span className="text-vibe">Protocol</span>
                </h1>
              </div>
            </div>

            <p className="text-xl text-muted-foreground mb-10 leading-relaxed font-medium">
              A systematic 9-phase industrial pipeline for building production-ready AI applications.
              From problem definition to monetization—master the art of AI Engineering.
            </p>

            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                onClick={() => navigate('/checkout?course=vibe')}
                size="lg"
                className="bg-vibe hover:bg-vibe/90 text-vibe-foreground h-16 px-12 rounded-2xl font-black text-lg shadow-2xl shadow-vibe/30 transition-all hover:scale-105"
              >
                {isCourseUnlocked('vibe') ? 'Access Granted' : 'Start Learning (₹199)'}
                {isCourseUnlocked('vibe') ? <CheckCircle2 className="ml-3 w-5 h-5" /> : <Sparkles className="ml-3 w-5 h-5" />}
              </Button>
              <div className="flex items-center px-6 text-[10px] font-black uppercase tracking-widest text-muted-foreground/60 border-l border-border ml-2">
                9 Phases • Industrial Pipeline
              </div>
            </div>
          </motion.div>

          <div className="lg:w-80 space-y-4">
            <div className="p-6 rounded-3xl bg-zinc-900 border border-white/10 shadow-2xl">
              <div className="text-[10px] font-black text-vibe uppercase tracking-widest mb-4">What You Will Build</div>
              <div className="space-y-4">
                {[
                  { label: 'Real Projects', val: '5+ Portfolio Pieces' },
                  { label: 'Tech Stack', val: 'Next.js + Supabase' },
                  { label: 'Deployment', val: 'Live URL' }
                ].map(m => (
                  <div key={m.label} className="flex justify-between items-center border-b border-white/5 pb-3 last:border-0 last:pb-0">
                    <span className="text-[10px] font-bold text-zinc-500 uppercase">{m.label}</span>
                    <span className="text-xs font-black text-white italic">{m.val}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
