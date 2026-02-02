
import { motion } from 'framer-motion';
import { ArrowRight, Code2, Brain, Sparkles, Zap, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';
import { CareerCounselingModal } from '@/components/CareerCounselingModal';

const HeroSection = () => {
  const navigate = useNavigate();

  return (
    <section className="min-h-screen pt-16 flex flex-col lg:flex-row relative">

      {/* Left Side - Vibe Coding (The Multiplier Layer) */}
      <motion.div
        initial={{ opacity: 0, x: -30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
        className="flex-1 bg-vibe-muted flex flex-col justify-center px-6 lg:px-12 py-12 lg:py-0 min-h-[50vh] lg:min-h-0 relative overflow-hidden"
      >
        <div className="absolute bottom-0 left-0 p-8 opacity-5">
          <Brain className="w-64 h-64 -rotate-12" />
        </div>
        <div className="max-w-xl mx-auto lg:mx-0 lg:ml-auto lg:mr-16 relative z-10">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 rounded-xl bg-vibe flex items-center justify-center shadow-lg shadow-vibe/20">
              <Brain className="w-7 h-7 text-vibe-foreground" />
            </div>
            <div className="flex flex-col">
              <span className="text-vibe font-heading font-black text-xs uppercase tracking-widest leading-none">The Multiplier Layer</span>
              <span className="text-[10px] font-bold text-muted-foreground mt-1 text-vibe">AI-NATIVE BUILDING SPEED</span>
            </div>
          </div>

          <h1 className="font-heading text-3xl sm:text-4xl lg:text-5xl font-bold mb-6 leading-tight">
            Build Apps at the <span className="text-vibe italic">Speed of Vibe</span>
          </h1>

          <p className="text-muted-foreground text-base sm:text-lg mb-8 leading-relaxed font-medium">
            Master the tools that let you architect entire products in minutes. Use AI, Agents, and Prompt Engineering to turn logic into market-ready MVPs.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 mb-8">
            <Button
              size="lg"
              className="bg-vibe hover:bg-vibe/90 text-vibe-foreground font-black w-full sm:w-auto h-14 rounded-2xl shadow-xl shadow-vibe/20 transition-all hover:scale-105 active:scale-95"
              onClick={() => navigate('/vibe-coding')}
            >
              Activate Vibe Protocol (₹499)
              <Sparkles className="ml-2 w-5 h-5" />
            </Button>

          </div>

          <div className="flex flex-wrap gap-4 text-[10px] font-black uppercase tracking-widest text-muted-foreground/60">
            <div className="flex items-center gap-2">
              <Zap className="w-3.5 h-3.5 text-vibe" />
              <span>Prompt Engineering</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-3.5 h-3.5 text-vibe" />
              <span>AI Agents</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap className="w-3.5 h-3.5 text-vibe" />
              <span>Product Launch</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Center Circle Trigger (Desktop) */}
      <div className="hidden lg:block absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-[40%] z-20">
        <CareerCounselingModal>
          <button className="group relative">
            <div className="w-40 h-40 rounded-full bg-background border-[6px] border-background shadow-2xl flex flex-col items-center justify-center relative overflow-hidden transition-transform duration-500 hover:scale-110 hover:rotate-3">
              <div className="absolute inset-0 bg-gradient-to-br from-vibe/10 via-background to-python/10 opacity-100 group-hover:opacity-90 transition-opacity" />

              <div className="relative z-10 flex flex-col items-center gap-2 p-4 text-center">
                <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center mb-1">
                  <span className="text-lg">🤔</span>
                </div>
                <div>
                  <span className="block text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-0.5">Confused?</span>
                  <span className="block text-sm font-black leading-tight">Find Your <br /> Path</span>
                </div>
              </div>
            </div>
          </button>
        </CareerCounselingModal>
      </div>

      {/* Center Trigger (Mobile) */}
      <div className="lg:hidden relative z-20 -my-6 flex justify-center pointer-events-none">
        <div className="pointer-events-auto">
          <CareerCounselingModal>
            <button className="relative group">
              <div className="w-16 h-16 rounded-full bg-background border-4 border-background shadow-xl flex items-center justify-center relative overflow-hidden">
                <div className="text-xl">🤔</div>
              </div>
              <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground whitespace-nowrap bg-background/80 px-2 py-0.5 rounded-full backdrop-blur-sm">
                Find Path
              </span>
            </button>
          </CareerCounselingModal>
        </div>
      </div>

      {/* Right Side - Python Course (The Logic Layer) */}
      <motion.div
        initial={{ opacity: 0, x: 30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="flex-1 bg-python-muted flex flex-col justify-center px-6 lg:px-12 py-12 lg:py-0 min-h-[50vh] lg:min-h-0 relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 p-8 opacity-5">
          <Code2 className="w-64 h-64 rotate-12" />
        </div>
        <div className="max-w-xl mx-auto lg:mx-0 lg:ml-16 relative z-10">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 rounded-xl bg-python flex items-center justify-center shadow-lg shadow-python/20">
              <Code2 className="w-7 h-7 text-python-foreground" />
            </div>
            <div className="flex flex-col">
              <span className="text-python font-heading font-black text-xs uppercase tracking-widest leading-none">The Logic Layer</span>
              <span className="text-[10px] font-bold text-muted-foreground mt-1">CORE ENGINEERING FOUNDATION</span>
            </div>
          </div>

          <h1 className="font-heading text-3xl sm:text-4xl lg:text-5xl font-bold mb-6 leading-tight">
            Master the <span className="text-python italic">Engine</span> of AI
          </h1>

          <p className="text-muted-foreground text-base sm:text-lg mb-8 leading-relaxed font-medium">
            Master the syntax, algorithms, and deep logic that powers modern software. Before you build fast with AI, you must understand the laws of production logic.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 mb-8">
            <Button
              size="lg"
              className="bg-python hover:bg-python/90 text-python-foreground font-black w-full sm:w-auto h-14 rounded-2xl shadow-xl shadow-python/20 transition-all hover:scale-105 active:scale-95"
              onClick={() => navigate('/python-course')}
            >
              Activate Identity Protocol (₹365)
              <ArrowRight className="ml-2 w-5 h-5" />
            </Button>

          </div>

          <div className="flex flex-wrap gap-4 text-[10px] font-black uppercase tracking-widest text-muted-foreground/60">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-success" />
              <span>Python Syntax</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-success" />
              <span>DS & Algorithms</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-success" />
              <span>ML/AI Engineering</span>
            </div>
          </div>
        </div>
      </motion.div>
    </section >
  );
};

export default HeroSection;

