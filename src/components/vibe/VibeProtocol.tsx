import { Search, Brain, Cpu, Rocket, Target } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { VibeProtocolStep } from '@/data/courses/vibe';

interface VibeProtocolProps {
  steps: VibeProtocolStep[];
}

export const VibeProtocol = ({ steps }: VibeProtocolProps) => {
  const protocolIcons: Record<string, any> = {
    RESEARCH: Search,
    LOGIC: Brain,
    BUILD: Cpu,
    DEPLOY: Rocket
  };

  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="font-heading text-3xl lg:text-4xl font-black mb-4 uppercase tracking-tighter">
            The <span className="text-vibe">Protocol</span> Manual
          </h2>
          <p className="text-muted-foreground font-medium max-w-xl mx-auto">
            Every artifact we build follows these 9 verified production phases.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-4 max-w-7xl mx-auto">
          {steps.map((step, idx) => {
            const Icon = protocolIcons[step.category] || Target;
            return (
              <Link to={`/phase/${step.promptRef}`} key={idx} className="p-6 rounded-[2rem] bg-card border border-border/50 group hover:border-vibe transition-all shadow-sm">
                <div className="text-[10px] font-black text-vibe/30 mb-4 tracking-widest">PHASE {String(step.id).padStart(2, '0')}</div>
                <div className="w-10 h-10 rounded-xl bg-muted flex items-center justify-center mb-4 group-hover:bg-vibe group-hover:text-vibe-foreground transition-all">
                  <Icon className="w-5 h-5" />
                </div>
                <h4 className="font-black text-sm uppercase mb-2 leading-tight h-10">{step.title}</h4>
                <p className="text-[10px] text-muted-foreground font-medium leading-relaxed mb-4 group-hover:text-foreground transition-colors">{step.description}</p>
                <div className="flex flex-wrap gap-1 opacity-40">
                  {step.tools.map(t => (
                    <span key={t} className="text-[8px] font-black px-1.5 py-0.5 bg-muted rounded border border-border">{t}</span>
                  ))}
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
};
