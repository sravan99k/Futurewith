import { motion } from 'framer-motion';
import { ShieldCheck } from 'lucide-react';
import { Github, Linkedin } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { CoursePassport } from '@/components/CoursePassport';
import type { SkillsPassportFeature } from '@/data/courses/vibe';

interface VibeSkillsPassportProps {
  features: SkillsPassportFeature[];
}

export const VibeSkillsPassport = ({ features }: VibeSkillsPassportProps) => {
  return (
    <section className="py-24 bg-muted/30 relative overflow-hidden border-t border-border">
      <div className="absolute top-0 left-0 w-full h-full bg-grid-pattern opacity-5 pointer-events-none" />
      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center mb-20">
          <div className="flex items-center justify-center gap-2 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-vibe flex items-center justify-center shadow-lg shadow-vibe/20">
              <ShieldCheck className="w-7 h-7 text-white" />
            </div>
          </div>
          <h2 className="font-heading text-4xl lg:text-5xl font-black mb-6">
            Architecture <span className="text-vibe">DNA</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-lg font-medium leading-relaxed">
            Our engine validates your proficiency across the entire AI-native lifecycle,
            converting your vibes into verifiable market authority.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-16 items-center max-w-6xl mx-auto">
          <div className="space-y-6">
            <h3 className="text-2xl font-black tracking-tight mb-8 uppercase italic border-b border-border pb-4 w-fit">Vibe Verification Nodes</h3>
            {features.map((feature, index) => (
              <div key={feature.id} className="flex items-start gap-5 p-6 rounded-[2rem] bg-card border border-border hover:border-vibe/40 smooth-transition group">
                <div className="mt-1 p-2 bg-vibe/10 rounded-xl group-hover:bg-vibe group-hover:text-white transition-colors">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <div>
                  <h4 className="font-black text-sm uppercase tracking-tight mb-1">{feature.title}</h4>
                  <p className="text-xs text-muted-foreground font-medium leading-relaxed">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>

          <div className="relative">
            <div className="absolute -inset-10 bg-gradient-to-br from-vibe/20 to-blue-500/20 rounded-[3rem] blur-3xl opacity-50" />
            <CoursePassport />
            <div className="mt-12 flex justify-center gap-4">
              <Button variant="outline" className="rounded-full h-12 px-6 font-black uppercase text-[10px] tracking-widest border-border hover:bg-zinc-100">
                <Github className="w-4 h-4 mr-2" />
                Connect GitHub
              </Button>
              <Button variant="outline" className="rounded-full h-12 px-6 font-black uppercase text-[10px] tracking-widest border-border hover:bg-zinc-100">
                <Linkedin className="w-4 h-4 mr-2 text-blue-600" />
                Post Verification
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
