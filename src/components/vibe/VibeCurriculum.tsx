import { motion } from 'framer-motion';
import { Clock, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

const curriculumPhases = [
  { id: 1, title: 'AI Tool Foundation', description: 'Master Cursor, v0.dev, and modern AI IDEs', duration: 'Week 1' },
  { id: 2, title: 'Prompt Engineering', description: 'CO-STAR, Chain-of-Thought, and advanced techniques', duration: 'Week 1-2' },
  { id: 3, title: 'Web Development', description: 'Build stunning UIs with v0, Framer, and Bolt.new', duration: 'Week 2' },
  { id: 4, title: 'App Development', description: 'Create mobile apps with AI-assisted coding', duration: 'Week 2-3' },
  { id: 5, title: 'Backend & APIs', description: 'Connect databases and build REST APIs', duration: 'Week 3' },
  { id: 6, title: 'Automation & Scraping', description: 'Build workflows and data extraction pipelines', duration: 'Week 3-4' },
  { id: 7, title: 'Production Deployment', description: 'Deploy to Vercel, Netlify, and cloud platforms', duration: 'Week 4-5' },
  { id: 8, title: 'Portfolio & Launch', description: 'Ship your product and build your portfolio', duration: 'Week 5-6' }
];

export const VibeCurriculum = () => {
  const scrollToToolStack = () => {
    const element = document.getElementById('tool-stack');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="py-24 bg-black text-white border-t border-white/5">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="font-heading text-3xl lg:text-4xl font-black mb-4 uppercase tracking-tighter">
            8-Phase <span className="text-vibe">Vibe Curriculum</span>
          </h2>
          <p className="text-zinc-500 font-medium max-w-2xl mx-auto">
            From tool mastery to shipping real products
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="grid gap-3">
            {curriculumPhases.map((phase) => (
              <motion.div
                key={phase.id}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                className="flex items-center gap-4 p-4 rounded-xl bg-[#1E1E1E] border border-white/5 hover:border-vibe/30 transition-all cursor-pointer group"
              >
                <div className="w-12 h-12 rounded-lg bg-vibe flex items-center justify-center font-black text-vibe-foreground text-lg shrink-0">
                  {phase.id}
                </div>
                <div className="flex-1">
                  <h4 className="font-black uppercase text-sm text-white group-hover:text-vibe transition-colors">{phase.title}</h4>
                  <p className="text-xs text-zinc-500">{phase.description}</p>
                </div>
                <div className="text-right shrink-0">
                  <div className="flex items-center gap-1.5 text-xs text-zinc-500">
                    <Clock className="w-3.5 h-3.5" />
                    {phase.duration}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="text-center mt-10">
            <Button
              onClick={scrollToToolStack}
              className="bg-vibe hover:bg-vibe/90 text-vibe-foreground h-12 px-10 rounded-xl font-black text-sm uppercase tracking-widest inline-flex items-center gap-2"
            >
              Explore Full Curriculum
              <ArrowRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};
