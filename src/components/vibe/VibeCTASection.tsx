import { useNavigate } from 'react-router-dom';
import { Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const VibeCTASection = () => {
  const navigate = useNavigate();

  return (
    <section className="py-32 relative overflow-hidden bg-zinc-950">
      <div className="absolute inset-0 bg-vibe opacity-5 blur-[100px] rounded-full scale-150" />
      <div className="container mx-auto px-4 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl lg:text-7xl font-black text-white italic uppercase tracking-tighter mb-8 leading-none">
            Become a <span className="text-vibe">Unicorn</span> Architect
          </h2>
          <p className="text-zinc-400 text-xl font-medium mb-12 max-w-3xl mx-auto leading-relaxed">
            The era of slow coding is over. The era of architectural vibes has begun.
            Claim your seat in the elite 1% of AI-native developers.
          </p>
          <div className="flex flex-col sm:flex-row gap-6 justify-center">
            <Button
              onClick={() => navigate('/checkout?course=vibe')}
              size="lg"
              className="bg-vibe text-vibe-foreground hover:bg-vibe/90 h-16 px-12 rounded-2xl font-black text-xl shadow-2xl transition-all hover:scale-105 active:scale-95"
            >
              Activate Evolution (₹199)
              <Zap className="ml-3 w-6 h-6" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="border-white/10 text-white hover:bg-white/10 h-16 px-12 rounded-2xl font-black text-xl backdrop-blur-sm"
              onClick={() => navigate('/python-course')}
            >
              Logic Foundations
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};
