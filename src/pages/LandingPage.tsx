import Header from '@/components/Header';
import Footer from '@/components/Footer';
import HeroSection from "@/components/HeroSection";

import { PublicProofMarquee } from '@/components/PublicProofMarquee';

const LandingPage = () => {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Header />
      <main>
        <HeroSection />
        <PublicProofMarquee />

      </main>
      <Footer />
    </div>
  );
};

export default LandingPage;
