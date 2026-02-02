import { Link, useNavigate } from 'react-router-dom';
import { Code2, Brain, Github, Twitter, Linkedin, Mail } from 'lucide-react';

const Footer = () => {
  const navigate = useNavigate();

  return (
    <footer className="bg-foreground text-background py-16">
      <div className="container mx-auto px-4">
        <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div>
            <Link to="/" className="flex items-center gap-2 mb-4">
              <div className="flex items-center gap-1">
                <div className="w-8 h-8 rounded-lg bg-python flex items-center justify-center">
                  <Code2 className="w-5 h-5 text-python-foreground" />
                </div>
                <div className="w-8 h-8 rounded-lg bg-vibe flex items-center justify-center">
                  <Brain className="w-5 h-5 text-vibe-foreground" />
                </div>
              </div>
              <span className="font-heading font-bold text-xl">LearnAI</span>
            </Link>
            <p className="text-background/60 text-sm">
              Master Python, AI, and ML. Build faster with Vibe Coding.
              Showcase your skills with AI Passport.
            </p>
          </div>

          {/* Python Course */}
          <div>
            <h4 className="font-heading font-semibold mb-4">Python Course</h4>
            <ul className="space-y-2 text-sm text-background/60">
              <li>
                <button onClick={() => navigate('/phase/phase-1')} className="hover:text-background smooth-transition">
                  Python Foundations
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/phase/phase-2')} className="hover:text-background smooth-transition">
                  Data Structures
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/phase/phase-4')} className="hover:text-background smooth-transition">
                  Machine Learning
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/phase/phase-5')} className="hover:text-background smooth-transition">
                  Deep Learning
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/phase/phase-7')} className="hover:text-background smooth-transition">
                  MLOps
                </button>
              </li>
            </ul>
          </div>

          {/* Vibe Coding */}
          <div>
            <h4 className="font-heading font-semibold mb-4">Vibe Coding</h4>
            <ul className="space-y-2 text-sm text-background/60">
              <li>
                <button onClick={() => navigate('/vibe-coding')} className="hover:text-background smooth-transition">
                  AI Tools
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/vibe-coding')} className="hover:text-background smooth-transition">
                  Prompt Engineering
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/vibe-coding')} className="hover:text-background smooth-transition">
                  Skills Passport
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/')} className="hover:text-background smooth-transition">
                  Career Path Quiz
                </button>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-heading font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm text-background/60">
              <li>
                <button onClick={() => navigate('/python-course')} className="hover:text-background smooth-transition">
                  Full Curriculum
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/phase/phase-8')} className="hover:text-background smooth-transition">
                  Generative AI
                </button>
              </li>
              <li>
                <button onClick={() => navigate('/phase/phase-9')} className="hover:text-background smooth-transition">
                  Portfolio Building
                </button>
              </li>
              <li>
                <button className="hover:text-background smooth-transition">
                  Community
                </button>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-background/10 pt-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-background/60">
            © 2025 LearnAI. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <button className="text-background/60 hover:text-background smooth-transition">
              <Github className="w-5 h-5" />
            </button>
            <button className="text-background/60 hover:text-background smooth-transition">
              <Twitter className="w-5 h-5" />
            </button>
            <button className="text-background/60 hover:text-background smooth-transition">
              <Linkedin className="w-5 h-5" />
            </button>
            <button className="text-background/60 hover:text-background smooth-transition">
              <Mail className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
