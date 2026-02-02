import { useParams, Link, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, ArrowRight, Clock, BookOpen, Code, Wrench, ShieldCheck, CheckCircle2, Folder, ChevronDown, ChevronUp, FileText, Search, Menu, X, Cpu, Terminal, Zap, FileSearch, Lock } from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import { allPhases } from '@/data/courses';
import { Button } from '@/components/ui/button';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { InteractiveQuiz } from '@/components/InteractiveQuiz';
import { useCourseProgress } from '@/hooks/useCourseProgress';
import Header from '@/components/Header';
import Footer from '@/components/Footer';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { FloatingPassport } from '@/components/FloatingPassport';
import { GuidedProjectRenderer } from '@/components/GuidedProjectRenderer';
import { InterviewSimulator } from '@/components/InterviewSimulator';

const typeIcons = {
  theory: FileSearch,
  practice: Terminal,
  exercise: Zap,
  project: ShieldCheck,
  assessment: CheckCircle2,
  interview: Cpu,
};

const typeColors = {
  theory: 'text-python bg-python/5 border-python/20',
  practice: 'text-zinc-400 bg-zinc-900 border-white/10',
  exercise: 'text-amber-500 bg-amber-500/5 border-amber-500/10',
  project: 'text-python bg-python/10 border-python/30',
  assessment: 'text-emerald-500 bg-emerald-500/5 border-emerald-500/10',
  interview: 'text-zinc-100 bg-zinc-900 border-white/20',
};

const PhasePage = () => {
  const { phaseId } = useParams();
  const navigate = useNavigate();
  const phase = allPhases.find(p => p.id === phaseId);
  const [expandedFolder, setExpandedFolder] = useState<string | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSidebar, setShowSidebar] = useState(true);

  const { completedTopics, markComplete, stats, setCurrentPhase } = useCourseProgress();
  const isTopicCompleted = (path: string | null) => path ? completedTopics.has(path) : false;

  // Update current phase when entering a phase page
  useEffect(() => {
    if (phase) {
      setCurrentPhase(phase.number);
    }
  }, [phase, setCurrentPhase]);

  // Find current topic object
  const currentTopicObject = useMemo(() => {
    if (!selectedTopic || !phase) return null;
    const direct = phase.topics.find(t => t.markdownPath === selectedTopic);
    if (direct) return direct;
    return (phase.folders || []).flatMap(f => f.topics).find(t => t.markdownPath === selectedTopic);
  }, [selectedTopic, phase]);

  if (!phase) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-4 py-32 text-center">
          <h1 className="font-heading text-3xl font-bold mb-4">Phase Not Found</h1>
          <p className="text-muted-foreground mb-8">The requested phase doesn't exist.</p>
          <Button onClick={() => navigate('/python-course')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Course
          </Button>
        </div>
        <Footer />
      </div>
    );
  }

  const isUnlocked = true; // Bypass verification for testing

  if (false) { // Skip verification check
    return (
      <div className="min-h-screen bg-background flex flex-col">
        <Header />
        <main className="flex-1 flex items-center justify-center p-6">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-xl w-full text-center"
          >
            <div className="w-24 h-24 rounded-[2rem] bg-zinc-950 border border-white/10 flex items-center justify-center mx-auto mb-8 shadow-2xl relative">
              <Lock className="w-10 h-10 text-python" />
              <div className="absolute inset-0 bg-python/10 blur-2xl rounded-full -z-10" />
            </div>

            <h1 className="text-4xl font-black uppercase italic tracking-tighter mb-4">Verification Required</h1>
            <p className="text-muted-foreground text-lg mb-10 font-medium leading-relaxed">
              Phase {phase.number} is part of the <span className="text-white font-bold">Standard Authority Protocol</span>.
              To access this logic, you must either complete the preceding phases or pass the architecture diagnostic.
            </p>

            <div className="grid gap-4">
              <Button
                onClick={() => navigate(`/interview-simulator?mode=diagnostic&phase=${phase.number}`)}
                className="bg-python hover:bg-python/90 text-python-foreground h-16 rounded-2xl font-black text-lg shadow-xl shadow-python/20 uppercase tracking-tight"
              >
                Start Logic Verification
                <Zap className="w-5 h-5 ml-2" />
              </Button>
              <Button
                variant="outline"
                onClick={() => navigate('/python-course')}
                className="h-14 rounded-2xl font-bold border-white/10 hover:bg-white/5"
              >
                Back to Curriculum
              </Button>
            </div>

            <p className="mt-8 text-[10px] font-black uppercase tracking-[0.3em] text-muted-foreground/40">
              Authority Level Insufficient for Protocol Access
            </p>
          </motion.div>
        </main>
        <Footer />
      </div>
    );
  }

  const currentIndex = allPhases.findIndex(p => p.id === phaseId);
  const prevPhase = currentIndex > 0 ? allPhases[currentIndex - 1] : null;
  const nextPhase = currentIndex < allPhases.length - 1 ? allPhases[currentIndex + 1] : null;

  const hasfolders = phase.folders && phase.folders.length > 0;

  // Flatten all topics for progress calculation
  const allTopicPaths = useMemo(() => {
    const direct = phase.topics.map(t => ({ markdownPath: t.markdownPath }));
    const folderTopics = (phase.folders || []).flatMap(f => f.topics.map(t => ({ markdownPath: t.markdownPath })));
    return [...direct, ...folderTopics];
  }, [phase]);

  const progress = useMemo(() => {
    if (allTopicPaths.length === 0) return 0;
    const completedCount = allTopicPaths.filter(t => completedTopics[t.markdownPath]).length;
    return Math.floor((completedCount / allTopicPaths.length) * 100);
  }, [allTopicPaths, completedTopics]);

  const filteredFolders = useMemo(() => {
    if (!hasfolders || !searchQuery) return phase.folders || [];
    const query = searchQuery.toLowerCase();
    return (phase.folders || []).filter(folder =>
      folder.name.toLowerCase().includes(query) ||
      folder.description.toLowerCase().includes(query) ||
      folder.topics.some(t => t.title.toLowerCase().includes(query))
    );
  }, [hasfolders, phase.folders, searchQuery]);

  const filteredTopics = useMemo(() => {
    if (hasfolders || !searchQuery) return phase.topics || [];
    const query = searchQuery.toLowerCase();
    return phase.topics.filter(topic =>
      topic.title.toLowerCase().includes(query) ||
      topic.description.toLowerCase().includes(query)
    );
  }, [hasfolders, phase.topics, searchQuery]);

  const Sidebar = () => (
    <div className="h-full overflow-y-auto pr-3 custom-scrollbar">
      {/* Search Section - Glass Effect */}
      <div className="mb-6 relative">
        <div className="absolute inset-0 bg-gradient-to-r from-python/10 to-transparent rounded-xl blur-lg -z-10" />
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-python/70" />
          <Input
            placeholder="Search topics..."
            className="pl-11 h-12 bg-zinc-900/60 backdrop-blur-xl border border-white/10 rounded-xl text-white placeholder:text-zinc-500 focus-visible:ring-python/50 focus-visible:border-python/30 transition-all"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Navigation Content */}
      <div className="space-y-4">
        {hasfolders ? (
          filteredFolders.map((folder) => (
            <div key={folder.id} className="space-y-2">
              {/* Module Card - Glass Effect */}
              <button
                onClick={() => setExpandedFolder(expandedFolder === folder.id ? null : folder.id)}
                className={`w-full relative overflow-hidden rounded-xl p-4 transition-all duration-300 group ${
                  expandedFolder === folder.id
                    ? 'bg-gradient-to-br from-python/20 via-python/10 to-transparent border border-python/30 shadow-[0_0_30px_rgba(249,115,22,0.15)]'
                    : 'bg-white/5 backdrop-blur-lg border border-white/10 hover:bg-white/10 hover:border-white/20 hover:shadow-[0_0_20px_rgba(249,115,22,0.08)]'
                }`}
              >
                {/* Glow Effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-python/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                
                <div className="relative flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 ${
                      expandedFolder === folder.id
                        ? 'bg-python/30 shadow-[0_0_15px_rgba(249,115,22,0.3)]'
                        : 'bg-white/10 group-hover:bg-python/20'
                    }`}>
                      <Folder className={`w-5 h-5 transition-colors duration-300 ${
                        expandedFolder === folder.id ? 'text-python' : 'text-zinc-400 group-hover:text-python'
                      }`} />
                    </div>
                    <div className="text-left">
                      <span className={`font-semibold text-sm block transition-colors duration-300 ${
                        expandedFolder === folder.id ? 'text-python' : 'text-white group-hover:text-python/80'
                      }`}>
                        {folder.name}
                      </span>
                      <span className="text-xs text-zinc-500">
                        {folder.topics.length} lessons
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {expandedFolder === folder.id ? (
                      <ChevronUp className="w-5 h-5 text-python animate-pulse" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-zinc-500 group-hover:text-python/70 transition-colors" />
                    )}
                  </div>
                </div>
              </button>

              {/* Topics - Glass Cards */}
              <AnimatePresence>
                {expandedFolder === folder.id && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}
                    className="overflow-hidden space-y-2 pl-2"
                  >
                    {folder.topics.map((topic) => {
                      const completed = isTopicCompleted(topic.markdownPath);
                      const isActive = selectedTopic === topic.markdownPath;
                      
                      return (
                        <motion.button
                          key={topic.id}
                          initial={{ x: -10, opacity: 0 }}
                          animate={{ x: 0, opacity: 1 }}
                          transition={{ duration: 0.2 }}
                          onClick={() => setSelectedTopic(topic.markdownPath)}
                          className={`w-full relative overflow-hidden rounded-lg p-3 transition-all duration-300 group ${
                            isActive
                              ? 'bg-gradient-to-r from-python/15 to-transparent border-l-2 border-python shadow-[0_0_15px_rgba(249,115,22,0.1)]'
                              : 'bg-zinc-900/40 backdrop-blur-sm border border-white/5 hover:bg-white/5 hover:border-white/10'
                          } ${completed ? 'opacity-60' : ''}`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="flex items-center gap-3 min-w-0">
                              <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-300 ${
                                isActive
                                  ? 'bg-python/20'
                                  : completed
                                  ? 'bg-success/10'
                                  : 'bg-white/5 group-hover:bg-python/10'
                              }`}>
                                {completed ? (
                                  <CheckCircle2 className="w-4 h-4 text-success" />
                                ) : (
                                  <FileText className={`w-4 h-4 transition-colors duration-300 ${
                                    isActive ? 'text-python' : 'text-zinc-500 group-hover:text-python/70'
                                  }`} />
                                )}
                              </div>
                              <span className={`truncate text-sm text-left transition-colors duration-300 ${
                                isActive ? 'text-python font-medium' : 'text-zinc-300 group-hover:text-white'
                              }`}>
                                {topic.title}
                              </span>
                            </div>
                            <div className="flex-shrink-0">
                              {completed && (
                                <div className="w-6 h-6 rounded-full bg-success/20 flex items-center justify-center animate-[pulse_2s_ease-in-out_infinite]">
                                  <CheckCircle2 className="w-3 h-3 text-success" />
                                </div>
                              )}
                            </div>
                          </div>
                        </motion.button>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))
        ) : (
          // Flat Topics - Glass Cards
          filteredTopics.map((topic, index) => {
            const completed = isTopicCompleted(topic.markdownPath);
            const isActive = selectedTopic === topic.markdownPath;
            
            return (
              <motion.button
                key={topic.id}
                initial={{ x: -10, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.05, duration: 0.2 }}
                onClick={() => setSelectedTopic(topic.markdownPath)}
                className={`w-full relative overflow-hidden rounded-xl p-4 transition-all duration-300 group ${
                  isActive
                    ? 'bg-gradient-to-br from-python/20 via-python/10 to-transparent border border-python/30 shadow-[0_0_30px_rgba(249,115,22,0.15)]'
                    : 'bg-white/5 backdrop-blur-lg border border-white/10 hover:bg-white/10 hover:border-white/20 hover:shadow-[0_0_20px_rgba(249,115,22,0.08)]'
                } ${completed ? 'opacity-60' : ''}`}
              >
                {/* Glow Effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-python/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                
                <div className="relative flex items-center justify-between gap-2">
                  <div className="flex items-center gap-3 min-w-0">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-all duration-300 ${
                      isActive
                        ? 'bg-python/30 shadow-[0_0_15px_rgba(249,115,22,0.3)]'
                        : completed
                        ? 'bg-success/10'
                        : 'bg-white/10 group-hover:bg-python/20'
                    }`}>
                      {completed ? (
                        <CheckCircle2 className="w-5 h-5 text-success" />
                      ) : (
                        <FileText className={`w-5 h-5 transition-colors duration-300 ${
                          isActive ? 'text-python' : 'text-zinc-400 group-hover:text-python'
                        }`} />
                      )}
                    </div>
                    <div className="min-w-0">
                      <span className={`truncate text-sm font-medium block transition-colors duration-300 ${
                        isActive ? 'text-python' : 'text-white group-hover:text-python/80'
                      }`}>
                        {topic.title}
                      </span>
                      <span className="text-xs text-zinc-500">
                        {topic.duration}
                      </span>
                    </div>
                  </div>
                  {completed && (
                    <div className="w-8 h-8 rounded-full bg-success/20 flex items-center justify-center flex-shrink-0 animate-[pulse_2s_ease-in-out_infinite]">
                      <CheckCircle2 className="w-4 h-4 text-success" />
                    </div>
                  )}
                </div>
              </motion.button>
            );
          })
        )}
      </div>
      
      {/* Progress Summary Card */}
      <div className="mt-8 p-4 bg-gradient-to-br from-python/10 via-python/5 to-transparent rounded-xl border border-python/20">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-python/80 uppercase tracking-wider">Progress</span>
          <span className="text-lg font-bold text-python">{progress}%</span>
        </div>
        <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className="h-full bg-gradient-to-r from-python to-python/60 rounded-full shadow-[0_0_10px_rgba(249,115,22,0.5)]"
          />
        </div>
        <p className="mt-3 text-xs text-zinc-500">
          {allTopicPaths.filter(t => completedTopics[t.markdownPath]).length} of {allTopicPaths.length} topics completed
        </p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="pt-20 pb-16">
        {/* Hero Section - Only show when no topic selected */}
        {!selectedTopic && (
          <section className="bg-python-muted py-12 lg:py-16">
            <div className="container mx-auto px-4">
              <Link
                to="/python-course"
                className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground mb-6 smooth-transition"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Curriculum
              </Link>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="flex flex-wrap items-center gap-3 mb-4">
                  <span className="px-4 py-2 rounded-full bg-python text-python-foreground font-heading font-bold">
                    Phase {phase.number}
                  </span>
                  <span className="flex items-center gap-1 text-muted-foreground">
                    <Clock className="w-4 h-4" />
                    {phase.duration}
                  </span>
                  <span className="text-muted-foreground">
                    {hasfolders ? `${phase.folders!.length} modules` : `${phase.topics.length} lessons`}
                  </span>
                </div>

                <h1 className="font-heading text-3xl sm:text-4xl lg:text-5xl font-bold mb-4">
                  {phase.title}
                </h1>

                <div className="max-w-md mb-8">
                  <div className="flex justify-between items-center text-sm mb-2">
                    <span className="text-muted-foreground font-medium">Phase Progress</span>
                    <span className="text-python font-bold">{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2 bg-python/10" />
                </div>

                <p className="text-lg text-muted-foreground max-w-3xl mb-8">
                  {phase.description}
                </p>

                <div className="flex flex-wrap gap-2">
                  {phase.skills.map((skill) => (
                    <span
                      key={skill}
                      className="flex items-center gap-1 px-3 py-1 rounded-full bg-python/10 text-python text-sm"
                    >
                      <CheckCircle2 className="w-3 h-3" />
                      {skill}
                    </span>
                  ))}
                </div>

                {/* Search Bar - Main View */}
                <div className="mt-8 relative max-w-md">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    placeholder="Search for a topic..."
                    className="pl-10 h-12 bg-background border-border focus-visible:ring-python"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>

                {phase.readmePath && !searchQuery && (
                  <div className="mt-12 p-6 bg-background border border-border rounded-xl">
                    <h2 className="font-heading text-xl font-bold mb-4 flex items-center gap-2">
                      <FileText className="w-5 h-5 text-python" />
                      Phase Overview
                    </h2>
                    <MarkdownRenderer markdownPath={phase.readmePath} className="prose-sm max-h-96 overflow-y-auto" />
                  </div>
                )}
              </motion.div>
            </div>
          </section>
        )}

        {/* Content Section */}
        <section className={`py-8 ${selectedTopic ? 'px-0' : 'lg:py-16 container mx-auto px-4'}`}>
          {selectedTopic ? (
            <div className="flex min-h-[calc(100vh-12rem)] relative">
              {/* Sidebar Toggle (Mobile) */}
              <button
                className="lg:hidden fixed bottom-6 right-6 z-50 w-14 h-14 rounded-2xl bg-gradient-to-br from-python to-python/80 backdrop-blur-lg border border-white/20 shadow-[0_0_30px_rgba(249,115,22,0.4)] flex items-center justify-center hover:shadow-[0_0_40px_rgba(249,115,22,0.5)] transition-all duration-300 active:scale-95"
                onClick={() => setShowSidebar(!showSidebar)}
              >
                <div className="relative">
                  {showSidebar ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
                  <div className="absolute inset-0 bg-white/20 blur-lg rounded-full -z-10" />
                </div>
              </button>

              {/* Sidebar */}
              <motion.aside
                initial={false}
                animate={{
                  width: showSidebar ? '320px' : '0px',
                  opacity: showSidebar ? 1 : 0,
                  x: showSidebar ? 0 : -320
                }}
                transition={{ duration: 0.3, ease: 'easeOut' }}
                className={`fixed lg:sticky top-24 left-0 z-40 bg-zinc-950/80 backdrop-blur-xl border-r border-white/10 h-[calc(100vh-6rem)] lg:h-[calc(100vh-8rem)] transition-all duration-300 overflow-hidden`}
              >
                <div className="w-[320px] h-full p-4">
                  {/* Sidebar Background Glow */}
                  <div className="absolute top-0 right-0 w-64 h-64 bg-python/5 blur-3xl rounded-full pointer-events-none" />
                  <div className="absolute bottom-0 left-0 w-48 h-48 bg-teal-500/5 blur-3xl rounded-full pointer-events-none" />
                  
                  {/* Sidebar Content */}
                  <div className="relative z-10 h-full">
                    <Sidebar />
                  </div>
                </div>
              </motion.aside>

              {/* Overlay for mobile sidebar */}
              <AnimatePresence>
                {showSidebar && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => setShowSidebar(false)}
                    className="lg:hidden fixed inset-0 bg-black/50 z-30"
                  />
                )}
              </AnimatePresence>

              {/* Content Area */}
              <div className="flex-1 px-6 lg:px-12 max-w-5xl mx-auto w-full">
                <nav className="mb-6 flex items-center justify-between">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedTopic(null)}
                    className="text-[10px] font-black uppercase tracking-widest text-muted-foreground hover:text-foreground"
                  >
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Exit Protocol
                  </Button>

                  <div className="text-sm font-medium text-muted-foreground truncate max-w-[150px] lg:max-w-xs flex items-center gap-3">
                    <div className="hidden sm:block flex-1 min-w-[60px] lg:min-w-[100px]">
                      <Progress value={progress} className="h-1.5 bg-python/10" />
                    </div>
                    <span>{progress}%</span>
                  </div>
                </nav>

                <motion.div
                  key={selectedTopic}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-card border border-border rounded-2xl p-6 lg:p-10 shadow-sm min-h-[60vh] overflow-hidden group"
                >
                  <div className="flex justify-end mb-6">
                    {(!currentTopicObject || currentTopicObject.type !== 'assessment') && (
                      <Button
                        onClick={() => markComplete(selectedTopic)}
                        variant={isTopicCompleted(selectedTopic) ? "outline" : "default"}
                        className={isTopicCompleted(selectedTopic) ? "border-success text-success hover:bg-success/5" : "bg-success hover:bg-success/90 text-white"}
                        size="sm"
                      >
                        {isTopicCompleted(selectedTopic) ? (
                          <>
                            <CheckCircle2 className="w-4 h-4 mr-2" />
                            Completed
                          </>
                        ) : (
                          "Mark as Complete"
                        )}
                      </Button>
                    )}
                  </div>

                  {currentTopicObject?.type === 'assessment' ? (
                    <InteractiveQuiz
                      markdownPath={selectedTopic}
                      onComplete={() => {
                        if (!isTopicCompleted(selectedTopic)) {
                          markComplete(selectedTopic);
                        }
                      }}
                    />
                  ) : currentTopicObject?.type === 'project' ? (
                    <GuidedProjectRenderer
                      markdownPath={selectedTopic}
                      onComplete={() => {
                        if (!isTopicCompleted(selectedTopic)) {
                          markComplete(selectedTopic);
                        }
                      }}
                    />
                  ) : currentTopicObject?.type === 'interview' ? (
                    <InterviewSimulator
                      markdownPath={selectedTopic}
                      onComplete={() => {
                        if (!isTopicCompleted(selectedTopic)) {
                          markComplete(selectedTopic);
                        }
                      }}
                    />
                  ) : (
                    <MarkdownRenderer markdownPath={selectedTopic} />
                  )}

                  <div className="mt-12 flex justify-center">
                    <Button
                      onClick={() => {
                        if (!isTopicCompleted(selectedTopic)) {
                          markComplete(selectedTopic!);
                        }
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className="bg-python hover:bg-python/90 text-python-foreground px-10 py-8 rounded-2xl text-xl font-black italic uppercase tracking-tighter shadow-[0_20px_40px_rgba(55,118,171,0.3)] transition-all active:scale-95"
                    >
                      {isTopicCompleted(selectedTopic) ? "Synchronize Next Logic" : "Commit Progress to Pipeline"}
                    </Button>
                  </div>
                </motion.div>

                {/* Bottom Topic Navigation */}
                <div className="mt-12 pt-8 border-t border-border flex items-center justify-between">
                  <Button
                    variant="outline"
                    onClick={() => setSelectedTopic(null)}
                  >
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Module List
                  </Button>

                  <Button
                    variant="ghost"
                    onClick={() => {
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    }}
                  >
                    Back to Top
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto">
              <h2 className="font-heading text-2xl font-bold mb-8">
                {searchQuery ? `Search Results for "${searchQuery}"` : 'Course Content'}
              </h2>

              <div className="space-y-4">
                {hasfolders ? (
                  // FOLDER-BASED VIEW
                  filteredFolders.map((folder, folderIndex) => {
                    const isExpanded = expandedFolder === folder.id;

                    return (
                      <motion.div
                        key={folder.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: folderIndex * 0.05 }}
                        className="bg-card border border-border rounded-xl overflow-hidden hover:shadow-lg smooth-transition"
                      >
                        <div
                          className="p-6 cursor-pointer"
                          onClick={() => setExpandedFolder(isExpanded ? null : folder.id)}
                        >
                          <div className="flex items-start gap-4">
                            <div className="w-12 h-12 rounded-xl bg-python/10 flex items-center justify-center flex-shrink-0">
                              <Folder className="w-6 h-6 text-python" />
                            </div>

                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-2">
                                <h3 className="font-heading font-semibold text-lg">
                                  {folder.name}
                                </h3>
                                <span className="text-xs px-2 py-0.5 rounded bg-muted">
                                  {folder.topics.length} {folder.topics.length === 1 ? 'lesson' : 'lessons'}
                                </span>
                              </div>

                              <p className="text-muted-foreground text-sm">
                                {folder.description}
                              </p>
                            </div>

                            <div className="flex items-center gap-2 flex-shrink-0">
                              {isExpanded ? (
                                <ChevronUp className="w-5 h-5 text-muted-foreground" />
                              ) : (
                                <ChevronDown className="w-5 h-5 text-muted-foreground" />
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Folder Contents */}
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="border-t border-border bg-muted/30"
                          >
                            <div className="p-6 space-y-3">
                              {folder.topics.map((topic, topicIndex) => {
                                const Icon = typeIcons[topic.type] || BookOpen;
                                const colorClass = typeColors[topic.type];

                                return (
                                  <div
                                    key={topic.id}
                                    className="flex items-center gap-3 p-4 rounded-lg bg-background border border-border hover:border-python/30 smooth-transition cursor-pointer"
                                    onClick={() => setSelectedTopic(topic.markdownPath)}
                                  >
                                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${colorClass}`}>
                                      <Icon className="w-5 h-5" />
                                    </div>

                                    <div className="flex-1 min-w-0">
                                      <div className="flex items-center gap-2 mb-1">
                                        <span className="text-xs text-muted-foreground">
                                          Step {topicIndex + 1}
                                        </span>
                                        <span className="text-xs px-2 py-0.5 rounded bg-muted capitalize">
                                          {topic.type}
                                        </span>
                                        {isTopicCompleted(topic.markdownPath) && (
                                          <span className="flex items-center gap-1 text-[10px] bg-success/10 text-success px-1.5 py-0.5 rounded-full font-bold">
                                            <CheckCircle2 className="w-3 h-3" />
                                            DONE
                                          </span>
                                        )}
                                        <span className="text-xs text-muted-foreground flex items-center gap-1">
                                          <Clock className="w-3 h-3" />
                                          {topic.duration}
                                        </span>
                                      </div>
                                      <p className={`font-medium text-sm ${isTopicCompleted(topic.markdownPath) ? 'text-muted-foreground line-through decoration-success/30' : ''}`}>{topic.title}</p>
                                      <p className="text-xs text-muted-foreground">{topic.description}</p>
                                    </div>

                                    {isTopicCompleted(topic.markdownPath) ? (
                                      <CheckCircle2 className="w-5 h-5 text-success flex-shrink-0" />
                                    ) : (
                                      <FileText className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </motion.div>
                        )}
                      </motion.div>
                    );
                  })
                ) : (
                  // FLAT TOPICS VIEW
                  filteredTopics.map((topic, index) => {
                    const Icon = typeIcons[topic.type] || BookOpen;
                    const colorClass = typeColors[topic.type];

                    return (
                      <motion.div
                        key={topic.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className="bg-card border border-border rounded-xl overflow-hidden hover:shadow-lg smooth-transition cursor-pointer p-6"
                        onClick={() => setSelectedTopic(topic.markdownPath)}
                      >
                        <div className="flex items-start gap-4">
                          <div className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 ${colorClass}`}>
                            <Icon className="w-6 h-6" />
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center flex-wrap gap-2 mb-2">
                              <span className="text-xs text-muted-foreground font-medium">
                                Step {index + 1}
                              </span>
                              <span className="text-xs px-2 py-0.5 rounded bg-muted capitalize">
                                {topic.type}
                              </span>
                              <span className="text-xs text-muted-foreground flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {topic.duration}
                              </span>
                            </div>

                            <h3 className="font-heading font-semibold text-lg mb-2">
                              {topic.title}
                            </h3>

                            <p className="text-muted-foreground text-sm">
                              {topic.description}
                            </p>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })
                )}

                {filteredFolders.length === 0 && filteredTopics.length === 0 && (
                  <div className="text-center py-12">
                    <p className="text-muted-foreground">No topics found matching "{searchQuery}"</p>
                    <Button variant="link" onClick={() => setSearchQuery('')} className="mt-2 text-python">
                      Clear search
                    </Button>
                  </div>
                )}
              </div>

              {/* Navigation */}
              {!searchQuery && (
                <div className="mt-12 flex flex-col sm:flex-row items-center justify-between gap-4">
                  {prevPhase ? (
                    <Button
                      variant="outline"
                      onClick={() => navigate(`/phase/${prevPhase.id}`)}
                    >
                      <ArrowLeft className="w-4 h-4 mr-2" />
                      Phase {prevPhase.number}: {prevPhase.title}
                    </Button>
                  ) : (
                    <Button
                      variant="outline"
                      onClick={() => navigate('/python-course')}
                    >
                      <ArrowLeft className="w-4 h-4 mr-2" />
                      Back to Course
                    </Button>
                  )}

                  {nextPhase && (
                    <Button
                      className="bg-python hover:bg-python/90 text-python-foreground"
                      onClick={() => navigate(`/phase/${nextPhase.id}`)}
                    >
                      Phase {nextPhase.number}: {nextPhase.title}
                      <ArrowRight className="w-4 h-4 ml-2" />
                    </Button>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      <Footer />
      <FloatingPassport />
    </div>
  );
};

export default PhasePage;
