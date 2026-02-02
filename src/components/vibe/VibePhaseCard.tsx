import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronDown, ChevronUp, Target, Code2, 
  CheckCircle2, Check, Code, Folder 
} from 'lucide-react';
import type { Phase, TrackType } from '@/data/courses/vibe';

interface VibePhaseCardProps {
  phase: Phase;
  track: TrackType;
  isExpanded: boolean;
  onToggle: () => void;
  isLocked: boolean;
  trackIcon: React.ElementType;
}

export const VibePhaseCard = ({ 
  phase, 
  track, 
  isExpanded, 
  onToggle, 
  isLocked, 
  trackIcon: TrackIcon 
}: VibePhaseCardProps) => {
  const filteredFiles = phase.fileGuidance?.filter(fg => fg.track === 'BOTH' || fg.track === track) || [];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl overflow-hidden transition-all ${
        isLocked 
          ? 'bg-zinc-900/50 border border-white/5' 
          : 'bg-zinc-900 border border-vibe/20 hover:border-vibe/40'
      }`}
    >
      {/* Phase Header */}
      <button
        onClick={onToggle}
        className="w-full p-6 flex items-center gap-4 text-left hover:bg-white/5 transition-colors"
      >
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center font-black text-lg ${
          phase.isFree 
            ? 'bg-success/20 text-success' 
            : 'bg-vibe/20 text-vibe'
        }`}>
          {phase.id}
        </div>
        
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="font-black text-lg uppercase tracking-tight text-white">{phase.title}</h4>
            {phase.isFree && (
              <span className="text-[9px] font-bold bg-success/20 text-success px-2 py-0.5 rounded-full">FREE</span>
            )}
            {isLocked && (
              <span className="text-[9px] font-bold bg-vibe/20 text-vibe px-2 py-0.5 rounded-full">LOCKED</span>
            )}
          </div>
          <p className="text-sm text-zinc-400">{phase.subtitle}</p>
        </div>

        <div className="flex items-center gap-3">
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-zinc-500" />
          ) : (
            <ChevronDown className="w-5 h-5 text-zinc-500" />
          )}
        </div>
      </button>

      {/* Phase Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-6 border-t border-white/5">
              {/* Description */}
              <p className="text-sm text-zinc-400 mb-6 pt-4">{phase.description}</p>

              {/* What To Do */}
              <div className="mb-6">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-vibe" />
                  <h5 className="text-[10px] font-black uppercase tracking-widest text-vibe">What To Do</h5>
                </div>
                <p className="text-sm text-zinc-300 pl-6">{phase.whatToDo}</p>
              </div>

              {/* Tools Used */}
              <div className="mb-6">
                <div className="flex items-center gap-2 mb-2">
                  <Code2 className="w-4 h-4 text-vibe" />
                  <h5 className="text-[10px] font-black uppercase tracking-widest text-vibe">Tools To Use</h5>
                </div>
                <div className="flex flex-wrap gap-2 pl-6">
                  {phase.tools?.map((tool, idx) => (
                    <span key={idx} className="px-3 py-1 rounded-full bg-vibe/10 text-vibe text-xs font-medium">
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              {/* Track Specific Advice */}
              {phase.trackSpecifics && (
                <div className="mb-6">
                  <div className="flex items-center gap-2 mb-2">
                    <TrackIcon className="w-4 h-4 text-vibe" />
                    <h5 className="text-[10px] font-black uppercase tracking-widest text-vibe">
                      {track === 'WEBSITE' ? 'Website Path' : 'App Path'}
                    </h5>
                  </div>
                  <p className="text-sm text-zinc-300 pl-6">
                    {track === 'WEBSITE' ? phase.trackSpecifics.website : phase.trackSpecifics.app}
                  </p>
                </div>
              )}

              {/* File Structure */}
              {filteredFiles.length > 0 && (
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Folder className="w-4 h-4 text-yellow-500" />
                    <h5 className="text-[10px] font-black uppercase tracking-widest text-yellow-500">Files You Will Create</h5>
                  </div>
                  <div className="pl-6 space-y-2">
                    {filteredFiles.map((file, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <Code className="w-4 h-4 text-zinc-600 mt-0.5" />
                        <div>
                          <span className="font-mono text-xs text-yellow-400">{file.filename}</span>
                          <p className="text-xs text-zinc-500">{file.purpose}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Deliverables */}
              {phase.deliverables && (
                <div className="pt-4 border-t border-white/5">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle2 className="w-4 h-4 text-success" />
                    <h5 className="text-[10px] font-black uppercase tracking-widest text-success">You Will Have</h5>
                  </div>
                  <div className="flex flex-wrap gap-2 pl-6">
                    {phase.deliverables.map((del, idx) => (
                      <span key={idx} className="text-xs text-zinc-400 flex items-center gap-1">
                        <Check className="w-3 h-3 text-success" />
                        {del}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};
