import { motion } from 'framer-motion';
import { ShieldCheck, ExternalLink, Cpu, Zap, Search } from 'lucide-react';

const verifications = [
    { id: 1, user: "Alex M.", artifact: "Legal AI Agent", impact: "Risk Mapping 99%", time: "2m ago", category: "LOGIC" },
    { id: 2, user: "Sara K.", artifact: "v0 Dashboard", impact: "UI Fidelity 100%", time: "5m ago", category: "BUILD" },
    { id: 3, user: "Rahal P.", artifact: "Sentiment Engine", impact: "Signal Latency <50ms", time: "12m ago", category: "DEPLOY" },
    { id: 4, user: "James W.", artifact: "Content Pipeline", impact: "50+ Artifacts/hr", time: "15m ago", category: "RESEARCH" },
    { id: 5, user: "Priay S.", artifact: "Healthcare RAG", impact: "Accuracy Verified", time: "20m ago", category: "LOGIC" }
];

export const PublicProofMarquee = () => {
    return (
        <div className="bg-zinc-950 border-y border-white/5 py-4 overflow-hidden relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-zinc-950 via-transparent to-zinc-950 z-10 pointer-events-none" />

            <div className="flex animate-marquee whitespace-nowrap gap-8 items-center">
                {[...verifications, ...verifications].map((v, idx) => (
                    <div key={`${v.id}-${idx}`} className="flex items-center gap-4 px-6 py-2 rounded-full bg-white/5 border border-white/10 hover:border-vibe/50 transition-colors cursor-default">
                        <div className="flex items-center gap-2">
                            <ShieldCheck className="w-3.5 h-3.5 text-vibe" />
                            <span className="text-[10px] font-black text-white uppercase tracking-widest">{v.user}</span>
                        </div>
                        <div className="h-4 w-px bg-white/10" />
                        <span className="text-[10px] font-bold text-zinc-400">{v.artifact}</span>
                        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-vibe/20 border border-vibe/30">
                            <Zap className="w-2.5 h-2.5 text-vibe" />
                            <span className="text-[8px] font-black text-vibe uppercase">{v.impact}</span>
                        </div>
                        <span className="text-[9px] font-medium text-zinc-600 italic">{v.time}</span>
                    </div>
                ))}
            </div>

            <style>{`
                @keyframes marquee {
                    0% { transform: translateX(0); }
                    100% { transform: translateX(-50%); }
                }
                .animate-marquee {
                    display: flex;
                    width: max-content;
                    animation: marquee 40s linear infinite;
                }
                .group:hover .animate-marquee {
                    animation-play-state: paused;
                }
            `}</style>
        </div>
    );
};
