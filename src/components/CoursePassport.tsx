import { motion, AnimatePresence } from 'framer-motion';
import { 
    BookOpen, 
    Clock, 
    Target, 
    TrendingUp, 
    Settings,
    Zap,
    Brain,
    Code,
    Lightbulb,
    Timer,
    BarChart3,
    User,
    Briefcase,
    GraduationCap
} from 'lucide-react';
import { useState, useMemo } from 'react';
import { useCourseProgress } from '../hooks/useCourseProgress';
import { StudentCategory } from '../hooks/useCourseProgress';

// Category options
const categoryOptions = [
    { value: 'fresher' as StudentCategory, label: 'Fresher', icon: GraduationCap, description: 'No work experience, building career' },
    { value: 'job-seeker' as StudentCategory, label: 'Job Seeker', icon: Briefcase, description: 'Actively looking for opportunities' },
    { value: 'career-switcher' as StudentCategory, label: 'Career Switcher', icon: User, description: 'Transitioning from different field' }
];

interface CoursePassportProps {
    courseId?: string;
    onProfileUpdate?: (profile: any) => void;
}

export const CoursePassport = ({ courseId = 'python-ai', onProfileUpdate }: CoursePassportProps) => {
    // Use the hook for all state and logic
    const {
        studentProfile,
        updateStudentProfile,
        engagement,
        skillScores,
        trackScores,
        stats,
        markComplete
    } = useCourseProgress();
    
    const [showSettings, setShowSettings] = useState(false);

    // Callback when profile updates
    const handleProfileUpdate = (updates: any) => {
        updateStudentProfile(updates);
        if (onProfileUpdate) {
            onProfileUpdate({ ...studentProfile, ...updates });
        }
    };

    // Get recommendation based on category and performance
    const getRecommendation = useMemo(() => {
        const { category } = studentProfile;
        const { overall } = skillScores;

        if (category === 'fresher') {
            if (overall < 30) return { text: 'Focus on fundamentals', priority: 'high' as const };
            if (overall < 60) return { text: 'Build more practice projects', priority: 'medium' as const };
            return { text: 'Start building portfolio', priority: 'low' as const };
        }
        
        if (category === 'job-seeker') {
            if (overall < 50) return { text: 'Accelerate interview prep', priority: 'high' as const };
            if (overall < 75) return { text: 'Practice mock interviews', priority: 'medium' as const };
            return { text: 'Ready for applications', priority: 'low' as const };
        }
        
        if (category === 'career-switcher') {
            if (overall < 40) return { text: 'Bridge technical gaps', priority: 'high' as const };
            if (overall < 70) return { text: 'Build transferability proof', priority: 'medium' as const };
            return { text: 'Prepare transition case', priority: 'low' as const };
        }

        return { text: 'Continue learning', priority: 'medium' as const };
    }, [studentProfile, skillScores]);

    // Get track comparison for display
    const trackComparison = useMemo(() => {
        const vibe = trackScores.vibeCoding.overall;
        const python = trackScores.pythonLogic.overall;
        const diff = vibe - python;
        
        if (Math.abs(diff) < 10) return { label: 'Balanced', direction: 'neutral' };
        if (diff > 0) return { label: 'Vibe Coding Focus', direction: 'vibe' };
        return { label: 'Python Logic Focus', direction: 'python' };
    }, [trackScores]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-zinc-900/50 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden"
        >
            {/* Header */}
            <div className="p-6 border-b border-white/5">
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border border-emerald-500/30 flex items-center justify-center">
                            <BookOpen className="w-5 h-5 text-emerald-400" />
                        </div>
                        <div>
                            <h3 className="text-lg font-semibold text-white">Course Passport</h3>
                            <p className="text-xs text-zinc-500">Your learning journey tracker</p>
                        </div>
                    </div>
                    <button
                        onClick={() => setShowSettings(!showSettings)}
                        className="p-2 rounded-lg hover:bg-white/5 transition-colors"
                    >
                        <Settings className="w-5 h-5 text-zinc-400" />
                    </button>
                </div>

                {/* Progress Bar */}
                <div className="relative h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${stats.progressPercentage}%` }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
                    />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-[10px] font-medium text-white/80">
                            {stats.progressPercentage}% Complete
                        </span>
                    </div>
                </div>
            </div>

            {/* Settings Panel */}
            <AnimatePresence>
                {showSettings && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="border-b border-white/5 overflow-hidden"
                    >
                        <div className="p-6 space-y-4">
                            <h4 className="text-sm font-medium text-zinc-300">Student Profile</h4>
                            
                            {/* Category Selection */}
                            <div className="grid grid-cols-3 gap-2">
                                {categoryOptions.map((option) => (
                                    <button
                                        key={option.value}
                                        onClick={() => handleProfileUpdate({ category: option.value })}
                                        className={`p-3 rounded-xl border transition-all text-left ${
                                            studentProfile.category === option.value
                                                ? 'border-emerald-500/50 bg-emerald-500/10'
                                                : 'border-white/10 hover:border-white/20 bg-white/5'
                                        }`}
                                    >
                                        <option.icon className={`w-5 h-5 mb-2 ${
                                            studentProfile.category === option.value ? 'text-emerald-400' : 'text-zinc-400'
                                        }`} />
                                        <p className={`text-xs font-medium ${
                                            studentProfile.category === option.value ? 'text-emerald-400' : 'text-zinc-300'
                                        }`}>
                                            {option.label}
                                        </p>
                                        <p className="text-[10px] text-zinc-500 mt-1">{option.description}</p>
                                    </button>
                                ))}
                            </div>

                            {/* Target Role */}
                            <div>
                                <label className="text-xs text-zinc-500 mb-1.5 block">Target Role</label>
                                <input
                                    type="text"
                                    value={studentProfile.targetRole}
                                    onChange={(e) => handleProfileUpdate({ targetRole: e.target.value })}
                                    className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-emerald-500/50"
                                    placeholder="e.g., ML Engineer, Data Scientist"
                                />
                            </div>

                            {/* Experience Level */}
                            <div>
                                <label className="text-xs text-zinc-500 mb-1.5 block">Experience</label>
                                <select
                                    value={studentProfile.experience}
                                    onChange={(e) => handleProfileUpdate({ experience: e.target.value })}
                                    className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm text-white focus:outline-none focus:border-emerald-500/50"
                                >
                                    <option value="0-1 years">0-1 years</option>
                                    <option value="1-3 years">1-3 years</option>
                                    <option value="3-5 years">3-5 years</option>
                                    <option value="5+ years">5+ years</option>
                                </select>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Main Content */}
            <div className="p-6 space-y-6">
                {/* Quick Stats */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                        <div className="flex items-center gap-2 mb-2">
                            <Clock className="w-4 h-4 text-emerald-400" />
                            <span className="text-xs text-zinc-400">Time Invested</span>
                        </div>
                        <p className="text-2xl font-bold text-white">
                            {Math.round(engagement.totalSessionMinutes / 60)}h
                        </p>
                        <p className="text-[10px] text-zinc-500 mt-1">across {engagement.streakDays} day streak</p>
                    </div>

                    <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                        <div className="flex items-center gap-2 mb-2">
                            <Target className="w-4 h-4 text-blue-400" />
                            <span className="text-xs text-zinc-400">Topics Done</span>
                        </div>
                        <p className="text-2xl font-bold text-white">
                            {stats.completedCount}/{stats.totalTopics}
                        </p>
                        <p className="text-[10px] text-zinc-500 mt-1">{stats.estimatedHoursRemaining}h remaining</p>
                    </div>
                </div>

                {/* Track Comparison Badge */}
                <div className="p-3 rounded-xl bg-gradient-to-r from-violet-500/10 to-fuchsia-500/10 border border-violet-500/20">
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-violet-400 font-medium">Learning Style</span>
                        <span className={`text-xs font-medium ${
                            trackComparison.direction === 'vibe' ? 'text-orange-400' :
                            trackComparison.direction === 'python' ? 'text-purple-400' :
                            'text-zinc-400'
                        }`}>
                            {trackComparison.label}
                        </span>
                    </div>
                    <div className="flex items-center gap-4 mt-2 text-[10px] text-zinc-500">
                        <span>Vibe: {trackScores.vibeCoding.overall}%</span>
                        <span>Python: {trackScores.pythonLogic.overall}%</span>
                    </div>
                </div>

                {/* Skills Radar */}
                <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="text-sm font-medium text-zinc-300">Skill Profile</h4>
                        <span className="text-xs font-medium text-emerald-400">{skillScores.overall}% Overall</span>
                    </div>
                    
                    <div className="grid grid-cols-5 gap-2">
                        {[
                            { label: 'Coding', value: skillScores.coding, icon: Code, color: 'text-blue-400' },
                            { label: 'Logic', value: skillScores.logic, icon: Brain, color: 'text-purple-400' },
                            { label: 'Depth', value: skillScores.depth, icon: Lightbulb, color: 'text-amber-400' },
                            { label: 'Speed', value: skillScores.speed, icon: Zap, color: 'text-orange-400' },
                            { label: 'Consistency', value: skillScores.consistency, icon: Timer, color: 'text-cyan-400' },
                        ].map((skill) => (
                            <div key={skill.label} className="text-center">
                                <div className="relative w-12 h-12 mx-auto mb-2">
                                    <svg className="w-full h-full transform -rotate-90">
                                        <circle
                                            cx="24"
                                            cy="24"
                                            r="20"
                                            stroke="currentColor"
                                            strokeWidth="3"
                                            fill="none"
                                            className="text-zinc-700"
                                        />
                                        <motion.circle
                                            cx="24"
                                            cy="24"
                                            r="20"
                                            stroke="currentColor"
                                            strokeWidth="3"
                                            fill="none"
                                            className={skill.color}
                                            strokeLinecap="round"
                                            initial={{ strokeDasharray: '126' }}
                                            animate={{ strokeDasharray: `${(skill.value / 100) * 126} 126` }}
                                            transition={{ duration: 1, ease: 'easeOut' }}
                                        />
                                    </svg>
                                    <skill.icon className={`absolute inset-0 w-4 h-4 m-auto ${skill.color}`} />
                                </div>
                                <p className="text-[10px] font-medium text-zinc-400">{skill.label}</p>
                                <p className="text-[9px] text-zinc-500">{skill.value}%</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Performance Summary */}
                <div className="p-4 rounded-2xl bg-white/5 border border-white/5">
                    <div className="flex items-center gap-2 mb-4">
                        <BarChart3 className="w-4 h-4 text-emerald-400" />
                        <h4 className="text-sm font-medium text-zinc-300">Performance</h4>
                    </div>

                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-zinc-400">Quiz Average</span>
                            <span className="text-xs font-medium text-white">{skillScores.logic}%</span>
                        </div>
                        <div className="h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${skillScores.logic}%` }}
                                transition={{ duration: 0.5 }}
                                className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full"
                            />
                        </div>

                        <div className="flex items-center justify-between">
                            <span className="text-xs text-zinc-400">Current Phase</span>
                            <span className="text-xs font-medium text-white">Phase {stats.currentPhase} of {stats.totalPhases}</span>
                        </div>
                        <div className="h-1.5 bg-zinc-700 rounded-full overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(stats.currentPhase / stats.totalPhases) * 100}%` }}
                                transition={{ duration: 0.5 }}
                                className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full"
                            />
                        </div>
                    </div>
                </div>

                {/* Recommendation */}
                <div className={`p-4 rounded-2xl border ${
                    getRecommendation.priority === 'high' 
                        ? 'bg-amber-500/10 border-amber-500/20' 
                        : getRecommendation.priority === 'medium'
                        ? 'bg-blue-500/10 border-blue-500/20'
                        : 'bg-emerald-500/10 border-emerald-500/20'
                }`}>
                    <div className="flex items-center gap-2 mb-2">
                        <TrendingUp className={`w-4 h-4 ${
                            getRecommendation.priority === 'high' 
                                ? 'text-amber-400' 
                                : getRecommendation.priority === 'medium'
                                ? 'text-blue-400'
                                : 'text-emerald-400'
                        }`} />
                        <span className={`text-xs font-medium ${
                            getRecommendation.priority === 'high' 
                                ? 'text-amber-400' 
                                : getRecommendation.priority === 'medium'
                                ? 'text-blue-400'
                                : 'text-emerald-400'
                        }`}>
                            {studentProfile.category === 'fresher' ? 'Fresher Path' : 
                             studentProfile.category === 'job-seeker' ? 'Job Seeker Path' : 
                             'Career Switcher Path'}
                        </span>
                    </div>
                    <p className="text-sm text-zinc-300">{getRecommendation.text}</p>
                </div>

                {/* Category Indicator */}
                <div className="flex items-center gap-2 pt-2 border-t border-white/5">
                    {categoryOptions.find(o => o.value === studentProfile.category)?.icon && (
                        <>
                            {(() => {
                                const Icon = categoryOptions.find(o => o.value === studentProfile.category)!.icon;
                                return <Icon className="w-4 h-4 text-zinc-400" />;
                            })()}
                        </>
                    )}
                    <span className="text-xs text-zinc-500">
                        Targeting <span className="text-zinc-300">{studentProfile.targetRole}</span>
                    </span>
                </div>
            </div>
        </motion.div>
    );
};

export default CoursePassport;
