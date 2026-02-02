import { useState, useEffect, useMemo, useCallback } from 'react';
import { allPhases } from '../data/courses';

// Types matching CoursePassport.tsx
export type StudentCategory = 'fresher' | 'job-seeker' | 'career-switcher';

export interface StudentProfile {
    category: StudentCategory;
    targetRole: string;
    experience: string;
    availability: string;
}

export interface EngagementData {
    timePerTopic: Record<string, number>; // minutes spent per topic
    scrollDepth: Record<string, number>; // percentage scrolled per topic
    lastActive: Date;
    streakDays: number;
    totalSessionMinutes: number;
}

export interface PerformanceData {
    quizScores: Record<string, number>;
    projectScores: Record<string, number>;
    averageScore: number;
    strongTopics: string[];
    weakTopics: string[];
}

export interface SkillScores {
    coding: number;
    logic: number;
    depth: number;
    speed: number;
    consistency: number;
    overall: number;
}

export interface CourseStats {
    completedCount: number;
    totalTopics: number;
    progressPercentage: number;
    estimatedHoursRemaining: number;
    currentPhase: number;
    totalPhases: number;
}

export interface TrackScores {
    vibeCoding: {
        speed: number;
        projects: number;
        output: number;
        overall: number;
    };
    pythonLogic: {
        algorithms: number;
        problemSolving: number;
        depth: number;
        overall: number;
    };
}

export interface UseCourseProgressReturn {
    // Profile
    studentProfile: StudentProfile;
    updateStudentProfile: (updates: Partial<StudentProfile>) => void;
    
    // Engagement
    engagement: EngagementData;
    trackEngagement: (topicId: string, timeMinutes: number, scrollPercent: number) => void;
    
    // Performance
    performance: PerformanceData;
    trackQuizScore: (quizId: string, score: number) => void;
    trackProjectScore: (projectId: string, score: number) => void;
    
    // Progress
    completedTopics: Set<string>;
    markComplete: (topicId: string) => void;
    setCurrentPhase: (phase: number) => void;
    
    // Calculated metrics
    skillScores: SkillScores;
    trackScores: TrackScores;
    stats: CourseStats;
    
    // Utility
    calculateStreak: () => void;
    isCourseUnlocked: (courseId: string) => boolean;
}

// Dynamic course metrics calculation
const calculateCourseMetrics = () => {
    let topics = 0;
    let phases = 0;

    allPhases.forEach(phase => {
        phases++;
        phase.topics.forEach(() => topics++);
        if (phase.folders) {
            phase.folders.forEach(folder => {
                folder.topics.forEach(() => topics++);
            });
        }
    });

    return { totalTopics: topics, totalPhases: phases };
};

// Calculate speed score based on completion velocity
const calculateSpeedScore = (completedTopics: number, totalTopics: number, timeSpent: number): number => {
    const completionRate = completedTopics / Math.max(totalTopics, 1);
    const expectedTime = completionRate * totalTopics * 120;
    const timeRatio = expectedTime > 0 ? Math.min(timeSpent / expectedTime, 1.5) : 1;
    return Math.min(50 + (timeRatio * 50), 100);
};

// Calculate consistency score
const calculateConsistencyScore = (streakDays: number, completedTopics: number, totalTopics: number): number => {
    const streakBonus = Math.min(streakDays * 2, 30);
    const completionBonus = Math.min((completedTopics / Math.max(totalTopics, 1)) * 50, 50);
    return Math.min(streakBonus + completionBonus, 100);
};

// Calculate strong and weak topics based on quiz scores
const calculateStrongWeakTopics = (quizScores: Record<string, number>): { strong: string[], weak: string[] } => {
    const entries = Object.entries(quizScores);
    if (entries.length < 3) return { strong: [], weak: [] };
    
    const sorted = entries.sort((a, b) => b[1] - a[1]);
    const strong = sorted.slice(0, 3).map(([topic]) => topic);
    const weak = sorted.slice(-2).map(([topic]) => topic);
    
    return { strong, weak };
};

export const useCourseProgress = (): UseCourseProgressReturn => {
    // Get dynamic course metrics
    const { totalTopics: dynamicTotalTopics, totalPhases: dynamicTotalPhases } = useMemo(
        () => calculateCourseMetrics(),
        []
    );

    // Student Profile State
    const [studentProfile, setStudentProfile] = useState<StudentProfile>({
        category: 'job-seeker',
        targetRole: 'ML Engineer',
        experience: '0-1 years',
        availability: 'Immediate'
    });

    // Engagement Tracking State
    const [engagement, setEngagement] = useState<EngagementData>({
        timePerTopic: {},
        scrollDepth: {},
        lastActive: new Date(),
        streakDays: 0,
        totalSessionMinutes: 0
    });

    // Performance Tracking State
    const [performance, setPerformance] = useState<PerformanceData>({
        quizScores: {},
        projectScores: {},
        averageScore: 0,
        strongTopics: [],
        weakTopics: []
    });

    // Progress Tracking State
    const [completedTopics, setCompletedTopics] = useState<Set<string>>(new Set());
    const [currentPhase, setCurrentPhaseState] = useState(1);

    // Check if course is unlocked
    const isCourseUnlocked = useCallback((courseId: string): boolean => {
        // For now, we'll return true to allow access
        // In a real implementation, this would check if the user has access to the course
        return true;
    }, []);

    // Load saved data on mount
    useEffect(() => {
        const savedProfile = localStorage.getItem('student_profile');
        const savedEngagement = localStorage.getItem('engagement_data');
        const savedPerformance = localStorage.getItem('performance_data');
        const savedProgress = localStorage.getItem('completed_topics');

        if (savedProfile) {
            try {
                setStudentProfile(JSON.parse(savedProfile));
            } catch (e) {
                console.error('Failed to parse profile');
            }
        }

        if (savedEngagement) {
            try {
                const parsed = JSON.parse(savedEngagement);
                setEngagement({ ...parsed, lastActive: new Date(parsed.lastActive) });
            } catch (e) {
                console.error('Failed to parse engagement');
            }
        }

        if (savedPerformance) {
            try {
                setPerformance(JSON.parse(savedPerformance));
            } catch (e) {
                console.error('Failed to parse performance');
            }
        }

        if (savedProgress) {
            try {
                setCompletedTopics(new Set(JSON.parse(savedProgress)));
            } catch (e) {
                console.error('Failed to parse progress');
            }
        }

        calculateStreak();
    }, []);

    // Save data on change
    useEffect(() => {
        localStorage.setItem('student_profile', JSON.stringify(studentProfile));
        localStorage.setItem('engagement_data', JSON.stringify(engagement));
        localStorage.setItem('performance_data', JSON.stringify(performance));
        localStorage.setItem('completed_topics', JSON.stringify(Array.from(completedTopics)));
    }, [studentProfile, engagement, performance, completedTopics]);

    // Calculate streak
    const calculateStreak = useCallback(() => {
        const savedDate = localStorage.getItem('last_active_date');
        if (!savedDate) {
            setEngagement(prev => ({ ...prev, streakDays: 1 }));
            return;
        }

        const lastActive = new Date(savedDate);
        const today = new Date();
        const diffDays = Math.floor((today.getTime() - lastActive.getTime()) / (1000 * 60 * 60 * 24));

        if (diffDays === 0) {
            // Same day, keep streak
        } else if (diffDays === 1) {
            // Consecutive day, increment streak
            setEngagement(prev => ({ ...prev, streakDays: (prev.streakDays || 1) + 1 }));
        } else {
            // Streak broken
            setEngagement(prev => ({ ...prev, streakDays: 1 }));
        }

        // Update last active
        setEngagement(prev => ({ ...prev, lastActive: new Date() }));
        localStorage.setItem('last_active_date', new Date().toISOString());
    }, []);

    // Calculate skill scores based on multiple factors
    const skillScores = useMemo((): SkillScores => {
        const completionRate = dynamicTotalTopics > 0 ? completedTopics.size / dynamicTotalTopics : 0;
        
        // Calculate individual scores
        const coding = Math.min(completedTopics.size * 3 + (performance.averageScore * 0.5), 100);
        const logic = Math.min(performance.averageScore + 20, 100);
        const depth = Math.min(completionRate * 100, 100);
        const speed = calculateSpeedScore(completedTopics.size, dynamicTotalTopics, engagement.totalSessionMinutes);
        const consistency = calculateConsistencyScore(engagement.streakDays, completedTopics.size, dynamicTotalTopics);
        
        // Weighted overall
        const overall = Math.round(
            (coding * 0.25) +
            (logic * 0.20) +
            (depth * 0.20) +
            (speed * 0.15) +
            (consistency * 0.20)
        );

        return {
            coding: Math.round(coding),
            logic: Math.round(logic),
            depth: Math.round(depth),
            speed: Math.round(speed),
            consistency: Math.round(consistency),
            overall
        };
    }, [completedTopics, performance, engagement, dynamicTotalTopics]);

    // Calculate track-specific scores (Vibe Coding vs Python Logic)
    const trackScores = useMemo((): TrackScores => {
        const completionRate = dynamicTotalTopics > 0 ? completedTopics.size / dynamicTotalTopics : 0;
        
        // Vibe Coding Track - emphasizes speed and output
        const vibeSpeed = calculateSpeedScore(completedTopics.size, dynamicTotalTopics, engagement.totalSessionMinutes);
        const vibeProjects = Math.min((Object.keys(performance.projectScores).length * 15) + (performance.averageScore * 0.3), 100);
        const vibeOutput = Math.min(completionRate * 120, 100);
        const vibeOverall = Math.round((vibeSpeed * 0.4) + (vibeProjects * 0.3) + (vibeOutput * 0.3));
        
        // Python Logic Track - emphasizes depth and problem solving
        const pyAlgorithms = Math.min(performance.averageScore + 25, 100);
        const pyProblemSolving = Math.min(performance.averageScore + 15, 100);
        const pyDepth = Math.min(completionRate * 100 + 20, 100);
        const pyOverall = Math.round((pyAlgorithms * 0.35) + (pyProblemSolving * 0.35) + (pyDepth * 0.30));
        
        return {
            vibeCoding: {
                speed: Math.round(vibeSpeed),
                projects: Math.round(vibeProjects),
                output: Math.round(vibeOutput),
                overall: vibeOverall
            },
            pythonLogic: {
                algorithms: Math.round(pyAlgorithms),
                problemSolving: Math.round(pyProblemSolving),
                depth: Math.round(pyDepth),
                overall: pyOverall
            }
        };
    }, [completedTopics, performance, engagement, dynamicTotalTopics]);

    // Calculate course statistics
    const stats = useMemo((): CourseStats => {
        const progressPercentage = dynamicTotalTopics > 0 
            ? Math.round((completedTopics.size / dynamicTotalTopics) * 100) 
            : 0;
        
        // Calculate estimated time remaining
        const remaining = dynamicTotalTopics - completedTopics.size;
        const avgMinutes = engagement.totalSessionMinutes / Math.max(completedTopics.size, 1);
        const totalMinutes = remaining * avgMinutes;
        const estimatedHoursRemaining = Math.round(totalMinutes / 60);

        return {
            completedCount: completedTopics.size,
            totalTopics: dynamicTotalTopics,
            progressPercentage,
            estimatedHoursRemaining,
            currentPhase,
            totalPhases: dynamicTotalPhases
        };
    }, [completedTopics, engagement, currentPhase, dynamicTotalTopics, dynamicTotalPhases]);

    // Update engagement metrics
    const trackEngagement = useCallback((topicId: string, timeMinutes: number, scrollPercent: number) => {
        setEngagement(prev => ({
            ...prev,
            timePerTopic: {
                ...prev.timePerTopic,
                [topicId]: (prev.timePerTopic[topicId] || 0) + timeMinutes
            },
            scrollDepth: {
                ...prev.scrollDepth,
                [topicId]: Math.max(prev.scrollDepth[topicId] || 0, scrollPercent)
            },
            totalSessionMinutes: prev.totalSessionMinutes + timeMinutes,
            lastActive: new Date()
        }));

        calculateStreak();
    }, [calculateStreak]);

    // Track quiz performance
    const trackQuizScore = useCallback((quizId: string, score: number) => {
        setPerformance(prev => {
            const newScores = { ...prev.quizScores, [quizId]: score };
            const scores = Object.values(newScores);
            const averageScore = scores.length > 0 
                ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length)
                : 0;
            
            // Calculate strong/weak topics
            const { strong, weak } = calculateStrongWeakTopics(newScores);
            
            return {
                ...prev,
                quizScores: newScores,
                averageScore,
                strongTopics: strong,
                weakTopics: weak
            };
        });
    }, []);

    // Track project performance
    const trackProjectScore = useCallback((projectId: string, score: number) => {
        setPerformance(prev => ({
            ...prev,
            projectScores: {
                ...prev.projectScores,
                [projectId]: score
            }
        }));
    }, []);

    // Mark topic as complete
    const markComplete = useCallback((topicId: string) => {
        setCompletedTopics(prev => {
            const newSet = new Set(prev);
            newSet.add(topicId);
            return newSet;
        });

        // Auto-update engagement on completion
        trackEngagement(topicId, 30, 100);
    }, [trackEngagement]);

    // Update student profile
    const updateStudentProfile = useCallback((updates: Partial<StudentProfile>) => {
        setStudentProfile(prev => ({ ...prev, ...updates }));
    }, []);

    // Set current phase
    const setCurrentPhase = useCallback((phase: number) => {
        setCurrentPhaseState(Math.max(1, Math.min(phase, dynamicTotalPhases)));
    }, [dynamicTotalPhases]);

    return {
        studentProfile,
        updateStudentProfile,
        engagement,
        trackEngagement,
        performance,
        trackQuizScore,
        trackProjectScore,
        completedTopics,
        markComplete,
        setCurrentPhase,
        skillScores,
        trackScores,
        stats,
        calculateStreak,
        isCourseUnlocked
    };
};

// Legacy compatibility hook - returns simplified stats for existing components
export const useLegacyCourseProgress = () => {
    const hook = useCourseProgress();
    
    return {
        stats: {
            completedCount: hook.stats.completedCount,
            buildHours: Math.round(hook.stats.estimatedHoursRemaining * 0.75),
            marketAuthority: hook.skillScores.overall,
            strengthProfile: {
                research: hook.skillScores.depth,
                logic: hook.skillScores.logic,
                build: hook.skillScores.coding,
                integrity: hook.skillScores.consistency,
                business: 0
            },
            unlockedPhases: {}
        }
    };
};

export default useCourseProgress;
