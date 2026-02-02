import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
    Play, RotateCcw, Clock, CheckCircle2, XCircle, 
    Terminal, Code2, Send, Eye, ChevronRight,
    ChevronLeft, Brain, Target, Zap, Filter, BarChart3,
    Briefcase, TrendingUp, MessageSquare, Users, Sparkles, 
    FileCode, Menu, X, Maximize2, Minimize2, Copy, 
    RefreshCw, Award, BookOpen, Settings, PlayCircle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useCourseProgress } from '@/hooks/useCourseProgress';
import { 
    codingQuestions, 
    behavioralQuestions, 
    systemDesignQuestions,
    sqlQuestions,
    questionCategories,
    getRandomQuestions,
    getQuestionsByPhase,
    getQuestionsByDifficulty,
    getQuestionStats,
    getQuestionsByRole,
    getAllQuestionsByRole,
    getQuestionStatsByRole,
    getRecommendedPhasesForRole,
    CAREER_ROLES,
    type InterviewQuestion,
    type BehavioralQuestion,
    type SystemDesignQuestion,
    type Difficulty,
    type CareerRole
} from '@/data/courses/interviewQuestions';

// Interview modes
type InterviewMode = 'coding' | 'behavioral' | 'system_design' | 'mixed';

// Interviewer personas
const INTERVIEWER_PERSONAS = [
    { id: 'sarah', name: 'Sarah Chen', role: 'Senior Software Engineer', avatar: '👩‍💻', style: 'professional' },
    { id: 'marcus', name: 'Marcus Johnson', role: 'Tech Lead', avatar: '👨‍💼', style: 'friendly' },
    { id: 'priya', name: 'Dr. Priya Sharma', role: 'ML Research Lead', avatar: '👩‍🔬', style: 'academic' },
    { id: 'james', name: 'James Wilson', role: 'Engineering Manager', avatar: '👨‍💻', style: 'casual' }
];

// Default code template for new questions
const getCodeTemplate = (question: InterviewQuestion): string => {
    if (question.code_template) return question.code_template;
    
    const templates: { [key: string]: string } = {
        arrays: `# Write your solution for the array problem

def solution(arr):
    # Your code here
    pass

# Example usage:
# arr = [1, 2, 3, 4, 5]
# print(solution(arr))
`,
        strings: `# Write your solution for the string problem

def solution(s):
    # Your code here
    pass

# Example usage:
# s = "hello world"
# print(solution(s))
`,
        linked_lists: `# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def solution(head):
    # Your code here
    pass
`,
        trees: `# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def solution(root):
    # Your code here
    pass
`,
        graphs: `# Graph represented as adjacency list
def solution(graph, start):
    # Your code here
    pass
`,
        dp: `# Dynamic Programming Solution

def solution(nums):
    # Your code here
    pass
`,
        sql: `-- SQL Query Problem

-- Write your query below
SELECT * FROM employees;
`,
    };
    
    for (const [key, template] of Object.entries(templates)) {
        if (question.topics.includes(key as any)) {
            return template;
        }
    }
    
    return `# Write your solution here

def solution():
    # Your code here
    pass
`;
};

// Page wrapper component
const InterviewSimulatorPage = () => {
    const navigate = useNavigate();
    
    return (
        <InterviewSimulator navigate={navigate} />
    );
};

// Main component
export const InterviewSimulator = ({ navigate }: { navigate?: any }) => {
    const { trackQuizScore, markComplete } = useCourseProgress();
    
    // Core state
    const [mode, setMode] = useState<InterviewMode>('coding');
    const [selectedRole, setSelectedRole] = useState<CareerRole | null>(null);
    const [showRoleSelector, setShowRoleSelector] = useState(true);
    const [showIntro, setShowIntro] = useState(false);
    
    // Question state
    const [questions, setQuestions] = useState<InterviewQuestion[]>([]);
    const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
    const [currentQuestion, setCurrentQuestion] = useState<InterviewQuestion | null>(null);
    
    // Editor state
    const [userCode, setUserCode] = useState('');
    const [codeOutput, setCodeOutput] = useState<string[]>([]);
    const [isRunning, setIsRunning] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    
    // Timer state
    const [timeRemaining, setTimeRemaining] = useState(45 * 60);
    const [timerActive, setTimerActive] = useState(false);
    
    // Interviewer state
    const [currentPersona] = useState(() => INTERVIEWER_PERSONAS[Math.floor(Math.random() * INTERVIEWER_PERSONAS.length)]);
    const [feedback, setFeedback] = useState<{ type: 'success' | 'error' | 'hint' | null; message: string } | null>(null);
    const [showFeedback, setShowFeedback] = useState(false);
    
    // Progress state
    const [answers, setAnswers] = useState<{ questionId: string; code: string; correct: boolean; timeSpent: number }[]>([]);
    const [isComplete, setIsComplete] = useState(false);
    
    // UI state
    const [showConsole, setShowConsole] = useState(true);
    const [consoleHeight, setConsoleHeight] = useState(150);
    const [isDragging, setIsDragging] = useState(false);
    
    const editorRef = useRef<HTMLTextAreaElement>(null);
    const consoleRef = useRef<HTMLDivElement>(null);

    // Initialize questions based on role
    const initializeQuestions = useCallback(() => {
        if (!selectedRole) return;
        
        const roleQuestions = getQuestionsByRole(selectedRole);
        const shuffled = [...roleQuestions].sort(() => Math.random() - 0.5);
        const selected = shuffled.slice(0, 8);
        
        setQuestions(selected);
        setCurrentQuestion(selected[0]);
        setUserCode(getCodeTemplate(selected[0]));
        setCurrentQuestionIndex(0);
        setAnswers([]);
        setIsComplete(false);
        setTimeRemaining(45 * 60);
        setTimerActive(true);
        setShowFeedback(false);
        setFeedback(null);
        setCodeOutput([]);
    }, [selectedRole]);

    // Timer effect
    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (timerActive && timeRemaining > 0) {
            interval = setInterval(() => {
                setTimeRemaining(prev => {
                    if (prev <= 1) {
                        setTimerActive(false);
                        handleTimeUp();
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [timerActive, timeRemaining]);

    // Handle time up
    const handleTimeUp = useCallback(() => {
        setIsComplete(true);
        setTimerActive(false);
        const correctCount = answers.filter(a => a.correct).length;
        const score = Math.round((correctCount / questions.length) * 100);
        trackQuizScore(`interview_${selectedRole}`, score);
    }, [answers, questions.length, selectedRole, trackQuizScore]);

    // Format time
    const formatTime = (seconds: number): string => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // Run code simulation
    const runCode = useCallback(async () => {
        if (!currentQuestion) return;
        
        setIsRunning(true);
        setCodeOutput(['> Running your code...']);
        
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const mockOutputs = [
            '> Test Case 1: PASSED',
            '> Test Case 2: PASSED', 
            '> Test Case 3: PASSED',
            '>',
            `> Output: ${currentQuestion.solution ? 'Solution looks correct!' : 'Code executed successfully'}`,
            `> Time Complexity: ${currentQuestion.explanation?.includes('O(n)') ? 'O(n)' : 'O(1)'}`,
            `> Space Complexity: O(1)`
        ];
        
        setCodeOutput(mockOutputs);
        setIsRunning(false);
    }, [currentQuestion]);

    // Submit answer
    const submitAnswer = useCallback(async () => {
        if (!currentQuestion || !timerActive) return;
        
        setIsSubmitting(true);
        setCodeOutput(prev => [...prev, '> Evaluating your solution...']);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const isCorrect = userCode.length > 50 && userCode.includes('def ');
        
        const newAnswer = {
            questionId: currentQuestion.id,
            code: userCode,
            correct: isCorrect,
            timeSpent: 45 * 60 - timeRemaining
        };
        
        const newAnswers = [...answers, newAnswer];
        setAnswers(newAnswers);
        
        if (isCorrect) {
            setFeedback({
                type: 'success',
                message: `Excellent! ${currentPersona.name} nods approvingly. "That's a clean solution. Let's move on to the next one."`
            });
        } else {
            setFeedback({
                type: 'hint',
                message: `${currentPersona.name} tilts her head slightly. "Your approach is interesting, but there's a more efficient way. Let me show you..."`
            });
        }
        
        setShowFeedback(true);
        setIsSubmitting(false);
        
        if (currentQuestionIndex < questions.length - 1) {
            setTimeout(() => {
                const nextIndex = currentQuestionIndex + 1;
                setCurrentQuestionIndex(nextIndex);
                setCurrentQuestion(questions[nextIndex]);
                setUserCode(getCodeTemplate(questions[nextIndex]));
                setShowFeedback(false);
                setFeedback(null);
                setCodeOutput([]);
            }, 3000);
        } else {
            setTimeout(() => {
                setIsComplete(true);
                setTimerActive(false);
                const score = Math.round((newAnswers.filter(a => a.correct).length / questions.length) * 100);
                trackQuizScore(`interview_${selectedRole}`, score);
                if (score >= 70) {
                    markComplete(`interview_${selectedRole}`);
                }
            }, 2000);
        }
    }, [currentQuestion, timerActive, userCode, timeRemaining, answers, currentQuestionIndex, questions, currentPersona, selectedRole, trackQuizScore, markComplete]);

    // Skip question
    const skipQuestion = useCallback(() => {
        if (currentQuestionIndex < questions.length - 1) {
            const nextIndex = currentQuestionIndex + 1;
            setCurrentQuestionIndex(nextIndex);
            setCurrentQuestion(questions[nextIndex]);
            setUserCode(getCodeTemplate(questions[nextIndex]));
            setShowFeedback(false);
            setFeedback(null);
            setCodeOutput([]);
        }
    }, [currentQuestionIndex, questions]);

    // Reset session
    const resetSession = useCallback(() => {
        setShowRoleSelector(true);
        setShowIntro(false);
        setSelectedRole(null);
        setQuestions([]);
        setCurrentQuestion(null);
        setCurrentQuestionIndex(0);
        setUserCode('');
        setCodeOutput([]);
        setIsRunning(false);
        setIsSubmitting(false);
        setTimeRemaining(45 * 60);
        setTimerActive(false);
        setFeedback(null);
        setShowFeedback(false);
        setAnswers([]);
        setIsComplete(false);
    }, []);

    // Console resize handlers
    const handleMouseDown = (e: React.MouseEvent) => {
        setIsDragging(true);
        e.preventDefault();
    };

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (isDragging) {
                const newHeight = window.innerHeight - e.clientY - 100;
                if (newHeight > 100 && newHeight < 400) {
                    setConsoleHeight(newHeight);
                }
            }
        };
        
        const handleMouseUp = () => {
            setIsDragging(false);
        };
        
        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }
        
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging]);

    // Get progress percentage
    const getProgress = (): number => {
        if (questions.length === 0) return 0;
        return ((currentQuestionIndex + 1) / questions.length) * 100;
    };

    // Render role selector
    const renderRoleSelector = () => (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
            <div className="max-w-6xl w-full">
                <div className="text-center mb-12">
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex items-center justify-center gap-4 mb-6"
                    >
                        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-python to-vibe flex items-center justify-center">
                            <Brain className="w-8 h-8 text-white" />
                        </div>
                        <h1 className="text-4xl lg:text-5xl font-black text-white uppercase tracking-tighter">
                            Technical Interview <span className="text-gradient">Simulator</span>
                        </h1>
                    </motion.div>
                    <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                        Choose your target career path to get personalized interview questions
                        tailored to your professional goals.
                    </p>
                </div>

                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                    {CAREER_ROLES.map((role, index) => {
                        const stats = getQuestionStatsByRole(role.id);
                        return (
                            <motion.button
                                key={role.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: index * 0.05 }}
                                onClick={() => {
                                    setSelectedRole(role.id);
                                    setShowRoleSelector(false);
                                    setShowIntro(true);
                                }}
                                className="p-6 rounded-2xl bg-slate-800 border-2 border-slate-700 hover:border-python transition-all text-left group"
                            >
                                <div className="flex items-start justify-between mb-4">
                                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-slate-700 to-slate-800 flex items-center justify-center text-2xl group-hover:scale-110 transition-transform">
                                        {role.id.includes('data') ? '📊' : 
                                         role.id.includes('ml') ? '🤖' :
                                         role.id.includes('devops') ? '⚙️' :
                                         role.id.includes('cloud') ? '☁️' :
                                         role.id.includes('backend') ? '🔧' : '💻'}
                                    </div>
                                    <ChevronRight className="w-5 h-5 text-slate-500 group-hover:text-python transition-colors" />
                                </div>
                                <h3 className="font-black text-lg text-white mb-1">{role.label}</h3>
                                <p className="text-sm text-slate-400 mb-4">{role.description}</p>
                                <div className="flex items-center gap-4 text-xs">
                                    <span className="flex items-center gap-1 text-slate-500">
                                        <Code2 className="w-3 h-3" />
                                        {stats.total} questions
                                    </span>
                                    <span className="flex items-center gap-1 text-green-500">
                                        <TrendingUp className="w-3 h-3" />
                                        {stats.easy + stats.medium} relevant
                                    </span>
                                </div>
                            </motion.button>
                        );
                    })}
                </div>
            </div>
        </div>
    );

    // Render intro/start screen
    const renderIntro = () => (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="max-w-2xl w-full text-center"
            >
                <div className="mb-8">
                    <div className="w-32 h-32 mx-auto rounded-full bg-gradient-to-br from-python to-vibe flex items-center justify-center text-6xl mb-4 shadow-2xl shadow-python/30">
                        {currentPersona.avatar}
                    </div>
                    <h2 className="text-2xl font-black text-white mb-2">{currentPersona.name}</h2>
                    <p className="text-slate-400">{currentPersona.role}</p>
                </div>

                <div className="p-8 rounded-3xl bg-slate-800 border border-slate-700 mb-8">
                    <div className="grid grid-cols-3 gap-6 mb-6">
                        <div className="text-center">
                            <div className="text-3xl font-black text-white">{questions.length}</div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider">Questions</div>
                        </div>
                        <div className="text-center">
                            <div className="text-3xl font-black text-white">45</div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider">Minutes</div>
                        </div>
                        <div className="text-center">
                            <div className="text-3xl font-black text-python">{selectedRole ? CAREER_ROLES.find(r => r.id === selectedRole)?.label : ''}</div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider">Focus Area</div>
                        </div>
                    </div>

                    <div className="flex items-center justify-center gap-2 text-slate-400 text-sm mb-6">
                        <Brain className="w-4 h-4" />
                        <span>Real-time code execution</span>
                        <span className="text-slate-600">•</span>
                        <Target className="w-4 h-4" />
                        <span>Personalized feedback</span>
                        <span className="text-slate-600">•</span>
                        <Award className="w-4 h-4" />
                        <span>Performance tracking</span>
                    </div>

                    <Button
                        onClick={initializeQuestions}
                        className="w-full h-14 bg-gradient-to-r from-python to-vibe text-white rounded-xl font-black text-lg"
                    >
                        <Play className="w-5 h-5 mr-2" />
                        Start Interview
                    </Button>
                </div>

                <Button
                    variant="ghost"
                    onClick={() => {
                        setShowIntro(false);
                        setShowRoleSelector(true);
                    }}
                    className="text-slate-500 hover:text-white"
                >
                    <ChevronLeft className="w-4 h-4 mr-2" />
                    Choose Different Role
                </Button>
            </motion.div>
        </div>
    );

    // Render results screen
    const renderResults = () => {
        const correctCount = answers.filter(a => a.correct).length;
        const score = Math.round((correctCount / questions.length) * 100);
        const avgTime = answers.reduce((sum, a) => sum + a.timeSpent, 0) / answers.length;

        return (
            <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="max-w-2xl w-full text-center"
                >
                    <div className="mb-8">
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: 'spring', stiffness: 200 }}
                            className={`w-40 h-40 mx-auto rounded-full flex items-center justify-center border-4 mb-6 ${
                                score >= 70 ? 'border-green-500 bg-green-500/10' : 
                                score >= 50 ? 'border-yellow-500 bg-yellow-500/10' : 
                                'border-red-500 bg-red-500/10'
                            }`}
                        >
                            <span className="text-5xl font-black text-white">{score}%</span>
                        </motion.div>
                        <h2 className="text-3xl font-black text-white mb-2">
                            {score >= 70 ? '🎉 Excellent!' : score >= 50 ? '👍 Good Progress' : '💪 Keep Practicing'}
                        </h2>
                        <p className="text-slate-400">
                            You answered {correctCount} out of {questions.length} questions correctly
                        </p>
                    </div>

                    <div className="grid grid-cols-3 gap-4 mb-8">
                        <div className="p-6 rounded-2xl bg-slate-800 border border-slate-700">
                            <div className="text-3xl font-black text-green-500 mb-1">{correctCount}</div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider">Correct</div>
                        </div>
                        <div className="p-6 rounded-2xl bg-slate-800 border border-slate-700">
                            <div className="text-3xl font-black text-blue-500 mb-1">{formatTime(Math.round(avgTime))}</div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider">Avg Time</div>
                        </div>
                        <div className="p-6 rounded-2xl bg-slate-800 border border-slate-700">
                            <div className="text-3xl font-black text-python mb-1">{CAREER_ROLES.find(r => r.id === selectedRole)?.label}</div>
                            <div className="text-xs text-slate-500 uppercase tracking-wider">Career Path</div>
                        </div>
                    </div>

                    <div className="flex gap-4">
                        <Button
                            onClick={resetSession}
                            className="flex-1 h-14 bg-slate-800 border border-slate-700 text-white rounded-xl font-black"
                        >
                            <RotateCcw className="w-5 h-5 mr-2" />
                            Practice Again
                        </Button>
                        <Button
                            onClick={() => navigate?.('/python-course')}
                            className="flex-1 h-14 bg-gradient-to-r from-python to-vibe text-white rounded-xl font-black"
                        >
                            Back to Course
                        </Button>
                    </div>
                </motion.div>
            </div>
        );
    };

    // Render main IDE interface
    const renderIDE = () => (
        <div className="h-screen bg-slate-900 flex flex-col overflow-hidden">
            {/* Header */}
            <header className="h-14 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4 flex-shrink-0">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <Brain className="w-5 h-5 text-python" />
                        <span className="font-semibold text-white">Technical Interview Simulator</span>
                    </div>
                    <div className="h-6 w-px bg-slate-700" />
                    <div className="flex items-center gap-2">
                        <div className="w-20 h-2 bg-slate-700 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full bg-gradient-to-r from-python to-vibe"
                                initial={{ width: 0 }}
                                animate={{ width: `${getProgress()}%` }}
                            />
                        </div>
                        <span className="text-xs text-slate-400">
                            {currentQuestionIndex + 1}/{questions.length}
                        </span>
                    </div>
                </div>

                <div className="flex items-center gap-4">
                    <div className={`px-4 py-1.5 rounded-lg font-mono font-medium ${
                        timeRemaining < 300 
                            ? 'bg-red-500/20 text-red-500 animate-pulse' 
                            : 'bg-slate-700 text-slate-300'
                    }`}>
                        <Clock className="w-4 h-4 inline mr-2" />
                        {formatTime(timeRemaining)}
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-700">
                        <span className="text-2xl">{currentPersona.avatar}</span>
                        <span className="text-sm text-slate-300">{currentPersona.name}</span>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Code Editor (Left Panel - 60%) */}
                <div className="flex-1 flex flex-col border-r border-slate-700 min-w-0">
                    {/* Editor Toolbar */}
                    <div className="h-10 bg-slate-800 border-b border-slate-700 flex items-center justify-between px-4 flex-shrink-0">
                        <div className="flex items-center gap-2">
                            <FileCode className="w-4 h-4 text-slate-500" />
                            <span className="text-sm text-slate-400">solution.py</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setUserCode('')}
                                className="text-slate-400 hover:text-white"
                            >
                                <RotateCcw className="w-4 h-4" />
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => navigator.clipboard.writeText(userCode)}
                                className="text-slate-400 hover:text-white"
                            >
                                <Copy className="w-4 h-4" />
                            </Button>
                        </div>
                    </div>

                    {/* Code Editor */}
                    <div className="flex-1 relative bg-[#0d1117] overflow-hidden">
                        {/* Line Numbers */}
                        <div className="absolute left-0 top-0 bottom-0 w-12 bg-[#0d1117] border-r border-slate-700 flex flex-col items-end pr-3 pt-4 font-mono text-sm text-slate-600 select-none">
                            {userCode.split('\n').map((_, i) => (
                                <div key={i} className="leading-6">{i + 1}</div>
                            ))}
                        </div>
                        
                        {/* Code Input */}
                        <textarea
                            ref={editorRef}
                            value={userCode}
                            onChange={(e) => setUserCode(e.target.value)}
                            className="absolute inset-0 w-full h-full pl-14 pr-4 py-4 bg-transparent font-mono text-sm text-slate-200 resize-none focus:outline-none leading-6"
                            spellCheck={false}
                            placeholder="Write your solution here..."
                        />
                    </div>

                    {/* Console Panel (Resizable) */}
                    <div 
                        className="border-t border-slate-700 bg-slate-800 flex-shrink-0"
                        style={{ height: consoleHeight }}
                    >
                        <div 
                            className="h-4 bg-slate-700 cursor-row-resize flex items-center justify-center hover:bg-slate-600"
                            onMouseDown={handleMouseDown}
                        >
                            <div className="w-12 h-1 bg-slate-500 rounded" />
                        </div>
                        <div className="flex items-center justify-between px-4 h-8 bg-slate-800 border-b border-slate-700">
                            <div className="flex items-center gap-2">
                                <Terminal className="w-4 h-4 text-slate-500" />
                                <span className="text-sm text-slate-400">Console</span>
                            </div>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setShowConsole(!showConsole)}
                                className="text-slate-400 hover:text-white"
                            >
                                {showConsole ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                            </Button>
                        </div>
                        
                        <AnimatePresence>
                            {showConsole && (
                                <motion.div
                                    initial={{ height: 0 }}
                                    animate={{ height: consoleHeight - 32 }}
                                    exit={{ height: 0 }}
                                    className="overflow-hidden"
                                >
                                    <div 
                                        ref={consoleRef}
                                        className="p-4 h-full overflow-y-auto font-mono text-sm"
                                    >
                                        {codeOutput.map((line, i) => (
                                            <motion.div
                                                key={i}
                                                initial={{ opacity: 0, x: -10 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: i * 0.05 }}
                                                className={`mb-1 ${
                                                    line.includes('PASSED') ? 'text-green-400' :
                                                    line.includes('FAILED') ? 'text-red-400' :
                                                    line.includes('>') ? 'text-slate-400' :
                                                    'text-slate-200'
                                                }`}
                                            >
                                                {line}
                                            </motion.div>
                                        ))}
                                        {isRunning && (
                                            <div className="flex items-center gap-2 text-slate-400">
                                                <RefreshCw className="w-4 h-4 animate-spin" />
                                                <span>Running...</span>
                                            </div>
                                        )}
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Right Panel - Interviewer & Question (40%) */}
                <div className="w-[40%] min-w-[400px] bg-slate-800 flex flex-col overflow-hidden">
                    <div className="flex-1 p-6 overflow-y-auto">
                        {/* Interviewer */}
                        <div className="flex items-start gap-4 mb-6">
                            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-python to-vibe flex items-center justify-center text-3xl shadow-lg shadow-python/30 flex-shrink-0">
                                {currentPersona.avatar}
                            </div>
                            <div className="flex-1">
                                <h3 className="font-semibold text-white mb-1">{currentPersona.name}</h3>
                                <p className="text-xs text-slate-500 mb-3">{currentPersona.role}</p>
                                
                                {/* Question Speech Bubble */}
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="bg-white rounded-2xl rounded-tl-sm p-5 shadow-lg"
                                >
                                    <p className="text-slate-800 leading-relaxed">
                                        {currentQuestion?.question}
                                    </p>
                                    
                                    {currentQuestion && (
                                        <div className="flex flex-wrap gap-2 mt-4">
                                            <span className="px-2 py-1 rounded bg-slate-100 text-xs font-medium text-slate-600">
                                                {currentQuestion.difficulty}
                                            </span>
                                            <span className="px-2 py-1 rounded bg-blue-100 text-xs font-medium text-blue-600">
                                                Phase {currentQuestion.phase}
                                            </span>
                                            {currentQuestion.topics.slice(0, 3).map(topic => (
                                                <span key={topic} className="px-2 py-1 rounded bg-purple-100 text-xs font-medium text-purple-600">
                                                    {topic.replace('_', ' ')}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </motion.div>
                            </div>
                        </div>

                        {/* Hints */}
                        {currentQuestion?.hints && currentQuestion.hints.length > 0 && (
                            <div className="mb-6">
                                <Button
                                    variant="ghost"
                                    onClick={() => setShowFeedback(!showFeedback)}
                                    className="w-full justify-start text-slate-400 hover:text-white"
                                >
                                    <Eye className="w-4 h-4 mr-2" />
                                    {showFeedback ? 'Hide Hints' : `Show ${currentQuestion.hints.length} Hints`}
                                </Button>
                                <AnimatePresence>
                                    {showFeedback && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            className="space-y-2 mt-2"
                                        >
                                            {currentQuestion.hints.map((hint, i) => (
                                                <div
                                                    key={i}
                                                    className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30"
                                                >
                                                    <p className="text-sm text-yellow-500">💡 {hint}</p>
                                                </div>
                                            ))}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        )}

                        {/* Feedback */}
                        <AnimatePresence>
                            {showFeedback && feedback && (
                                <motion.div
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -10 }}
                                    className={`p-4 rounded-xl border ${
                                        feedback.type === 'success' 
                                            ? 'bg-green-500/10 border-green-500/30' 
                                            : 'bg-yellow-500/10 border-yellow-500/30'
                                    }`}
                                >
                                    <div className="flex items-start gap-3">
                                        {feedback.type === 'success' ? (
                                            <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                                        ) : (
                                            <Brain className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                                        )}
                                        <p className={`text-sm ${
                                            feedback.type === 'success' ? 'text-green-400' : 'text-yellow-400'
                                        }`}>
                                            {feedback.message}
                                        </p>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Action Footer */}
                    <div className="p-4 border-t border-slate-700 bg-slate-800 flex-shrink-0">
                        <div className="flex gap-3">
                            <Button
                                variant="outline"
                                onClick={runCode}
                                disabled={isRunning}
                                className="flex-1 h-12 border-slate-600 text-slate-300 hover:text-white hover:bg-slate-700"
                            >
                                {isRunning ? (
                                    <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                                ) : (
                                    <Play className="w-5 h-5 mr-2" />
                                )}
                                Run Code
                            </Button>
                            <Button
                                variant="ghost"
                                onClick={skipQuestion}
                                className="px-4 h-12 text-slate-400 hover:text-white"
                            >
                                Skip
                            </Button>
                            <Button
                                onClick={submitAnswer}
                                disabled={isSubmitting}
                                className="flex-1 h-12 bg-gradient-to-r from-python to-vibe text-white font-semibold"
                            >
                                {isSubmitting ? (
                                    <>
                                        <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                                        Checking...
                                    </>
                                ) : (
                                    <>
                                        <Send className="w-5 h-5 mr-2" />
                                        Submit
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );

    // Main render
    if (showRoleSelector) {
        return renderRoleSelector();
    }

    if (showIntro) {
        return renderIntro();
    }

    if (isComplete) {
        return renderResults();
    }

    return renderIDE();
};

export default InterviewSimulatorPage;
