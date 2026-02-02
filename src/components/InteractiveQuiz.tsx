import { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle2, XCircle, ArrowRight, RotateCcw, Award, HelpCircle, ChevronRight, ChevronLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useCourseProgress } from '@/hooks/useCourseProgress';

interface Question {
    id: number;
    question: string;
    code?: string;
    options: {
        key: string;
        text: string;
    }[];
    answer: string;
    explanation: string;
}

interface InteractiveQuizProps {
    markdownPath: string;
    onComplete?: () => void;
}

export const InteractiveQuiz = ({ markdownPath, onComplete }: InteractiveQuizProps) => {
    const [questions, setQuestions] = useState<Question[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [selectedOption, setSelectedOption] = useState<string | null>(null);
    const [isAnswered, setIsAnswered] = useState(false);
    const [score, setScore] = useState(0);
    const [quizComplete, setQuizComplete] = useState(false);
    const [loading, setLoading] = useState(true);
    const [mistakes, setMistakes] = useState(0);
    const [failedAttempts, setFailedAttempts] = useState(0);

    useEffect(() => {
        const fetchAndParse = async () => {
            try {
                setLoading(true);
                const response = await fetch(markdownPath);
                const text = await response.text();

                // Parse the markdown into questions
                const rawBlocks = text.split(/\n\*\*?\d+\.\s*/g);

                const parsed: Question[] = [];
                let qId = 1;

                // Skip first block (header)
                for (let i = 1; i < rawBlocks.length; i++) {
                    const block = rawBlocks[i];

                    // Extract question text - everything until options or code
                    const qMatch = block.match(/^([\s\S]*?)(?:\n\s*[a-d]\)|```python)/);
                    let questionText = qMatch ? qMatch[1].trim() : block.split('\n')[0].trim();
                    questionText = questionText.replace(/\*\*$/, '').trim(); // Remove trailing markers

                    // Extract code
                    const codeMatch = block.match(/```python([\s\S]*?)```/);
                    const code = codeMatch ? codeMatch[1].trim() : undefined;

                    // Extract options
                    const options: { key: string, text: string }[] = [];
                    const optionMatches = block.matchAll(/^\s*([a-d])\)\s*(.*?)\n/gm);
                    for (const m of optionMatches) {
                        options.push({ key: m[1], text: m[2].trim() });
                    }

                    if (options.length === 0) {
                        // Try alternate format: a) without newline
                        const altOptionMatches = block.matchAll(/\b([a-d])\)\s*([^✅\*]*?)(?=\s*[a-d]\)|✅|\*\*Answer|\*Explanation|$)/g);
                        for (const m of altOptionMatches) {
                            options.push({ key: m[1], text: m[2].trim() });
                        }
                    }

                    if (options.length === 0) continue;

                    // Extract answer
                    const ansMatch = block.match(/(?:✅\s*)?\*\*?Answer:\s*([a-d])\*\*?/i);
                    const answer = ansMatch ? ansMatch[1].toLowerCase() : '';

                    // Extract explanation
                    const expMatch = block.match(/(?:📝\s*)?\*\*?Explanation\*\*?:\s*([\s\S]*?)(?:\n\s*---|\n\n|$)/i) ||
                        block.match(/\*Explanation:\s*([\s\S]*?)\*/i);
                    const explanation = expMatch ? expMatch[1].trim() : '';

                    parsed.push({
                        id: qId++,
                        question: questionText,
                        code,
                        options,
                        answer,
                        explanation
                    });
                }

                setQuestions(parsed);
            } catch (err) {
                console.error("Failed to parse quiz:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchAndParse();
    }, [markdownPath]);

    const currentQuestion = questions[currentIndex];

    const handleOptionSelect = (key: string) => {
        if (isAnswered) return;
        setSelectedOption(key);
    };

    const handleCheck = () => {
        if (!selectedOption || isAnswered) return;
        setIsAnswered(true);
        if (selectedOption === currentQuestion.answer) {
            setScore(s => s + 1);
        } else {
            setMistakes(m => m + 1);
        }
    };

    const { trackQuizScore } = useCourseProgress();

    const handleNext = () => {
        if (currentIndex < questions.length - 1) {
            setCurrentIndex(c => c + 1);
            setSelectedOption(null);
            setIsAnswered(false);
        } else {
            const finalScore = score + (selectedOption === currentQuestion.answer ? 1 : 0); // Include current Q
            const percent = Math.round((finalScore / questions.length) * 100);

            setQuizComplete(true);

            // Track quiz score to the student profile
            if (markdownPath) {
                // Extract a quiz identifier from the markdown path
                const quizId = markdownPath.split('/').pop()?.replace('.md', '') || 'unknown';
                trackQuizScore(quizId, percent);
            }

            // Only trigger onComplete if they pass the 70% threshold
            if (percent >= 70 && onComplete) {
                onComplete();
            } else {
                setFailedAttempts(f => f + 1);
            }
        }
    };

    const resetQuiz = () => {
        setCurrentIndex(0);
        setSelectedOption(null);
        setIsAnswered(false);
        setScore(0);
        setQuizComplete(false);
    };

    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center p-20 gap-4">
                <div className="w-12 h-12 border-4 border-python border-t-transparent rounded-full animate-spin" />
                <p className="text-muted-foreground animate-pulse font-medium">Preparing your assessment...</p>
            </div>
        );
    }

    if (questions.length === 0) {
        return (
            <div className="p-12 text-center bg-muted/20 rounded-2xl border border-dashed border-border">
                <HelpCircle className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                <h3 className="text-lg font-semibold mb-2">Quiz format not recognized</h3>
                <p className="text-muted-foreground text-sm">We couldn't automatically convert this content into an interactive quiz.</p>
            </div>
        );
    }

    if (quizComplete) {
        const percent = Math.round((score / questions.length) * 100);
        const hasPassed = percent >= 70;

        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center p-8 lg:p-12 bg-white dark:bg-zinc-900 rounded-[2.5rem] border border-border shadow-2xl relative overflow-hidden"
            >
                <div className={`absolute top-0 left-0 w-full h-3 ${hasPassed ? 'bg-success' : 'bg-destructive'}`} />

                <div className="w-24 h-24 rounded-3xl bg-muted flex items-center justify-center mx-auto mb-8 relative rotate-3">
                    {hasPassed ? (
                        <>
                            <Award className="w-12 h-12 text-python" />
                            <div className="absolute -top-2 -right-2 p-1.5 rounded-full bg-success text-white shadow-lg">
                                <CheckCircle2 className="w-4 h-4" />
                            </div>
                        </>
                    ) : (
                        <>
                            <RotateCcw className="w-12 h-12 text-destructive animate-pulse" />
                            <div className="absolute -top-2 -right-2 p-1.5 rounded-full bg-destructive text-white shadow-lg">
                                <XCircle className="w-4 h-4" />
                            </div>
                        </>
                    )}
                </div>

                <h2 className="text-4xl font-black mb-3 italic">
                    {hasPassed ? "LOGIC INTEGRITY VERIFIED" : "INTEGRITY CHECK FAILED"}
                </h2>
                <p className="text-muted-foreground mb-10 max-w-sm mx-auto font-medium">
                    {hasPassed
                        ? `Elite performance detected. You've successfully internalized ${score} critical logic nodes.`
                        : `Current logic accuracy (${percent}%) is below the Professional Threshold of 70%. Protocol reset required.`}
                </p>

                <div className="grid grid-cols-2 gap-4 mb-10">
                    <div className={`p-6 rounded-3xl border-2 transition-all ${hasPassed ? 'bg-success/5 border-success/10' : 'bg-destructive/5 border-destructive/10'}`}>
                        <div className={`text-4xl font-black mb-1 ${hasPassed ? 'text-success' : 'text-destructive'}`}>{percent}%</div>
                        <div className="text-[10px] font-black uppercase tracking-widest opacity-50">Precision Score</div>
                    </div>
                    <div className="p-6 rounded-3xl bg-muted/50 border border-border">
                        <div className="text-4xl font-black mb-1">{mistakes}</div>
                        <div className="text-[10px] font-black uppercase tracking-widest opacity-50">Logic Faults</div>
                    </div>
                </div>

                <div className="flex flex-col gap-4">
                    {!hasPassed && (
                        <Button
                            onClick={resetQuiz}
                            className="bg-destructive hover:bg-destructive/90 text-white h-16 rounded-2xl font-black text-xl shadow-xl shadow-destructive/20"
                        >
                            <RotateCcw className="w-6 h-6 mr-3" />
                            Initialize Resimulation
                        </Button>
                    )}
                    {hasPassed && (
                        <Button
                            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                            className="bg-python hover:bg-python/90 text-python-foreground h-16 rounded-2xl font-black text-xl shadow-xl shadow-python/20"
                        >
                            Next Core Module
                            <ArrowRight className="w-6 h-6 ml-3" />
                        </Button>
                    )}
                    <Button variant="ghost" onClick={resetQuiz} className="h-12 rounded-xl font-bold text-muted-foreground">
                        Review Protocol Manual
                    </Button>
                </div>
            </motion.div>
        );
    }

    return (
        <div className="max-w-3xl mx-auto space-y-6">
            {/* Header Info */}
            <div className="flex items-center justify-between px-2">
                <div className="space-y-1">
                    <span className="text-xs font-bold text-python uppercase tracking-widest">Question {currentIndex + 1} of {questions.length}</span>
                    <div className="w-48 lg:w-64">
                        <Progress value={((currentIndex + 1) / questions.length) * 100} className="h-1.5 bg-python/5" />
                    </div>
                </div>
                <div className="flex items-center gap-4">
                    <div className="text-right">
                        <div className="text-sm font-bold">{score} Correct</div>
                        <div className="text-[10px] text-muted-foreground">CURRENT ACCURACY</div>
                    </div>
                </div>
            </div>

            <AnimatePresence mode="wait">
                <motion.div
                    key={currentIndex}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="space-y-6"
                >
                    {/* Question Box */}
                    <div className="p-6 lg:p-8 bg-card border border-border rounded-3xl shadow-sm">
                        <h3 className="text-xl font-bold leading-relaxed mb-6">
                            {currentQuestion.question}
                        </h3>

                        {currentQuestion.code && (
                            <div className="mb-6 rounded-2xl overflow-hidden border border-white/5 bg-[#1e1e1e]">
                                <div className="flex items-center gap-1.5 px-4 py-2 bg-[#252525] border-b border-white/5">
                                    <div className="w-2 h-2 rounded-full bg-red-500/50" />
                                    <div className="w-2 h-2 rounded-full bg-yellow-500/50" />
                                    <div className="w-2 h-2 rounded-full bg-green-500/50" />
                                    <span className="text-[10px] font-mono text-muted-foreground/40 ml-2">code_snippet.py</span>
                                </div>
                                <pre className="p-5 overflow-x-auto text-sm font-mono text-python-muted whitespace-pre-wrap leading-relaxed">
                                    {currentQuestion.code}
                                </pre>
                            </div>
                        )}

                        <div className="grid gap-3">
                            {currentQuestion.options.map((opt) => {
                                const isSelected = selectedOption === opt.key;
                                const isCorrect = isAnswered && opt.key === currentQuestion.answer;
                                const isWrong = isAnswered && isSelected && opt.key !== currentQuestion.answer;

                                return (
                                    <button
                                        key={opt.key}
                                        onClick={() => handleOptionSelect(opt.key)}
                                        disabled={isAnswered}
                                        className={`
                      w-full flex items-center text-left p-4 rounded-2xl border-2 transition-all group
                      ${isSelected ? 'border-python bg-python/5' : 'border-border bg-muted/5 hover:border-python/30'}
                      ${isCorrect ? 'border-success bg-success/10 text-success' : ''}
                      ${isWrong ? 'border-destructive bg-destructive/10 text-destructive' : ''}
                      ${isAnswered && !isSelected && !isCorrect ? 'opacity-40' : ''}
                    `}
                                    >
                                        <div className={`
                      w-8 h-8 rounded-xl flex items-center justify-center font-bold text-sm mr-4 transition-colors
                      ${isSelected ? 'bg-python text-python-foreground' : 'bg-muted text-muted-foreground'}
                      ${isCorrect ? 'bg-success text-white' : ''}
                      ${isWrong ? 'bg-destructive text-white' : ''}
                    `}>
                                            {opt.key.toUpperCase()}
                                        </div>
                                        <span className="flex-1 font-medium">{opt.text}</span>
                                        {isCorrect && <CheckCircle2 className="w-5 h-5 text-success ml-2" />}
                                        {isWrong && <XCircle className="w-5 h-5 text-destructive ml-2" />}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Correct/Incorrect Feedback & Explanation */}
                    <AnimatePresence>
                        {isAnswered && (
                            <motion.div
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`p-6 rounded-3xl border-2 ${selectedOption === currentQuestion.answer ? 'bg-success/5 border-success/20' : 'bg-destructive/5 border-destructive/20'}`}
                            >
                                <div className="flex items-center gap-3 mb-3">
                                    {selectedOption === currentQuestion.answer ? (
                                        <div className="p-2 rounded-full bg-success/20 text-success">
                                            <CheckCircle2 className="w-5 h-5" />
                                        </div>
                                    ) : (
                                        <div className="p-2 rounded-full bg-destructive/20 text-destructive">
                                            <XCircle className="w-5 h-5" />
                                        </div>
                                    )}
                                    <h4 className={`font-bold ${selectedOption === currentQuestion.answer ? 'text-success' : 'text-destructive'}`}>
                                        {selectedOption === currentQuestion.answer ? "Excellent! That's correct." : "Not quite right!"}
                                    </h4>
                                </div>
                                <div className="pl-12">
                                    <p className="text-sm leading-relaxed text-muted-foreground italic mb-1 uppercase tracking-widest text-[10px] opacity-60">The logic behind this:</p>
                                    <p className="text-sm/relaxed font-medium">
                                        {currentQuestion.explanation}
                                    </p>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Actions */}
                    <div className="flex justify-between items-center bg-card p-4 rounded-3xl border border-border shadow-sm">
                        <div className="flex items-center gap-2">
                            <Button variant="ghost" size="icon" onClick={() => setCurrentIndex(c => Math.max(0, c - 1))} disabled={currentIndex === 0 || isAnswered} className="rounded-xl">
                                <ChevronLeft className="w-5 h-5" />
                            </Button>
                            <Button variant="ghost" size="icon" onClick={() => setCurrentIndex(c => Math.min(questions.length - 1, c + 1))} disabled={currentIndex === questions.length - 1 || isAnswered} className="rounded-xl">
                                <ChevronRight className="w-5 h-5" />
                            </Button>
                        </div>

                        {!isAnswered ? (
                            <Button
                                onClick={handleCheck}
                                disabled={!selectedOption}
                                className="bg-python hover:bg-python/90 text-python-foreground font-bold px-8 rounded-xl h-12"
                            >
                                Confirm Answer
                            </Button>
                        ) : (
                            <Button
                                onClick={handleNext}
                                className="bg-python hover:bg-python/90 text-python-foreground font-bold px-8 rounded-xl h-12"
                            >
                                {currentIndex < questions.length - 1 ? "Next Question" : "View Results"}
                                <ArrowRight className="w-4 h-4 ml-2" />
                            </Button>
                        )}
                    </div>
                </motion.div>
            </AnimatePresence>
        </div>
    );
};
