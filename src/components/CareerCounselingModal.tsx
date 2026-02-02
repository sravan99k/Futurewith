
import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ChevronLeft, CheckCircle2, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { counselingQuestions, learningPaths, type LearningPath } from '@/data/courses/counseling';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
    DialogTrigger,
} from "@/components/ui/dialog";

interface CareerCounselingModalProps {
    children: React.ReactNode;
}

export const CareerCounselingModal = ({ children }: CareerCounselingModalProps) => {
    const navigate = useNavigate();
    const [currentStep, setCurrentStep] = useState(0);
    const [answers, setAnswers] = useState<Record<string, string>>({});
    const [recommendedPath, setRecommendedPath] = useState<LearningPath | null>(null);
    const [showResults, setShowResults] = useState(false);
    const [isOpen, setIsOpen] = useState(false);

    const currentQuestion = counselingQuestions[currentStep];

    const handleAnswer = useCallback((questionId: string, value: string) => {
        setAnswers(prev => ({ ...prev, [questionId]: value }));
    }, []);

    const handleNext = useCallback(() => {
        if (currentStep < counselingQuestions.length - 1) {
            setCurrentStep(prev => prev + 1);
        } else {
            // Calculate recommended path
            const answerValues = Object.values(answers);
            let bestPath = learningPaths[0];
            let maxScore = 0;

            learningPaths.forEach(path => {
                const score = path.bestFor.filter(bf => answerValues.includes(bf)).length;
                if (score > maxScore) {
                    maxScore = score;
                    bestPath = path;
                }
            });

            setRecommendedPath(bestPath);
            setShowResults(true);
        }
    }, [currentStep, answers]);

    const handleBack = useCallback(() => {
        if (currentStep > 0) {
            setCurrentStep(prev => prev - 1);
        }
    }, [currentStep]);

    const handleReset = useCallback(() => {
        setCurrentStep(0);
        setAnswers({});
        setRecommendedPath(null);
        setShowResults(false);
    }, []);

    const handleStartPath = useCallback(() => {
        setIsOpen(false); // Close dialog
        if (recommendedPath) {
            if (recommendedPath.track === 'vibe') {
                navigate('/vibe-coding');
            } else {
                navigate('/python-course');
            }
        }
    }, [recommendedPath, navigate]);

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
                {children}
            </DialogTrigger>
            <DialogContent className="max-w-2xl bg-card border-border p-0 overflow-hidden sm:rounded-2xl">
                <DialogHeader className="px-6 pt-6 pb-2">
                    <DialogTitle className="text-2xl font-heading font-bold text-center">
                        Find Your <span className="text-python">Learning Path</span>
                    </DialogTitle>
                    <DialogDescription className="text-center font-medium">
                        Answer a few questions to get a personalized recommendation
                    </DialogDescription>
                </DialogHeader>

                <div className="p-6 pt-2">
                    <AnimatePresence mode="wait">
                        {!showResults ? (
                            <motion.div
                                key="questions"
                                initial={{ opacity: 0, x: 20 }}
                                animate={{ opacity: 1, x: 0 }}
                                exit={{ opacity: 0, x: -20 }}
                                className="w-full"
                            >
                                {/* Progress */}
                                <div className="flex items-center gap-2 mb-6">
                                    {counselingQuestions.map((_, index) => (
                                        <div
                                            key={index}
                                            className={`flex-1 h-2 rounded-full transition-colors ${index <= currentStep ? 'bg-python' : 'bg-muted'
                                                }`}
                                        />
                                    ))}
                                </div>

                                {/* Question */}
                                <div className="mb-6">
                                    <span className="text-sm text-muted-foreground mb-1 block">
                                        Question {currentStep + 1} of {counselingQuestions.length}
                                    </span>
                                    <h3 className="font-heading text-xl font-semibold">
                                        {currentQuestion.question}
                                    </h3>
                                </div>

                                {/* Options */}
                                <div className="space-y-3 mb-6">
                                    {currentQuestion.options.map((option) => (
                                        <button
                                            key={option.id}
                                            onClick={() => handleAnswer(currentQuestion.id, option.value)}
                                            className={`w-full text-left p-3.5 rounded-xl border-2 smooth-transition ${answers[currentQuestion.id] === option.value
                                                ? 'border-python bg-python/5'
                                                : 'border-border hover:border-python/30'
                                                }`}
                                        >
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <div className="font-medium mb-0.5">{option.label}</div>
                                                    <div className="text-sm text-muted-foreground">{option.description}</div>
                                                </div>
                                                {answers[currentQuestion.id] === option.value && (
                                                    <CheckCircle2 className="w-5 h-5 text-python flex-shrink-0" />
                                                )}
                                            </div>
                                        </button>
                                    ))}
                                </div>

                                {/* Navigation */}
                                <div className="flex items-center justify-between mt-auto">
                                    <Button
                                        variant="ghost"
                                        onClick={handleBack}
                                        disabled={currentStep === 0}
                                    >
                                        <ChevronLeft className="w-4 h-4 mr-1" />
                                        Back
                                    </Button>
                                    <Button
                                        onClick={handleNext}
                                        disabled={!answers[currentQuestion.id]}
                                        className="bg-python hover:bg-python/90 text-python-foreground"
                                    >
                                        {currentStep === counselingQuestions.length - 1 ? 'Get Recommendation' : 'Next'}
                                        <ChevronRight className="w-4 h-4 ml-1" />
                                    </Button>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="results"
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="text-center py-4"
                            >
                                <div className="w-16 h-16 rounded-full bg-success/20 flex items-center justify-center mx-auto mb-4">
                                    <CheckCircle2 className="w-8 h-8 text-success" />
                                </div>

                                <h3 className="font-heading text-2xl font-bold mb-2">
                                    Your Recommended Path
                                </h3>
                                <p className="text-muted-foreground mb-6">
                                    Based on your goals and availability
                                </p>

                                {recommendedPath && (
                                    <div className="bg-python/5 border border-python/20 rounded-xl p-6 mb-8 text-left">
                                        <h4 className="font-heading text-xl font-bold text-python mb-2">
                                            {recommendedPath.title}
                                        </h4>
                                        <p className="text-muted-foreground mb-4">{recommendedPath.description}</p>
                                        <div className="flex flex-wrap items-center gap-3 text-sm">
                                            <span className="px-3 py-1 rounded-full bg-python/10 text-python">
                                                {recommendedPath.duration}
                                            </span>
                                            <span className="px-3 py-1 rounded-full bg-muted">
                                                {recommendedPath.phases.length} Phases
                                            </span>
                                            <span className="px-3 py-1 rounded-full bg-muted capitalize">
                                                {recommendedPath.track === 'both' ? 'Full Stack AI' : recommendedPath.track} Track
                                            </span>
                                        </div>
                                    </div>
                                )}

                                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                                    <Button variant="outline" onClick={handleReset}>
                                        Retake Quiz
                                    </Button>
                                    <Button
                                        className="bg-python hover:bg-python/90 text-python-foreground w-full sm:w-auto"
                                        onClick={handleStartPath}
                                    >
                                        Start Learning
                                        <ArrowRight className="ml-2 w-4 h-4" />
                                    </Button>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </DialogContent>
        </Dialog>
    );
};
