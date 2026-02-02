import Header from "@/components/Header";
import { InterviewSimulator } from "@/components/InterviewSimulator";
import { FloatingPassport } from "@/components/FloatingPassport";

const InterviewSimulatorPage = () => {
    return (
        <div className="min-h-screen bg-background">
            <Header />
            <main className="container mx-auto px-4 pt-32 pb-20">
                <InterviewSimulator markdownPath="/src/data/courses/phase-6-interview-skills/step2-coding_interview_patterns/01_coding_interview_patterns_theory.md" />
            </main>
            <FloatingPassport />
        </div>
    );
};

export default InterviewSimulatorPage;
