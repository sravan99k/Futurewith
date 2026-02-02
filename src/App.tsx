import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import LandingPage from "./pages/LandingPage";
import PythonLogicPage from "./pages/PythonLogicPage";
import VibeCodingPage from "./pages/VibeCodingPage";
import PhasePage from "./pages/PhasePage";
import InterviewSimulatorPage from "./pages/InterviewSimulatorPage";
import NotFound from "./pages/NotFound";

import ScrollToTop from "@/components/ScrollToTop";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ScrollToTop />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/python-course" element={<PythonLogicPage />} />
          <Route path="/vibe-coding" element={<VibeCodingPage />} />
          <Route path="/phase/:phaseId" element={<PhasePage />} />
          <Route path="/interview-simulator" element={<InterviewSimulatorPage />} />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
