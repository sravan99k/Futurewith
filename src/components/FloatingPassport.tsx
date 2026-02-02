import React, { useState } from 'react';
import { X, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';

export const FloatingPassport = () => {
  const [isOpen, setIsOpen] = useState(false);

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-gradient-to-br from-python to-python/80 backdrop-blur-lg border-2 border-white/20 shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300 flex items-center justify-center"
        aria-label="Open course passport"
      >
        <BookOpen className="w-6 h-6 text-white" />
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-80 bg-background rounded-xl shadow-2xl border border-border overflow-hidden">
      <div className="bg-gradient-to-r from-python to-python/90 p-4 flex justify-between items-center">
        <h3 className="font-bold text-white">Course Passport</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="text-white hover:bg-white/20 p-1 rounded-full"
          aria-label="Close passport"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      
      <div className="p-4 max-h-[60vh] overflow-y-auto">
        <div className="space-y-4">
          <div className="flex items-center gap-3 p-3 bg-muted/30 rounded-lg">
            <div className="w-10 h-10 rounded-full bg-python/10 flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-python" />
            </div>
            <div>
              <p className="font-medium">Track your progress</p>
              <p className="text-sm text-muted-foreground">View completed topics and achievements</p>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Course Progress</span>
              <span className="font-medium">0%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-python w-0" style={{ width: '0%' }}></div>
            </div>
          </div>
          
          <div className="pt-2">
            <Button className="w-full" variant="outline">
              View Full Progress
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FloatingPassport;
