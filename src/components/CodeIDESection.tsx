import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Terminal, Copy, Check, Trash2, Loader2, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { getPyodide } from '@/services/pyodideService';

// Declare pyodide on window for TypeScript
declare global {
  interface Window {
    loadPyodide: any;
  }
}

const CodeIDESection = () => {
  const [code, setCode] = useState(`# Welcome to the Real Python IDE!
# This is running a full Python 3.12 environment in your browser.

def fibonacci(n):
    a, b = 0, 1
    result = []
    for _ in range(n):
        result.append(a)
        a, b = b, a + b
    return result

# Run it!
print("Numbers:", fibonacci(10))

# Try something advanced:
import math
print(f"PI is {math.pi}")
`);
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isPyodideLoading, setIsPyodideLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const pyodideRef = useRef<any>(null);

  // Initialize Pyodide
  useEffect(() => {
    const initPyodide = async () => {
      try {
        const pyodide = await getPyodide();
        pyodideRef.current = pyodide;
        setIsPyodideLoading(false);
      } catch (error) {
        console.error("Failed to load Pyodide:", error);
        setOutput("Error: Failed to load Python runtime. Please check your internet connection.");
        setIsPyodideLoading(false);
      }
    };
    initPyodide();
  }, []);

  const runCode = async () => {
    if (!pyodideRef.current || isRunning) return;

    setIsRunning(true);
    let consoleOutput = '';

    // Redirect stdout to our local variable
    pyodideRef.current.setStdout({
      batched: (str: string) => {
        consoleOutput += str + '\n';
      }
    });

    try {
      await pyodideRef.current.runPythonAsync(code);
      setOutput(consoleOutput || 'Code executed successfully (no output).');
    } catch (err: any) {
      setOutput(`Error:\n${err.message}`);
    } finally {
      setIsRunning(false);
    }
  };

  const clearOutput = () => setOutput('');

  const copyCode = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="w-10 h-10 rounded-lg bg-foreground flex items-center justify-center">
              <Terminal className="w-6 h-6 text-background" />
            </div>
          </div>
          <h2 className="font-heading text-3xl lg:text-4xl font-bold mb-4">
            Interactive <span className="text-python">Python Shell</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto text-lg/relaxed">
            A full Python 3.12 environment running right in your browser.
            No installs, no servers — just code.
          </p>

          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-muted-foreground bg-blue-500/5 py-2 px-4 rounded-full w-fit mx-auto border border-blue-500/10">
            <Info className="w-3 h-3 text-blue-500" />
            <span>Powered by Pyodide (WebAssembly)</span>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto relative"
        >
          {/* Loading Overlay */}
          <AnimatePresence>
            {isPyodideLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 z-50 rounded-2xl bg-black/40 backdrop-blur-md flex flex-col items-center justify-center text-white"
              >
                <Loader2 className="w-12 h-12 animate-spin mb-4 text-python" />
                <p className="font-heading font-medium text-lg">Waking up Python...</p>
                <p className="text-sm opacity-70 mt-2">Loading core runtime libraries</p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className={`ide-container relative ${isPyodideLoading ? 'blur-sm grayscale scale-[0.98]' : ''} transition-all duration-500`}>
            {/* IDE Header */}
            <div className="ide-header flex items-center justify-between px-4 py-3 bg-[#1e1e1e] border-b border-white/5 rounded-t-2xl">
              <div className="flex items-center gap-2">
                <div className="ide-dot bg-[#ff5f56]" />
                <div className="ide-dot bg-[#ffbd2e]" />
                <div className="ide-dot bg-[#27c93f]" />
                <span className="text-[#858585] text-xs ml-4 font-mono select-none">play_with_python.py</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={copyCode}
                  className="p-1.5 rounded hover:bg-white/10 text-[#858585] transition-colors"
                  title="Copy code"
                >
                  {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Code Editor */}
            <div className="bg-[#1e1e1e] p-4 min-h-[300px]">
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full min-h-[300px] bg-transparent text-[#d4d4d4] resize-none focus:outline-none font-mono text-sm leading-relaxed caret-python"
                spellCheck={false}
              />
            </div>

            {/* Toolbar */}
            <div className="px-4 py-3 bg-[#1e1e1e] border-t border-white/5 flex items-center justify-between">
              <Button
                onClick={runCode}
                disabled={isRunning || isPyodideLoading}
                className="bg-python hover:bg-python/90 text-python-foreground font-bold shadow-lg shadow-python/20 px-6"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin mr-2" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2 fill-current" />
                    Execute Code
                  </>
                )}
              </Button>

              <div className="flex items-center gap-2 text-xs text-[#858585] font-mono">
                <span className="hidden sm:inline">Python 3.12</span>
                <span className="w-1 h-1 rounded-full bg-success" />
                <span>Ready</span>
              </div>
            </div>

            {/* Output Area */}
            <AnimatePresence>
              {output && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="border-t border-white/5 bg-[#151515] p-5 rounded-b-2xl"
                >
                  <div className="flex items-center justify-between mb-3 text-xs font-mono uppercase tracking-wider text-[#858585]">
                    <span>Output Console</span>
                    <button
                      onClick={clearOutput}
                      className="hover:text-white flex items-center gap-1 transition-colors"
                    >
                      <Trash2 className="w-3 h-3" />
                      Clear
                    </button>
                  </div>
                  <pre className={`font-mono text-sm whitespace-pre-wrap leading-relaxed ${output.startsWith('Error:') ? 'text-red-400' : 'text-python-muted'}`}>
                    {output}
                  </pre>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default CodeIDESection;
