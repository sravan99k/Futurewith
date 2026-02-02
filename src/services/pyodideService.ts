/**
 * Global Pyodide service to share a single Python runtime across all interactive components.
 */

let pyodideInstance: any = null;
let initializationPromise: Promise<any> | null = null;

export const getPyodide = async () => {
    if (pyodideInstance) {
        return pyodideInstance;
    }

    if (initializationPromise) {
        return initializationPromise;
    }

    initializationPromise = (async () => {
        if (typeof window === 'undefined' || !window.loadPyodide) {
            // Wait a bit if script isn't loaded yet
            await new Promise(resolve => setTimeout(resolve, 500));
            if (!window.loadPyodide) {
                throw new Error("Pyodide script not found. Ensure it's in index.html");
            }
        }

        try {
            const pyodide = await window.loadPyodide({
                indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/"
            });
            pyodideInstance = pyodide;
            return pyodide;
        } catch (err) {
            initializationPromise = null;
            throw err;
        }
    })();

    return initializationPromise;
};
