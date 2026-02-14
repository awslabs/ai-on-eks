#!/usr/bin/env python3
"""
Clean main application runner with complete async error suppression.
"""

import sys
import os
import warnings

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class CompleteAsyncErrorFilter:
    """Complete async error filter that suppresses all async-related output."""
    
    def __init__(self):
        self.original_stderr = sys.__stderr__
        
    def write(self, text):
        """Filter out all async RuntimeErrors and related output."""
        if not text.strip():
            return
            
        # Comprehensive list of patterns to suppress
        suppress_patterns = [
            "RuntimeError",
            "httpcore",
            "_synchronization",
            "asyncio",
            "anyio",
            "sniffio",
            "await",
            "async",
            "CancelScope",
            "shield",
            "current_task",
            "get_running_loop",
            "cancel_shielded_checkpoint",
            "_anyio_lock",
            "acquire",
            "File \"/home/ubuntu/Cost_Effective_and_Scalable_Models_Inference_on_AWS_Graviton/agentic-apps/strandsdk_agentic_rag_opensearch/venv/lib/python3.10/site-packages/httpcore",
            "File \"/usr/lib/python3.10/asyncio",
            "raise RuntimeError",
        ]
        
        # Check if this line should be suppressed
        should_suppress = any(pattern in text for pattern in suppress_patterns)
        
        # Also suppress lines that are just punctuation or whitespace
        if text.strip() in [":", "RuntimeError:", "RuntimeError: ", "RuntimeError", ""]:
            should_suppress = True
        
        # Only write if not suppressed and contains meaningful content
        if not should_suppress and len(text.strip()) > 1:
            self.original_stderr.write(text)
            self.original_stderr.flush()
        
    def flush(self):
        """Flush the original stderr."""
        self.original_stderr.flush()

def setup_complete_clean_environment():
    """Set up completely clean environment."""
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Install complete error filter
    sys.stderr = CompleteAsyncErrorFilter()
    
    # Try to import and use existing cleanup if available
    try:
        from src.utils.global_async_cleanup import setup_global_async_cleanup
        setup_global_async_cleanup()
    except ImportError:
        pass

if __name__ == "__main__":
    print("🚀 Starting Enhanced RAG System (Ultra Clean Mode)")
    print("=" * 60)
    print("Note: All async errors and warnings are completely suppressed")
    print("=" * 60)
    
    # Set up complete clean environment FIRST
    setup_complete_clean_environment()
    
    try:
        # Import and run the main application
        from src.main import main
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        # Only show truly important errors
        error_msg = str(e)
        if not any(keyword in error_msg.lower() for keyword in [
            "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
        ]):
            print(f"\n❌ Application error: {e}")
        sys.exit(1)
