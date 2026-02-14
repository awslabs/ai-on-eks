"""
Utility functions for handling async cleanup and suppressing warnings.
"""

import warnings
import sys
import logging
from contextlib import contextmanager

# Configure logging to suppress specific async warnings
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)

@contextmanager
def suppress_async_warnings():
    """Context manager to suppress async-related warnings during RAGAs evaluation."""
    
    # Store original warning filters
    original_filters = warnings.filters[:]
    
    # Store original stderr
    original_stderr = sys.stderr
    
    try:
        # Suppress specific async warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*async generator ignored GeneratorExit.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Attempted to exit cancel scope.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*no running event loop.*")
        
        # Suppress HTTP connection warnings
        warnings.filterwarnings("ignore", message=".*HTTP11ConnectionByteStream.*")
        warnings.filterwarnings("ignore", message=".*HTTP11Connection.*")
        
        # Create a custom stderr that filters out specific error messages
        class FilteredStderr:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
                
            def write(self, text):
                # Filter out specific async error messages
                if any(phrase in text for phrase in [
                    "async generator ignored GeneratorExit",
                    "Attempted to exit cancel scope",
                    "no running event loop",
                    "HTTP11ConnectionByteStream",
                    "coroutine object HTTP11ConnectionByteStream.aclose"
                ]):
                    return  # Don't write these messages
                
                self.original_stderr.write(text)
                
            def flush(self):
                self.original_stderr.flush()
                
            def __getattr__(self, name):
                return getattr(self.original_stderr, name)
        
        # Replace stderr temporarily
        sys.stderr = FilteredStderr(original_stderr)
        
        yield
        
    finally:
        # Restore original settings
        warnings.filters[:] = original_filters
        sys.stderr = original_stderr

def setup_async_environment():
    """Set up the environment to minimize async warnings."""
    
    # Configure logging levels
    loggers_to_quiet = [
        "httpcore",
        "httpx", 
        "anyio",
        "asyncio",
        "urllib3.connectionpool"
    ]
    
    for logger_name in loggers_to_quiet:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        
    # Set global warning filters
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*async generator ignored GeneratorExit.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*")
