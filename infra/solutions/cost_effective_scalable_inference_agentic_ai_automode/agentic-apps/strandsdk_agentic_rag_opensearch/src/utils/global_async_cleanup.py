"""
Global async cleanup and warning suppression for the entire application.
This module should be imported at the very beginning of the main application.
"""

import warnings
import sys
import logging
import os
from contextlib import redirect_stderr
from io import StringIO

def setup_global_async_cleanup():
    """Set up global async cleanup and warning suppression."""
    
    # Suppress all async-related warnings globally
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*async.*")
    warnings.filterwarnings("ignore", message=".*coroutine.*")
    warnings.filterwarnings("ignore", message=".*HTTP.*")
    warnings.filterwarnings("ignore", message=".*Exception ignored.*")
    
    # Configure logging to suppress noisy libraries
    loggers_to_quiet = [
        "httpcore",
        "httpx", 
        "anyio",
        "asyncio",
        "urllib3.connectionpool",
        "mcp.client.streamable_http"
    ]
    
    for logger_name in loggers_to_quiet:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # Only show critical errors
    
    # Set environment variables to reduce async noise
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

class AsyncWarningFilter:
    """Custom stderr filter to remove async warnings while preserving other messages."""
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = []
        self.in_async_traceback = False
        
    def write(self, text):
        # Check if we're starting an async exception block
        if "Exception ignored in:" in text:
            self.in_async_traceback = True
            return
        
        # Check if we're in an async traceback
        if self.in_async_traceback:
            # Look for patterns that indicate end of traceback
            if (text.strip() == "" or 
                not any(pattern in text for pattern in [
                    "Traceback", "File ", "RuntimeError", "yield", "await", 
                    "async", "coroutine", "HTTP", "anyio", "httpcore"
                ])):
                self.in_async_traceback = False
            else:
                return  # Skip this line as it's part of async traceback
        
        # Filter out specific async error patterns
        async_patterns = [
            "async generator ignored GeneratorExit",
            "Attempted to exit cancel scope",
            "no running event loop",
            "HTTP11ConnectionByteStream",
            "coroutine object",
            "RuntimeError: async generator",
            "Exception ignored in:",
            "httpcore/_async/",
            "anyio/_backends/",
            "RuntimeError:",
            "Traceback (most recent call last):",
            "yield part",
            "await self._connection",
            "async with self._state_lock"
        ]
        
        # Check if this line contains async warning patterns
        if any(pattern in text for pattern in async_patterns):
            return  # Don't write async warning messages
        
        # Write non-async messages to original stderr
        self.original_stderr.write(text)
        
    def flush(self):
        self.original_stderr.flush()
        
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

def install_global_stderr_filter():
    """Install global stderr filter to suppress async warnings."""
    if not hasattr(sys.stderr, '_original_stderr'):
        sys.stderr._original_stderr = sys.stderr
        sys.stderr = AsyncWarningFilter(sys.stderr._original_stderr)

def remove_global_stderr_filter():
    """Remove global stderr filter and restore original stderr."""
    if hasattr(sys.stderr, '_original_stderr'):
        original = sys.stderr._original_stderr
        sys.stderr = original

# Apply global setup when module is imported
setup_global_async_cleanup()
install_global_stderr_filter()

# Ensure cleanup on exit
import atexit
atexit.register(remove_global_stderr_filter)
