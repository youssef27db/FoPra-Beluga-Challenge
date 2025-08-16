"""!
@file __init__.py
@brief MCTS package initialization

This package provides Monte Carlo Tree Search (MCTS) implementation
for the Beluga Challenge container optimization problem.
Supports both Python and C++ implementations with automatic fallback.
"""

import os
import warnings

# Global variables to store implementations
_cpp_mcts = None
_python_mcts = None
_cpp_available = False

# Try to load C++ implementation
try:
    from . import mcts_cpp
    _cpp_mcts = mcts_cpp
    _cpp_available = True
    print("Using fast C++ MCTS implementation")
except ImportError as e:
    print(f"C++ MCTS not available: {e}. Using Python implementation.")

# Always load Python implementation as fallback
from . import mcts as python_mcts
from . import mcts_node as python_mcts_node
_python_mcts = python_mcts

def _should_use_cpp():
    """Check if C++ implementation should be used based on environment variable."""
    return os.environ.get('USE_CPP_MCTS', 'true').lower() in ('true', '1', 'yes', 'on')

def _get_implementation():
    """Get the appropriate MCTS implementation based on current settings."""
    if _should_use_cpp() and _cpp_available:
        return _cpp_mcts, "C++"
    else:
        return _python_mcts, "Python"

class MCTS:
    """Dynamic MCTS wrapper that selects implementation at runtime."""
    
    def __init__(self, *args, **kwargs):
        impl_module, impl_name = _get_implementation()
        self._impl = impl_module.MCTS(*args, **kwargs)
        self._impl_name = impl_name
    
    def search(self):
        return self._impl.search()
    
    def getBestPath(self):
        return self._impl.getBestPath()
    
    def getRoot(self):
        return self._impl.getRoot()
    
    def getDepth(self):
        return self._impl.getDepth()
    
    def getSimulations(self):
        return self._impl.getSimulations()
    
    def isDebug(self):
        return self._impl.isDebug()

class MCTSNode:
    """Dynamic MCTSNode wrapper that selects implementation at runtime."""
    
    def __init__(self, *args, **kwargs):
        impl_module, impl_name = _get_implementation()
        if impl_name == "C++":
            self._impl = impl_module.MCTSNode(*args, **kwargs)
        else:
            self._impl = python_mcts_node.MCTSNode(*args, **kwargs)
        self._impl_name = impl_name
    
    def __getattr__(self, name):
        # Delegate all other method calls to the implementation
        return getattr(self._impl, name)

def get_mcts_implementation_info():
    """Get information about current MCTS implementation."""
    impl_module, impl_name = _get_implementation()
    return {
        'cpp_available': _cpp_available,
        'default_implementation': impl_name,
        'fallback_available': True
    }

# Export for convenience
__all__ = ['MCTS', 'MCTSNode', 'get_mcts_implementation_info']