"""!
@file __init__.py
@brief MCTS package initialization

This package provides Monte Carlo Tree Search (MCTS) implementation
for the Beluga Challenge container optimization problem.
Supports both Python and C++ implementations with automatic fallback.
"""

import os
import warnings

# Check if C++ implementation should be used
USE_CPP_MCTS = os.environ.get('USE_CPP_MCTS', 'true').lower() in ('true', '1', 'yes', 'on')

if USE_CPP_MCTS:
    try:
        # Try to import C++ implementation
        from .mcts_cpp import MCTS, MCTSNode, get_mcts_implementation_info
        print("🚀 Using fast C++ MCTS implementation")
    except ImportError as e:
        warnings.warn(f"C++ MCTS not available: {e}. Using Python implementation.")
        from .mcts import MCTS
        from .mcts_node import MCTSNode
        def get_mcts_implementation_info():
            return {'cpp_available': False, 'default_implementation': 'Python', 'fallback_available': False}
else:
    # Use Python implementation
    from .mcts import MCTS
    from .mcts_node import MCTSNode
    def get_mcts_implementation_info():
        return {'cpp_available': False, 'default_implementation': 'Python (forced)', 'fallback_available': False}

# Export for convenience
__all__ = ['MCTS', 'MCTSNode', 'get_mcts_implementation_info']