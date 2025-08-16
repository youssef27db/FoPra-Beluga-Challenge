"""
C++ MCTS integration module for the Beluga Challenge.
Provides a drop-in replacement for the Python MCTS implementation.
"""

import sys
import os
import warnings

# Add the mcts_fast directory to the path
mcts_fast_dir = os.path.join(os.path.dirname(__file__), 'mcts_fast')
sys.path.insert(0, mcts_fast_dir)

try:
    import mcts_fast
    from python_interface import run_cpp_mcts, create_cpp_mcts_node
    CPP_MCTS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"C++ MCTS not available: {e}. Falling back to Python implementation.")
    CPP_MCTS_AVAILABLE = False


class MCTSNode:
    """
    Drop-in replacement for the Python MCTSNode class using C++ implementation.
    Maintains the same API as the original Python version.
    """
    
    def __init__(self, state, parent=None, action=None, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self._cpp_node = None
        
        if CPP_MCTS_AVAILABLE:
            try:
                # Convert action to the format expected by C++
                cpp_action = None
                if action is not None:
                    cpp_action = (action[0], action[1]) if isinstance(action, tuple) else action
                
                self._cpp_node = create_cpp_mcts_node(state, cpp_action)
            except Exception as e:
                warnings.warn(f"Failed to create C++ node: {e}. Using Python fallback.")
                self._cpp_node = None
    
    def is_terminal(self):
        """Check if this node represents a terminal state."""
        return self.state.is_terminal()
    
    def is_root(self):
        """Check if this node is the root of the tree."""
        return self.parent is None
    
    def copy(self):
        """Create a copy of this node."""
        return MCTSNode(self.state.copy(), self.parent, self.action, self.depth)


class MCTS:
    """
    Drop-in replacement for the Python MCTS class using C++ implementation.
    Maintains the same API as the original Python version.
    """
    
    def __init__(self, root, depth=5, n_simulations=300, debug=False):
        self.root = root
        self.depth = depth
        self.n_simulations = n_simulations
        self.debug = debug
        self.use_cpp = CPP_MCTS_AVAILABLE and hasattr(root, '_cpp_node') and root._cpp_node is not None
        
        if self.debug:
            impl_type = "C++" if self.use_cpp else "Python"
            print(f"[MCTS] Using {impl_type} implementation")
    
    def search(self):
        """
        Run the MCTS search algorithm.
        Returns the best child node found after all simulations.
        """
        if self.use_cpp and CPP_MCTS_AVAILABLE:
            return self._search_cpp()
        else:
            return self._search_python_fallback()
    
    def _search_cpp(self):
        """Run MCTS search using C++ implementation."""
        try:
            # Run C++ MCTS search
            result = run_cpp_mcts(
                self.root.state, 
                self.root.action,
                depth=self.depth,
                n_simulations=self.n_simulations,
                debug=self.debug
            )
            
            if result is None:
                return None
            
            action_name, params = result
            
            # Create a mock node with the best action for compatibility
            best_node = MCTSNode(
                state=self.root.state,
                parent=self.root,
                action=(action_name, params),
                depth=self.root.depth + 1
            )
            
            return best_node
            
        except Exception as e:
            if self.debug:
                print(f"[MCTS] C++ search failed: {e}. Falling back to Python.")
            return self._search_python_fallback()
    
    def _search_python_fallback(self):
        """Fallback to Python implementation if C++ fails."""
        # Import the original Python MCTS implementation
        from .mcts import MCTS as PythonMCTS
        from .mcts_node import MCTSNode as PythonMCTSNode
        
        # Create Python MCTS node
        python_root = PythonMCTSNode(
            state=self.root.state,
            parent=None,
            action=self.root.action,
            depth=self.root.depth
        )
        
        # Run Python MCTS
        python_mcts = PythonMCTS(
            root=python_root,
            depth=self.depth,
            n_simulations=self.n_simulations,
            debug=self.debug
        )
        
        python_best = python_mcts.search()
        
        if python_best is None:
            return None
        
        # Convert back to our wrapper format
        best_node = MCTSNode(
            state=python_best.state,
            parent=self.root,
            action=python_best.action,
            depth=python_best.depth
        )
        
        return best_node
    
    def get_best_path(self):
        """Get the path of best actions from the root."""
        # This method is not commonly used in the trainer, so we'll just return empty list
        return []


def get_mcts_implementation_info():
    """Get information about the available MCTS implementations."""
    info = {
        'cpp_available': CPP_MCTS_AVAILABLE,
        'default_implementation': 'C++' if CPP_MCTS_AVAILABLE else 'Python',
        'fallback_available': True
    }
    
    if CPP_MCTS_AVAILABLE:
        try:
            # Test if we can create a simple C++ object
            jig_type = mcts_fast.createJigType("typeA")
            info['cpp_functional'] = True
        except Exception as e:
            info['cpp_functional'] = False
            info['cpp_error'] = str(e)
    else:
        info['cpp_functional'] = False
    
    return info


# Export the classes for drop-in replacement
__all__ = ['MCTS', 'MCTSNode', 'get_mcts_implementation_info']