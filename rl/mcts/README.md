# MCTS Implementation for Beluga Challenge

This module provides both Python and C++ implementations of Monte Carlo Tree Search (MCTS) for the Beluga Challenge container optimization problem.

## Features

- **High-Performance C++ Implementation**: Significantly faster than Python for intensive MCTS searches
- **Automatic Fallback**: Seamlessly falls back to Python implementation if C++ is unavailable
- **Drop-in Replacement**: No code changes required in existing trainer or evaluation scripts
- **Compatible API**: Maintains the same interface as the original Python implementation

## Installation

### Building the C++ Extension

```bash
cd rl/mcts/mcts_fast
python setup.py build_ext --inplace
```

Requirements:
- Python 3.7+
- pybind11 (`pip install pybind11`)
- C++17 compatible compiler

## Usage

### Automatic Selection (Recommended)

The system automatically uses the C++ implementation if available:

```python
from rl.mcts import MCTS, MCTSNode

# Will use C++ implementation if available, Python otherwise
root = MCTSNode(state=problem_state, action=("left_stack_rack", None))
mcts = MCTS(root, depth=5, n_simulations=300)
best_node = mcts.search()
```

### Force Python Implementation

To explicitly use the Python implementation:

```bash
export USE_CPP_MCTS=false
python -m rl.main --mode train
```

Or in Python:

```python
import os
os.environ['USE_CPP_MCTS'] = 'false'

from rl.mcts import MCTS, MCTSNode
# Will use Python implementation
```

### Check Implementation Status

```python
from rl.mcts import get_mcts_implementation_info

info = get_mcts_implementation_info()
print(f"Using: {info['default_implementation']}")
print(f"C++ available: {info['cpp_available']}")
```

## Performance

The C++ implementation provides significant speedup:

- **2-10x faster** for typical MCTS searches
- **Lower memory usage** for large state spaces
- **Better scaling** with increased simulation counts

## Integration with Trainer

No changes required to existing code. The trainer automatically uses the faster implementation:

```python
# In trainer.py - this automatically uses C++ MCTS if available
root = MCTSNode(state=self.env.state, action=(high_level_action_str, None))
mcts = MCTS(root, depth=5, n_simulations=60)
best_node = mcts.search()
```

## Troubleshooting

### C++ Compilation Issues

1. **Missing pybind11**: Install with `pip install pybind11`
2. **Compiler errors**: Ensure you have a C++17 compatible compiler
3. **macOS**: May need Xcode command line tools: `xcode-select --install`

### Runtime Issues

If you encounter issues with the C++ implementation:

1. **Force Python fallback**: Set `USE_CPP_MCTS=false`
2. **Check compilation**: Run `python test_cpp_mcts.py` in the `mcts_fast` directory
3. **Debug mode**: Enable debug output with `debug=True` in MCTS constructor

### Verification

Run the integration test to verify everything works:

```bash
python test_integration.py
```

This tests:
- C++ implementation functionality
- Automatic fallback to Python
- Trainer compatibility
- Real problem state handling

## API Reference

### MCTS Class

```python
class MCTS:
    def __init__(self, root, depth=5, n_simulations=300, debug=False):
        """
        Initialize MCTS search.
        
        Args:
            root: MCTSNode root of the search tree
            depth: Maximum search depth
            n_simulations: Number of simulations to run
            debug: Enable debug output
        """
    
    def search(self):
        """
        Run MCTS search algorithm.
        
        Returns:
            MCTSNode: Best child node found, or None if no solution
        """
```

### MCTSNode Class

```python
class MCTSNode:
    def __init__(self, state, parent=None, action=None, depth=0):
        """
        Initialize MCTS node.
        
        Args:
            state: ProblemState for this node
            parent: Parent MCTSNode (None for root)
            action: Action taken to reach this state
            depth: Depth in the search tree
        """
```

## Contributing

When modifying the MCTS implementation:

1. **Update both implementations**: Ensure Python and C++ versions stay in sync
2. **Test thoroughly**: Run both `test_cpp_mcts.py` and `test_integration.py`
3. **Maintain API compatibility**: Don't break existing trainer code
4. **Document changes**: Update this README with any new features

## License

Same license as the main Beluga Challenge project.