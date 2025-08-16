"""
Python interface utilities for the C++ MCTS implementation.
Provides conversion functions between Python and C++ state representations.
"""

from typing import List, Dict, Any, Optional, Tuple
import mcts_fast


def convert_python_state_to_cpp(python_state) -> mcts_fast.ProblemState:
    """
    Convert a Python ProblemState to C++ ProblemState.
    
    Args:
        python_state: Python ProblemState object from rl.env.state
        
    Returns:
        mcts_fast.ProblemState: C++ equivalent
    """
    # Convert jigs
    cpp_jigs = []
    for jig in python_state.jigs:
        jig_type = mcts_fast.createJigType(jig.jig_type.name)
        cpp_jig = mcts_fast.Jig(jig_type, jig.empty)
        cpp_jigs.append(cpp_jig)
    
    # Convert belugas
    cpp_belugas = []
    for beluga in python_state.belugas:
        cpp_outgoing_types = []
        for jig_type in beluga.outgoing:
            cpp_outgoing_types.append(mcts_fast.createJigType(jig_type.name))
        cpp_beluga = mcts_fast.Beluga(beluga.current_jigs[:], cpp_outgoing_types)
        cpp_belugas.append(cpp_beluga)
    
    # Convert trailers (handle None values)
    def convert_trailer_list(trailer_list):
        result = []
        for trailer in trailer_list:
            if trailer is None:
                result.append(None)
            else:
                result.append(int(trailer))
        return result
    
    cpp_trailers_beluga = convert_trailer_list(python_state.trailers_beluga)
    cpp_trailers_factory = convert_trailer_list(python_state.trailers_factory)
    
    # Convert racks
    cpp_racks = []
    for rack in python_state.racks:
        cpp_rack = mcts_fast.Rack(rack.size, rack.current_jigs[:])
        cpp_racks.append(cpp_rack)
    
    # Convert production lines
    cpp_production_lines = []
    for pl in python_state.production_lines:
        cpp_pl = mcts_fast.ProductionLine(pl.scheduled_jigs[:])
        cpp_production_lines.append(cpp_pl)
    
    # Convert hangars
    cpp_hangars = convert_trailer_list(python_state.hangars)
    
    return mcts_fast.ProblemState(
        cpp_jigs, cpp_belugas, cpp_trailers_beluga, cpp_trailers_factory,
        cpp_racks, cpp_production_lines, cpp_hangars
    )


def create_cpp_mcts_node(python_state, action: Optional[Tuple[str, Any]] = None) -> mcts_fast.MCTSNode:
    """
    Create a C++ MCTSNode from a Python state.
    
    Args:
        python_state: Python ProblemState object
        action: Optional action tuple (action_name, params)
        
    Returns:
        mcts_fast.MCTSNode: Root node for MCTS search
    """
    cpp_state = convert_python_state_to_cpp(python_state)
    
    # Convert action if provided
    cpp_action = None
    if action is not None:
        action_name, params = action
        if params is None:
            cpp_params = []
        elif isinstance(params, dict):
            cpp_params = list(params.values())
        else:
            cpp_params = list(params) if hasattr(params, '__iter__') else [params]
        cpp_action = (action_name, cpp_params)
    
    return mcts_fast.MCTSNode(cpp_state, None, cpp_action, 0)


def run_cpp_mcts(python_state, action: Optional[Tuple[str, Any]] = None, 
                 depth: int = 5, n_simulations: int = 300, debug: bool = False) -> Optional[Tuple[str, Any]]:
    """
    Run C++ MCTS on a Python state and return the best action.
    
    Args:
        python_state: Python ProblemState object
        action: Optional action tuple for root node
        depth: MCTS search depth
        n_simulations: Number of MCTS simulations
        debug: Enable debug output
        
    Returns:
        Optional[Tuple[str, Any]]: Best action as (action_name, params) or None
    """
    try:
        # Create root node
        root_node = create_cpp_mcts_node(python_state, action)
        
        # Create MCTS instance
        mcts = mcts_fast.MCTS(root_node, depth, n_simulations, debug)
        
        # Run search
        best_node = mcts.search()
        
        if best_node is None:
            return None
        
        # Get action from best node
        action_opt = best_node.getAction()
        if action_opt is None:
            return None
        
        action_name, cpp_params = action_opt
        
        # Convert parameters back to Python format
        if not cpp_params:
            params = {}
        else:
            # Convert based on action type
            if action_name in ["left_stack_rack", "right_stack_rack", "left_unstack_rack", "right_unstack_rack"]:
                if len(cpp_params) >= 2:
                    params = {"rack": cpp_params[0], "trailer": cpp_params[1]}
                else:
                    params = {"rack": cpp_params[0]} if cpp_params else {}
            elif action_name == "load_beluga":
                params = {"trailer": cpp_params[0]} if cpp_params else {}
            elif action_name in ["get_from_hangar", "deliver_to_hangar"]:
                if len(cpp_params) >= 2:
                    params = {"hangar": cpp_params[0], "trailer": cpp_params[1]}
                else:
                    params = {"hangar": cpp_params[0]} if cpp_params else {}
            else:
                params = tuple(cpp_params) if cpp_params else {}
        
        return (action_name, params)
        
    except Exception as e:
        print(f"Error in C++ MCTS: {e}")
        return None


# For backward compatibility with existing code
def create_mcts_node_from_state(python_state, action=None):
    """Backward compatibility function."""
    return create_cpp_mcts_node(python_state, action)

def run_mcts_search(python_state, action=None, depth=5, n_simulations=300, debug=False):
    """Backward compatibility function."""
    return run_cpp_mcts(python_state, action, depth, n_simulations, debug)