import mcts_fast
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import cpu_count
import re

# Import Python MCTS for comparison
from .mcts import MCTS
from .mcts_node import MCTSNode
from rl.env.state import load_from_json as py_load_from_json

def standardize_action_format(action_str):
    """
    Standardize action string format for better comparison between implementations.
    
    Args:
        action_str: The action string from any implementation
        
    Returns:
        A standardized string representation of the action
    """
    if action_str is None:
        return "None"
        
    # If it's already a memory address, return as is (we'll handle this specially in comparison)
    if action_str.startswith('<') and '0x' in action_str and '>' in action_str:
        return action_str
        
    # Remove all spaces
    action_str = action_str.replace(' ', '')
    
    # Extract action name and params if available
    match = re.match(r"([a-zA-Z_]+)(?:\(|\[)(.*?)(?:\)|\])", action_str)
    if match:
        action_name = match.group(1)
        params = match.group(2)
        return f"{action_name}[{params}]"  # Standardize to use square brackets
    
    # If no parameters found, return just the action name
    return action_str

def benchmark_comparison(problem_path, depth=50, simulations=100, num_threads=1, runs=3):
    """
    Benchmark comparing C++ (pybind) and native Python MCTS implementations.
    Runs multiple iterations for more accurate timing.
    
    Notes:
    - The C++ implementation uses parallelization where only the children of the root
      are merged, resulting in fewer total nodes in the final tree
    - The Python implementation is single-threaded and builds a full tree
    """
    cpp_times = []
    py_times = []
    cpp_nodes_list = []
    py_nodes_list = []
    cpp_sims_per_sec = []
    py_sims_per_sec = []
    cpp_best_actions = []
    py_best_actions = []
    decision_agreements = []
    cpp_values = []
    py_values = []
    cpp_visits = []
    py_visits = []
    
    print(f"\nBenchmark for: {problem_path}")
    print(f"Parameters: depth={depth}, simulations={simulations}, threads={num_threads}, runs={runs}")
    print("-" * 60)
    print(f"Note: C++ uses {num_threads} threads with root parallelization (merged subtrees)")
    print(f"      Python is single-threaded with a traditional full tree\n")
    
    for run in range(runs):
        print(f"Run {run+1}/{runs}:")
        
        # C++ MCTS via pybind
        print("Running C++ MCTS...")
        start_time = time.perf_counter()
        cpp_state = mcts_fast.ProblemState.load_from_json(problem_path)
        cpp_state.apply_action("unload_beluga", [])  # Standard action to put in same state
        cpp_root = mcts_fast.MCTSNode(cpp_state)
        cpp_mcts = mcts_fast.MCTS(cpp_root, depth=depth, n_simulations=simulations,
                                debug=False, num_threads=num_threads)
        cpp_best = cpp_mcts.search()
        cpp_time = time.perf_counter() - start_time
        cpp_nodes = cpp_mcts.count_total_nodes()
        cpp_times.append(cpp_time)
        cpp_nodes_list.append(cpp_nodes)
        cpp_sims_per_sec.append(simulations / cpp_time if cpp_time > 0 else 0)
        
        # Get best action info from C++ implementation
        cpp_best_action = None
        cpp_best_reward = None
        cpp_best_visits = None
        
        # Try different approaches to extract meaningful action info
        try:
            if hasattr(cpp_best, 'get_action_name'):
                cpp_best_action = cpp_best.get_action_name()
            elif hasattr(cpp_best, 'action_name'):
                cpp_best_action = cpp_best.action_name
            
            if hasattr(cpp_best, 'get_action_params'):
                params = cpp_best.get_action_params()
                if params:
                    cpp_best_action = f"{cpp_best_action}({params})"
            elif hasattr(cpp_best, 'action_params'):
                params = cpp_best.action_params
                if params:
                    cpp_best_action = f"{cpp_best_action}({params})"
                    
            # Extract from action property (used by mcts_fast) 
            elif hasattr(cpp_best, 'action'):
                action = cpp_best.action
                action_str = str(action)
                
                # Parse the tuple representation ('action_name', [params])
                if "(" in action_str and ")" in action_str:
                    # Remove parentheses
                    content = action_str.strip("()")
                    
                    # Find the position of the first comma after the action name
                    name_end = content.find(",")
                    if name_end != -1:
                        # Extract action name
                        action_name = content[:name_end].strip("'\"")
                        cpp_best_action = action_name
                        
                        # Extract parameters if available
                        params_str = content[name_end+1:].strip()
                        if params_str.startswith("[") and params_str.endswith("]"):
                            cpp_best_action = f"{action_name}{params_str}"
                    else:
                        cpp_best_action = content.strip("'\"")
                else:
                    cpp_best_action = action_str
            
            if hasattr(cpp_best, 'get_reward'):
                cpp_best_reward = cpp_best.get_reward()
            elif hasattr(cpp_best, 'reward'):
                cpp_best_reward = cpp_best.reward
            elif hasattr(cpp_best, 'value'):
                cpp_best_reward = cpp_best.value
            elif hasattr(cpp_best, 'total_reward') and hasattr(cpp_best, 'visits') and cpp_best.visits > 0:
                # Calculate average reward from total reward and visits
                cpp_best_reward = cpp_best.total_reward / cpp_best.visits
            
            # Debug: inspect available attributes on the node
            print("  Debug: C++ node attributes:")
            for attr_name in dir(cpp_best):
                if not attr_name.startswith("__"):  # Skip internal attributes
                    try:
                        attr_value = getattr(cpp_best, attr_name)
                        if not callable(attr_value):  # Skip methods
                            print(f"    - {attr_name}: {attr_value}")
                    except Exception as e:
                        print(f"    - {attr_name}: <error accessing>")
            print("  ---")
                
            if hasattr(cpp_best, 'get_visit_count'):
                cpp_best_visits = cpp_best.get_visit_count()
            elif hasattr(cpp_best, 'visit_count'):
                cpp_best_visits = cpp_best.visit_count
            elif hasattr(cpp_best, 'visits'):
                cpp_best_visits = cpp_best.visits
        except Exception as e:
            print(f"  Warning: Error extracting C++ action info: {e}")
            
        # Special approach for mcts_fast - get the best action directly from the MCTS object
        if cpp_best_action is None and hasattr(cpp_mcts, 'get_best_action'):
            try:
                best_action = cpp_mcts.get_best_action()
                if best_action:
                    cpp_best_action = str(best_action)
            except Exception as e:
                pass
                
        # If all attempts failed, try using __str__ or __repr__
        if cpp_best_action is None:
            try:
                # Try to get action information from child state
                if hasattr(cpp_best, 'get_state'):
                    state = cpp_best.get_state()
                    if hasattr(state, 'get_last_action'):
                        cpp_best_action = state.get_last_action()
                    elif hasattr(state, 'last_action'):
                        cpp_best_action = state.last_action
            except Exception as e:
                pass
                
        # Last resort
        if cpp_best_action is None:
            cpp_best_action = str(cpp_best)
        
        print(f"  C++ time: {cpp_time:.3f}s, nodes: {cpp_nodes}")
        print(f"  C++ simulations/sec: {simulations / cpp_time:.1f}")
        # Standardize the action format
        cpp_std_action = standardize_action_format(cpp_best_action)
        
        print(f"  C++ best action: {cpp_best_action}")
        print(f"  C++ standardized: {cpp_std_action}")
        
        # Always print reward/value information, even if None
        value_str = f"{cpp_best_reward:.4f}" if cpp_best_reward is not None else "unknown"
        print(f"  C++ value: {value_str}")
        print(f"  C++ visits: {cpp_best_visits if cpp_best_visits is not None else 'unknown'}")
        
        # Native Python MCTS
        print("Running Python MCTS...")
        start_time = time.perf_counter()
        py_state = py_load_from_json(problem_path)
        py_state.apply_action("unload_beluga", [])  # Same standard action
        py_root = MCTSNode(py_state)
        py_mcts = MCTS(py_root, depth=depth, n_simulations=simulations, debug=False)
        py_best = py_mcts.search()
        py_time = time.perf_counter() - start_time
        py_nodes = py_mcts.count_total_nodes()
        py_times.append(py_time)
        py_nodes_list.append(py_nodes)
        py_sims_per_sec.append(simulations / py_time if py_time > 0 else 0)
        
        # Get best action info from Python implementation
        py_best_action = None
        py_best_reward = None
        py_best_visits = None
        
        # Try different approaches to extract meaningful action info
        try:
            if hasattr(py_best, 'get_action_name'):
                py_best_action = py_best.get_action_name()
            elif hasattr(py_best, 'action_name'):
                py_best_action = py_best.action_name
            
            if hasattr(py_best, 'get_action_params'):
                params = py_best.get_action_params()
                if params:
                    py_best_action = f"{py_best_action}({params})"
            elif hasattr(py_best, 'action_params'):
                params = py_best.action_params
                if params:
                    py_best_action = f"{py_best_action}({params})"
                    
            if hasattr(py_best, 'get_reward'):
                py_best_reward = py_best.get_reward()
            elif hasattr(py_best, 'reward'):
                py_best_reward = py_best.reward
            elif hasattr(py_best, 'value'):
                py_best_reward = py_best.value
            elif hasattr(py_best, 'total_reward') and hasattr(py_best, 'visits') and py_best.visits > 0:
                # Calculate average reward from total reward and visits
                py_best_reward = py_best.total_reward / py_best.visits
                
            # Debug: inspect available attributes on the node
            print("  Debug: Python node attributes:")
            for attr_name in dir(py_best):
                if not attr_name.startswith("__"):  # Skip internal attributes
                    try:
                        attr_value = getattr(py_best, attr_name)
                        if not callable(attr_value):  # Skip methods
                            print(f"    - {attr_name}: {attr_value}")
                    except Exception as e:
                        print(f"    - {attr_name}: <error accessing>")
            print("  ---")
                
            if hasattr(py_best, 'get_visit_count'):
                py_best_visits = py_best.get_visit_count()
            elif hasattr(py_best, 'visit_count'):
                py_best_visits = py_best.visit_count
            elif hasattr(py_best, 'visits'):
                py_best_visits = py_best.visits
        except Exception as e:
            print(f"  Warning: Error extracting Python action info: {e}")
            
        # Special approach for Python MCTS - get the best action directly from the MCTS object
        if py_best_action is None and hasattr(py_mcts, 'get_best_action'):
            try:
                best_action = py_mcts.get_best_action()
                if best_action:
                    py_best_action = str(best_action)
            except Exception as e:
                pass
                
        # Try extracting action from the node directly if it's an action container
        if py_best_action is None and hasattr(py_best, 'action'):
            try:
                action = py_best.action
                if hasattr(action, 'name'):
                    py_best_action = action.name
                    if hasattr(action, 'params'):
                        py_best_action += f"({action.params})"
                else:
                    action_str = str(action)
                    
                    # Check if it's a tuple representation ('action_name', [params])
                    if "(" in action_str and ")" in action_str:
                        # Remove parentheses
                        content = action_str.strip("()")
                        
                        # Find the position of the first comma after the action name
                        name_end = content.find(",")
                        if name_end != -1:
                            # Extract action name
                            action_name = content[:name_end].strip("'\"")
                            py_best_action = action_name
                            
                            # Extract parameters if available
                            params_str = content[name_end+1:].strip()
                            if params_str.startswith("[") and params_str.endswith("]"):
                                py_best_action = f"{action_name}{params_str}"
                        else:
                            py_best_action = content.strip("'\"")
                    else:
                        py_best_action = action_str
            except Exception as e:
                print(f"  Warning: Error extracting Python action from node: {e}")
                
        # If all attempts failed, try using __str__ or __repr__
        if py_best_action is None:
            try:
                # Try to get action information from child state
                if hasattr(py_best, 'get_state'):
                    state = py_best.get_state()
                    if hasattr(state, 'get_last_action'):
                        py_best_action = state.get_last_action()
                    elif hasattr(state, 'last_action'):
                        py_best_action = state.last_action
                elif hasattr(py_best, 'state') and hasattr(py_best.state, 'last_action'):
                    py_best_action = py_best.state.last_action
            except Exception as e:
                pass
                
        # Last resort
        if py_best_action is None:
            py_best_action = str(py_best)
        
        print(f"  Python time: {py_time:.3f}s, nodes: {py_nodes}")
        print(f"  Python simulations/sec: {simulations / py_time:.1f}")
        # Standardize the action format
        py_std_action = standardize_action_format(py_best_action)
        
        print(f"  Python best action: {py_best_action}")
        print(f"  Python standardized: {py_std_action}")
        
        # Always print reward/value information, even if None
        value_str = f"{py_best_reward:.4f}" if py_best_reward is not None else "unknown"
        print(f"  Python value: {value_str}")
        print(f"  Python visits: {py_best_visits if py_best_visits is not None else 'unknown'}")
            
        # Compare decisions using standardized format for more reliable comparison
        decisions_match = False
        
        # Special case: if either is just a memory address, we can't compare directly
        if (py_std_action.startswith('<') and '0x' in py_std_action) or \
           (cpp_std_action.startswith('<') and '0x' in cpp_std_action):
            print(f"  Decision agreement: ⚠️ Cannot compare (object reference)")
        else:
            decisions_match = py_std_action == cpp_std_action
            print(f"  Decision agreement: {'✓ Same' if decisions_match else '✗ Different'}")
        
        # Calculate speedup
        speedup = py_time / cpp_time if cpp_time > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        
        # Track decision quality and value information
        cpp_best_actions.append(cpp_best_action)
        py_best_actions.append(py_best_action)
        decision_agreements.append(decisions_match)
        
        # Store value and visit information for analysis
        if cpp_best_reward is not None:
            cpp_values.append(cpp_best_reward)
        if py_best_reward is not None:
            py_values.append(py_best_reward)
        if cpp_best_visits is not None:
            cpp_visits.append(cpp_best_visits)
        if py_best_visits is not None:
            py_visits.append(py_best_visits)
            
        print()
    
    # Calculate averages
    avg_cpp_time = sum(cpp_times) / len(cpp_times)
    avg_py_time = sum(py_times) / len(py_times)
    avg_cpp_nodes = sum(cpp_nodes_list) / len(cpp_nodes_list)
    avg_py_nodes = sum(py_nodes_list) / len(py_nodes_list)
    avg_speedup = avg_py_time / avg_cpp_time if avg_cpp_time > 0 else 0
    avg_cpp_sims_per_sec = sum(cpp_sims_per_sec) / len(cpp_sims_per_sec)
    avg_py_sims_per_sec = sum(py_sims_per_sec) / len(py_sims_per_sec)
    
    # Decision quality metrics
    agreement_rate = sum(decision_agreements) / len(decision_agreements) if decision_agreements else 0
    cpp_decision_consistency = len(set(cpp_best_actions)) == 1 if cpp_best_actions else False
    py_decision_consistency = len(set(py_best_actions)) == 1 if py_best_actions else False
    
    # Calculate average values and visits if available
    avg_cpp_value = sum(cpp_values) / len(cpp_values) if cpp_values else None
    avg_py_value = sum(py_values) / len(py_values) if py_values else None
    avg_cpp_visits = sum(cpp_visits) / len(cpp_visits) if cpp_visits else None
    avg_py_visits = sum(py_visits) / len(py_visits) if py_visits else None
    
    # Calculate standard deviations
    std_cpp_time = np.std(cpp_times) if len(cpp_times) > 1 else 0
    std_py_time = np.std(py_times) if len(py_times) > 1 else 0
    std_cpp_sims = np.std(cpp_sims_per_sec) if len(cpp_sims_per_sec) > 1 else 0
    std_py_sims = np.std(py_sims_per_sec) if len(py_sims_per_sec) > 1 else 0
    
    print("\nSummary:")
    print(f"  Average C++ time: {avg_cpp_time:.3f}s ± {std_cpp_time:.3f}s")
    print(f"  Average Python time: {avg_py_time:.3f}s ± {std_py_time:.3f}s")
    print(f"  Average C++ nodes: {avg_cpp_nodes:.1f} (root parallelization merges subtrees)")
    print(f"  Average Python nodes: {avg_py_nodes:.1f} (full tree)")
    print(f"  Average C++ simulations/sec: {avg_cpp_sims_per_sec:.1f} ± {std_cpp_sims:.1f}")
    print(f"  Average Python simulations/sec: {avg_py_sims_per_sec:.1f} ± {std_py_sims:.1f}")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Simulation throughput ratio: {avg_cpp_sims_per_sec/avg_py_sims_per_sec if avg_py_sims_per_sec > 0 else 0:.2f}x")
    
    # Decision quality section
    print("\nDecision Quality:")
    print(f"  C++ decisions: {'consistent across runs' if cpp_decision_consistency else 'varied across runs'}")
    print(f"  Python decisions: {'consistent across runs' if py_decision_consistency else 'varied across runs'}")
    print(f"  Agreement rate: {agreement_rate*100:.1f}% ({sum(decision_agreements)}/{len(decision_agreements)} runs)")
    
    # If decisions are consistent, show them
    if cpp_decision_consistency:
        print(f"  C++ consistently chose: {cpp_best_actions[0]}")
    if py_decision_consistency:
        print(f"  Python consistently chose: {py_best_actions[0]}")
    
    # Show value comparison if available
    print("\nValue & Visit Comparison:")
    if avg_cpp_value is not None:
        print(f"  C++ average value: {avg_cpp_value:.4f}")
    else:
        print("  C++ average value: unknown")
        
    if avg_py_value is not None:
        print(f"  Python average value: {avg_py_value:.4f}")
    else:
        print("  Python average value: unknown")
        
    if avg_cpp_value is not None and avg_py_value is not None and avg_py_value != 0:
        value_ratio = avg_cpp_value / avg_py_value
        print(f"  Value ratio (C++/Python): {value_ratio:.4f}")
        
    # Show visit counts if available
    if avg_cpp_visits is not None:
        print(f"  C++ average visits: {avg_cpp_visits:.1f}")
    if avg_py_visits is not None:
        print(f"  Python average visits: {avg_py_visits:.1f}")
    
    # Show different decisions if there was disagreement
    if not all(decision_agreements):
        print("\nDecision differences:")
        for i, (cpp_action, py_action, agreement) in enumerate(zip(cpp_best_actions, py_best_actions, decision_agreements)):
            if not agreement:
                print(f"  Run {i+1}: C++: {cpp_action} | Python: {py_action}")
    
    return {
        'cpp_times': cpp_times,
        'py_times': py_times,
        'cpp_nodes': cpp_nodes_list,
        'py_nodes': py_nodes_list,
        'avg_cpp_time': avg_cpp_time,
        'avg_py_time': avg_py_time,
        'avg_speedup': avg_speedup,
        'std_cpp_time': std_cpp_time,
        'std_py_time': std_py_time,
        'avg_cpp_sims_per_sec': avg_cpp_sims_per_sec,
        'avg_py_sims_per_sec': avg_py_sims_per_sec,
        'std_cpp_sims': std_cpp_sims,
        'std_py_sims': std_py_sims,
        'sim_throughput_ratio': avg_cpp_sims_per_sec/avg_py_sims_per_sec if avg_py_sims_per_sec > 0 else 0,
        'cpp_best_actions': cpp_best_actions,
        'py_best_actions': py_best_actions,
        'decision_agreements': decision_agreements,
        'agreement_rate': agreement_rate,
        'cpp_decision_consistency': cpp_decision_consistency,
        'py_decision_consistency': py_decision_consistency,
        # Value and visit information
        'cpp_values': cpp_values,
        'py_values': py_values,
        'avg_cpp_value': avg_cpp_value,
        'avg_py_value': avg_py_value,
        'cpp_visits': cpp_visits,
        'py_visits': py_visits,
        'avg_cpp_visits': avg_cpp_visits,
        'avg_py_visits': avg_py_visits,
    }

def plot_results(results, problem_name):
    """Plot benchmark results comparing C++ and Python."""
    try:
        plt.figure(figsize=(14, 12))  # Made taller to accommodate additional plots
        
        # 1. Bar chart comparing average times
        plt.subplot(4, 2, 1)  # Changed from 3,2 to 4,2
        bars = plt.bar(['C++', 'Python'], 
                       [results['avg_cpp_time'], results['avg_py_time']], 
                       yerr=[results['std_cpp_time'], results['std_py_time']])
        bars[0].set_color('green')
        bars[1].set_color('blue')
        plt.ylabel('Time (seconds)')
        plt.title('Average Execution Time')
        
        # Add the values above the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.2f}s', ha='center')
        
        # 2. Individual run times
        plt.subplot(4, 2, 2)  # Changed from 3,2 to 4,2
        runs = list(range(1, len(results['cpp_times'])+1))
        plt.plot(runs, results['cpp_times'], 'o-', label='C++', color='green')
        plt.plot(runs, results['py_times'], 'o-', label='Python', color='blue')
        plt.xlabel('Run Number')
        plt.ylabel('Time (seconds)')
        plt.title('Time per Run')
        plt.legend()
        plt.grid(True)
        
        # 3. Node counts (with explanation)
        plt.subplot(4, 2, 3)  # Changed from 3,2 to 4,2
        avg_cpp_nodes = sum(results['cpp_nodes']) / len(results['cpp_nodes'])
        avg_py_nodes = sum(results['py_nodes']) / len(results['py_nodes'])
        bars = plt.bar(['C++\n(root parallelization)', 'Python\n(full tree)'], 
                       [avg_cpp_nodes, avg_py_nodes])
        bars[0].set_color('green')
        bars[1].set_color('blue')
        plt.ylabel('Nodes')
        plt.title('Average Node Count')
        
        # Add the values above the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.1f}', ha='center')
        
        # 4. Simulations per second (more fair comparison)
        plt.subplot(4, 2, 4)  # Changed from 3,2 to 4,2
        bars = plt.bar(['C++', 'Python'], 
                     [results['avg_cpp_sims_per_sec'], results['avg_py_sims_per_sec']],
                     yerr=[results['std_cpp_sims'], results['std_py_sims']])
        bars[0].set_color('green')
        bars[1].set_color('blue')
        plt.ylabel('Simulations/second')
        plt.title('Simulation Throughput')
        
        # Add the values above the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.1f}', ha='center')
        
        # 5. Speedup chart
        plt.subplot(4, 2, 5)  # Changed from 3,2 to 4,2
        plt.bar(['Time Speedup'], [results['avg_speedup']], color='red')
        plt.ylabel('Speedup Factor (x times)')
        plt.title('C++ vs Python Time Speedup')
        plt.grid(axis='y')
        
        # Add the value above the bar
        plt.text(0, results['avg_speedup'] + 0.1, 
                 f"{results['avg_speedup']:.2f}x", ha='center')
        
        # 6. Simulation throughput ratio
        plt.subplot(4, 2, 6)  # Changed from 3,2 to 4,2
        sim_ratio = results.get('sim_throughput_ratio', 
                              results['avg_cpp_sims_per_sec']/results['avg_py_sims_per_sec'] 
                              if results['avg_py_sims_per_sec'] > 0 else 0)
        plt.bar(['Simulation Throughput Ratio'], [sim_ratio], color='orange')
        plt.ylabel('Ratio (x times)')
        plt.title('C++ vs Python Simulation Throughput')
        plt.grid(axis='y')
        
        # 7. Decision Quality
        plt.subplot(4, 2, 7)
        if 'agreement_rate' in results and 'decision_agreements' in results:
            agreement_rate = results['agreement_rate']
            
            # Create a pie chart for decision agreement
            labels = ['Agreement', 'Disagreement']
            sizes = [agreement_rate, 1 - agreement_rate]
            colors = ['lightgreen', 'lightcoral']
            explode = (0.1, 0)  # explode the 1st slice (Agreement)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Decision Agreement Between C++ and Python')
        else:
            plt.text(0.5, 0.5, 'Decision quality data not available', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            
        # 8. Decision Consistency
        plt.subplot(4, 2, 8)
        if 'cpp_decision_consistency' in results and 'py_decision_consistency' in results:
            consistency_data = [
                int(results['cpp_decision_consistency']), 
                int(results['py_decision_consistency'])
            ]
            
            bars = plt.bar(['C++ Consistency', 'Python Consistency'], 
                         consistency_data,
                         color=['green', 'blue'])
            
            plt.ylim(0, 1.2)  # Set y-limit to make room for text
            plt.yticks([0, 1], ['Inconsistent', 'Consistent'])
            plt.title('Decision Consistency Across Runs')
            
            # Add labels
            for bar, val in zip(bars, consistency_data):
                text = "Consistent" if val == 1 else "Inconsistent"
                plt.text(bar.get_x() + bar.get_width()/2., 1.05, 
                         text, ha='center', fontsize=10)
        
        # Add the value above the bar
        plt.text(0, sim_ratio + 0.1, f"{sim_ratio:.2f}x", ha='center')
        
        plt.tight_layout()
        plt.suptitle(f'MCTS Benchmark: C++ vs Python\n{problem_name}', fontsize=16)
        plt.subplots_adjust(top=0.92)
        
        # Save plot
        filename = f"benchmark_{problem_name.split('/')[-1].replace('.json', '')}.png"
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
        
        try:
            plt.show()
        except Exception:
            pass
            
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Problem to benchmark
    problem_path = './problemset3(BigProblems)/problem20_j95_r9_b10_pl3.json'
    
    # Benchmark parameters
    depth = 50
    simulations = 500  # Reduced to make Python implementation finish in reasonable time
    num_threads = 4
    runs = 3  # Number of runs for averaging
    
    print("=== MCTS Benchmark: C++ (pybind) vs Native Python ===")
    
    # Check if problem exists
    if os.path.exists(problem_path):
        results = benchmark_comparison(problem_path, depth, simulations, num_threads, runs)
        plot_results(results, problem_path)
    else:
        print(f"Problem not found: {problem_path}")
        
        # Try to find alternative problems
        print("Looking for available problems...")
        alternatives = []
        
        # Check in problemset2
        problemset2_dir = "./problemset2"
        if os.path.exists(problemset2_dir):
            problems = [os.path.join(problemset2_dir, f) for f in os.listdir(problemset2_dir) 
                      if f.endswith('.json')]
            if problems:
                alternatives.extend(problems[:2])  # Just add a couple
        
        if alternatives:
            print(f"Found {len(alternatives)} alternative problems. Using first one.")
            problem_path = alternatives[0]
            print(f"Using alternative problem: {problem_path}")
            results = benchmark_comparison(problem_path, depth, simulations, num_threads, runs)
            plot_results(results, problem_path)
        else:
            print("No alternative problems found.")

if __name__ == "__main__":
    main()

