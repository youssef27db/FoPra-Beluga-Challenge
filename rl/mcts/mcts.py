from .mcts_node import MCTSNode
from rl.env.state import *
from rl.env import *
import random

class MCTS:
    def __init__(self, root: MCTSNode, depth: int = 100, n_simulations: int = 1000):
        self.root = root
        self.depth = depth
        self.n_simulations = n_simulations
        self.debug = False

    def search(self):
        for i in range(self.n_simulations):
            if self.debug:
                print(f"\nIteration {i+1}/{self.n_simulations}")
            
            # 1. Selection
            node = self.select(self.root)
            if self.debug:
                print(f"Selected node: depth={node.depth}, action={node.action}")
            
            # 2. Expansion
            if not node.is_terminal():
                untried_actions = node.get_untried_action()
                if untried_actions:
                    action = random.choice(untried_actions)
                    if self.debug:
                        print(f"Expanding node with action: {action}")
                    node = node.expand(action)
                else:
                    if self.debug:
                        print("No untried actions available, skipping expansion.")
            
            # 3. Simulation
            reward = self.rollout(node)
            if self.debug:
                print(f"Rollout reward: {reward}")
            
            # 4. Backpropagation
            node.backpropagate(reward)
    
        # Final selection
        print("\nFinal selection:")
        best_child = self.root.best_child(exploration_weight=0)
        if best_child is None:
            print("WARNING: Root has no children!")
            return None
        else:
            print(f"Best child: action={best_child.action}, visits={best_child.visits}, reward={best_child.total_reward}")
            return best_child


    def select(self, node):
        """Traversiere den Baum, bis wir einen nicht vollständig expandierten Knoten oder ein Terminal-Node finden."""
        current_depth = 0
        while not node.is_terminal() and node.is_fully_expanded() and current_depth < self.depth:
            next_node = node.best_child()
            if next_node is None:
                
                break  # Falls keine Kinder vorhanden
            node = next_node
            current_depth += 1
        return node

    def rollout(self, node):
        """Simulate random actions from the node until we reach a terminal state or max depth."""
        state: ProblemState = node.state.copy()
        depth = node.depth  # Start from the node's current depth
        
        #print(f"DEBUG - Starting rollout from depth {depth}")
        rollout_actions = []
        
        while not state.is_terminal() and depth < self.depth:
            # Get possible actions as (action_name, params) tuples
            possible_actions = state.get_possible_actions()
            
            if not possible_actions:
                print(f"DEBUG - No possible actions at depth {depth}")
                break
            
            # Choose a random action
            action_name, params = random.choice(possible_actions)
            
            # Apply the action and update state
            #print(f"DEBUG - Rollout action: {action_name} with params {params}")
            rollout_actions.append((action_name, params))
            
            # Apply action to state
            state.apply_action(action_name, params)
            # Update subgoals after each action
            state.upddate_subgoals()
            
            depth += 1
        
        if self.debug:
            print(f"DEBUG - Rollout completed with {len(rollout_actions)} actions")
            print(f"DEBUG - Final rollout actions: {rollout_actions[:5]}{'...' if len(rollout_actions) > 5 else ''}")
        
        # Calculate reward based on final state
        reward = state.evaluate(depth)
        #print(f"DEBUG - Evaluation result: {reward}")
        
        return reward

    def get_best_path(self):
        """
        Gibt den Pfad der bestbesuchten Kindknoten ab der Wurzel zurück.
        """
        path = []
        node = self.root
        while True:
            best_child = node.best_child(exploration_weight=0)
            if best_child is None:
                break
            path.append(best_child.action)
            node = best_child
        return path