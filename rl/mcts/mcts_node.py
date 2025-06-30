import math
import random
from rl.env.state import ProblemState  # Assuming State is defined in state.py


# list of complex actions

complex_actions = ["left_stack_rack", "right_stack_rack", "left_unstack_rack", "right_unstack_rack"]



class MCTSNode:
    def __init__(self, state, parent=None, action=None, depth=0):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.action = action  # tuple: (action_name, param_tuple)
        self.depth = depth
        # check if action is complex e.g. in complex_actions
        self.parent_node_complex = action is not None and action[0] in complex_actions

    def is_root(self):
        return self.parent is None


    def is_terminal(self):
        return self.state.is_terminal()


    def is_fully_expanded(self):
        if self.parent_node_complex:
            # Wenn der Knoten komplexe Aktionen hat, prüfen wir, ob alle Parameter ausprobiert wurden
            return len(self.get_untried_param_candidates()) == 0
        else:
            return len(self.get_untried_action_candidates()) == 0
        

    def get_untried_param_candidates(self):
        all_legal_params = self.state.enumerate_valid_params(self.action[0])
        print(f"DEBUG - All legal params for {self.action}: {all_legal_params}")
        tried_params = [child.action[1] for child in self.children if child.action is not None]
        return [p for p in all_legal_params if p not in tried_params]
    

    def get_untried_action_candidates(self):
        all_actions = self.state.get_possible_actions()
        tried_actions = [child.action[0] for child in self.children if child.action is not None]
        return [a for a in all_actions if a not in tried_actions]


    def expand(self, candidate: tuple):
        action_name, params = candidate
        is_complex_action = action_name in complex_actions

        if is_complex_action:
            # High-Level: State bleibt gleich, depth bleibt gleich
            new_state = self.state.copy()
            child_node = MCTSNode(state=new_state, parent=self, action=candidate, depth=self.depth)
            self.add_child(child_node)
            return child_node

        # Parameter-Aktion: State ändert sich, depth inkrementieren
        new_state = self.state.copy()
        new_state.apply_action(candidate)
        child_node = MCTSNode(state=new_state, parent=self, action=candidate, depth=self.depth + 1)
        self.add_child(child_node)
        return child_node


    def add_child(self, child):
        self.children.append(child)


    def best_child(self, exploration_weight=1.0):
        # Beispiel: UCT-Formel
        best_score = float('-inf')
        best = None
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best = child
        return best


    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)




class MCTS:
    def __init__(self, root: MCTSNode, depth: int = 100, n_simulations: int = 1000):
        self.root = root
        self.depth = depth
        self.n_simulations = n_simulations

    def search(self):
        for i in range(self.n_simulations):
            print(f"\nIteration {i+1}/{self.n_simulations}")
            
            # 1. Selection
            node = self.select(self.root)
            print(f"Selected node: depth={node.depth}, action={node.action}")
            
            # 2. Expansion
            if not node.is_terminal():
                if node.parent_node_complex:
                    untried_candidates = node.get_untried_param_candidates()
                    print(f"Complex node, untried params: {len(untried_candidates)}")
                    if untried_candidates:
                        candidate = random.choice(untried_candidates)
                        print(f"Expanding with param: {candidate}")
                        node = node.expand((node.action[0], candidate))
                else:
                    untried_candidates = node.get_untried_action_candidates()
                    print(f"Standard node, untried actions: {len(untried_candidates)}")
                    if untried_candidates:
                        candidate = random.choice(untried_candidates)
                        print(f"Expanding with action: {candidate}")
                        node = node.expand((candidate, None))
            
            # 3. Simulation
            reward = self.rollout(node)
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
        state = node.state.copy()
        depth = node.depth  # Start from the node's current depth
        
        while not state.is_terminal() and depth < self.depth:
            # Get possible actions
            possible_actions = state.get_possible_actions()
            if not possible_actions:
                break
                
            # Choose random action
            action_name = random.choice(possible_actions)
            
            # Check if it's a complex action
            if action_name in complex_actions:
                params = state.enumerate_valid_params((action_name, None))
                if params:
                    param = random.choice(params)
                    state.apply_action((action_name, param))
            else:
                state.apply_action((action_name, None))
            
            depth += 1
        
        # Calculate reward based on the final state
        return state.evaluate(depth)