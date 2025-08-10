"""!
@file mcts_node.py
@brief MCTS Node implementation for the Beluga Challenge

This module implements the Monte Carlo Tree Search node structure
for choosing the best parameters for complex actions in the Beluga Challenge.
"""

import math
from rl.env.state import ProblemState  # Assuming State is defined in state.py

# List of complex actions that require parameter selection
complex_actions = ["left_stack_rack", "right_stack_rack", "left_unstack_rack", "right_unstack_rack"]

class MCTSNode:
    """!
    @brief Node class for Monte Carlo Tree Search
    
    Each node represents a state in the search tree and maintains
    information about visits, rewards, and possible actions.
    """
    
    def __init__(self, state, parent=None, action=None, depth=0):
        """!
        @brief Initialize a new MCTS node
        @param state Current problem state
        @param parent Parent node in the search tree
        @param action Action taken to reach this node
        @param depth Depth of this node in the tree
        """
        self.state = state # Current state (later observation)
        self.parent = parent # Parent node
        self.action = action # Action taken to reach this node
        self.depth = depth # Depth in the tree
        self.children = [] # List of child nodes
        self.visits = 0 # Number of visits to this node
        self.total_reward = 0.0 # Total reward accumulated from this node

    def is_root(self):
        """!
        @brief Check if this node is the root of the tree
        @return True if this is the root node, False otherwise
        """
        return self.parent is None

    def is_terminal(self):
        """!
        @brief Check if this node represents a terminal state
        @return True if this is a terminal state, False otherwise
        """
        return self.state.is_terminal()

    def is_fully_expanded(self):
        """!
        @brief Check if all possible actions have been tried from this node
        @return True if fully expanded, False otherwise
        """
        return len(self.get_untried_action()) == 0
        
    def get_untried_action(self):
        """!
        @brief Get list of actions that haven't been tried yet from this node
        @return List of untried (action_name, parameters) tuples
        """
        # If root node has a specific action without parameters
        if self.is_root() and self.action is not None and self.action[1] is None:
            # Only return parameters for this specific action
            action_name = self.action[0]
            all_params = self.state.enumerate_valid_params(action_name)
            tried_params = [child.action[1] for child in self.children]
            # Return only untried parameters
            return [(action_name, param) for param in all_params if param not in tried_params]
        else:
            # Normal behavior for other nodes
            all_possible_actions = self.state.get_possible_actions()
            tried_actions = [child.action for child in self.children]
            return [action for action in all_possible_actions if action not in tried_actions]

    def expand(self, candidate: tuple):
        """!
        @brief Expand the node by adding a new child for the given action
        @param candidate Tuple of (action_name, parameters) to expand
        @return The newly created child node
        """
        new_state = self.state.copy()  # Create copy
        new_state.apply_action(candidate[0], candidate[1])  # Apply action on copy
        child_node = MCTSNode(state=new_state, parent=self, action=candidate, depth=self.depth + 1)
        self.add_child(child_node)
        return child_node

    def add_child(self, child):
        """!
        @brief Add a child node to this node
        @param child Child node to add
        """
        self.children.append(child)

    def best_child(self, exploration_weight=1.0):
        """!
        @brief Select the best child node using UCT (Upper Confidence bounds applied to Trees)
        @param exploration_weight Weight for exploration vs exploitation trade-off
        @return Best child node according to UCT formula
        """
        # Example: UCT formula
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
        """!
        @brief Backpropagate the reward up the tree to all ancestors
        @param reward Reward value to propagate
        """
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)
