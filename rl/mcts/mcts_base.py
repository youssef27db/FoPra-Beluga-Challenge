from mcts.node import MCTSNode
import random

class MCTSBase:
    def __init__(self, root_state, simulations=100, max_depth=10):
        # Initialize the root node and the number of simulations
        self.root = MCTSNode(state=root_state)
        self.simulations = simulations
        self.max_depth = max_depth

    def search(self):
        # Perform multiple simulations to explore and refine the search tree
        for _ in range(self.simulations):
            node = self.select(self.root)
            if not node.is_terminal():
                self.expand(node)
                value = self.simulate(node)
            else:
                value = node.state.evaluate()
            self.backpropagate(node, value)
        # Return the action from the best child of the root
        return self.root.best_child(c_param=0.0).action

    def select(self, node, depth=0):
        # Stop if node is expanded, not terminal, and below max depth
        while node.is_expanded and not node.is_terminal() and depth < self.max_depth:
            node = node.best_child()
            depth += 1
        return node

    def expand(self, node):
        # Expand the node if it hasn't been expanded before
        if not node.is_expanded:
            node.expand()

    def simulate(self, node):
        # Perform a rollout (random playout) until reaching a terminal state
        rollout_state = node.state.clone()
        while not rollout_state.is_terminal():
            actions = rollout_state.get_all_possible_high_level_actions()
            action = self.rollout_policy(rollout_state, actions)
            rollout_state.apply_high_level_action(action)
        return rollout_state.evaluate()

    def rollout_policy(self, state, actions):
        # Simple random policy selecting from possible actions
        return random.choice(actions)

    def backpropagate(self, node, value):
        # Propagate the value back up the tree to update node statistics
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent