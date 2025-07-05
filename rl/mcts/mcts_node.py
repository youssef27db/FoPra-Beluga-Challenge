import math
from rl.env.state import ProblemState  # Assuming State is defined in state.py


# list of complex actions

complex_actions = ["left_stack_rack", "right_stack_rack", "left_unstack_rack", "right_unstack_rack"]



class MCTSNode:
    def __init__(self, state, parent=None, action=None, depth=0):
        self.state = state # später observation
        self.parent = parent # Parent node
        self.action = action # Action taken to reach this node
        self.depth = depth # Depth in the tree
        self.children = [] # List of child nodes
        self.visits = 0 # Number of visits to this node
        self.total_reward = 0.0 # Total reward accumulated from this node

    def is_root(self):
        return self.parent is None


    def is_terminal(self):
        return self.state.is_terminal()


    def is_fully_expanded(self):
        return len(self.get_untried_action()) == 0
        

    def get_untried_action(self):
        # Falls der Root-Node eine spezifische Aktion ohne Parameter hat
        if self.is_root() and self.action is not None and self.action[1] is None:
            # Nur Parameter für diese spezifische Aktion zurückgeben
            action_name = self.action[0]
            all_params = self.state.enumerate_valid_params(action_name)
            tried_params = [child.action[1] for child in self.children]
            # Nur noch nicht probierte Parameter zurückgeben
            return [(action_name, param) for param in all_params if param not in tried_params]
        else:
            # Normales Verhalten für andere Knoten
            all_possible_actions = self.state.get_possible_actions()
            tried_actions = [child.action for child in self.children]
            return [action for action in all_possible_actions if action not in tried_actions]


    def expand(self, candidate: tuple):
        new_state = self.state.copy()  # Kopie erstellen
        new_state.apply_action(candidate[0], candidate[1])  # Aktion auf Kopie anwenden
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
