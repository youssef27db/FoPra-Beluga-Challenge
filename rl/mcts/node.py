class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.action = action  # hier: Parameter-Action, z. B. (rack, pos)
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.is_expanded = False

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_parameter_actions())

    def expand(self):
        actions = self.state.get_possible_parameter_actions()
        for action in actions:
            new_state = self.state.clone()
            new_state.apply_parameter_action(action)
            child_node = MCTSNode(new_state, parent=self, action=action)
            self.children.append(child_node)
        self.is_expanded = True

    def best_child(self, c_param=1.4):
        from math import sqrt, log
        best_score = -float('inf')
        best_node = None
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.value / child.visits
                explore = c_param * sqrt(log(self.visits + 1) / (child.visits + 1))
                score = exploit + explore
            if score > best_score:
                best_score = score
                best_node = child
        return best_node