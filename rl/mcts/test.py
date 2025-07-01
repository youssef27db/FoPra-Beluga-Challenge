from .mcts_node import MCTSNode, MCTS
from rl.env.environment import Env
from rl.env.state import ProblemState, load_from_json
from rl.env.action import left_stack_rack, right_stack_rack, left_unstack_rack, right_unstack_rack


complex_action = "left_stack_rack"
env = Env()
env.reset()  # Reset the environment to its initial state

env.step("unload_beluga")
print(env.state) # Beispielaktion, um den Zustand zu ändern
env.step("unload_beluga")  # Weitere Beispielaktion
print(env.state)
env.step("unload_beluga")
print(env.state)
env.step("left_stack_rack", 0,0)
print(env.state)  # Beispielaktion, um den Zustand zu ändern


# Komplexe Aktion, für die wir Parameter finden wollen
root = MCTSNode(state=env.state, action=(complex_action, None))
mcts = MCTS(root, depth=5, n_simulations=1)
best_node = mcts.search()
if best_node:
    print("Best action:", best_node.action)
    print("Reward:", best_node.total_reward)
else:
    print("Keine gültige Aktion gefunden!")