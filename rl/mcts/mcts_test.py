from .mcts_node import MCTSNode, MCTS
from rl.env.state import ProblemState, load_from_json
from rl.env.action import left_stack_rack, right_stack_rack, left_unstack_rack, right_unstack_rack


complex_action = "left_stack_rack"

# Annahme: ProblemState ist korrekt importiert und initialisiert
initial_state = load_from_json("rl/mcts/problem.json") # ggf. mit Parametern
initial_state.apply_action(("unload_beluga", None))
initial_state.apply_action(("unload_beluga", None))
initial_state.apply_action(("unload_beluga", None))
initial_state.upddate_subgoals()
print("anzahl unloaded:",initial_state.belugas_unloaded)
print(initial_state.evaluate(0))

# Komplexe Aktion, für die wir Parameter finden wollen
root = MCTSNode(state=initial_state, action=(complex_action, None))
mcts = MCTS(root, depth=5, n_simulations=80)

best_node = mcts.search()
if best_node:
    print("Best action:", best_node.action)
    print("Reward:", best_node.total_reward/best_node.visits)
else:
    print("Keine gültige Aktion gefunden!")