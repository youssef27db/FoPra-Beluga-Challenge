from .mcts_node import MCTSNode, MCTS
from rl.env.state import ProblemState, load_from_json
from rl.env.action import left_stack_rack, right_stack_rack, left_unstack_rack, right_unstack_rack


complex_action = "left_stack_rack"

# Annahme: ProblemState ist korrekt importiert und initialisiert
initial_state = load_from_json("rl/mcts/problem.json") # ggf. mit Parametern
left_stack_rack_v = left_stack_rack(initial_state,0,0)
print("bool-value:", left_stack_rack_v)
valid_params = initial_state.enumerate_valid_params(complex_action)
print("Verfügbare Parameter für Aktion", complex_action, ":", valid_params)

initial_state.apply_action(("unload_beluga", None))
valid_params = initial_state.enumerate_valid_params(complex_action)
print("Verfügbare Parameter für Aktion", complex_action, ":", valid_params)



initial_state.apply_action(("unload_beluga", None))
valid_params = initial_state.enumerate_valid_params(complex_action)
print("Verfügbare Parameter für Aktion", complex_action, ":", valid_params)

wait = input("Drücke Enter, um fortzufahren...")

# Komplexe Aktion, für die wir Parameter finden wollen
root = MCTSNode(state=initial_state, action=(complex_action, None))
mcts = MCTS(root, depth=10, n_simulations=20)
best_node = mcts.search()
if best_node:
    print("Best action:", best_node.action)
    print("Reward:", best_node.total_reward)
else:
    print("Keine gültige Aktion gefunden!")