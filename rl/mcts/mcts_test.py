from .mcts_node import MCTSNode
from .mcts import MCTS
from rl.env.state import ProblemState, load_from_json
from rl.env.action import left_stack_rack, right_stack_rack, left_unstack_rack, right_unstack_rack
import cProfile


# Annahme: ProblemState ist korrekt importiert und initialisiert
initial_state = load_from_json("rl/mcts/problem.json") # ggf. mit Parametern
initial_state.apply_action("unload_beluga", [])
initial_state.apply_action("unload_beluga", [])
#initial_state.apply_action(("unload_beluga", None))
initial_state.upddate_subgoals()
param_actions = [
            "left_stack_rack",
            "right_stack_rack",
            "left_unstack_rack",
            "right_unstack_rack",
            "load_beluga",
            "get_from_hangar",
            "deliver_to_hangar"
        ]

for action in param_actions:
    print("-" * 20)
    print(f"Starte MCTS für Aktion: {action}")
    specific_action = action
    # Erstelle den Root-Node mit dieser Aktion (ohne Parameter)
    root = MCTSNode(state=initial_state, action=(specific_action, None))

    # MCTS mit diesem Root-Node starten
    mcts = MCTS(root, depth=10, n_simulations=10)
    
    # Profiling nur für die Suche
    cProfile.run('mcts.search()', sort='cumtime')
    # Alternativ:
    # import pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # best_node = mcts.search()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(20)

    best_node = mcts.search()
    if best_node:
        best_params = best_node.action[1]
        print(f"Beste Parameter für {specific_action}: {best_params}")


