import mcts_fast


state = mcts_fast.ProblemState.load_from_json("./problemset3(BigProblems)/problem4_j68_r9_b17_pl2.json")
state.apply_action("unload_beluga", [])
state.apply_action("unload_beluga", [])
root = mcts_fast.MCTSNode(state=state, action=("left_stack_rack", []))

# MCTS mit diesem Root-Node starten
mcts = mcts_fast.MCTS(root, depth=100, n_simulations=1000, debug=False)
best_node = mcts.search()

if best_node:
  params = best_node.action[1]
  print(f"Best params: {params}")
