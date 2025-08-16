#include "mcts.h"
#include <iostream>
#include <chrono>

namespace mcts_fast {

MCTS::MCTS(
    std::shared_ptr<MCTSNode> root,
    int depth,
    int n_simulations,
    bool debug
) : root_(root), depth_(depth), n_simulations_(n_simulations), debug_(debug),
    rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

std::shared_ptr<MCTSNode> MCTS::search() {
    bool terminal_node_found = false;
    
    for (int i = 0; i < n_simulations_; ++i) {
        if (debug_) {
            debugPrint("Iteration " + std::to_string(i + 1) + "/" + std::to_string(n_simulations_));
        }
        
        // 1. Selection
        auto node = select(root_);
        if (debug_) {
            debugPrint("Selected node: depth=" + std::to_string(node->getDepth()) + 
                      ", visits=" + std::to_string(node->getVisits()));
        }
        
        // 2. Expansion
        if (!node->isTerminal()) {
            auto untried_actions = node->getUntriedActions();
            if (!untried_actions.empty()) {
                // Choose random untried action
                std::uniform_int_distribution<size_t> dist(0, untried_actions.size() - 1);
                const auto& action = untried_actions[dist(rng_)];
                
                if (debug_) {
                    debugPrint("Expanding node with action: " + action.first);
                }
                
                node = node->expand(action);
                
                if (node->getState().isTerminal()) {
                    if (debug_) {
                        debugPrint("Terminal state reached! Solution found.");
                    }
                    terminal_node_found = true;
                    double reward = node->getState().evaluate(node->getDepth());
                    node->backpropagate(reward);
                    if (debug_) {
                        debugPrint("Rollout reward: " + std::to_string(reward));
                    }
                    break; // Abort MCTS
                }
            } else {
                // Abort if no untried actions are available
                if (node->getChildren().empty() || node->getDepth() >= depth_ - 1) {
                    if (debug_) {
                        debugPrint("No further actions possible at depth " + 
                                 std::to_string(node->getDepth()) + ". Aborting MCTS.");
                    }
                    // Final selection as at the end of the method
                    debugPrint("Final selection (early):");
                    auto best_child = root_->bestChild(0.0);
                    return best_child;
                } else {
                    if (debug_) {
                        debugPrint("No untried actions available, skipping expansion.");
                    }
                }
            }
        }
        
        // 3. Simulation
        double reward = rollout(node);
        if (debug_) {
            debugPrint("Rollout reward: " + std::to_string(reward));
        }
        
        // 4. Backpropagation
        node->backpropagate(reward);
    }
    
    // Final selection
    debugPrint("Final selection:");
    auto best_child = root_->bestChild(0.0);
    if (!best_child) {
        debugPrint("WARNING: Root has no children!");
        return nullptr;
    } else {
        double avg_reward = best_child->getVisits() > 0 ? 
            best_child->getTotalReward() / best_child->getVisits() : 0.0;
        debugPrint("Best child: visits=" + std::to_string(best_child->getVisits()) + 
                  ", reward=" + std::to_string(avg_reward));
        if (terminal_node_found) {
            debugPrint("Note: A terminal state was found!");
        }
        return best_child;
    }
}

std::shared_ptr<MCTSNode> MCTS::select(std::shared_ptr<MCTSNode> node) {
    int current_depth = 0;
    while (!node->isTerminal() && node->isFullyExpanded() && current_depth < depth_) {
        auto next_node = node->bestChild();
        if (!next_node) {
            break; // If no children are present
        }
        node = next_node;
        current_depth++;
    }
    return node;
}

double MCTS::rollout(std::shared_ptr<MCTSNode> node) {
    ProblemState state = node->getState().copy();
    int depth = node->getDepth(); // Start from the node's current depth
    
    std::vector<ActionTuple> rollout_actions;
    
    while (!state.isTerminal() && depth < depth_) {
        // Get possible actions
        auto possible_actions = state.getPossibleActions();
        
        if (possible_actions.empty()) {
            if (debug_) {
                debugPrint("No possible actions at depth " + std::to_string(depth));
            }
            break;
        }
        
        // Choose a random action
        std::uniform_int_distribution<size_t> dist(0, possible_actions.size() - 1);
        const auto& action = possible_actions[dist(rng_)];
        
        rollout_actions.push_back(action);
        
        // Apply action to state
        state.applyAction(action.first, action.second);
        depth++;
    }
    
    if (debug_) {
        debugPrint("Rollout completed with " + std::to_string(rollout_actions.size()) + " actions");
        if (rollout_actions.size() > 5) {
            debugPrint("Final rollout actions: first 5 of " + std::to_string(rollout_actions.size()));
        } else {
            debugPrint("All rollout actions logged");
        }
    }
    
    // Calculate reward based on final state
    double reward = state.evaluate(depth);
    if (debug_) {
        debugPrint("Rollout ended at depth " + std::to_string(depth) + 
                  ", final reward: " + std::to_string(reward));
        
        auto subgoals = state.getSubgoals();
        std::string subgoal_str = "Subgoals: ";
        for (const auto& pair : subgoals) {
            subgoal_str += pair.first + "=" + std::to_string(pair.second) + " ";
        }
        debugPrint(subgoal_str);
    }
    
    // After rollout:
    if (state.isTerminal() && debug_) {
        debugPrint("Terminal state reached in rollout!");
    }
    
    return reward;
}

std::vector<ActionTuple> MCTS::getBestPath() const {
    std::vector<ActionTuple> path;
    auto node = root_;
    
    while (true) {
        auto best_child = node->bestChild(0.0);
        if (!best_child) {
            break;
        }
        if (best_child->getAction().has_value()) {
            path.push_back(best_child->getAction().value());
        }
        node = best_child;
    }
    
    return path;
}

void MCTS::debugPrint(const std::string& message) const {
    if (debug_) {
        std::cout << "[MCTS] " << message << std::endl;
    }
}

} // namespace mcts_fast