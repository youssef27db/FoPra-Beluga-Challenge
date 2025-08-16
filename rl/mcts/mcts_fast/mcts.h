#pragma once
#include "types.h"
#include "mcts_node.h"
#include <memory>
#include <random>

namespace mcts_fast {

class MCTS {
public:
    MCTS(
        std::shared_ptr<MCTSNode> root,
        int depth = 5,
        int n_simulations = 300,
        bool debug = false
    );
    
    // Main MCTS algorithm
    std::shared_ptr<MCTSNode> search();
    
    // MCTS phases
    std::shared_ptr<MCTSNode> select(std::shared_ptr<MCTSNode> node);
    double rollout(std::shared_ptr<MCTSNode> node);
    
    // Utility functions
    std::vector<ActionTuple> getBestPath() const;
    
    // Getters
    std::shared_ptr<MCTSNode> getRoot() const { return root_; }
    int getDepth() const { return depth_; }
    int getSimulations() const { return n_simulations_; }
    bool isDebug() const { return debug_; }

private:
    std::shared_ptr<MCTSNode> root_;
    int depth_;
    int n_simulations_;
    bool debug_;
    
    // Random number generation
    mutable std::mt19937 rng_;
    
    // Helper functions
    void debugPrint(const std::string& message) const;
};

} // namespace mcts_fast