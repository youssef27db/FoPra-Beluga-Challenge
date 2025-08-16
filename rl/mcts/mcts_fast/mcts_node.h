#pragma once
#include "types.h"
#include "problem_state.h"
#include <vector>
#include <memory>
#include <optional>

namespace mcts_fast {

class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    MCTSNode(
        const ProblemState& state, 
        std::shared_ptr<MCTSNode> parent = nullptr,
        const std::optional<ActionTuple>& action = std::nullopt,
        int depth = 0
    );
    
    // Tree structure
    std::shared_ptr<MCTSNode> getParent() const { return parent_; }
    const std::vector<std::shared_ptr<MCTSNode>>& getChildren() const { return children_; }
    void addChild(std::shared_ptr<MCTSNode> child);
    
    // Node properties
    const ProblemState& getState() const { return state_; }
    const std::optional<ActionTuple>& getAction() const { return action_; }
    int getDepth() const { return depth_; }
    int getVisits() const { return visits_; }
    double getTotalReward() const { return total_reward_; }
    
    // MCTS operations
    bool isRoot() const;
    bool isTerminal() const;
    bool isFullyExpanded() const;
    
    std::vector<ActionTuple> getUntriedActions() const;
    std::shared_ptr<MCTSNode> expand(const ActionTuple& action);
    std::shared_ptr<MCTSNode> bestChild(double exploration_weight = 1.0) const;
    void backpropagate(double reward);
    
    // UCT calculation
    double getUCTValue(double exploration_weight = 1.0, int parent_visits = 1) const;
    
    std::string toString() const;

private:
    ProblemState state_;
    std::shared_ptr<MCTSNode> parent_;
    std::vector<std::shared_ptr<MCTSNode>> children_;
    std::optional<ActionTuple> action_;
    int depth_;
    int visits_;
    double total_reward_;
    
    mutable std::vector<ActionTuple> cached_possible_actions_;
    mutable bool actions_cached_;
};

} // namespace mcts_fast