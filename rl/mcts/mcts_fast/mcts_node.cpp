#include "mcts_node.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace mcts_fast {

MCTSNode::MCTSNode(
    const ProblemState& state,
    std::shared_ptr<MCTSNode> parent,
    const std::optional<ActionTuple>& action,
    int depth
) : state_(state), parent_(parent), action_(action), depth_(depth), 
    visits_(0), total_reward_(0.0), actions_cached_(false) {
}

void MCTSNode::addChild(std::shared_ptr<MCTSNode> child) {
    children_.push_back(child);
}

bool MCTSNode::isRoot() const {
    return parent_ == nullptr;
}

bool MCTSNode::isTerminal() const {
    return state_.isTerminal();
}

bool MCTSNode::isFullyExpanded() const {
    return getUntriedActions().empty();
}

std::vector<ActionTuple> MCTSNode::getUntriedActions() const {
    if (!actions_cached_) {
        cached_possible_actions_ = state_.getPossibleActions();
        actions_cached_ = true;
    }
    
    std::vector<ActionTuple> untried_actions;
    
    // If root node has a specific action without parameters
    if (isRoot() && action_.has_value() && action_->second.empty()) {
        const std::string& action_name = action_->first;
        auto all_params = state_.enumerateValidParams(action_name);
        
        std::vector<ActionParams> tried_params;
        for (const auto& child : children_) {
            if (child->action_.has_value()) {
                tried_params.push_back(child->action_->second);
            }
        }
        
        for (const auto& param : all_params) {
            if (std::find(tried_params.begin(), tried_params.end(), param) == tried_params.end()) {
                untried_actions.emplace_back(action_name, param);
            }
        }
        return untried_actions;
    }
    
    // Normal behavior for other nodes
    std::vector<ActionTuple> tried_actions;
    for (const auto& child : children_) {
        if (child->action_.has_value()) {
            tried_actions.push_back(child->action_.value());
        }
    }
    
    for (const auto& action : cached_possible_actions_) {
        if (std::find(tried_actions.begin(), tried_actions.end(), action) == tried_actions.end()) {
            untried_actions.push_back(action);
        }
    }
    
    return untried_actions;
}

std::shared_ptr<MCTSNode> MCTSNode::expand(const ActionTuple& action) {
    ProblemState new_state = state_.copy();
    new_state.applyAction(action.first, action.second);
    
    auto child_node = std::make_shared<MCTSNode>(
        new_state, shared_from_this(), action, depth_ + 1);
    addChild(child_node);
    
    return child_node;
}

std::shared_ptr<MCTSNode> MCTSNode::bestChild(double exploration_weight) const {
    if (children_.empty()) {
        return nullptr;
    }
    
    double best_score = std::numeric_limits<double>::lowest();
    std::shared_ptr<MCTSNode> best_child = nullptr;
    
    for (const auto& child : children_) {
        double score;
        if (child->visits_ == 0) {
            score = std::numeric_limits<double>::infinity();
        } else {
            double exploitation = child->total_reward_ / child->visits_;
            double exploration = exploration_weight * 
                std::sqrt(std::log(visits_) / child->visits_);
            score = exploitation + exploration;
        }
        
        if (score > best_score) {
            best_score = score;
            best_child = child;
        }
    }
    
    return best_child;
}

void MCTSNode::backpropagate(double reward) {
    visits_++;
    total_reward_ += reward;
    
    if (parent_) {
        parent_->backpropagate(reward);
    }
}

double MCTSNode::getUCTValue(double exploration_weight, int parent_visits) const {
    if (visits_ == 0) {
        return std::numeric_limits<double>::infinity();
    }
    
    double exploitation = total_reward_ / visits_;
    double exploration = exploration_weight * std::sqrt(std::log(parent_visits) / visits_);
    return exploitation + exploration;
}

std::string MCTSNode::toString() const {
    std::ostringstream oss;
    oss << "MCTSNode(depth=" << depth_ << ", visits=" << visits_ 
        << ", total_reward=" << total_reward_;
    
    if (action_.has_value()) {
        oss << ", action=" << action_->first << " [";
        for (size_t i = 0; i < action_->second.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << action_->second[i];
        }
        oss << "]";
    }
    
    oss << ", children=" << children_.size() << ")";
    return oss.str();
}

} // namespace mcts_fast