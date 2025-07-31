#include "MCTSNode.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

MCTSNode::MCTSNode(const ProblemState& state, 
                   std::shared_ptr<MCTSNode> parent,
                   const std::pair<std::string, std::vector<int>>& action,
                   int depth)
    : state(state), parent(parent), action(action), depth(depth), 
      visits(0), total_reward(0.0f) {} 

bool MCTSNode::is_root() const {
    return parent.expired();
}

bool MCTSNode::is_terminal() const {
    return state.is_terminal();
}

bool MCTSNode::is_fully_expanded() const {
    return get_untried_actions().empty();
}

std::vector<std::pair<std::string, std::vector<int>>> MCTSNode::get_untried_actions() const {
    // For root node with specific action without parameters
    if (is_root() && !action.first.empty() && action.second.empty()) {
        std::string action_name = action.first;
        auto all_params = state.enumerate_valid_params(action_name);
        
        std::vector<std::pair<std::string, std::vector<int>>> untried;
        for (const auto& params : all_params) {
            std::pair<std::string, std::vector<int>> candidate = {action_name, params};
            if (!action_tried(candidate)) {
                untried.push_back(candidate);
            }
        }
        return untried;
    } else {
        // Normal behavior for other nodes
        auto all_possible_actions = get_all_possible_actions();
        std::vector<std::pair<std::string, std::vector<int>>> untried;
        
        for (const auto& action : all_possible_actions) {
            if (!action_tried(action)) {
                untried.push_back(action);
            }
        }
        return untried;
    }
}

std::shared_ptr<MCTSNode> MCTSNode::expand(const std::pair<std::string, std::vector<int>>& candidate) {
    // Create copy using move semantics for better performance
    ProblemState new_state = state.copy();
    new_state.apply_action(candidate.first, candidate.second);
    
    // Use make_shared for better memory efficiency and reduced allocations
    auto child_node = std::make_shared<MCTSNode>(std::move(new_state), shared_from_this(), candidate, depth + 1);
    add_child(child_node);
    return child_node;
}

void MCTSNode::add_child(std::shared_ptr<MCTSNode> child) {
    // Reserve capacity for typical branching factor to avoid reallocations
    if (children.empty()) {
        children.reserve(8);  // Typical branching factor estimate
    }
    children.push_back(std::move(child));
}

std::shared_ptr<MCTSNode> MCTSNode::best_child(float exploration_weight) const {
    if (children.empty()) {
        return nullptr;
    }
    
    float best_score = std::numeric_limits<float>::lowest();
    std::shared_ptr<MCTSNode> best = nullptr;
    
    for (const auto& child : children) {
        float score;
        if (child->visits == 0) {
            score = std::numeric_limits<float>::infinity();
        } else {
            // Standard UCT formula
            float exploitation = child->total_reward / child->visits;
            float exploration = exploration_weight * std::sqrt(std::log(visits) / child->visits);
            score = exploitation + exploration;
        }
        
        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }
    
    return best;
}

void MCTSNode::backpropagate(float reward) {
    // Wir brauchen einen shared_ptr auf den aktuellen Knoten, um ihn im Loop zu halten
    std::shared_ptr<MCTSNode> current_node = shared_from_this();

    while (current_node != nullptr) {
        current_node->visits += 1;
        current_node->total_reward += reward;
        
        // Um zum Parent zu gelangen, müssen wir den weak_ptr "sperren",
        // um einen temporären shared_ptr zu erhalten.
        current_node = current_node->parent.lock();
    }
}

std::vector<std::pair<std::string, std::vector<int>>> MCTSNode::get_all_possible_actions() const {
    return state.get_possible_actions();
}

bool MCTSNode::action_tried(const std::pair<std::string, std::vector<int>>& action) const {
    for (const auto& child : children) {
        if (child->action.first == action.first && child->action.second == action.second) {
            return true;
        }
    }
    return false;
}

// Virtual loss methods removed - not needed for root parallelization
