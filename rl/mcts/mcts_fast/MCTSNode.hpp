#pragma once
#include "ProblemState.hpp"
#include <vector>
#include <memory>

/**
 * @class MCTSNode
 * @brief Represents a node in the Monte Carlo Tree Search.
 */
class MCTSNode : public std::enable_shared_from_this<MCTSNode> {
public:
    ProblemState state;                      // State represented by this node
    std::weak_ptr<MCTSNode> parent;        // Parent node
    std::pair<std::string, std::vector<int>> action;  // Action taken to reach this node
    int depth;                               // Depth in the tree
    std::vector<std::shared_ptr<MCTSNode>> children;  // List of child nodes
    int visits;                              // Number of visits to this node
    float total_reward;                      // Total reward accumulated from this node

    /**
     * @brief Constructor for MCTSNode.
     * @param state The problem state
     * @param parent Parent node (nullptr for root)
     * @param action Action taken to reach this node
     * @param depth Depth in the tree
     */
    MCTSNode(const ProblemState& state, 
             std::shared_ptr<MCTSNode> parent = nullptr,
             const std::pair<std::string, std::vector<int>>& action = {"", {}},
             int depth = 0);

    /**
     * @brief Check if this is the root node.
     * @return True if root, false otherwise
     */
    bool is_root() const;

    /**
     * @brief Check if this is a terminal node.
     * @return True if terminal, false otherwise
     */
    bool is_terminal() const;

    /**
     * @brief Check if this node is fully expanded.
     * @return True if fully expanded, false otherwise
     */
    bool is_fully_expanded() const;

    /**
     * @brief Get untried actions for this node.
     * @return Vector of untried actions
     */
    std::vector<std::pair<std::string, std::vector<int>>> get_untried_actions() const;

    /**
     * @brief Expand this node with the given action.
     * @param candidate The action to expand with
     * @return Pointer to the new child node
     */
    std::shared_ptr<MCTSNode> expand(const std::pair<std::string, std::vector<int>>& candidate);

    /**
     * @brief Add a child node.
     * @param child The child node to add
     */
    void add_child(std::shared_ptr<MCTSNode> child);

    /**
     * @brief Backpropagate reward through the tree.
     * @param reward The reward to backpropagate
     */
    void backpropagate(float reward);

    /**
     * @brief Select the best child using UCT formula.
     * @param exploration_weight The exploration weight (C parameter)
     * @return Pointer to the best child, or nullptr if no children
     */
    std::shared_ptr<MCTSNode> best_child(float exploration_weight = 1.0f) const;

private:
    // Helper method to get all possible actions for this state
    std::vector<std::pair<std::string, std::vector<int>>> get_all_possible_actions() const;
    
    // Helper method to check if an action has been tried
    bool action_tried(const std::pair<std::string, std::vector<int>>& action) const;
};
