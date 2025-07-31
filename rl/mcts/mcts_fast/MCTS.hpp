#pragma once
#include "MCTSNode.hpp"
#include <memory>
#include <vector>
#include <random>
#include <thread>

/**
 * @class MCTS
 * @brief Monte Carlo Tree Search implementation.
 */
class MCTS {
public:
    std::shared_ptr<MCTSNode> root;
    int depth;
    int n_simulations;
    bool debug;
    std::mt19937 rng;
    int num_threads;

    /**
     * @brief Constructor for MCTS.
     * @param root The root node of the search tree
     * @param depth Maximum search depth
     * @param n_simulations Number of simulations to run
     * @param debug Enable debug output
     * @param num_threads Number of threads to use for parallelization (0 = auto-detect)
     */
    MCTS(std::shared_ptr<MCTSNode> root, int depth = 5, int n_simulations = 300, bool debug = false, int num_threads = 0);

    /**
     * @brief Perform the MCTS search.
     * @return The best child node found, or nullptr if no solution
     */
    std::shared_ptr<MCTSNode> search();

    /**
     * @brief Select a node for expansion using UCT.
     * @param node Starting node for selection
     * @return Selected node
     */
    std::shared_ptr<MCTSNode> select(std::shared_ptr<MCTSNode> node);

    /**
     * @brief Perform a rollout simulation from the given node.
     * @param node Starting node for rollout
     * @return Reward from the simulation
     */
    float rollout(std::shared_ptr<MCTSNode> node);

    /**
     * @brief Single rollout implementation.
     * @param node Starting node for rollout
     * @param rng Random number generator
     * @return Reward from the simulation
     */
    float rollout_single(std::shared_ptr<MCTSNode> node, std::mt19937& rng);

    /**
     * @brief Get the best path from root to leaves.
     * @return Vector of actions representing the best path
     */
    std::vector<std::pair<std::string, std::vector<int>>> get_best_path();

    /**
     * @brief Collect training data from the MCTS tree.
     * @param max_samples Maximum number of samples to collect
     * @param min_visits Minimum visits required for a node to be included
     * @return Vector of (observation, value) pairs
     */
    std::vector<std::pair<std::vector<float>, float>> collect_training_data(int max_samples = 1000, int min_visits = 5);

    /**
     * @brief Count total number of nodes in the tree.
     * @return Total node count
     */
    int count_total_nodes() const;

    /**
     * @brief Get the maximum depth of the tree.
     * @return Maximum depth
     */
    int get_tree_depth() const;

    /**
     * @brief Perform a single MCTS iteration.
     * @return True if terminal node found, false otherwise
     */
    bool search_single_iteration();

    /**
     * @brief Root parallelization - creates multiple independent MCTS instances.
     * @param thread_count Number of threads to use (0 = use instance's num_threads)
     * @return The best child node found, or nullptr if no solution
     */
    std::shared_ptr<MCTSNode> search_root_parallel(int thread_count = 0);
    
    /**
     * @brief Set the number of threads to use for parallelization.
     * @param threads Number of threads (0 = auto-detect)
     */
    void set_num_threads(int threads);

private:
    /**
     * @brief Helper function to count nodes recursively.
     * @param node Starting node
     * @return Count of nodes in subtree
     */
    int count_nodes(std::shared_ptr<MCTSNode> node) const;

    /**
     * @brief Helper function to get maximum depth recursively.
     * @param node Starting node
     * @return Maximum depth in subtree
     */
    int get_max_depth(std::shared_ptr<MCTSNode> node) const;
};
