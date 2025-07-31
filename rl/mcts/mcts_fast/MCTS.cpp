#include "MCTS.hpp"
#include <iostream>
#include <algorithm>
#include <queue>
#include <map>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <iomanip>
#include <future>
#include <vector>
#include <numeric>

// Get number of hardware threads available
inline unsigned int get_hardware_threads() {
    unsigned int threads = std::thread::hardware_concurrency();
    return threads > 0 ? threads : 1; // Fallback to 1 if detection fails
}

MCTS::MCTS(std::shared_ptr<MCTSNode> root, int depth, int n_simulations, bool debug, int num_threads)
    : root(root), depth(depth), n_simulations(n_simulations), debug(debug) {
    std::random_device rd;
    rng.seed(rd());
    set_num_threads(num_threads); // Set number of threads (0 will auto-detect)
}

void MCTS::set_num_threads(int threads) {
    // Auto-detect if threads is 0
    if (threads <= 0) {
        num_threads = get_hardware_threads();
        if (debug) {
            std::cout << "Auto-detected " << num_threads << " hardware threads." << std::endl;
        }
    } else {
        num_threads = threads;
    }
}

std::shared_ptr<MCTSNode> MCTS::search() {
    // Use root parallelization if we have multiple threads and enough simulations
    if (num_threads > 1 && n_simulations >= 100) {
        return search_root_parallel();
    }
    
    // Otherwise use the original sequential implementation
    bool terminal_node_found = false;
    
    for (int sim = 0; sim < n_simulations; ++sim) {
        if (debug) {
            std::cout << "\nIteration " << (sim + 1) << "/" << n_simulations << std::endl;
        }
        
        // 1. Selection (sequential - follows UCT path)
        auto node = select(root);
        if (debug) {
            std::cout << "Selected node: depth=" << node->depth << ", action=" << node->action.first << std::endl;
        }
        
        // 2. Expansion (sequential - modifies tree structure)
        if (!node->is_terminal()) {
            auto untried_actions = node->get_untried_actions();
            if (!untried_actions.empty()) {
                std::uniform_int_distribution<> dist(0, untried_actions.size() - 1);
                auto action = untried_actions[dist(rng)];
                
                if (debug) {
                    std::cout << "Expanding node with action: " << action.first << std::endl;
                }
                
                node = node->expand(action);
                
                if (node->state.is_terminal()) {
                    terminal_node_found = true;
                }
            }
        }
        
        // 3. Rollout - now using simple rollout since parallelization is at root level
        float reward = rollout(node);
        if (debug) {
            std::cout << "Rollout reward: " << reward << std::endl;
        }
        
        // 4. Backpropagation (sequential - updates path to root)
        node->backpropagate(reward);
    }

    // Final selection
    if (debug) {
        std::cout << "\nFinal selection:" << std::endl;
    }
    auto best_child = root->best_child(0.0f);
    if (best_child == nullptr) {
        if (debug) {
            std::cout << "WARNING: Root has no children!" << std::endl;
        }
        return nullptr;
    } else {
        if (debug) {
            float avg_reward = best_child->visits > 0 ? best_child->total_reward / best_child->visits : 0;
            std::cout << "Best child: action=" << best_child->action.first 
                      << ", visits=" << best_child->visits 
                      << ", reward=" << avg_reward << std::endl;
            if (terminal_node_found) {
                std::cout << "Hinweis: Ein Terminal-Zustand wurde gefunden!" << std::endl;
            }
        }
        return best_child;
    }
}

std::shared_ptr<MCTSNode> MCTS::select(std::shared_ptr<MCTSNode> node) {
    int current_depth = 0;
    while (!node->is_terminal() && node->is_fully_expanded() && current_depth < depth) {
        auto next_node = node->best_child(1.0f); // Standard UCT selection
        if (next_node == nullptr) {
            break; // If no children available
        }
        node = next_node;
        current_depth++;
    }
    return node;
}

float MCTS::rollout(std::shared_ptr<MCTSNode> node) {
    // Just do a single rollout - parallel execution is now handled at root level
    return rollout_single(node, rng);
}

float MCTS::rollout_single(std::shared_ptr<MCTSNode> node, std::mt19937& rng) {
    // Single rollout implementation - this is what actually gets parallelized
    ProblemState state = node->state.copy();
    int current_depth = node->depth;
    
    const int max_rollout_steps = std::min(50, depth - current_depth);
    int actions_taken = 0;
    
    while (!state.is_terminal() && current_depth < depth && actions_taken < max_rollout_steps) {
        auto possible_actions = state.get_possible_actions();
        
        if (possible_actions.empty()) {
            break;
        }
        
        // Use faster random selection - avoid creating distribution each time
        size_t action_idx = rng() % possible_actions.size();
        const auto& action = possible_actions[action_idx];
        
        state.apply_action(action.first, action.second);
        current_depth++;
        actions_taken++;
    }
    
    return state.evaluate(current_depth);
}

// This method has been removed as it's no longer needed for root parallelization

std::vector<std::pair<std::string, std::vector<int>>> MCTS::get_best_path() {
    std::vector<std::pair<std::string, std::vector<int>>> path;
    auto node = root;
    while (true) {
        auto best_child = node->best_child(0.0f);
        if (best_child == nullptr) {
            break;
        }
        path.push_back(best_child->action);
        node = best_child;
    }
    return path;
}

std::vector<std::pair<std::vector<float>, float>> MCTS::collect_training_data(int max_samples, int min_visits) {
    std::vector<std::pair<std::vector<float>, float>> training_data;
    std::queue<std::shared_ptr<MCTSNode>> nodes_to_process;
    nodes_to_process.push(root);
    
    // Statistics for diagnosis
    int total_nodes_checked = 0;
    int nodes_below_threshold = 0;
    
    while (!nodes_to_process.empty() && static_cast<int>(training_data.size()) < max_samples) {
        auto current_node = nodes_to_process.front();
        nodes_to_process.pop();
        total_nodes_checked++;
        
        // Only consider nodes with enough visits (for more stable estimates)
        if (current_node->visits >= min_visits) {
            // Get observation of current state
            auto observation = current_node->state.get_observation_high_level();
            
            // Calculate value (Q-value of the node)
            float value = current_node->visits > 0 ? current_node->total_reward / current_node->visits : 0.0f;
            
            // Add as training pair
            training_data.push_back({observation, value});
        } else {
            nodes_below_threshold++;
        }
        
        // Add children for processing
        for (const auto& child : current_node->children) {
            nodes_to_process.push(child);
        }
    }
    
    // Diagnostic output
    std::cout << "Gesammelte Trainingsdaten: " << training_data.size() << " Samples" << std::endl;
    std::cout << "Geprüfte Knoten insgesamt: " << total_nodes_checked << std::endl;
    std::cout << "Knoten unter min_visits=" << min_visits << ": " << nodes_below_threshold << std::endl;
    std::cout << "Anzahl Root-Kinder: " << root->children.size() << std::endl;
    
    // If too few nodes were collected, try with lower min_visits
    if (training_data.size() < 10 && min_visits > 1) {
        std::cout << "Zu wenige Trainingsdaten! Versuche mit min_visits=1..." << std::endl;
        return collect_training_data(max_samples, 1);
    }
    
    return training_data;
}

int MCTS::count_total_nodes() const {
    return count_nodes(root);
}

int MCTS::get_tree_depth() const {
    return get_max_depth(root);
}

bool MCTS::search_single_iteration() {
    // 1. Selection
    auto node = select(root);
    if (debug) {
        std::cout << "Selected node: depth=" << node->depth << ", action=" << node->action.first << std::endl;
    }
    
    // 2. Expansion
    if (!node->is_terminal()) {
        auto untried_actions = node->get_untried_actions();
        if (!untried_actions.empty()) {
            std::uniform_int_distribution<> dist(0, untried_actions.size() - 1);
            auto action = untried_actions[dist(rng)];
            
            if (debug) {
                std::cout << "Expanding node with action: " << action.first << std::endl;
            }
            
            node = node->expand(action);
            
            if (node->state.is_terminal()) {
                if (debug) {
                    std::cout << "Terminal-Zustand erreicht! Lösung gefunden." << std::endl;
                }
                float reward = node->state.evaluate(node->depth);
                node->backpropagate(reward);
                if (debug) {
                    std::cout << "Rollout reward: " << reward << std::endl;
                }
                return true; // Terminal node found
            }
        }
    }
    
    // 3. Rollout (only if not terminal)
    if (!node->is_terminal()) {
        float reward = rollout(node);
        if (debug) {
            std::cout << "Rollout reward: " << reward << std::endl;
        }
        
        // 4. Backpropagation
        node->backpropagate(reward);
    }
    
    return false; // No terminal node found
}

int MCTS::count_nodes(std::shared_ptr<MCTSNode> node) const {
    // Use iterative approach to avoid stack overflow with deep trees
    int count = 0;
    std::queue<std::shared_ptr<MCTSNode>> nodes_to_count;
    nodes_to_count.push(node);
    
    while (!nodes_to_count.empty()) {
        auto current = nodes_to_count.front();
        nodes_to_count.pop();
        count++;
        
        // Add children to queue
        for (const auto& child : current->children) {
            nodes_to_count.push(child);
        }
        
        // Safety check to prevent infinite loops or excessive memory usage
        if (count > 1000000) { // 1 million nodes limit
            std::cerr << "Warning: Node count exceeded 1M, stopping count." << std::endl;
            break;
        }
    }
    return count;
}

int MCTS::get_max_depth(std::shared_ptr<MCTSNode> node) const {
    // Use iterative approach to avoid stack overflow
    int max_depth = 0;
    std::queue<std::shared_ptr<MCTSNode>> nodes_to_check;
    nodes_to_check.push(node);
    
    while (!nodes_to_check.empty()) {
        auto current = nodes_to_check.front();
        nodes_to_check.pop();
        max_depth = std::max(max_depth, current->depth);
        
        // Add children to queue
        for (const auto& child : current->children) {
            nodes_to_check.push(child);
        }
        
        // Safety check
        if (nodes_to_check.size() > 100000) { // Prevent excessive queue size
            std::cerr << "Warning: Queue size exceeded 100K, stopping depth calculation." << std::endl;
            break;
        }
    }
    return max_depth;
}

std::shared_ptr<MCTSNode> MCTS::search_root_parallel(int thread_count) {
    // Use instance's thread count if not specified
    if (thread_count <= 0) {
        thread_count = this->num_threads;
    }
    
    // Ensure we have at least one thread
    thread_count = std::max(1, thread_count);
    
    if (debug) {
        std::cout << "Starting root parallelization with " << thread_count << " threads..." << std::endl;
    }
    
    // If only one thread, just use sequential search
    if (thread_count == 1) {
        if (debug) {
            std::cout << "Using single thread - reverting to standard search" << std::endl;
        }
        
        // Sequential search - this is the existing method
        for (int sim = 0; sim < n_simulations; ++sim) {
            if (debug && sim % 50 == 0) {
                std::cout << "Iteration " << sim << "/" << n_simulations << std::endl;
            }
            search_single_iteration();
        }
        
        return root->best_child(0.0f);
    }
    
    // Divide simulations across threads
    int sims_per_thread = n_simulations / thread_count;
    
    // Need at least 1 simulation per thread
    if (sims_per_thread < 1) {
        sims_per_thread = 1;
        if (debug) {
            std::cout << "Warning: More threads than simulations. Setting minimum 1 simulation per thread." << std::endl;
        }
    }
    
    if (debug) {
        std::cout << "Running " << sims_per_thread << " simulations per thread" << std::endl;
        std::cout << "Total simulations to run: " << (sims_per_thread * thread_count) << std::endl;
    }
    
    // Create vector to store thread results
    std::vector<std::future<std::shared_ptr<MCTSNode>>> tree_futures;
    std::mutex debug_mutex; // For synchronized debug output
    
    // Launch threads - each with its own MCTS instance
    for (int t = 0; t < thread_count; t++) {
        tree_futures.push_back(std::async(std::launch::async, [this, t, sims_per_thread, &debug_mutex]() {
            // Make a deep copy of the root state for this thread
            ProblemState thread_state = this->root->state.copy();
            
            // Create a new root node for this thread
            auto thread_root = std::make_shared<MCTSNode>(thread_state);
            
            // Use different random seed for each thread
            std::mt19937 thread_rng(std::random_device{}() + t * 1000);
            
            // Create thread-specific MCTS instance
            MCTS thread_mcts(thread_root, this->depth, sims_per_thread, false); // Debug off for workers
            
            // Print thread start message
            if (this->debug) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                std::cout << "Thread " << t << " started with " << sims_per_thread << " simulations" << std::endl;
            }
            
            // Run thread-specific search (fully independent)
            thread_mcts.search();
            
            // Print thread completion message
            if (this->debug) {
                std::lock_guard<std::mutex> lock(debug_mutex);
                std::cout << "Thread " << t << " completed search with " 
                          << thread_mcts.count_total_nodes() << " nodes" << std::endl;
            }
            
            // Return the thread's result (just the root node after search)
            return thread_root;
        }));
    }
    
    if (debug) {
        std::cout << "All threads launched, waiting for completion..." << std::endl;
    }
    
    // Collect results from all threads
    std::vector<std::shared_ptr<MCTSNode>> thread_roots;
    for (auto& future : tree_futures) {
        thread_roots.push_back(future.get());
    }
    
    if (debug) {
        std::cout << "All threads completed successfully." << std::endl;
        std::cout << "Merging results from " << thread_roots.size() << " trees..." << std::endl;
    }
    
    // Create a map to merge results from all threads by action
    std::map<std::pair<std::string, std::vector<int>>, std::pair<float, int>> merged_results;
    
    // Process each thread's root node
    for (const auto& thread_root : thread_roots) {
        // For each child of the thread's root
        for (const auto& child : thread_root->children) {
            // Combine statistics by action
            const auto& action = child->action;
            auto& stats = merged_results[action];
            stats.first += child->total_reward;  // Sum rewards
            stats.second += child->visits;       // Sum visits
        }
    }
    
    if (debug) {
        std::cout << "Found " << merged_results.size() << " unique actions across all threads" << std::endl;
    }
    
    // Transfer the merged statistics back to the main root's children
    for (const auto& merged_action : merged_results) {
        const auto& action = merged_action.first;
        const auto& stats = merged_action.second;
        
        // Find if this action exists in the main root
        auto existing_child = std::find_if(
            root->children.begin(), root->children.end(),
            [&action](const std::shared_ptr<MCTSNode>& child) {
                return child->action == action;
            }
        );
        
        if (existing_child != root->children.end()) {
            // Update existing child's statistics
            (*existing_child)->total_reward = stats.first;
            (*existing_child)->visits = stats.second;
        } else {
            // Create a new child if it doesn't exist in main root
            ProblemState new_state = root->state.copy();
            new_state.apply_action(action.first, action.second);
            auto new_child = std::make_shared<MCTSNode>(new_state, root, action);
            new_child->total_reward = stats.first;
            new_child->visits = stats.second;
            root->children.push_back(new_child);
        }
    }
    
    // Ensure the root node's visit count is correct (sum of children)
    root->visits = 0;
    float total_reward = 0.0f;
    for (const auto& child : root->children) {
        root->visits += child->visits;
        total_reward += child->total_reward;
    }
    root->total_reward = total_reward;
    
    if (debug) {
        std::cout << "Root parallelization complete." << std::endl;
        std::cout << "Final tree: " << root->children.size() << " root children, " 
                  << root->visits << " total visits." << std::endl;
    }
    
    // Return the best child based on visits
    auto best_child = root->best_child(0.0f);
    
    if (debug && best_child) {
        std::cout << "Best action: " << best_child->action.first 
                  << ", value: " << (best_child->total_reward / best_child->visits)
                  << ", visits: " << best_child->visits << std::endl;
    }
    
    return best_child;
}
