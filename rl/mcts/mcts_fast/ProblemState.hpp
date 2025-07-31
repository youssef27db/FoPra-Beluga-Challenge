#pragma once
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <unordered_map>
#include <memory> // For std::shared_ptr
#include "Jig.hpp"
#include "Beluga.hpp"
#include "Rack.hpp"
#include "ProductionLine.hpp"

/**
 * @class ProblemState
 * @brief Represents the state of the problem for MCTS.
 *
 * This class encapsulates the state of the environment, including jigs, belugas,
 * racks, production lines, and hangars. It provides methods for cloning, evaluating,
 * and applying actions, as well as checking terminal states.
 * 
 * Implementation uses Copy-on-Write (CoW) pattern with std::shared_ptr to minimize
 * unnecessary deep copies and improve MCTS performance.
 */
class ProblemState {
public:
    // State variables are now private, access through methods
private:
    std::shared_ptr<std::vector<Jig>> _jigs;
    std::shared_ptr<std::vector<Beluga>> _belugas;
    std::shared_ptr<std::vector<int>> _trailers_beluga;   // -1 for None, otherwise jig ID
    std::shared_ptr<std::vector<int>> _trailers_factory;  // -1 for None, otherwise jig ID
    std::shared_ptr<std::vector<Rack>> _racks;
    std::shared_ptr<std::vector<ProductionLine>> _production_lines;
    std::shared_ptr<std::vector<int>> _hangars;          // -1 for None, otherwise jig ID

public:
    // Accessor methods for backward compatibility
    const std::vector<Jig>& jigs() const { return *_jigs; }
    const std::vector<Beluga>& belugas() const { return *_belugas; }
    const std::vector<int>& trailers_beluga() const { return *_trailers_beluga; }
    const std::vector<int>& trailers_factory() const { return *_trailers_factory; }
    const std::vector<Rack>& racks() const { return *_racks; }
    const std::vector<ProductionLine>& production_lines() const { return *_production_lines; }
    const std::vector<int>& hangars() const { return *_hangars; }

    // Subgoals for evaluation
    int belugas_unloaded;
    int belugas_finished;
    int production_lines_finished;
    int total_lines;
    int total_belugas;
    bool problem_solved;

    /**
     * @brief Constructor for ProblemState.
     * @param jigs List of jigs in the environment.
     * @param belugas List of belugas in the environment.
     * @param trailers_beluga List of trailers for belugas.
     * @param trailers_factory List of trailers for the factory.
     * @param racks List of racks in the environment.
     * @param production_lines List of production lines in the environment.
     * @param hangars List of hangars in the environment.
     */
    ProblemState(const std::vector<Jig>& jigs,
                 const std::vector<Beluga>& belugas,
                 const std::vector<int>& trailers_beluga,
                 const std::vector<int>& trailers_factory,
                 const std::vector<Rack>& racks,
                 const std::vector<ProductionLine>& production_lines,
                 const std::vector<int>& hangars);

    /**
     * @brief Default constructor.
     */
    ProblemState();

    /**
     * @brief Copy constructor - implements copy-on-write, very fast.
     */
    ProblemState(const ProblemState& other) = default;

    /**
     * @brief Assignment operator - implements copy-on-write, very fast.
     */
    ProblemState& operator=(const ProblemState& other) = default;
    
    /**
     * @brief Helper template to ensure a vector is unique before modification
     * @param vec The shared_ptr to the vector to make unique if necessary
     */
    template<typename T>
    void ensure_unique(std::shared_ptr<std::vector<T>>& vec) {
        if (!vec.unique()) {
            vec = std::make_shared<std::vector<T>>(*vec);
        }
    }
    
    /**
     * @brief Helper template to ensure all vectors are unique before modification
     * Call this before any operation that modifies the state
     */
    void ensure_all_unique() {
        ensure_unique(_jigs);
        ensure_unique(_belugas);
        ensure_unique(_trailers_beluga);
        ensure_unique(_trailers_factory);
        ensure_unique(_racks);
        ensure_unique(_production_lines);
        ensure_unique(_hangars);
    }
    
    // Modifier methods - these handle the CoW pattern
    void set_jig(size_t index, const Jig& jig);
    void set_beluga(size_t index, const Beluga& beluga);
    void set_trailer_beluga(size_t index, int value);
    void set_trailer_factory(size_t index, int value);
    void set_rack(size_t index, const Rack& rack);
    void set_production_line(size_t index, const ProductionLine& line);
    void set_hangar(size_t index, int value);
    
    // Vector modification methods
    void add_jig(const Jig& jig);
    void remove_jig(size_t index);
    void add_beluga(const Beluga& beluga);
    void remove_beluga(size_t index);
    void add_rack(const Rack& rack);
    void remove_rack(size_t index);
    void add_production_line(const ProductionLine& line);
    void remove_production_line(size_t index);

    /**
     * @brief Creates a "copy" of the current state - with CoW this is very fast.
     * @return A new ProblemState object that shares data with the current state.
     */
    ProblemState clone() const;

    /**
     * @brief Creates a "copy" of the current state (alias for clone).
     * @return A new ProblemState object that shares data with the current state.
     */
    ProblemState copy() const;

    /**
     * @brief Checks if the current state is terminal.
     * @return True if the state is terminal, false otherwise.
     */
    bool is_terminal() const;

    /**
     * @brief Evaluates the current state.
     * @param depth The depth of the current state in the MCTS tree.
     * @param mu A penalty factor for deeper states.
     * @return A float score representing the evaluation of the state.
     */
    float evaluate(int depth, float mu = 0.05f) const;

    /**
     * @brief Gets the subgoals for evaluation.
     * @return A map of subgoal names to their scores.
     */
    std::unordered_map<std::string, float> get_subgoals() const;

    /**
     * @brief Applies an action to the state.
     * @param action_name The name of the action to apply.
     * @param params The parameters for the action.
     * @return True if the action was successfully applied, false otherwise.
     */
    bool apply_action(const std::string& action_name, const std::vector<int>& params);

    /**
     * @brief Checks if an action is valid without modifying the state.
     * @param action_name The name of the action to check.
     * @param params The parameters for the action.
     * @return True if the action is valid, false otherwise.
     */
    bool check_action_valid(const std::string& action_name, const std::vector<int>& params) const;

    /**
     * @brief Enumerates all valid parameter combinations for a given action.
     * @param action The action name.
     * @return A vector of valid parameter combinations.
     */
    std::vector<std::vector<int>> enumerate_valid_params(const std::string& action) const;

    /**
     * @brief Gets all possible actions with their parameters.
     * @return A vector of pairs containing action names and their parameters.
     */
    std::vector<std::pair<std::string, std::vector<int>>> get_possible_actions() const;

    /**
     * @brief Marks a beluga as complete and removes it.
     * @return True if a beluga was completed, false otherwise.
     */
    bool beluga_complete();

    /**
     * @brief Gets the observation for high-level agents.
     * @return A vector representing the current state observation.
     */
    std::vector<float> get_observation_high_level() const;

    /**
     * @brief String representation of the problem state.
     */
    std::string to_string() const;

    /**
     * @brief Equality operator.
     */
    bool operator==(const ProblemState& other) const;

    /**
     * @brief Hash function for the state.
     */
    size_t hash() const;

    /**
     * @brief Load state from JSON file.
     */
    static ProblemState load_from_json(const std::string& path);
};