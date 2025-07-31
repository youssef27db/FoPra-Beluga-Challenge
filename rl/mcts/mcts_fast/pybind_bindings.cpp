#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "Jig.hpp"
#include "Beluga.hpp"
#include "Rack.hpp"
#include "ProductionLine.hpp"
#include "ProblemState.hpp"
#include "Action.hpp"
#include "MCTSNode.hpp"
#include "MCTS.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mcts_fast, m) {
    m.doc() = "Python bindings for the Beluga problem state classes";

    // JigType class
    py::class_<JigType>(m, "JigType")
        .def(py::init<const std::string&, int, int>(),
             "Constructor", py::arg("name"), py::arg("size_empty"), py::arg("size_loaded"))
        .def(py::init<>(), "Default constructor")
        .def_readwrite("name", &JigType::name, "Name of the jig type")
        .def_readwrite("size_empty", &JigType::size_empty, "Size when empty")
        .def_readwrite("size_loaded", &JigType::size_loaded, "Size when loaded")
        .def("__eq__", &JigType::operator==, "Equality operator")
        .def("__ne__", &JigType::operator!=, "Inequality operator")
        .def("__str__", &JigType::to_string, "String representation")
        .def("__repr__", &JigType::to_string, "String representation");

    // Jig class
    py::class_<Jig>(m, "Jig")
        .def(py::init<const JigType&, bool>(),
             "Constructor", py::arg("jig_type"), py::arg("empty"))
        .def(py::init<>(), "Default constructor")
        .def_readwrite("jig_type", &Jig::jig_type, "Type of the jig")
        .def_readwrite("empty", &Jig::empty, "Whether the jig is empty")
        .def("copy", &Jig::copy, "Create a copy of the jig")
        .def("__str__", &Jig::to_string, "String representation")
        .def("__repr__", &Jig::to_string, "String representation");

    // Beluga class
    py::class_<Beluga>(m, "Beluga")
        .def(py::init<const std::vector<int>&, const std::vector<JigType>&>(),
             "Constructor", py::arg("current_jigs"), py::arg("outgoing"))
        .def(py::init<>(), "Default constructor")
        .def_readwrite("current_jigs", &Beluga::current_jigs, "Current jig IDs")
        .def_readwrite("outgoing", &Beluga::outgoing, "Outgoing jig types")
        .def("copy", &Beluga::copy, "Create a copy of the beluga")
        .def("__str__", &Beluga::to_string, "String representation")
        .def("__repr__", &Beluga::to_string, "String representation");

    // Rack class
    py::class_<Rack>(m, "Rack")
        .def(py::init<int, const std::vector<int>&>(),
             "Constructor", py::arg("size"), py::arg("current_jigs"))
        .def(py::init<>(), "Default constructor")
        .def_readwrite("size", &Rack::size, "Size capacity of the rack")
        .def_readwrite("current_jigs", &Rack::current_jigs, "Current jig IDs in the rack")
        .def("get_free_space", &Rack::get_free_space, "Get free space in the rack",
             py::arg("all_jigs"))
        .def("copy", &Rack::copy, "Create a copy of the rack")
        .def("__str__", &Rack::to_string, "String representation")
        .def("__repr__", &Rack::to_string, "String representation");

    // ProductionLine class
    py::class_<ProductionLine>(m, "ProductionLine")
        .def(py::init<const std::vector<int>&>(),
             "Constructor", py::arg("scheduled_jigs"))
        .def(py::init<>(), "Default constructor")
        .def_readwrite("scheduled_jigs", &ProductionLine::scheduled_jigs, "Scheduled jig IDs")
        .def("copy", &ProductionLine::copy, "Create a copy of the production line")
        .def("__str__", &ProductionLine::to_string, "String representation")
        .def("__repr__", &ProductionLine::to_string, "String representation");

    // ProblemState class
    py::class_<ProblemState>(m, "ProblemState")
        .def(py::init<const std::vector<Jig>&, const std::vector<Beluga>&,
                      const std::vector<int>&, const std::vector<int>&,
                      const std::vector<Rack>&, const std::vector<ProductionLine>&,
                      const std::vector<int>&>(),
             "Constructor",
             py::arg("jigs"), py::arg("belugas"), py::arg("trailers_beluga"),
             py::arg("trailers_factory"), py::arg("racks"), py::arg("production_lines"),
             py::arg("hangars"))
        .def(py::init<>(), "Default constructor")
        .def_static("load_from_json", &ProblemState::load_from_json, "Lädt einen Zustand aus einer JSON-Datei.")
        
        // State variables - using property for Copy-on-Write access
        .def_property_readonly("jigs", &ProblemState::jigs, "List of jigs")
        .def_property_readonly("belugas", &ProblemState::belugas, "List of belugas")
        .def_property_readonly("trailers_beluga", &ProblemState::trailers_beluga, "Beluga trailers")
        .def_property_readonly("trailers_factory", &ProblemState::trailers_factory, "Factory trailers")
        .def_property_readonly("racks", &ProblemState::racks, "List of racks")
        .def_property_readonly("production_lines", &ProblemState::production_lines, "Production lines")
        .def_property_readonly("hangars", &ProblemState::hangars, "List of hangars")
        
        // Subgoal counters
        .def_readwrite("belugas_unloaded", &ProblemState::belugas_unloaded, "Belugas unloaded counter")
        .def_readwrite("belugas_finished", &ProblemState::belugas_finished, "Belugas finished counter")
        .def_readwrite("production_lines_finished", &ProblemState::production_lines_finished, "Production lines finished counter")
        .def_readwrite("total_lines", &ProblemState::total_lines, "Total production lines")
        .def_readwrite("total_belugas", &ProblemState::total_belugas, "Total belugas")
        .def_readwrite("problem_solved", &ProblemState::problem_solved, "Problem solved flag")
        
        // MCTS API methods
        .def("clone", &ProblemState::clone, "Create a deep copy of the state")
        .def("copy", &ProblemState::copy, "Create a copy of the state (alias for clone)")
        .def("is_terminal", &ProblemState::is_terminal, "Check if state is terminal")
        .def("evaluate", &ProblemState::evaluate, "Evaluate the state",
             py::arg("depth"), py::arg("mu") = 0.05f)
        .def("get_subgoals", &ProblemState::get_subgoals, "Get subgoals for evaluation")
        
        // Action methods
        .def("apply_action", &ProblemState::apply_action, "Apply an action to the state",
             py::arg("action_name"), py::arg("params"))
        .def("check_action_valid", &ProblemState::check_action_valid, "Check if action is valid",
             py::arg("action_name"), py::arg("params"))
        .def("enumerate_valid_params", &ProblemState::enumerate_valid_params,
             "Enumerate valid parameters for an action", py::arg("action"))
        .def("get_possible_actions", &ProblemState::get_possible_actions,
             "Get all possible actions with parameters")
        
        // Other methods
        .def("beluga_complete", &ProblemState::beluga_complete, "Mark beluga as complete")
        .def("get_observation_high_level", &ProblemState::get_observation_high_level,
             "Get observation for high-level agents")
        
        // Operators
        .def("__eq__", &ProblemState::operator==, "Equality operator")
        .def("__str__", &ProblemState::to_string, "String representation")
        .def("__repr__", &ProblemState::to_string, "String representation")
        .def("__hash__", &ProblemState::hash, "Hash function");

    // Utility functions
    // These were likely part of the original Python code but not implemented in C++
    // Commenting them out for now
    // m.def("get_type", &get_type_from_string, "Get jig type by name", py::arg("name"));
    // m.def("extract_id", &extract_id_from_string, "Extract jig ID from a string (e.g., 'jig001' -> 0)", py::arg("id_str"));
    m.def("get_name_from_id", [](int id) -> std::string {
        // Implementation matching the Python version
        std::string id_str = std::to_string(id + 1);  // Convert back to 1-based
        // Pad with leading zeros to make it 4 digits total
        std::string padded = std::string(4 - std::min(4, static_cast<int>(id_str.length())), '0') + id_str;
        return "jig" + padded;
    }, "Get name from ID (e.g., 0 -> 'jig0001')", py::arg("id"));

    // Action functions
    py::module action_module = m.def_submodule("Action", "Action functions for state transitions");
    action_module.def("load_beluga", &Action::load_beluga, "Load beluga from specific trailer",
                     py::arg("state"), py::arg("trailer_beluga"), py::arg("none") = -1);
    action_module.def("unload_beluga", &Action::unload_beluga, "Unload beluga",
                     py::arg("state"));
    action_module.def("get_from_hangar", &Action::get_from_hangar, "Get jig from hangar to trailer",
                     py::arg("state"), py::arg("hangar"), py::arg("trailer_factory"));
    action_module.def("deliver_to_hangar", &Action::deliver_to_hangar, "Deliver jig from trailer to hangar",
                     py::arg("state"), py::arg("hangar"), py::arg("trailer_factory"));
    action_module.def("left_stack_rack", &Action::left_stack_rack, "Stack jig on rack from left trailer",
                     py::arg("state"), py::arg("rack"), py::arg("trailer_id"));
    action_module.def("right_stack_rack", &Action::right_stack_rack, "Stack jig on rack from right trailer",
                     py::arg("state"), py::arg("rack"), py::arg("trailer_id"));
    action_module.def("left_unstack_rack", &Action::left_unstack_rack, "Unstack jig from rack to left trailer",
                     py::arg("state"), py::arg("rack"), py::arg("trailer_id"));
    action_module.def("right_unstack_rack", &Action::right_unstack_rack, "Unstack jig from rack to right trailer",
                     py::arg("state"), py::arg("rack"), py::arg("trailer_id"));

    // MCTSNode class
    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def(py::init<const ProblemState&, std::shared_ptr<MCTSNode>, 
                      const std::pair<std::string, std::vector<int>>&, int>(),
             "Constructor",
             py::arg("state"), py::arg("parent") = nullptr, 
             py::arg("action") = std::make_pair(std::string(""), std::vector<int>{}), 
             py::arg("depth") = 0)
        
        // Public members
        .def_readwrite("state", &MCTSNode::state, "Problem state")
        .def_property("parent",
            [](const MCTSNode& node) -> std::shared_ptr<MCTSNode> {
                return node.parent.lock();  // Convert weak_ptr to shared_ptr for Python
            },
            [](MCTSNode& node, std::shared_ptr<MCTSNode> parent) {
                node.parent = parent;  // Store the shared_ptr as weak_ptr
            },
            "Parent node")
        .def_readwrite("action", &MCTSNode::action, "Action taken to reach this node")
        .def_readwrite("depth", &MCTSNode::depth, "Depth in tree")
        .def_readwrite("children", &MCTSNode::children, "Child nodes")
        .def_readwrite("visits", &MCTSNode::visits, "Number of visits")
        .def_readwrite("total_reward", &MCTSNode::total_reward, "Total reward")
        
        // Methods
        .def("is_root", &MCTSNode::is_root, "Check if this is root node")
        .def("is_terminal", &MCTSNode::is_terminal, "Check if this is terminal node")
        .def("is_fully_expanded", &MCTSNode::is_fully_expanded, "Check if fully expanded")
        .def("get_untried_actions", &MCTSNode::get_untried_actions, "Get untried actions")
        .def("get_untried_action", &MCTSNode::get_untried_actions, "Get untried actions (alias for Python compatibility)")
        .def("expand", &MCTSNode::expand, "Expand with given action", py::arg("candidate"))
        .def("add_child", &MCTSNode::add_child, "Add child node", py::arg("child"))
        .def("best_child", &MCTSNode::best_child, "Get best child using UCT",
             py::arg("exploration_weight") = 1.0f)
        .def("backpropagate", &MCTSNode::backpropagate, "Backpropagate reward", py::arg("reward"));

    // MCTS class
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<std::shared_ptr<MCTSNode>, int, int, bool, int>(),
             "Constructor",
             py::arg("root"), py::arg("depth") = 5, py::arg("n_simulations") = 300, py::arg("debug") = false,
             py::arg("num_threads") = 0)
        
        // Public members
        .def_readwrite("root", &MCTS::root, "Root node")
        .def_readwrite("depth", &MCTS::depth, "Maximum search depth")
        .def_readwrite("n_simulations", &MCTS::n_simulations, "Number of simulations")
        .def_readwrite("debug", &MCTS::debug, "Debug flag")
        .def_readwrite("num_threads", &MCTS::num_threads, "Number of threads for parallelization")
        
        // Methods
        .def("search", &MCTS::search, "Perform MCTS search")
        .def("search_root_parallel", &MCTS::search_root_parallel, "Perform root-parallel MCTS search", 
             py::arg("thread_count") = 0)
        .def("set_num_threads", &MCTS::set_num_threads, "Set number of threads for parallelization", 
             py::arg("threads"))
        .def("select", &MCTS::select, "Select node for expansion", py::arg("node"))
        .def("rollout", &MCTS::rollout, "Perform rollout simulation", py::arg("node"))
        .def("get_best_path", &MCTS::get_best_path, "Get best path from root")
        .def("collect_training_data", &MCTS::collect_training_data, "Collect training data from tree",
             py::arg("max_samples") = 1000, py::arg("min_visits") = 5)
        .def("count_total_nodes", &MCTS::count_total_nodes, "Count total nodes in tree")
        .def("get_tree_depth", &MCTS::get_tree_depth, "Get maximum tree depth")
        .def("search_single_iteration", &MCTS::search_single_iteration, "Perform single MCTS iteration");
}
