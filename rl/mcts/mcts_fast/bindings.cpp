#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/operators.h>

#include "types.h"
#include "jig.h"
#include "beluga.h"
#include "rack.h"
#include "production_line.h"
#include "problem_state.h"
#include "mcts_node.h"
#include "mcts.h"

namespace py = pybind11;
using namespace mcts_fast;

PYBIND11_MODULE(mcts_fast, m) {
    m.doc() = "Fast C++ implementation of MCTS for the Beluga Challenge";
    
    // JigType class
    py::class_<JigType>(m, "JigType")
        .def(py::init<const std::string&, int, int>())
        .def("getName", &JigType::getName)
        .def("getSizeEmpty", &JigType::getSizeEmpty)
        .def("getSizeLoaded", &JigType::getSizeLoaded)
        .def("toString", &JigType::toString)
        .def(py::self == py::self)
        .def(py::self != py::self);
    
    // Jig class
    py::class_<Jig>(m, "Jig")
        .def(py::init<const JigType&, bool>())
        .def("getJigType", &Jig::getJigType, py::return_value_policy::reference)
        .def("isEmpty", &Jig::isEmpty)
        .def("setEmpty", &Jig::setEmpty)
        .def("getSize", &Jig::getSize)
        .def("toString", &Jig::toString);
    
    // Beluga class
    py::class_<Beluga>(m, "Beluga")
        .def(py::init<const std::vector<JigId>&, const std::vector<JigType>&>())
        .def("getCurrentJigs", 
             static_cast<const std::vector<JigId>& (Beluga::*)() const>(&Beluga::getCurrentJigs),
             py::return_value_policy::reference)
        .def("getOutgoing", 
             static_cast<const std::vector<JigType>& (Beluga::*)() const>(&Beluga::getOutgoing),
             py::return_value_policy::reference)
        .def("addCurrentJig", &Beluga::addCurrentJig)
        .def("removeCurrentJig", &Beluga::removeCurrentJig)
        .def("addOutgoing", &Beluga::addOutgoing)
        .def("removeOutgoing", &Beluga::removeOutgoing)
        .def("isEmpty", &Beluga::isEmpty)
        .def("toString", &Beluga::toString);
    
    // Rack class
    py::class_<Rack>(m, "Rack")
        .def(py::init<int, const std::vector<JigId>&>())
        .def("getSize", &Rack::getSize)
        .def("getCurrentJigs", 
             static_cast<const std::vector<JigId>& (Rack::*)() const>(&Rack::getCurrentJigs),
             py::return_value_policy::reference)
        .def("addJig", &Rack::addJig)
        .def("removeJig", &Rack::removeJig)
        .def("removeTopJig", &Rack::removeTopJig)
        .def("getTopJig", &Rack::getTopJig)
        .def("hasJigs", &Rack::hasJigs)
        .def("getFreeSpace", &Rack::getFreeSpace)
        .def("canFitJig", &Rack::canFitJig)
        .def("toString", &Rack::toString);
    
    // ProductionLine class
    py::class_<ProductionLine>(m, "ProductionLine")
        .def(py::init<const std::vector<JigId>&>())
        .def("getScheduledJigs", 
             static_cast<const std::vector<JigId>& (ProductionLine::*)() const>(&ProductionLine::getScheduledJigs),
             py::return_value_policy::reference)
        .def("addScheduledJig", &ProductionLine::addScheduledJig)
        .def("removeScheduledJig", &ProductionLine::removeScheduledJig)
        .def("removeFirstScheduledJig", &ProductionLine::removeFirstScheduledJig)
        .def("getNextJig", &ProductionLine::getNextJig)
        .def("hasScheduledJigs", &ProductionLine::hasScheduledJigs)
        .def("isEmpty", &ProductionLine::isEmpty)
        .def("toString", &ProductionLine::toString);
    
    // ProblemState class
    py::class_<ProblemState>(m, "ProblemState")
        .def(py::init<
             const std::vector<Jig>&,
             const std::vector<Beluga>&,
             const std::vector<std::optional<JigId>>&,
             const std::vector<std::optional<JigId>>&,
             const std::vector<Rack>&,
             const std::vector<ProductionLine>&,
             const std::vector<std::optional<JigId>>&>())
        .def("copy", &ProblemState::copy)
        .def("isTerminal", &ProblemState::isTerminal)
        .def("evaluate", &ProblemState::evaluate, 
             py::arg("depth"), py::arg("mu") = 0.05)
        .def("getPossibleActions", &ProblemState::getPossibleActions)
        .def("enumerateValidParams", &ProblemState::enumerateValidParams)
        .def("checkActionValid", &ProblemState::checkActionValid)
        .def("applyAction", &ProblemState::applyAction)
        .def("getSubgoals", &ProblemState::getSubgoals)
        .def("toString", &ProblemState::toString)
        // Getters
        .def("getJigs", 
             static_cast<const std::vector<Jig>& (ProblemState::*)() const>(&ProblemState::getJigs),
             py::return_value_policy::reference)
        .def("getBelugas", 
             static_cast<const std::vector<Beluga>& (ProblemState::*)() const>(&ProblemState::getBelugas),
             py::return_value_policy::reference)
        .def("getTrailersBeluga", 
             static_cast<const std::vector<std::optional<JigId>>& (ProblemState::*)() const>(&ProblemState::getTrailersBeluga),
             py::return_value_policy::reference)
        .def("getTrailersFactory", 
             static_cast<const std::vector<std::optional<JigId>>& (ProblemState::*)() const>(&ProblemState::getTrailersFactory),
             py::return_value_policy::reference)
        .def("getRacks", 
             static_cast<const std::vector<Rack>& (ProblemState::*)() const>(&ProblemState::getRacks),
             py::return_value_policy::reference)
        .def("getProductionLines", 
             static_cast<const std::vector<ProductionLine>& (ProblemState::*)() const>(&ProblemState::getProductionLines),
             py::return_value_policy::reference)
        .def("getHangars", 
             static_cast<const std::vector<std::optional<JigId>>& (ProblemState::*)() const>(&ProblemState::getHangars),
             py::return_value_policy::reference);
    
    // MCTSNode class
    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def(py::init<const ProblemState&, std::shared_ptr<MCTSNode>, 
                      const std::optional<ActionTuple>&, int>(),
             py::arg("state"), py::arg("parent") = nullptr, 
             py::arg("action") = std::nullopt, py::arg("depth") = 0)
        .def("getState", &MCTSNode::getState, py::return_value_policy::reference)
        .def("getAction", &MCTSNode::getAction)
        .def("getDepth", &MCTSNode::getDepth)
        .def("getVisits", &MCTSNode::getVisits)
        .def("getTotalReward", &MCTSNode::getTotalReward)
        .def("getParent", &MCTSNode::getParent)
        .def("getChildren", &MCTSNode::getChildren, py::return_value_policy::reference)
        .def("addChild", &MCTSNode::addChild)
        .def("isRoot", &MCTSNode::isRoot)
        .def("isTerminal", &MCTSNode::isTerminal)
        .def("isFullyExpanded", &MCTSNode::isFullyExpanded)
        .def("getUntriedActions", &MCTSNode::getUntriedActions)
        .def("expand", &MCTSNode::expand)
        .def("bestChild", &MCTSNode::bestChild, py::arg("exploration_weight") = 1.0)
        .def("backpropagate", &MCTSNode::backpropagate)
        .def("getUCTValue", &MCTSNode::getUCTValue, 
             py::arg("exploration_weight") = 1.0, py::arg("parent_visits") = 1)
        .def("toString", &MCTSNode::toString);
    
    // MCTS class
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<std::shared_ptr<MCTSNode>, int, int, bool>(),
             py::arg("root"), py::arg("depth") = 5, 
             py::arg("n_simulations") = 300, py::arg("debug") = false)
        .def("search", &MCTS::search)
        .def("select", &MCTS::select)
        .def("rollout", &MCTS::rollout)
        .def("getBestPath", &MCTS::getBestPath)
        .def("getRoot", &MCTS::getRoot)
        .def("getDepth", &MCTS::getDepth)
        .def("getSimulations", &MCTS::getSimulations)
        .def("isDebug", &MCTS::isDebug);
    
    // Utility functions
    m.def("createJigType", &createJigType, "Create a JigType from type name");
    
    // Type bindings for STL containers
    py::bind_vector<std::vector<JigId>>(m, "VectorJigId");
    py::bind_vector<std::vector<JigType>>(m, "VectorJigType");
    py::bind_vector<std::vector<Jig>>(m, "VectorJig");
    py::bind_vector<std::vector<Beluga>>(m, "VectorBeluga");
    py::bind_vector<std::vector<Rack>>(m, "VectorRack");
    py::bind_vector<std::vector<ProductionLine>>(m, "VectorProductionLine");
    py::bind_vector<std::vector<std::optional<JigId>>>(m, "VectorOptionalJigId");
    py::bind_vector<std::vector<ActionParams>>(m, "VectorActionParams");
    py::bind_vector<std::vector<ActionTuple>>(m, "VectorActionTuple");
    py::bind_vector<std::vector<std::shared_ptr<MCTSNode>>>(m, "VectorMCTSNode");
}