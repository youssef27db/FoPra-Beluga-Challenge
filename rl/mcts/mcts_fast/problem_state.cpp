#include "problem_state.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace mcts_fast {

ProblemState::ProblemState(
    const std::vector<Jig>& jigs,
    const std::vector<Beluga>& belugas,
    const std::vector<std::optional<JigId>>& trailers_beluga,
    const std::vector<std::optional<JigId>>& trailers_factory,
    const std::vector<Rack>& racks,
    const std::vector<ProductionLine>& production_lines,
    const std::vector<std::optional<JigId>>& hangars
) : jigs_(jigs), belugas_(belugas), trailers_beluga_(trailers_beluga),
    trailers_factory_(trailers_factory), racks_(racks), production_lines_(production_lines),
    hangars_(hangars), belugas_unloaded_(0), belugas_finished_(0),
    production_lines_finished_(0), total_lines_(production_lines.size()),
    total_belugas_(belugas.size()), problem_solved_(false) {
}

ProblemState::ProblemState(const ProblemState& other) 
    : jigs_(other.jigs_), belugas_(other.belugas_), trailers_beluga_(other.trailers_beluga_),
      trailers_factory_(other.trailers_factory_), racks_(other.racks_), 
      production_lines_(other.production_lines_), hangars_(other.hangars_),
      belugas_unloaded_(other.belugas_unloaded_), belugas_finished_(other.belugas_finished_),
      production_lines_finished_(other.production_lines_finished_), 
      total_lines_(other.total_lines_), total_belugas_(other.total_belugas_),
      problem_solved_(other.problem_solved_) {
}

ProblemState& ProblemState::operator=(const ProblemState& other) {
    if (this != &other) {
        jigs_ = other.jigs_;
        belugas_ = other.belugas_;
        trailers_beluga_ = other.trailers_beluga_;
        trailers_factory_ = other.trailers_factory_;
        racks_ = other.racks_;
        production_lines_ = other.production_lines_;
        hangars_ = other.hangars_;
        belugas_unloaded_ = other.belugas_unloaded_;
        belugas_finished_ = other.belugas_finished_;
        production_lines_finished_ = other.production_lines_finished_;
        total_lines_ = other.total_lines_;
        total_belugas_ = other.total_belugas_;
        problem_solved_ = other.problem_solved_;
    }
    return *this;
}

ProblemState ProblemState::copy() const {
    return ProblemState(*this);
}

bool ProblemState::isTerminal() const {
    return belugas_.empty() && production_lines_.empty();
}

double ProblemState::evaluate(int depth, double mu) const {
    double score = 0.0;
    auto subgoals = getSubgoals();
    for (const auto& pair : subgoals) {
        score += pair.second;
    }
    score -= mu * depth;
    return score;
}

std::unordered_map<std::string, double> ProblemState::getSubgoals() const {
    belugas_finished_ = total_belugas_ - static_cast<int>(belugas_.size());
    production_lines_finished_ = total_lines_ - static_cast<int>(production_lines_.size());
    
    if (belugas_.empty() && production_lines_.empty()) {
        problem_solved_ = true;
    }
    
    return {
        {"subgoal_1", belugas_unloaded_ * 15.0},
        {"subgoal_2", belugas_finished_ * 60.0},
        {"subgoal_3", production_lines_finished_ * 100.0},
        {"goal", problem_solved_ ? 1000.0 : 0.0}
    };
}

// Helper validation functions
bool ProblemState::isValidTrailerBeluga(TrailerId trailer_id) const {
    return trailer_id >= 0 && trailer_id < static_cast<int>(trailers_beluga_.size());
}

bool ProblemState::isValidTrailerFactory(TrailerId trailer_id) const {
    return trailer_id >= 0 && trailer_id < static_cast<int>(trailers_factory_.size());
}

bool ProblemState::isValidRack(RackId rack_id) const {
    return rack_id >= 0 && rack_id < static_cast<int>(racks_.size());
}

bool ProblemState::isValidHangar(HangarId hangar_id) const {
    return hangar_id >= 0 && hangar_id < static_cast<int>(hangars_.size());
}

// Action implementations
bool ProblemState::loadBeluga(TrailerId trailer_id) {
    if (!isValidTrailerBeluga(trailer_id) || belugas_.empty()) {
        return false;
    }
    
    if (!trailers_beluga_[trailer_id].has_value()) {
        return false;
    }
    
    JigId jig_id = trailers_beluga_[trailer_id].value();
    if (jig_id < 0 || jig_id >= static_cast<int>(jigs_.size())) {
        return false;
    }
    
    Beluga& beluga = belugas_[0];
    
    if (!jigs_[jig_id].isEmpty()) {
        return false;
    }
    
    if (beluga.getOutgoing().empty()) {
        return false;
    }
    
    if (!beluga.getCurrentJigs().empty()) {
        return false;
    }
    
    if (jigs_[jig_id].getJigType() != beluga.getOutgoing()[0]) {
        return false;
    }
    
    // Effects
    if (beluga.getOutgoing().size() == 1) {
        beluga.getOutgoing().clear();
        trailers_beluga_[trailer_id] = std::nullopt;
        belugaComplete();
        return true;
    }
    
    beluga.getOutgoing().erase(beluga.getOutgoing().begin());
    trailers_beluga_[trailer_id] = std::nullopt;
    return true;
}

bool ProblemState::unloadBeluga() {
    // Find first empty trailer slot
    TrailerId trailer_beluga = -1;
    for (size_t i = 0; i < trailers_beluga_.size(); ++i) {
        if (!trailers_beluga_[i].has_value()) {
            trailer_beluga = static_cast<TrailerId>(i);
            break;
        }
    }
    
    if (trailer_beluga == -1 || belugas_.empty()) {
        return false;
    }
    
    Beluga& beluga = belugas_[0];
    if (beluga.getCurrentJigs().empty()) {
        return false;
    }
    
    JigId jig_id = beluga.getCurrentJigs().back();
    beluga.getCurrentJigs().pop_back();
    
    if (beluga.getCurrentJigs().size() == 0) {
        belugas_unloaded_++;
        trailers_beluga_[trailer_beluga] = jig_id;
        
        if (beluga.getOutgoing().empty()) {
            belugaComplete();
        }
        return true;
    }
    
    trailers_beluga_[trailer_beluga] = jig_id;
    return true;
}

bool ProblemState::getFromHangar(HangarId hangar_id, TrailerId trailer_id) {
    if (!isValidHangar(hangar_id) || !isValidTrailerFactory(trailer_id)) {
        return false;
    }
    
    if (!hangars_[hangar_id].has_value() || trailers_factory_[trailer_id].has_value()) {
        return false;
    }
    
    JigId jig_id = hangars_[hangar_id].value();
    if (jig_id < 0 || jig_id >= static_cast<int>(jigs_.size())) {
        return false;
    }
    
    if (!jigs_[jig_id].isEmpty()) {
        return false;
    }
    
    // Effects
    trailers_factory_[trailer_id] = jig_id;
    hangars_[hangar_id] = std::nullopt;
    return true;
}

bool ProblemState::deliverToHangar(HangarId hangar_id, TrailerId trailer_id) {
    if (!isValidHangar(hangar_id) || !isValidTrailerFactory(trailer_id)) {
        return false;
    }
    
    if (hangars_[hangar_id].has_value() || !trailers_factory_[trailer_id].has_value()) {
        return false;
    }
    
    JigId jig_id = trailers_factory_[trailer_id].value();
    if (jig_id < 0 || jig_id >= static_cast<int>(jigs_.size())) {
        return false;
    }
    
    if (jigs_[jig_id].isEmpty()) {
        return false;
    }
    
    // Find corresponding production line
    int production_line_idx = -1;
    for (size_t i = 0; i < production_lines_.size(); ++i) {
        if (!production_lines_[i].getScheduledJigs().empty() && 
            jig_id == production_lines_[i].getScheduledJigs()[0]) {
            production_line_idx = static_cast<int>(i);
            break;
        }
    }
    
    if (production_line_idx == -1) {
        return false;
    }
    
    // Effects
    production_lines_[production_line_idx].getScheduledJigs().erase(
        production_lines_[production_line_idx].getScheduledJigs().begin());
    hangars_[hangar_id] = jig_id;
    jigs_[jig_id].setEmpty(true);
    trailers_factory_[trailer_id] = std::nullopt;
    
    if (production_lines_[production_line_idx].getScheduledJigs().empty()) {
        production_lines_.erase(production_lines_.begin() + production_line_idx);
    }
    
    return true;
}

bool ProblemState::leftStackRack(RackId rack_id, TrailerId trailer_id) {
    if (!isValidRack(rack_id) || !isValidTrailerBeluga(trailer_id)) {
        return false;
    }
    
    if (!trailers_beluga_[trailer_id].has_value()) {
        return false;
    }
    
    JigId jig_id = trailers_beluga_[trailer_id].value();
    if (jig_id < 0 || jig_id >= static_cast<int>(jigs_.size())) {
        return false;
    }
    
    Rack& rack = racks_[rack_id];
    if (!rack.canFitJig(jigs_[jig_id], jigs_)) {
        return false;
    }
    
    // Effects
    trailers_beluga_[trailer_id] = std::nullopt;
    rack.getCurrentJigs().insert(rack.getCurrentJigs().begin(), jig_id);
    return true;
}

bool ProblemState::rightStackRack(RackId rack_id, TrailerId trailer_id) {
    if (!isValidRack(rack_id) || !isValidTrailerFactory(trailer_id)) {
        return false;
    }
    
    if (!trailers_factory_[trailer_id].has_value()) {
        return false;
    }
    
    JigId jig_id = trailers_factory_[trailer_id].value();
    if (jig_id < 0 || jig_id >= static_cast<int>(jigs_.size())) {
        return false;
    }
    
    Rack& rack = racks_[rack_id];
    if (!rack.canFitJig(jigs_[jig_id], jigs_)) {
        return false;
    }
    
    // Effects
    trailers_factory_[trailer_id] = std::nullopt;
    rack.getCurrentJigs().push_back(jig_id);
    return true;
}

bool ProblemState::leftUnstackRack(RackId rack_id, TrailerId trailer_id) {
    if (!isValidRack(rack_id) || !isValidTrailerBeluga(trailer_id)) {
        return false;
    }
    
    if (trailers_beluga_[trailer_id].has_value() || racks_[rack_id].getCurrentJigs().empty()) {
        return false;
    }
    
    // Effects
    JigId jig_id = racks_[rack_id].getCurrentJigs().front();
    racks_[rack_id].getCurrentJigs().erase(racks_[rack_id].getCurrentJigs().begin());
    trailers_beluga_[trailer_id] = jig_id;
    return true;
}

bool ProblemState::rightUnstackRack(RackId rack_id, TrailerId trailer_id) {
    if (!isValidRack(rack_id) || !isValidTrailerFactory(trailer_id)) {
        return false;
    }
    
    if (trailers_factory_[trailer_id].has_value() || racks_[rack_id].getCurrentJigs().empty()) {
        return false;
    }
    
    // Effects
    JigId jig_id = racks_[rack_id].getCurrentJigs().back();
    racks_[rack_id].getCurrentJigs().pop_back();
    trailers_factory_[trailer_id] = jig_id;
    return true;
}

bool ProblemState::belugaComplete() {
    if (belugas_.empty()) {
        return false;
    }
    
    Beluga& beluga = belugas_[0];
    if (!beluga.getOutgoing().empty() || !beluga.getCurrentJigs().empty()) {
        return false;
    }
    
    // Effects
    belugas_.erase(belugas_.begin());
    return true;
}

bool ProblemState::checkActionValid(const std::string& action_name, const ActionParams& params) const {
    ProblemState copy_state = this->copy();
    return copy_state.applyAction(action_name, params);
}

bool ProblemState::applyAction(const std::string& action_name, const ActionParams& params) {
    try {
        if (action_name == "left_stack_rack" && params.size() >= 2) {
            return leftStackRack(params[0], params[1]);
        } else if (action_name == "right_stack_rack" && params.size() >= 2) {
            return rightStackRack(params[0], params[1]);
        } else if (action_name == "left_unstack_rack" && params.size() >= 2) {
            return leftUnstackRack(params[0], params[1]);
        } else if (action_name == "right_unstack_rack" && params.size() >= 2) {
            return rightUnstackRack(params[0], params[1]);
        } else if (action_name == "load_beluga" && params.size() >= 1) {
            return loadBeluga(params[0]);
        } else if (action_name == "unload_beluga") {
            return unloadBeluga();
        } else if (action_name == "get_from_hangar" && params.size() >= 2) {
            return getFromHangar(params[0], params[1]);
        } else if (action_name == "deliver_to_hangar" && params.size() >= 2) {
            return deliverToHangar(params[0], params[1]);
        }
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<ActionParams> ProblemState::enumerateValidParams(const std::string& action_name) const {
    std::vector<ActionParams> params;
    
    if (action_name == "left_stack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(racks_.size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_beluga_.size()); ++trailer_id) {
                ActionParams param = {rack_id, trailer_id};
                if (checkActionValid(action_name, param)) {
                    params.push_back(param);
                }
            }
        }
    } else if (action_name == "right_stack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(racks_.size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_factory_.size()); ++trailer_id) {
                ActionParams param = {rack_id, trailer_id};
                if (checkActionValid(action_name, param)) {
                    params.push_back(param);
                }
            }
        }
    } else if (action_name == "left_unstack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(racks_.size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_beluga_.size()); ++trailer_id) {
                ActionParams param = {rack_id, trailer_id};
                if (checkActionValid(action_name, param)) {
                    params.push_back(param);
                }
            }
        }
    } else if (action_name == "right_unstack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(racks_.size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_factory_.size()); ++trailer_id) {
                ActionParams param = {rack_id, trailer_id};
                if (checkActionValid(action_name, param)) {
                    params.push_back(param);
                }
            }
        }
    } else if (action_name == "load_beluga") {
        for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_beluga_.size()); ++trailer_id) {
            ActionParams param = {trailer_id};
            if (checkActionValid(action_name, param)) {
                params.push_back(param);
            }
        }
    } else if (action_name == "deliver_to_hangar") {
        for (int hangar_id = 0; hangar_id < static_cast<int>(hangars_.size()); ++hangar_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_factory_.size()); ++trailer_id) {
                ActionParams param = {hangar_id, trailer_id};
                if (checkActionValid(action_name, param)) {
                    params.push_back(param);
                }
            }
        }
    } else if (action_name == "get_from_hangar") {
        for (int hangar_id = 0; hangar_id < static_cast<int>(hangars_.size()); ++hangar_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(trailers_factory_.size()); ++trailer_id) {
                ActionParams param = {hangar_id, trailer_id};
                if (checkActionValid(action_name, param)) {
                    params.push_back(param);
                }
            }
        }
    }
    
    return params;
}

std::vector<ActionTuple> ProblemState::getPossibleActions() const {
    std::vector<ActionTuple> possible_actions;
    
    // Check unload_beluga (no parameters)
    if (checkActionValid("unload_beluga", {})) {
        possible_actions.emplace_back("unload_beluga", ActionParams{});
    }
    
    // Check actions with parameters
    std::vector<std::string> param_actions = {
        "left_stack_rack", "right_stack_rack", "left_unstack_rack",
        "right_unstack_rack", "load_beluga", "get_from_hangar", "deliver_to_hangar"
    };
    
    for (const auto& action : param_actions) {
        auto params_list = enumerateValidParams(action);
        for (const auto& params : params_list) {
            possible_actions.emplace_back(action, params);
        }
    }
    
    return possible_actions;
}

std::string ProblemState::toString() const {
    std::ostringstream oss;
    oss << "jigs:\n";
    for (size_t i = 0; i < jigs_.size(); ++i) {
        oss << "\t" << i << ": " << jigs_[i].toString() << "\n";
    }
    oss << "belugas:\n";
    for (size_t i = 0; i < belugas_.size(); ++i) {
        oss << "\t" << i << ": " << belugas_[i].toString() << "\n";
    }
    oss << "trailers_beluga: [";
    for (size_t i = 0; i < trailers_beluga_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (trailers_beluga_[i].has_value() ? std::to_string(trailers_beluga_[i].value()) : "None");
    }
    oss << "]\n";
    oss << "trailers_factory: [";
    for (size_t i = 0; i < trailers_factory_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (trailers_factory_[i].has_value() ? std::to_string(trailers_factory_[i].value()) : "None");
    }
    oss << "]\n";
    oss << "racks:\n";
    for (size_t i = 0; i < racks_.size(); ++i) {
        oss << "\t" << i << ": " << racks_[i].toString() << "\n";
    }
    oss << "production_lines:\n";
    for (size_t i = 0; i < production_lines_.size(); ++i) {
        oss << "\t" << i << ": " << production_lines_[i].toString() << "\n";
    }
    oss << "hangars: [";
    for (size_t i = 0; i < hangars_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << (hangars_[i].has_value() ? std::to_string(hangars_[i].value()) : "None");
    }
    oss << "]";
    return oss.str();
}

} // namespace mcts_fast