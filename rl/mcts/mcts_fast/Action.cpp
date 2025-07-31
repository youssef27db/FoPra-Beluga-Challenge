#include "Action.hpp"
#include <algorithm>
#include <vector> // Nötig für std::vector

// Hilfsfunktion zur Überprüfung von Indexgrenzen
template<typename T>
bool is_valid_index(int index, const std::vector<T>& vec) {
    return index >= 0 && static_cast<size_t>(index) < vec.size();
}

namespace Action {

bool load_beluga(ProblemState& state, int trailer_beluga, int none) {
    if (!is_valid_index(trailer_beluga, state.trailers_beluga())) return false;
    
    int jig_id = state.trailers_beluga()[trailer_beluga];
    if (jig_id == -1) return false;

    // Sicherheitscheck für jig_id
    if (!is_valid_index(jig_id, state.jigs())) return false;

    // ZUERST prüfen, ob Belugas überhaupt vorhanden sind
    if (state.belugas().empty()) return false;
    
    // JETZT ist es sicher, eine Referenz zu holen (wir müssen die Kopie holen, da wir später modifizieren)
    const Beluga& beluga = state.belugas()[0];

    // JETZT die restlichen Bedingungen prüfen
    if (!state.jigs()[jig_id].empty || beluga.outgoing.empty() || !beluga.current_jigs.empty()) {
        return false;
    }
    
    if (!(state.jigs()[jig_id].jig_type == beluga.outgoing[0])) {
        return false;
    }

    // Get a modifiable copy of the Beluga
    Beluga modifiedBeluga = state.belugas()[0];
    
    if (modifiedBeluga.outgoing.size() == 1) {
        modifiedBeluga.outgoing.erase(modifiedBeluga.outgoing.begin());
        state.set_beluga(0, modifiedBeluga);
        state.set_trailer_beluga(trailer_beluga, -1);
        state.beluga_complete();
        return true;
    }

    modifiedBeluga.outgoing.erase(modifiedBeluga.outgoing.begin());
    state.set_beluga(0, modifiedBeluga);
    state.set_trailer_beluga(trailer_beluga, -1);
    return true;
}

bool unload_beluga(ProblemState& state) {
    int trailer_beluga = -1;
    for (size_t i = 0; i < state.trailers_beluga().size(); ++i) {
        if (state.trailers_beluga()[i] == -1) {
            trailer_beluga = i;
            break;
        }
    }

    if (trailer_beluga == -1 || state.belugas().empty()) return false;

    const Beluga& beluga = state.belugas()[0];
    if (beluga.current_jigs.empty()) return false;

    // Create a modified copy
    Beluga modifiedBeluga = beluga;
    int jigToTransfer = modifiedBeluga.current_jigs.back();
    modifiedBeluga.current_jigs.pop_back();
    
    // Update state
    state.set_trailer_beluga(trailer_beluga, jigToTransfer);
    state.set_beluga(0, modifiedBeluga);

    if (modifiedBeluga.current_jigs.empty()) {
        state.belugas_unloaded += 1;
        if (modifiedBeluga.outgoing.empty()) {
            state.beluga_complete();
        }
    }
    
    return true;
}

bool get_from_hangar(ProblemState& state, int hangar, int trailer_factory) {
    if (!is_valid_index(hangar, state.hangars()) || !is_valid_index(trailer_factory, state.trailers_factory())) return false;
    if (state.hangars()[hangar] == -1 || state.trailers_factory()[trailer_factory] != -1) return false;

    int jig_id = state.hangars()[hangar];
    // Sicherheitscheck
    if (!is_valid_index(jig_id, state.jigs())) return false;
    if (!state.jigs()[jig_id].empty) return false;

    state.set_trailer_factory(trailer_factory, jig_id);
    state.set_hangar(hangar, -1);
    return true;
}

bool deliver_to_hangar(ProblemState& state, int hangar, int trailer_factory) {
    if (!is_valid_index(hangar, state.hangars()) || !is_valid_index(trailer_factory, state.trailers_factory())) return false;
    if (state.hangars()[hangar] != -1 || state.trailers_factory()[trailer_factory] == -1) return false;

    int jig_id = state.trailers_factory()[trailer_factory];
    // Sicherheitscheck
    if (!is_valid_index(jig_id, state.jigs())) return false;
    if (state.jigs()[jig_id].empty) return false;

    int production_line_idx = -1;
    for (size_t i = 0; i < state.production_lines().size(); ++i) {
        if (!state.production_lines()[i].scheduled_jigs.empty() && 
            state.production_lines()[i].scheduled_jigs[0] == jig_id) {
            production_line_idx = i;
            break;
        }
    }

    if (production_line_idx == -1) return false;

    // Create a modified production line
    ProductionLine modifiedLine = state.production_lines()[production_line_idx];
    modifiedLine.scheduled_jigs.erase(modifiedLine.scheduled_jigs.begin());
    
    // Create a modified jig
    Jig modifiedJig = state.jigs()[jig_id];
    modifiedJig.empty = true;
    
    // Update state
    state.set_production_line(production_line_idx, modifiedLine);
    state.set_hangar(hangar, jig_id);
    state.set_jig(jig_id, modifiedJig);
    state.set_trailer_factory(trailer_factory, -1);

    // Increment counter when a production line is finished
    if (modifiedLine.scheduled_jigs.empty()) {
        state.production_lines_finished += 1;
        state.remove_production_line(production_line_idx);
    }
    return true;
}

bool left_stack_rack(ProblemState& state, int rack, int trailer_id) {
    if (!is_valid_index(rack, state.racks()) || !is_valid_index(trailer_id, state.trailers_beluga())) return false;
    if (state.trailers_beluga()[trailer_id] == -1) return false;

    int jig_id = state.trailers_beluga()[trailer_id];
    // Sicherheitscheck
    if (!is_valid_index(jig_id, state.jigs())) return false;

    const Jig& jig = state.jigs()[jig_id];
    int jig_size = jig.empty ? jig.jig_type.size_empty : jig.jig_type.size_loaded;

    if (state.racks()[rack].get_free_space(state.jigs()) < jig_size) return false;

    // Optimize: modify in-place instead of creating copies
    state.set_trailer_beluga(trailer_id, -1);
    
    // Get current rack and modify it
    Rack rack_copy = state.racks()[rack];
    rack_copy.current_jigs.insert(rack_copy.current_jigs.begin(), jig_id);
    state.set_rack(rack, rack_copy);
    
    return true;
}

bool right_stack_rack(ProblemState& state, int rack, int trailer_id) {
    if (!is_valid_index(rack, state.racks()) || !is_valid_index(trailer_id, state.trailers_factory())) return false;
    if (state.trailers_factory()[trailer_id] == -1) return false;

    int jig_id = state.trailers_factory()[trailer_id];
    // Sicherheitscheck
    if (!is_valid_index(jig_id, state.jigs())) return false;

    const Jig& jig = state.jigs()[jig_id];
    int jig_size = jig.empty ? jig.jig_type.size_empty : jig.jig_type.size_loaded;

    if (state.racks()[rack].get_free_space(state.jigs()) < jig_size) return false;

    // Create a modified rack
    Rack modifiedRack = state.racks()[rack];
    modifiedRack.current_jigs.push_back(jig_id);
    
    // Update state
    state.set_trailer_factory(trailer_id, -1);
    state.set_rack(rack, modifiedRack);
    return true;
}

bool left_unstack_rack(ProblemState& state, int rack, int trailer_id) {
    if (!is_valid_index(rack, state.racks()) || !is_valid_index(trailer_id, state.trailers_beluga())) return false;
    if (state.trailers_beluga()[trailer_id] != -1 || state.racks()[rack].current_jigs.empty()) return false;

    // Create a modified rack
    Rack modifiedRack = state.racks()[rack];
    int jigToTransfer = modifiedRack.current_jigs.front();
    modifiedRack.current_jigs.erase(modifiedRack.current_jigs.begin());
    
    // Update state
    state.set_trailer_beluga(trailer_id, jigToTransfer);
    state.set_rack(rack, modifiedRack);
    return true;
}

bool right_unstack_rack(ProblemState& state, int rack, int trailer_id) {
    if (!is_valid_index(rack, state.racks()) || !is_valid_index(trailer_id, state.trailers_factory())) return false;
    if (state.trailers_factory()[trailer_id] != -1 || state.racks()[rack].current_jigs.empty()) return false;

    // Create a modified rack
    Rack modifiedRack = state.racks()[rack];
    int jigToTransfer = modifiedRack.current_jigs.back();
    modifiedRack.current_jigs.pop_back();
    
    // Update state
    state.set_trailer_factory(trailer_id, jigToTransfer);
    state.set_rack(rack, modifiedRack);
    return true;
}

// left_stack_rack, right_stack_rack, left_unstack_rack, and right_unstack_rack are already defined above

} // namespace Action
