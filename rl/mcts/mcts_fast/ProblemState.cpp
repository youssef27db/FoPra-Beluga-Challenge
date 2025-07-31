#include "ProblemState.hpp"
#include "Action.hpp"
#include <sstream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <fstream>
#include <string>
#include "json.hpp"  // Use the local json.hpp file

ProblemState::ProblemState()
    : _jigs(std::make_shared<std::vector<Jig>>()),
      _belugas(std::make_shared<std::vector<Beluga>>()),
      _trailers_beluga(std::make_shared<std::vector<int>>()),
      _trailers_factory(std::make_shared<std::vector<int>>()),
      _racks(std::make_shared<std::vector<Rack>>()),
      _production_lines(std::make_shared<std::vector<ProductionLine>>()),
      _hangars(std::make_shared<std::vector<int>>()),
      belugas_unloaded(0), belugas_finished(0), production_lines_finished(0),
      total_lines(0), total_belugas(0), problem_solved(false) {}

ProblemState::ProblemState(const std::vector<Jig>& jigs,
                           const std::vector<Beluga>& belugas,
                           const std::vector<int>& trailers_beluga,
                           const std::vector<int>& trailers_factory,
                           const std::vector<Rack>& racks,
                           const std::vector<ProductionLine>& production_lines,
                           const std::vector<int>& hangars)
    : _jigs(std::make_shared<std::vector<Jig>>(jigs)),
      _belugas(std::make_shared<std::vector<Beluga>>(belugas)),
      _trailers_beluga(std::make_shared<std::vector<int>>(trailers_beluga)),
      _trailers_factory(std::make_shared<std::vector<int>>(trailers_factory)),
      _racks(std::make_shared<std::vector<Rack>>(racks)),
      _production_lines(std::make_shared<std::vector<ProductionLine>>(production_lines)),
      _hangars(std::make_shared<std::vector<int>>(hangars)),
      belugas_unloaded(0), belugas_finished(0), production_lines_finished(0),
      total_lines(production_lines.size()), total_belugas(belugas.size()),
      problem_solved(false) {}

// Accessor methods for modifying elements - ensure CoW pattern
void ProblemState::set_jig(size_t index, const Jig& jig) {
    if (index >= _jigs->size()) return;
    ensure_unique(_jigs);
    (*_jigs)[index] = jig;
}

void ProblemState::set_beluga(size_t index, const Beluga& beluga) {
    if (index >= _belugas->size()) return;
    ensure_unique(_belugas);
    (*_belugas)[index] = beluga;
}

void ProblemState::set_trailer_beluga(size_t index, int value) {
    if (index >= _trailers_beluga->size()) return;
    ensure_unique(_trailers_beluga);
    (*_trailers_beluga)[index] = value;
}

void ProblemState::set_trailer_factory(size_t index, int value) {
    if (index >= _trailers_factory->size()) return;
    ensure_unique(_trailers_factory);
    (*_trailers_factory)[index] = value;
}

void ProblemState::set_rack(size_t index, const Rack& rack) {
    if (index >= _racks->size()) return;
    ensure_unique(_racks);
    (*_racks)[index] = rack;
}

void ProblemState::set_production_line(size_t index, const ProductionLine& line) {
    if (index >= _production_lines->size()) return;
    ensure_unique(_production_lines);
    (*_production_lines)[index] = line;
}

void ProblemState::set_hangar(size_t index, int value) {
    if (index >= _hangars->size()) return;
    ensure_unique(_hangars);
    (*_hangars)[index] = value;
}

// Vector modification methods
void ProblemState::add_jig(const Jig& jig) {
    ensure_unique(_jigs);
    _jigs->push_back(jig);
}

void ProblemState::remove_jig(size_t index) {
    if (index >= _jigs->size()) return;
    ensure_unique(_jigs);
    _jigs->erase(_jigs->begin() + index);
}

void ProblemState::add_beluga(const Beluga& beluga) {
    ensure_unique(_belugas);
    _belugas->push_back(beluga);
}

void ProblemState::remove_beluga(size_t index) {
    if (index >= _belugas->size()) return;
    ensure_unique(_belugas);
    _belugas->erase(_belugas->begin() + index);
}

void ProblemState::add_rack(const Rack& rack) {
    ensure_unique(_racks);
    _racks->push_back(rack);
}

void ProblemState::remove_rack(size_t index) {
    if (index >= _racks->size()) return;
    ensure_unique(_racks);
    _racks->erase(_racks->begin() + index);
}

void ProblemState::add_production_line(const ProductionLine& line) {
    ensure_unique(_production_lines);
    _production_lines->push_back(line);
}

void ProblemState::remove_production_line(size_t index) {
    if (index >= _production_lines->size()) return;
    ensure_unique(_production_lines);
    _production_lines->erase(_production_lines->begin() + index);
}

ProblemState ProblemState::clone() const {
    // With copy-on-write, this is just a shallow copy of shared_ptrs
    return ProblemState(*this);
}

ProblemState ProblemState::copy() const {
    return clone();
}

bool ProblemState::is_terminal() const {
    // Note the use of -> operator to access the shared_ptr's vector
    return _belugas->empty() && _production_lines->empty();
}

float ProblemState::evaluate(int depth, float mu) const {
    // Direct calculation instead of creating unordered_map
    int local_belugas_finished = total_belugas - _belugas->size();
    int local_production_lines_finished = total_lines - _production_lines->size();
    bool local_problem_solved = _belugas->empty() && _production_lines->empty();
    
    float score = 0.0f;
    score += static_cast<float>(belugas_unloaded) * 15.0f;           // subgoal_1
    score += static_cast<float>(local_belugas_finished) * 60.0f;     // subgoal_2  
    score += static_cast<float>(local_production_lines_finished) * 100.0f; // subgoal_3
    score += local_problem_solved ? 1000.0f : 0.0f;                 // goal
    
    // Penalty for depth
    score -= mu * depth;
    return score;
}

// Neue Version, die keine Member mehr ändert und const ist
std::unordered_map<std::string, float> ProblemState::get_subgoals() const {
    // Werte lokal berechnen, anstatt Member zu verändern
    int local_belugas_finished = total_belugas - _belugas->size();
    int local_production_lines_finished = total_lines - _production_lines->size();
    bool local_problem_solved = _belugas->empty() && _production_lines->empty();
    
    return {
        {"subgoal_1", static_cast<float>(belugas_unloaded) * 15.0f},
        {"subgoal_2", static_cast<float>(local_belugas_finished) * 60.0f},
        {"subgoal_3", static_cast<float>(local_production_lines_finished) * 100.0f},
        {"goal", local_problem_solved ? 1000.0f : 0.0f}
    };
}

bool ProblemState::apply_action(const std::string& action_name, const std::vector<int>& params) {
    // Ensure all vectors are unique before modifying any of them
    ensure_all_unique();
    
    // Use switch-like optimization with hash or enum
    // For now, optimize string comparisons with early returns and likely actions first
    if (action_name[0] == 'l') {  // left_* actions
        if (action_name == "left_stack_rack" && params.size() >= 2) {
            return Action::left_stack_rack(*this, params[0], params[1]);
        } else if (action_name == "left_unstack_rack" && params.size() >= 2) {
            return Action::left_unstack_rack(*this, params[0], params[1]);
        } else if (action_name == "load_beluga" && params.size() >= 1) {
            return Action::load_beluga(*this, params[0], params.size() > 1 ? params[1] : -1);
        }
    } else if (action_name[0] == 'r') {  // right_* actions
        if (action_name == "right_stack_rack" && params.size() >= 2) {
            return Action::right_stack_rack(*this, params[0], params[1]);
        } else if (action_name == "right_unstack_rack" && params.size() >= 2) {
            return Action::right_unstack_rack(*this, params[0], params[1]);
        }
    } else if (action_name[0] == 'u') {  // unload_beluga
        if (action_name == "unload_beluga") {
            return Action::unload_beluga(*this);
        }
    } else if (action_name[0] == 'g') {  // get_from_hangar
        if (action_name == "get_from_hangar" && params.size() >= 2) {
            return Action::get_from_hangar(*this, params[0], params[1]);
        }
    } else if (action_name[0] == 'd') {  // deliver_to_hangar
        if (action_name == "deliver_to_hangar" && params.size() >= 2) {
            return Action::deliver_to_hangar(*this, params[0], params[1]);
        }
    }
    return false;
}

bool ProblemState::check_action_valid(const std::string& action_name, const std::vector<int>& params) const {
    ProblemState state_copy = copy();
    return const_cast<ProblemState&>(state_copy).apply_action(action_name, params);
}

std::vector<std::vector<int>> ProblemState::enumerate_valid_params(const std::string& action) const {
    std::vector<std::vector<int>> params;
    
    // Fast path: direct validation for common actions
    // If we get wrong results, we fall back to check_action_valid()
    
    if (action == "left_stack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(_racks->size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_beluga->size()); ++trailer_id) {
                // Quick pre-check to avoid expensive validation
                if ((*_trailers_beluga)[trailer_id] != -1) {
                    int jig_id = (*_trailers_beluga)[trailer_id];
                    if (jig_id >= 0 && jig_id < static_cast<int>(_jigs->size())) {
                        const Jig& jig = (*_jigs)[jig_id];
                        int jig_size = jig.empty ? jig.jig_type.size_empty : jig.jig_type.size_loaded;
                        if ((*_racks)[rack_id].get_free_space(*_jigs) >= jig_size) {
                            // Fast validation passed, but let's double-check with full validation
                            std::vector<int> candidate = {rack_id, trailer_id};
                            if (check_action_valid(action, candidate)) {
                                params.push_back(candidate);
                            }
                        }
                    }
                }
            }
        }
    } else if (action == "right_stack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(_racks->size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_factory->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_factory)[trailer_id] != -1) {
                    int jig_id = (*_trailers_factory)[trailer_id];
                    if (jig_id >= 0 && jig_id < static_cast<int>(_jigs->size())) {
                        const Jig& jig = (*_jigs)[jig_id];
                        int jig_size = jig.empty ? jig.jig_type.size_empty : jig.jig_type.size_loaded;
                        if ((*_racks)[rack_id].get_free_space(*_jigs) >= jig_size) {
                            std::vector<int> candidate = {rack_id, trailer_id};
                            if (check_action_valid(action, candidate)) {
                                params.push_back(candidate);
                            }
                        }
                    }
                }
            }
        }
    } else if (action == "left_unstack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(_racks->size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_beluga->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_beluga)[trailer_id] == -1 && !(*_racks)[rack_id].current_jigs.empty()) {
                    std::vector<int> candidate = {rack_id, trailer_id};
                    if (check_action_valid(action, candidate)) {
                        params.push_back(candidate);
                    }
                }
            }
        }
    } else if (action == "right_unstack_rack") {
        for (int rack_id = 0; rack_id < static_cast<int>(_racks->size()); ++rack_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_factory->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_factory)[trailer_id] == -1 && !(*_racks)[rack_id].current_jigs.empty()) {
                    std::vector<int> candidate = {rack_id, trailer_id};
                    if (check_action_valid(action, candidate)) {
                        params.push_back(candidate);
                    }
                }
            }
        }
    } else if (action == "left_load") {
        for (int line_id = 0; line_id < static_cast<int>(_production_lines->size()); ++line_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_beluga->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_beluga)[trailer_id] != -1) {
                    int jig_id = (*_trailers_beluga)[trailer_id];
                    if (jig_id >= 0 && jig_id < static_cast<int>(_jigs->size()) && 
                        (*_jigs)[jig_id].empty && 
                        (*_production_lines)[line_id].scheduled_jigs.empty()) {
                        std::vector<int> candidate = {line_id, trailer_id};
                        if (check_action_valid(action, candidate)) {
                            params.push_back(candidate);
                        }
                    }
                }
            }
        }
    } else if (action == "right_load") {
        for (int line_id = 0; line_id < static_cast<int>(_production_lines->size()); ++line_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_factory->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_factory)[trailer_id] != -1) {
                    int jig_id = (*_trailers_factory)[trailer_id];
                    if (jig_id >= 0 && jig_id < static_cast<int>(_jigs->size()) && 
                        (*_jigs)[jig_id].empty && 
                        (*_production_lines)[line_id].scheduled_jigs.empty()) {
                        std::vector<int> candidate = {line_id, trailer_id};
                        if (check_action_valid(action, candidate)) {
                            params.push_back(candidate);
                        }
                    }
                }
            }
        }
    } else if (action == "left_unload") {
        for (int line_id = 0; line_id < static_cast<int>(_production_lines->size()); ++line_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_beluga->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_beluga)[trailer_id] == -1 && 
                    !(*_production_lines)[line_id].scheduled_jigs.empty()) {
                    std::vector<int> candidate = {line_id, trailer_id};
                    if (check_action_valid(action, candidate)) {
                        params.push_back(candidate);
                    }
                }
            }
        }
    } else if (action == "right_unload") {
        for (int line_id = 0; line_id < static_cast<int>(_production_lines->size()); ++line_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_factory->size()); ++trailer_id) {
                // Quick pre-check
                if ((*_trailers_factory)[trailer_id] == -1 && 
                    !(*_production_lines)[line_id].scheduled_jigs.empty()) {
                    std::vector<int> candidate = {line_id, trailer_id};
                    if (check_action_valid(action, candidate)) {
                        params.push_back(candidate);
                    }
                }
            }
        }
    } else if (action == "deliver_to_hangar") {
        for (int hangar_id = 0; hangar_id < static_cast<int>(_hangars->size()); ++hangar_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_factory->size()); ++trailer_id) {
                if ((*_hangars)[hangar_id] == -1 && (*_trailers_factory)[trailer_id] != -1) {
                    int jig_id = (*_trailers_factory)[trailer_id];
                    if (jig_id >= 0 && jig_id < static_cast<int>(_jigs->size()) && !(*_jigs)[jig_id].empty) {
                        // Check if jig is needed by any production line
                        bool needed = false;
                        for (const auto& pl : *_production_lines) {
                            if (!pl.scheduled_jigs.empty() && pl.scheduled_jigs[0] == jig_id) {
                                needed = true;
                                break;
                            }
                        }
                        if (needed) {
                            params.push_back({hangar_id, trailer_id});
                        }
                    }
                }
            }
        }
    } else if (action == "get_from_hangar") {
        for (int hangar_id = 0; hangar_id < static_cast<int>(_hangars->size()); ++hangar_id) {
            for (int trailer_id = 0; trailer_id < static_cast<int>(_trailers_factory->size()); ++trailer_id) {
                if ((*_hangars)[hangar_id] != -1 && (*_trailers_factory)[trailer_id] == -1) {
                    int jig_id = (*_hangars)[hangar_id];
                    if (jig_id >= 0 && jig_id < static_cast<int>(_jigs->size()) && (*_jigs)[jig_id].empty) {
                        params.push_back({hangar_id, trailer_id});
                    }
                }
            }
        }
    } else if (action == "unload_beluga") {
        // Check if unload is possible
        if (!_belugas->empty()) {
            const Beluga& beluga = (*_belugas)[0];
            if (!beluga.current_jigs.empty()) {
                // Check if there's an empty trailer
                for (int i = 0; i < static_cast<int>(_trailers_beluga->size()); ++i) {
                    if ((*_trailers_beluga)[i] == -1) {
                        params.push_back({}); // No parameters needed
                        break;
                    }
                }
            }
        }
    }
    
    return params;
}

std::vector<std::pair<std::string, std::vector<int>>> ProblemState::get_possible_actions() const {
    std::vector<std::pair<std::string, std::vector<int>>> possible_actions;
    
    // Pre-allocate reasonable capacity to avoid multiple reallocations
    possible_actions.reserve(20);  // Estimated capacity
    
    // Check unload_beluga (no parameters)
    if (check_action_valid("unload_beluga", {})) {
        possible_actions.emplace_back("unload_beluga", std::vector<int>{});
    }
    
    // Add other actions with parameters using static array for better cache performance
    static const std::vector<std::string> param_actions = {
        "left_stack_rack", "right_stack_rack", "left_unstack_rack",
        "right_unstack_rack", "load_beluga", "get_from_hangar", "deliver_to_hangar"
    };
    
    for (const std::string& action : param_actions) {
        auto params = enumerate_valid_params(action);
        for (auto& param : params) {
            possible_actions.emplace_back(action, std::move(param));
        }
    }
    
    return possible_actions;
}

bool ProblemState::beluga_complete() {
    if (_belugas->empty()) {
        return false;
    }
    
    // We need to ensure the belugas vector is unique before modifying it
    ensure_unique(_belugas);
    
    Beluga& beluga = (*_belugas)[0];
    if (!beluga.outgoing.empty() || !beluga.current_jigs.empty()) {
        return false;
    }
    
    // Increment the completed belugas counter
    belugas_finished++;
    
    // Remove the beluga
    _belugas->erase(_belugas->begin());
    return true;
}

std::vector<float> ProblemState::get_observation_high_level() const {
    const int n_racks = 10;
    std::vector<float> out(10 + 3 * n_racks, 0.0f);
    
    std::vector<JigType> needed_outgoing_types;
    std::vector<int> needed_in_production_lines;
    
    for (const auto& pl : *_production_lines) {
        if (!pl.scheduled_jigs.empty()) {
            needed_in_production_lines.push_back(pl.scheduled_jigs[0]);
        }
    }
    
    // First slot 0 beluga
    if (!_belugas->empty()) {
        out[0] = std::max(0.0f, std::min(static_cast<float>((*_belugas)[0].current_jigs.size()), 1.0f));
        if (out[0] == 0) {
            needed_outgoing_types = (*_belugas)[0].outgoing;
        }
    } else {
        out[0] = -1;
    }
    
    // Slots 1-3 Beluga Trailer
    for (int i = 0; i < 3; ++i) {
        if (i < static_cast<int>(_trailers_beluga->size())) {
            if ((*_trailers_beluga)[i] == -1) {
                out[1 + i] = 0.5f;
            } else {
                const Jig& jig = (*_jigs)[(*_trailers_beluga)[i]];
                if (jig.empty && out[0] == 0) {
                    bool found = std::find(needed_outgoing_types.begin(), needed_outgoing_types.end(), 
                                         jig.jig_type) != needed_outgoing_types.end();
                    out[1 + i] = found ? 0.0f : 0.25f;
                } else {
                    out[1 + i] = 1.0f;
                }
            }
        } else {
            out[1 + i] = -1;
        }
    }
    
    // Continue with other slots implementation...
    // This is a simplified version. Full implementation would mirror the Python code.
    
    return out;
}

std::string ProblemState::to_string() const {
    std::stringstream ss;
    ss << "jigs:\n";
    for (size_t i = 0; i < _jigs->size(); ++i) {
        ss << "\t" << i << ": " << (*_jigs)[i].to_string() << "\n";
    }
    ss << "belugas:\n";
    for (size_t i = 0; i < _belugas->size(); ++i) {
        ss << "\t" << i << ": " << (*_belugas)[i].to_string() << "\n";
    }
    // Add other components...
    return ss.str();
}

bool ProblemState::operator==(const ProblemState& other) const {
    return to_string() == other.to_string();
}

size_t ProblemState::hash() const {
    return std::hash<std::string>{}(to_string());
}



JigType get_type_from_string(const std::string& type_str) {
    if (type_str == "typeA") return { "typeA", 4, 4 };
    if (type_str == "typeB") return { "typeB", 8, 11 };
    if (type_str == "typeC") return { "typeC", 9, 18 };
    if (type_str == "typeD") return { "typeD", 18, 25 };
    if (type_str == "typeE") return { "typeE", 32, 32 };
    
    // Wenn kein Typ passt, wird ein Fehler geworfen.
    throw std::runtime_error("Unbekannter Jig-Typ: " + type_str);
}

// Diese Funktion extrahiert die ID aus einem String wie "jig1".
int extract_id_from_string(const std::string& id_str) {
    std::string prefix = "jig";
    // Prüfen, ob der String mit "jig" beginnt
    if (id_str.rfind(prefix, 0) == 0) {
        // Extrahiere den Teil nach "jig"
        std::string number_part = id_str.substr(prefix.length());
        // Konvertiere zu einer Ganzzahl und subtrahiere 1
        return std::stoi(number_part) - 1;
    }
    
    // Wenn das Format unerwartet ist, wird ein Fehler geworfen.
    throw std::runtime_error("Konnte ID nicht aus String extrahieren: " + id_str);
}


// === Implementierung der statischen Methode ===

ProblemState ProblemState::load_from_json(const std::string& path) {
    // 1. Datei öffnen und lesen
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Konnte Datei nicht öffnen: " + path);
    }
    // Explizite Verwendung von nlohmann::json
    nlohmann::json data = nlohmann::json::parse(file);
    file.close();

    // 2. Jigs erstellen
    std::vector<Jig> jigs;
    // In C++ iteriert man über Key-Value-Paare in einem JSON-Objekt so:
    for (auto const& [key, val] : data["jigs"].items()) {
        jigs.emplace_back(
            get_type_from_string(val["type"]),
            val["empty"]
        );
    }

    // 3. Belugas erstellen
    std::vector<Beluga> belugas;
    for (const auto& beluga_data : data["flights"]) {
        std::vector<int> incoming;
        for (const auto& entry : beluga_data["incoming"]) {
            incoming.push_back(extract_id_from_string(entry));
        }

        std::vector<JigType> outgoing;
        for (const auto& entry : beluga_data["outgoing"]) {
            outgoing.push_back(get_type_from_string(entry));
        }
        belugas.emplace_back(incoming, outgoing);
    }

    // 4. ProductionLines erstellen
    std::vector<ProductionLine> production_lines;
    for (const auto& pl_data : data["production_lines"]) {
        std::vector<int> schedule;
        for (const auto& entry : pl_data["schedule"]) {
            schedule.push_back(extract_id_from_string(entry));
        }
        production_lines.emplace_back(schedule);
    }

    // 5. Racks erstellen
    std::vector<Rack> racks;
    for (const auto& rack_data : data["racks"]) {
        std::vector<int> storage;
        for (const auto& entry : rack_data["jigs"]) {
            storage.push_back(extract_id_from_string(entry));
        }
        racks.emplace_back(rack_data["size"], storage);
    }

    // 6. Trailer und Hangars initialisieren
    // In C++ verwenden wir -1, um "None" oder "leer" darzustellen.
    std::vector<int> trailers_beluga(data["trailers_beluga"].size(), -1);
    std::vector<int> trailers_factory(data["trailers_factory"].size(), -1);
    std::vector<int> hangars(data["hangars"].size(), -1);

    // 7. ProblemState-Objekt mit allen Daten zurückgeben
    return ProblemState(jigs, belugas, trailers_beluga, trailers_factory, racks, production_lines, hangars);
}