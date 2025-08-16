#pragma once
#include "types.h"
#include "jig.h"
#include "beluga.h"
#include "rack.h"
#include "production_line.h"
#include <vector>
#include <unordered_map>
#include <optional>

namespace mcts_fast {

class ProblemState {
public:
    ProblemState(
        const std::vector<Jig>& jigs,
        const std::vector<Beluga>& belugas,
        const std::vector<std::optional<JigId>>& trailers_beluga,
        const std::vector<std::optional<JigId>>& trailers_factory,
        const std::vector<Rack>& racks,
        const std::vector<ProductionLine>& production_lines,
        const std::vector<std::optional<JigId>>& hangars
    );
    
    // Getters
    const std::vector<Jig>& getJigs() const { return jigs_; }
    const std::vector<Beluga>& getBelugas() const { return belugas_; }
    const std::vector<std::optional<JigId>>& getTrailersBeluga() const { return trailers_beluga_; }
    const std::vector<std::optional<JigId>>& getTrailersFactory() const { return trailers_factory_; }
    const std::vector<Rack>& getRacks() const { return racks_; }
    const std::vector<ProductionLine>& getProductionLines() const { return production_lines_; }
    const std::vector<std::optional<JigId>>& getHangars() const { return hangars_; }
    
    // Mutable getters
    std::vector<Jig>& getJigs() { return jigs_; }
    std::vector<Beluga>& getBelugas() { return belugas_; }
    std::vector<std::optional<JigId>>& getTrailersBeluga() { return trailers_beluga_; }
    std::vector<std::optional<JigId>>& getTrailersFactory() { return trailers_factory_; }
    std::vector<Rack>& getRacks() { return racks_; }
    std::vector<ProductionLine>& getProductionLines() { return production_lines_; }
    std::vector<std::optional<JigId>>& getHangars() { return hangars_; }
    
    // MCTS API
    ProblemState copy() const;
    bool isTerminal() const;
    double evaluate(int depth, double mu = 0.05) const;
    std::vector<ActionTuple> getPossibleActions() const;
    std::vector<ActionParams> enumerateValidParams(const std::string& action_name) const;
    bool checkActionValid(const std::string& action_name, const ActionParams& params) const;
    bool applyAction(const std::string& action_name, const ActionParams& params);
    
    // Action implementations
    bool loadBeluga(TrailerId trailer_id);
    bool unloadBeluga();
    bool getFromHangar(HangarId hangar_id, TrailerId trailer_id);
    bool deliverToHangar(HangarId hangar_id, TrailerId trailer_id);
    bool leftStackRack(RackId rack_id, TrailerId trailer_id);
    bool rightStackRack(RackId rack_id, TrailerId trailer_id);
    bool leftUnstackRack(RackId rack_id, TrailerId trailer_id);
    bool rightUnstackRack(RackId rack_id, TrailerId trailer_id);
    
    // Utility functions
    std::unordered_map<std::string, double> getSubgoals() const;
    bool belugaComplete();
    std::string toString() const;
    
    // Copy constructor and assignment
    ProblemState(const ProblemState& other);
    ProblemState& operator=(const ProblemState& other);

private:
    std::vector<Jig> jigs_;
    std::vector<Beluga> belugas_;
    std::vector<std::optional<JigId>> trailers_beluga_;
    std::vector<std::optional<JigId>> trailers_factory_;
    std::vector<Rack> racks_;
    std::vector<ProductionLine> production_lines_;
    std::vector<std::optional<JigId>> hangars_;
    
    // Subgoal counters
    mutable int belugas_unloaded_;
    mutable int belugas_finished_;
    mutable int production_lines_finished_;
    mutable int total_lines_;
    mutable int total_belugas_;
    mutable bool problem_solved_;
    
    // Helper functions for action validation
    bool isValidTrailerBeluga(TrailerId trailer_id) const;
    bool isValidTrailerFactory(TrailerId trailer_id) const;
    bool isValidRack(RackId rack_id) const;
    bool isValidHangar(HangarId hangar_id) const;
};

} // namespace mcts_fast