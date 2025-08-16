#pragma once
#include "types.h"
#include "jig.h"
#include <vector>

namespace mcts_fast {

class Beluga {
public:
    Beluga(const std::vector<JigId>& current_jigs, const std::vector<JigType>& outgoing);
    
    const std::vector<JigId>& getCurrentJigs() const { return current_jigs_; }
    const std::vector<JigType>& getOutgoing() const { return outgoing_; }
    
    std::vector<JigId>& getCurrentJigs() { return current_jigs_; }
    std::vector<JigType>& getOutgoing() { return outgoing_; }
    
    void addCurrentJig(JigId jig_id) { current_jigs_.push_back(jig_id); }
    void removeCurrentJig(JigId jig_id);
    void addOutgoing(const JigType& jig_type) { outgoing_.push_back(jig_type); }
    void removeOutgoing(const JigType& jig_type);
    
    bool isEmpty() const { return current_jigs_.empty() && outgoing_.empty(); }
    std::string toString() const;
    
    // Copy constructor and assignment
    Beluga(const Beluga& other);
    Beluga& operator=(const Beluga& other);

private:
    std::vector<JigId> current_jigs_;
    std::vector<JigType> outgoing_;
};

} // namespace mcts_fast