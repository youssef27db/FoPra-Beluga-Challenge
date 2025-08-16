#pragma once
#include "types.h"
#include "jig.h"
#include <vector>

namespace mcts_fast {

class Rack {
public:
    Rack(int size, const std::vector<JigId>& current_jigs);
    
    int getSize() const { return size_; }
    const std::vector<JigId>& getCurrentJigs() const { return current_jigs_; }
    std::vector<JigId>& getCurrentJigs() { return current_jigs_; }
    
    void addJig(JigId jig_id) { current_jigs_.push_back(jig_id); }
    void removeJig(JigId jig_id);
    void removeTopJig();
    JigId getTopJig() const;
    bool hasJigs() const { return !current_jigs_.empty(); }
    
    int getFreeSpace(const std::vector<Jig>& all_jigs) const;
    bool canFitJig(const Jig& jig, const std::vector<Jig>& all_jigs) const;
    
    std::string toString() const;
    
    // Copy constructor and assignment
    Rack(const Rack& other);
    Rack& operator=(const Rack& other);

private:
    int size_;
    std::vector<JigId> current_jigs_;
};

} // namespace mcts_fast