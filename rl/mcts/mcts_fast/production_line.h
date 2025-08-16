#pragma once
#include "types.h"
#include <vector>

namespace mcts_fast {

class ProductionLine {
public:
    ProductionLine(const std::vector<JigId>& scheduled_jigs);
    
    const std::vector<JigId>& getScheduledJigs() const { return scheduled_jigs_; }
    std::vector<JigId>& getScheduledJigs() { return scheduled_jigs_; }
    
    void addScheduledJig(JigId jig_id) { scheduled_jigs_.push_back(jig_id); }
    void removeScheduledJig(JigId jig_id);
    void removeFirstScheduledJig();
    
    JigId getNextJig() const;
    bool hasScheduledJigs() const { return !scheduled_jigs_.empty(); }
    bool isEmpty() const { return scheduled_jigs_.empty(); }
    
    std::string toString() const;
    
    // Copy constructor and assignment
    ProductionLine(const ProductionLine& other);
    ProductionLine& operator=(const ProductionLine& other);

private:
    std::vector<JigId> scheduled_jigs_;
};

} // namespace mcts_fast