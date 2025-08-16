#include "production_line.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace mcts_fast {

ProductionLine::ProductionLine(const std::vector<JigId>& scheduled_jigs)
    : scheduled_jigs_(scheduled_jigs) {
}

ProductionLine::ProductionLine(const ProductionLine& other)
    : scheduled_jigs_(other.scheduled_jigs_) {
}

ProductionLine& ProductionLine::operator=(const ProductionLine& other) {
    if (this != &other) {
        scheduled_jigs_ = other.scheduled_jigs_;
    }
    return *this;
}

void ProductionLine::removeScheduledJig(JigId jig_id) {
    auto it = std::find(scheduled_jigs_.begin(), scheduled_jigs_.end(), jig_id);
    if (it != scheduled_jigs_.end()) {
        scheduled_jigs_.erase(it);
    }
}

void ProductionLine::removeFirstScheduledJig() {
    if (!scheduled_jigs_.empty()) {
        scheduled_jigs_.erase(scheduled_jigs_.begin());
    }
}

JigId ProductionLine::getNextJig() const {
    if (scheduled_jigs_.empty()) {
        throw std::runtime_error("ProductionLine is empty, no next jig available");
    }
    return scheduled_jigs_.front();
}

std::string ProductionLine::toString() const {
    std::ostringstream oss;
    oss << "scheduled_jigs = [";
    for (size_t i = 0; i < scheduled_jigs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << scheduled_jigs_[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace mcts_fast