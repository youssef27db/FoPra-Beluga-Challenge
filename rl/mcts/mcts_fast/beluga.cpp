#include "beluga.h"
#include <algorithm>
#include <sstream>

namespace mcts_fast {

Beluga::Beluga(const std::vector<JigId>& current_jigs, const std::vector<JigType>& outgoing)
    : current_jigs_(current_jigs), outgoing_(outgoing) {
}

Beluga::Beluga(const Beluga& other)
    : current_jigs_(other.current_jigs_), outgoing_(other.outgoing_) {
}

Beluga& Beluga::operator=(const Beluga& other) {
    if (this != &other) {
        current_jigs_ = other.current_jigs_;
        outgoing_ = other.outgoing_;
    }
    return *this;
}

void Beluga::removeCurrentJig(JigId jig_id) {
    auto it = std::find(current_jigs_.begin(), current_jigs_.end(), jig_id);
    if (it != current_jigs_.end()) {
        current_jigs_.erase(it);
    }
}

void Beluga::removeOutgoing(const JigType& jig_type) {
    auto it = std::find(outgoing_.begin(), outgoing_.end(), jig_type);
    if (it != outgoing_.end()) {
        outgoing_.erase(it);
    }
}

std::string Beluga::toString() const {
    std::ostringstream oss;
    oss << "current_jigs = [";
    for (size_t i = 0; i < current_jigs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << current_jigs_[i];
    }
    oss << "] | outgoing = [";
    for (size_t i = 0; i < outgoing_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << outgoing_[i].toString();
    }
    oss << "]";
    return oss.str();
}

} // namespace mcts_fast