#include "rack.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace mcts_fast {

Rack::Rack(int size, const std::vector<JigId>& current_jigs)
    : size_(size), current_jigs_(current_jigs) {
}

Rack::Rack(const Rack& other)
    : size_(other.size_), current_jigs_(other.current_jigs_) {
}

Rack& Rack::operator=(const Rack& other) {
    if (this != &other) {
        size_ = other.size_;
        current_jigs_ = other.current_jigs_;
    }
    return *this;
}

void Rack::removeJig(JigId jig_id) {
    auto it = std::find(current_jigs_.begin(), current_jigs_.end(), jig_id);
    if (it != current_jigs_.end()) {
        current_jigs_.erase(it);
    }
}

void Rack::removeTopJig() {
    if (!current_jigs_.empty()) {
        current_jigs_.pop_back();
    }
}

JigId Rack::getTopJig() const {
    if (current_jigs_.empty()) {
        throw std::runtime_error("Rack is empty, no top jig available");
    }
    return current_jigs_.back();
}

int Rack::getFreeSpace(const std::vector<Jig>& all_jigs) const {
    int total_used_space = 0;
    for (JigId jig_id : current_jigs_) {
        if (jig_id >= 0 && jig_id < static_cast<int>(all_jigs.size())) {
            total_used_space += all_jigs[jig_id].getSize();
        }
    }
    return size_ - total_used_space;
}

bool Rack::canFitJig(const Jig& jig, const std::vector<Jig>& all_jigs) const {
    return getFreeSpace(all_jigs) >= jig.getSize();
}

std::string Rack::toString() const {
    std::ostringstream oss;
    oss << "size = " << size_ << " | current_jigs = [";
    for (size_t i = 0; i < current_jigs_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << current_jigs_[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace mcts_fast