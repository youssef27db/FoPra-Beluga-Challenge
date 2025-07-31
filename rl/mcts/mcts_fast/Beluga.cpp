#include "Beluga.hpp"
#include <sstream>

Beluga::Beluga(const std::vector<int>& current_jigs, const std::vector<JigType>& outgoing)
    : current_jigs(current_jigs), outgoing(outgoing) {}

Beluga Beluga::copy() const {
    return Beluga(current_jigs, outgoing);
}

std::string Beluga::to_string() const {
    std::stringstream ss;
    ss << "current_jigs = [";
    for (size_t i = 0; i < current_jigs.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << current_jigs[i];
    }
    ss << "] | outgoing = [";
    for (size_t i = 0; i < outgoing.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << outgoing[i].to_string();
    }
    ss << "]";
    return ss.str();
}
