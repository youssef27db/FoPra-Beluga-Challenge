#include "ProductionLine.hpp"
#include <sstream>

ProductionLine::ProductionLine(const std::vector<int>& scheduled_jigs)
    : scheduled_jigs(scheduled_jigs) {}

ProductionLine ProductionLine::copy() const {
    return ProductionLine(scheduled_jigs);
}

std::string ProductionLine::to_string() const {
    std::stringstream ss;
    ss << "scheduled_jigs = [";
    for (size_t i = 0; i < scheduled_jigs.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << scheduled_jigs[i];
    }
    ss << "]";
    return ss.str();
}
