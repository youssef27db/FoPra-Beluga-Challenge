#include "Rack.hpp"
#include <sstream>

Rack::Rack(int size, const std::vector<int>& current_jigs)
    : size(size), current_jigs(current_jigs) {}

int Rack::get_free_space(const std::vector<Jig>& all_jigs) const {
    int total_used_space = 0;
    for (int jig_id : current_jigs) {
        const Jig& jig = all_jigs[jig_id - 1];
        int jig_size = jig.empty ? jig.jig_type.size_empty : jig.jig_type.size_loaded;
        total_used_space += jig_size;
    }
    
    int remaining_space = size - total_used_space;
    return remaining_space;
}

Rack Rack::copy() const {
    return Rack(size, current_jigs);
}

std::string Rack::to_string() const {
    std::stringstream ss;
    ss << "size = " << size << " | current_jigs = [";
    for (size_t i = 0; i < current_jigs.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << current_jigs[i];
    }
    ss << "]";
    return ss.str();
}
