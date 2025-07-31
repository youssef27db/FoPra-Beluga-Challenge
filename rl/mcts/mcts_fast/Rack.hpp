#pragma once
#include <vector>
#include "Jig.hpp"

/**
 * @class Rack
 * @brief Represents a storage rack with size constraints and current jigs.
 */
class Rack {
public:
    int size;
    std::vector<int> current_jigs;

    /**
     * @brief Constructor for Rack.
     * @param size The maximum size capacity of the rack.
     * @param current_jigs List of current jig IDs in the rack.
     */
    Rack(int size, const std::vector<int>& current_jigs);

    /**
     * @brief Default constructor.
     */
    Rack() = default;

    /**
     * @brief Copy constructor.
     */
    Rack(const Rack& other) = default;

    /**
     * @brief Assignment operator.
     */
    Rack& operator=(const Rack& other) = default;

    /**
     * @brief Gets the free space in the rack.
     * @param all_jigs List of all jigs to calculate sizes.
     * @return The remaining free space in the rack.
     */
    int get_free_space(const std::vector<Jig>& all_jigs) const;

    /**
     * @brief Creates a copy of the rack.
     */
    Rack copy() const;

    /**
     * @brief String representation of the rack.
     */
    std::string to_string() const;
};
