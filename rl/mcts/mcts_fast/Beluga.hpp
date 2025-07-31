#pragma once
#include <vector>
#include "Jig.hpp"

/**
 * @class Beluga
 * @brief Represents a beluga aircraft with current jigs and outgoing jig types.
 */
class Beluga {
public:
    std::vector<int> current_jigs;
    std::vector<JigType> outgoing;

    /**
     * @brief Constructor for Beluga.
     * @param current_jigs List of current jig IDs.
     * @param outgoing List of outgoing jig types.
     */
    Beluga(const std::vector<int>& current_jigs, const std::vector<JigType>& outgoing);

    /**
     * @brief Default constructor.
     */
    Beluga() = default;

    /**
     * @brief Copy constructor.
     */
    Beluga(const Beluga& other) = default;

    /**
     * @brief Assignment operator.
     */
    Beluga& operator=(const Beluga& other) = default;

    /**
     * @brief Creates a copy of the beluga.
     */
    Beluga copy() const;

    /**
     * @brief String representation of the beluga.
     */
    std::string to_string() const;
};
