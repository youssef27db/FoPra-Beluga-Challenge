#pragma once
#include <vector>
#include <string>

/**
 * @class ProductionLine
 * @brief Represents a production line with scheduled jigs.
 */
class ProductionLine {
public:
    std::vector<int> scheduled_jigs;

    /**
     * @brief Constructor for ProductionLine.
     * @param scheduled_jigs List of scheduled jig IDs.
     */
    ProductionLine(const std::vector<int>& scheduled_jigs);

    /**
     * @brief Default constructor.
     */
    ProductionLine() = default;

    /**
     * @brief Copy constructor.
     */
    ProductionLine(const ProductionLine& other) = default;

    /**
     * @brief Assignment operator.
     */
    ProductionLine& operator=(const ProductionLine& other) = default;

    /**
     * @brief Creates a copy of the production line.
     */
    ProductionLine copy() const;

    /**
     * @brief String representation of the production line.
     */
    std::string to_string() const;
};
