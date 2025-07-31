#pragma once
#include <string>

/**
 * @class JigType
 * @brief Represents a type of jig with its size characteristics.
 */
class JigType {
public:
    std::string name;
    int size_empty;
    int size_loaded;

    /**
     * @brief Constructor for JigType.
     * @param name The name of the jig type.
     * @param size_empty The size when empty.
     * @param size_loaded The size when loaded.
     */
    JigType(const std::string& name, int size_empty, int size_loaded);

    /**
     * @brief Default constructor.
     */
    JigType() = default;

    /**
     * @brief Copy constructor.
     */
    JigType(const JigType& other) = default;

    /**
     * @brief Assignment operator.
     */
    JigType& operator=(const JigType& other) = default;

    /**
     * @brief Equality operator.
     */
    bool operator==(const JigType& other) const;

    /**
     * @brief Inequality operator.
     */
    bool operator!=(const JigType& other) const;

    /**
     * @brief String representation of the jig type.
     */
    std::string to_string() const;
};

/**
 * @class Jig
 * @brief Represents a jig with a type and empty/loaded state.
 */
class Jig {
public:
    JigType jig_type;
    bool empty;

    /**
     * @brief Constructor for Jig.
     * @param jig_type The type of the jig.
     * @param empty Whether the jig is empty.
     */
    Jig(const JigType& jig_type, bool empty);

    /**
     * @brief Default constructor.
     */
    Jig() = default;

    /**
     * @brief Copy constructor.
     */
    Jig(const Jig& other) = default;

    /**
     * @brief Assignment operator.
     */
    Jig& operator=(const Jig& other) = default;

    /**
     * @brief Creates a copy of the jig.
     */
    Jig copy() const;

    /**
     * @brief String representation of the jig.
     */
    std::string to_string() const;
};

// Utility functions for jig types
JigType get_type(const std::string& name);