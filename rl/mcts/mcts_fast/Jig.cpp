#include "Jig.hpp"

// JigType implementation
JigType::JigType(const std::string& name, int size_empty, int size_loaded)
    : name(name), size_empty(size_empty), size_loaded(size_loaded) {}

bool JigType::operator==(const JigType& other) const {
    return name == other.name;
}

bool JigType::operator!=(const JigType& other) const {
    return name != other.name;
}

std::string JigType::to_string() const {
    return name;
}

// Jig implementation
Jig::Jig(const JigType& jig_type, bool empty)
    : jig_type(jig_type), empty(empty) {}

Jig Jig::copy() const {
    return Jig(jig_type, empty);
}

std::string Jig::to_string() const {
    return jig_type.to_string() + " | " + (empty ? "true" : "false");
}

// Utility function
JigType get_type(const std::string& name) {
    if (name == "typeA") {
        return JigType("typeA", 4, 4);
    } else if (name == "typeB") {
        return JigType("typeB", 8, 11);
    } else if (name == "typeC") {
        return JigType("typeC", 9, 18);
    } else if (name == "typeD") {
        return JigType("typeD", 18, 25);
    } else if (name == "typeE") {
        return JigType("typeE", 32, 32);
    }
    return JigType("unknown", 0, 0);
}
