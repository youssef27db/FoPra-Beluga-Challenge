#include "jig.h"
#include <sstream>

namespace mcts_fast {

// JigType implementation
JigType::JigType(const std::string& name, int size_empty, int size_loaded)
    : name_(name), size_empty_(size_empty), size_loaded_(size_loaded) {
}

bool JigType::operator==(const JigType& other) const {
    return name_ == other.name_;
}

bool JigType::operator!=(const JigType& other) const {
    return name_ != other.name_;
}

std::string JigType::toString() const {
    return name_;
}

// Jig implementation
Jig::Jig(const JigType& jig_type, bool empty)
    : jig_type_(jig_type), empty_(empty) {
}

Jig::Jig(const Jig& other)
    : jig_type_(other.jig_type_), empty_(other.empty_) {
}

Jig& Jig::operator=(const Jig& other) {
    if (this != &other) {
        jig_type_ = other.jig_type_;
        empty_ = other.empty_;
    }
    return *this;
}

int Jig::getSize() const {
    return empty_ ? jig_type_.getSizeEmpty() : jig_type_.getSizeLoaded();
}

std::string Jig::toString() const {
    std::ostringstream oss;
    oss << jig_type_.toString() << " | " << (empty_ ? "empty" : "loaded");
    return oss.str();
}

// Utility functions
JigType createJigType(const std::string& type_name) {
    if (type_name == "typeA") {
        return JigType("typeA", 4, 4);
    } else if (type_name == "typeB") {
        return JigType("typeB", 8, 11);
    } else if (type_name == "typeC") {
        return JigType("typeC", 9, 18);
    } else if (type_name == "typeD") {
        return JigType("typeD", 18, 25);
    } else if (type_name == "typeE") {
        return JigType("typeE", 32, 32);
    } else {
        return JigType("unknown", 0, 0);
    }
}

} // namespace mcts_fast