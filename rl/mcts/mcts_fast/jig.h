#pragma once
#include "types.h"
#include <string>

namespace mcts_fast {

class JigType {
public:
    JigType(const std::string& name, int size_empty, int size_loaded);
    
    const std::string& getName() const { return name_; }
    int getSizeEmpty() const { return size_empty_; }
    int getSizeLoaded() const { return size_loaded_; }
    
    bool operator==(const JigType& other) const;
    bool operator!=(const JigType& other) const;
    
    std::string toString() const;

private:
    std::string name_;
    int size_empty_;
    int size_loaded_;
};

class Jig {
public:
    Jig(const JigType& jig_type, bool empty);
    
    const JigType& getJigType() const { return jig_type_; }
    bool isEmpty() const { return empty_; }
    void setEmpty(bool empty) { empty_ = empty; }
    
    int getSize() const;
    std::string toString() const;
    
    // Copy constructor and assignment
    Jig(const Jig& other);
    Jig& operator=(const Jig& other);

private:
    JigType jig_type_;
    bool empty_;
};

// Utility functions for JigType creation
JigType createJigType(const std::string& type_name);

} // namespace mcts_fast