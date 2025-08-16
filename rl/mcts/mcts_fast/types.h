#pragma once
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace mcts_fast {

// Forward declarations
class JigType;
class Jig;
class Beluga;
class Rack;
class ProductionLine;
class ProblemState;
class MCTSNode;
class MCTS;

// Type aliases for convenience
using JigId = int;
using RackId = int;
using TrailerId = int;
using HangarId = int;
using BelugaId = int;

// Action parameter types
using ActionParams = std::vector<int>;
using ActionTuple = std::pair<std::string, ActionParams>;

} // namespace mcts_fast