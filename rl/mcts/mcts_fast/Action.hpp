#pragma once
#include "ProblemState.hpp"

/**
 * @file Action.hpp
 * @brief Collection of functions that manage state transitions for various operations.
 * 
 * This file contains functions for managing operations on belugas, jigs, trailers, 
 * hangars, and racks. Each function follows these principles:
 * - Clear and descriptive function names
 * - Early precondition checks to prevent invalid state mutations
 * - Direct modification of the state object
 * - Boolean return values to signal success or failure
 */

namespace Action {

/**
 * @brief Load beluga from specific trailer.
 * @param state The problem state to modify
 * @param trailer_beluga The trailer index to load from
 * @param none Unused parameter (for API compatibility)
 * @return True if successful, false otherwise
 */
bool load_beluga(ProblemState& state, int trailer_beluga, int none = -1);

/**
 * @brief Unload beluga (no additional parameter besides state).
 * @param state The problem state to modify
 * @return True if successful, false otherwise
 */
bool unload_beluga(ProblemState& state);

/**
 * @brief Get jig from specific hangar to specific trailer.
 * @param state The problem state to modify
 * @param hangar The hangar index
 * @param trailer_factory The factory trailer index
 * @return True if successful, false otherwise
 */
bool get_from_hangar(ProblemState& state, int hangar, int trailer_factory);

/**
 * @brief Deliver jig from specific trailer to specific hangar.
 * @param state The problem state to modify
 * @param hangar The hangar index
 * @param trailer_factory The factory trailer index
 * @return True if successful, false otherwise
 */
bool deliver_to_hangar(ProblemState& state, int hangar, int trailer_factory);

/**
 * @brief Stack jig on rack from the left trailer (Beluga).
 * @param state The problem state to modify
 * @param rack The rack index
 * @param trailer_id The beluga trailer index
 * @return True if successful, false otherwise
 */
bool left_stack_rack(ProblemState& state, int rack, int trailer_id);

/**
 * @brief Stack jig on rack from the right trailer (Factory).
 * @param state The problem state to modify
 * @param rack The rack index
 * @param trailer_id The factory trailer index
 * @return True if successful, false otherwise
 */
bool right_stack_rack(ProblemState& state, int rack, int trailer_id);

/**
 * @brief Unstack jig from rack to the left trailer (Beluga).
 * @param state The problem state to modify
 * @param rack The rack index
 * @param trailer_id The beluga trailer index
 * @return True if successful, false otherwise
 */
bool left_unstack_rack(ProblemState& state, int rack, int trailer_id);

/**
 * @brief Unstack jig from rack to the right trailer (Factory).
 * @param state The problem state to modify
 * @param rack The rack index
 * @param trailer_id The factory trailer index
 * @return True if successful, false otherwise
 */
bool right_unstack_rack(ProblemState& state, int rack, int trailer_id);

} // namespace Action
