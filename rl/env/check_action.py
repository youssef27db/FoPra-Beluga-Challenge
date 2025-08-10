
"""!
@file check_action.py
@brief Action validation functions for the Beluga Challenge
"""

# 0
def check_load_beluga(state, obs) -> bool:
    """!
    @brief Check if a beluga can be loaded from a trailer
    @param state Current problem state
    @param obs Current observation array
    @return True if loading is possible, False otherwise
    """
    # Preconditions
    if obs[0] != 0:
        return False
    
    for i in range(3):
        if obs[1 + i] == 0:  # Trailer has matching empty jig
            return True

    return False


# 1
def check_unload_beluga(state, obs) -> bool:
    """!
    @brief Check if a beluga can be unloaded
    @param state Current problem state
    @param obs Current observation array
    @return True if unloading is possible, False otherwise
    """
    # Preconditions
    if obs[0] != 1:
        return False
    
    for i in range(3):
        if obs[1 + i] == 0.5:  # Trailer is empty
            return True

    return False


# 2
def check_get_from_hangar(state, obs) -> bool:
    """!
    @brief Check if a jig can be retrieved from hangar
    @param state Current problem state
    @param obs Current observation array
    @return True if retrieval is possible, False otherwise
    """
    # Preconditions
    for hangar_idx in range(3):
        if obs[7 + hangar_idx] == 1:
            for trailer_idx in range(3):
                    if obs[4 + trailer_idx] == 0.5:
                        return True
                    
    return False


# 3
def check_deliver_to_hangar(state, obs) -> bool:
    """!
    @brief Check if a jig can be delivered to hangar
    @param state Current problem state
    @param obs Current observation array
    @return True if delivery is possible, False otherwise
    """
    # Preconditions
    for trailer_idx in range(3):
                if obs[4 + trailer_idx] == 1:
                    for hangar_idx in range(3):
                        if obs[7 + hangar_idx] == 0:
                            return True

    return False

# 4
def check_left_stack_rack(state, obs) -> bool:
    """!
    @brief Check if a jig can be stacked from left (Beluga) trailer to rack
    @param state Current problem state
    @param obs Current observation array
    @return True if stacking is possible, False otherwise
    """
    # Preconditions
    for i in range(3):
        if obs[1 + i] != 0.5 and obs[1 + i] != -1:
            jig_id = state.trailers_beluga[i]
            jig = state.jigs[jig_id]
            jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded 
            for rack_idx in range(len(state.racks)):
                rack_obj = state.racks[rack_idx]
                if rack_obj.get_free_space(state.jigs) >= jig_size:
                    return True

    return False


# 5
def check_right_stack_rack(state, obs) -> bool:
    """!
    @brief Check if a jig can be stacked from right (Factory) trailer to rack
    @param state Current problem state
    @param obs Current observation array
    @return True if stacking is possible, False otherwise
    """
    # Preconditions
    for i in range(3):
        if obs[4 + i] != 0.5 and obs[4 + i] != -1:
            jig_id = state.trailers_factory[i]
            jig = state.jigs[jig_id]
            jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded 
            for rack_idx in range(len(state.racks)):
                rack_obj = state.racks[rack_idx]
                if rack_obj.get_free_space(state.jigs) >= jig_size:
                    return True

    return False


# 6
def check_left_unstack_rack(state, obs) -> bool:
    """!
    @brief Check if a jig can be unstacked from rack to left (Beluga) trailer
    @param state Current problem state
    @param obs Current observation array
    @return True if unstacking is possible, False otherwise
    """
    # Preconditions
    for trailer_idx in range(3):
        if obs[1 + trailer_idx] == 0.5:
             for rack_idx in range(len(state.racks)):
                if state.racks[rack_idx].current_jigs != []:
                    return True
                
    return False


# 7
def check_right_unstack_rack(state, obs) -> bool:
    """!
    @brief Check if a jig can be unstacked from rack to right (Factory) trailer
    @param state Current problem state
    @param obs Current observation array
    @return True if unstacking is possible, False otherwise
    """
    # Preconditions
    # Preconditions
    for trailer_idx in range(3):
        if obs[4 + trailer_idx] == 0.5:
             for rack_idx in range(len(state.racks)):
                if state.racks[rack_idx].current_jigs != []:
                    return True
                
    return False