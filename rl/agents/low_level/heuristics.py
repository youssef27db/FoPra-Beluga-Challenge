from typing import Tuple, Optional

def decide_parameters(obs, high_level_action):
    """!
    @brief Decide action parameters based on high-level action and current observation
    
    This function implements heuristic decision-making for low-level action parameters.
    It analyzes the current state observation to determine appropriate parameters
    for the given high-level action.
    
    @param obs Current state observation array
    @param high_level_action High-level action string to execute
    @return Tuple of (action_name, parameters) or ("None", []) if no action possible
    """
    n_racks = 10  # Number of available racks

    # Switch case for high-level agent action
    match high_level_action:
        # If unload_beluga, no parameters are returned
        case "unload_beluga":
            return "unload_beluga", []

        # If load_beluga, return trailer index
        case "load_beluga":
            for i in range(3):
                if obs[1 + i] == 0:  # Trailer has matching empty jig
                    return "load_beluga", {"trailer_beluga": i, "none": None}

        # If right_unstack_rack, return rack index and trailer ID
        case "right_unstack_rack":
            for rack_idx in range(n_racks):
                slot = 10 + rack_idx * 3
                if obs[slot + 1] == 1:
                    for trailer_idx in range(3):
                        if obs[4 + trailer_idx] == 0.5:
                            return "right_unstack_rack", {"rack": rack_idx, "trailer_id": trailer_idx}
                    
        # If left_unstack_rack, return rack index and trailer ID
        case "left_unstack_rack":
            for rack_idx in range(n_racks):
                slot = 10 + rack_idx * 3
                if obs[slot] == 1:
                    for trailer_idx in range(3):
                        if obs[1 + trailer_idx] == 0.5:
                            return "left_unstack_rack", {"rack": rack_idx, "trailer_id": trailer_idx}

        # If get_from_hangar, return hangar index and trailer factory index
        case "get_from_hangar":
            for hangar_idx in range(3):
                if obs[7 + hangar_idx] == 1:
                    for trailer_idx in range(3):
                        if obs[4 + trailer_idx] == 0.5:
                            return "get_from_hangar", {"hangar": hangar_idx, "trailer_factory": trailer_idx}

        # If deliver_to_hangar, return hangar index and trailer factory index
        case "deliver_to_hangar":
            for trailer_idx in range(3):
                if obs[4 + trailer_idx] == 1:
                    for hangar_idx in range(3):
                        if obs[7 + hangar_idx] == 0:
                            return "deliver_to_hangar", {"hangar": hangar_idx, "trailer_factory": trailer_idx}


        # No action available
        case _:
            return "None", []
        
    return "None", []