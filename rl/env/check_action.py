"""
--------------------------------
This file contains a collection of functions that manage state transitions for
various operations on "beluga" objects, jigs, trailers, hangars, and racks. Each
function is responsible for a specific action and follows these principles:

• Clearly defined and descriptive function names.
• Early precondition checks to prevent invalid state mutations.
• Direct modification of the 'state' object, ensuring that changes are localized.
• Returning Boolean values to signal the success or failure of an operation.
• Inline comments provide clarity regarding the preconditions and side effects.
• Separation of concerns: each function handles one piece of functionality.

Functions:
    0. load_beluga: Loads a beluga from a specified trailer slot.
    1. unload_beluga: Unloads a beluga and, if fully processed, removes it from the list.
    2. get_from_hangar: Retrieves a jig from a hangar and places it into a factory trailer.
    3. deliver_to_hangar: Delivers a jig from a trailer into a hangar after verifying production line data.
    4. left_stack_rack: Stacks a jig from the left (Beluga) trailer onto a rack.
    5. right_stack_rack: Stacks a jig from the right (Factory) trailer onto a rack.
    6. left_unstack_rack: Unstacks a jig from a rack into a left (Beluga) trailer.
    7. right_unstack_rack: Unstacks a jig from a rack into a right (Factory) trailer.
"""

# 0
def check_load_beluga(state, obs) -> bool:
    # Preconditions
    if obs[0] != 0:
        return False
    
    for i in range(3):
        if obs[1 + i] == 0:  # Trailer hat passenden leeren Jig
            return True

    return False


# 1
def check_unload_beluga(state, obs) -> bool:
    # Preconditions
    if obs[0] != 1:
        return False
    
    for i in range(3):
        if obs[1 + i] == 0.5:  # Trailer ist leer
            return True

    return False


# 2
def check_get_from_hangar(state, obs) -> bool:
    # Preconditions
    for hangar_idx in range(3):
        if obs[7 + hangar_idx] == 1:
            for trailer_idx in range(3):
                    if obs[4 + trailer_idx] == 0.5:
                        return True
                    
    return False


# 3
def check_deliver_to_hangar(state, obs) -> bool:
    # Preconditions
    for trailer_idx in range(3):
                if obs[4 + trailer_idx] == 1:
                    for hangar_idx in range(3):
                        if obs[7 + hangar_idx] == 0:
                            return True

    return False

# 4
def check_left_stack_rack(state, obs) -> bool:
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
    # Preconditions
    for trailer_idx in range(3):
        if obs[1 + trailer_idx] == 0.5:
             for rack_idx in range(len(state.racks)):
                if state.racks[rack_idx].current_jigs != []:
                    return True
                
    return False


# 7
def check_right_unstack_rack(state, obs) -> bool:
    # Preconditions
    for trailer_idx in range(3):
        if obs[4 + trailer_idx] == 0.5:
             for rack_idx in range(len(state.racks)):
                if state.racks[rack_idx].current_jigs != []:
                    return True
                
    return False