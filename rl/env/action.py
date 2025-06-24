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
def load_beluga(state, trailer_beluga: int) -> bool:
    """Load beluga from specific trailer."""
    jig_id = state.trailers_beluga[trailer_beluga]

    # Preconditions
    if len(state.belugas) == 0:
        return False

    beluga = state.belugas[0]

    if jig_id is None:  # Trailer is empty
        return False

    if not state.jigs[jig_id].empty:  # Jig must be empty
        return False

    if not beluga.outgoing:  # Beluga has no outgoing types
        return False

    if state.jigs[jig_id].jig_type != beluga.outgoing[0]:
        return False

    # Effects: Remove outgoing type and clear trailer slot
    beluga.outgoing.pop(0)
    state.trailers_beluga[trailer_beluga] = None
    return True


# 1
def unload_beluga(state) -> bool:
    """Unload beluga (no additional parameter besides state)."""
    # Find the first empty trailer slot
    trailer_beluga = None
    for i, trailer in enumerate(state.trailers_beluga):
        if trailer is None:
            trailer_beluga = i
            break

    if trailer_beluga is None or not state.belugas:
        return False

    beluga = state.belugas[0]
    if not beluga.current_jigs:
        return False

    # Effects: Unload the last jig from current_jigs into the trailer slot
    state.trailers_beluga[trailer_beluga] = beluga.current_jigs.pop(-1)

    if not beluga.current_jigs and not beluga.outgoing:
        # Effect: Remove beluga from the list if fully processed
        state.belugas.pop(0)

    return True


# 2
def get_from_hangar(state, hangar: int, trailer_factory: int) -> bool:
    """Get jig from specific hangar to specific trailer."""
    if hangar >= len(state.hangars) or trailer_factory >= len(state.trailers_factory):
        return False

    if state.hangars[hangar] is None or state.trailers_factory[trailer_factory] is not None:
        return False

    jig_id = state.hangars[hangar]
    if not state.jigs[jig_id].empty:
        return False

    # Effects: Move jig from hangar to factory trailer
    state.trailers_factory[trailer_factory] = jig_id
    state.hangars[hangar] = None
    return True


# 3
def deliver_to_hangar(state, hangar: int, trailer_factory: int) -> bool:
    """Deliver jig from specific trailer to specific hangar."""
    if hangar >= len(state.hangars) or trailer_factory >= len(state.trailers_factory):
        return False

    if state.hangars[hangar] is not None or state.trailers_factory[trailer_factory] is None:
        return False

    jig_id = state.trailers_factory[trailer_factory]
    if state.jigs[jig_id].empty:
        return False

    # Find corresponding production line for the jig
    production_line_idx = None
    for i, pl in enumerate(state.production_lines):
        if pl.scheduled_jigs and jig_id == pl.scheduled_jigs[0]:
            production_line_idx = i
            break

    if production_line_idx is None:
        return False

    # Effects: Remove jig from production line, deliver to hangar, and clear trailer slot
    state.production_lines[production_line_idx].scheduled_jigs.pop(0)
    state.hangars[hangar] = jig_id
    state.jigs[jig_id].empty = True
    state.trailers_factory[trailer_factory] = None

    if not state.production_lines[production_line_idx].scheduled_jigs:
        state.production_lines.pop(production_line_idx)
    return True


# 4
def left_stack_rack(state, rack: int, trailer_id: int) -> bool:
    """Stack jig on rack from the left trailer (Beluga)."""
    if rack >= len(state.racks):
        return False

    trailers = state.trailers_beluga
    if trailer_id >= len(trailers) or trailers[trailer_id] is None:
        return False

    jig_id = trailers[trailer_id]
    jig = state.jigs[jig_id]
    jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded

    rack_obj = state.racks[rack]
    if rack_obj.get_free_space(state.jigs) < jig_size:
        return False

    # Effects: Clear trailer slot and stack jig onto the rack
    trailers[trailer_id] = None
    rack_obj.current_jigs.insert(0, jig_id)
    return True


# 5
def right_stack_rack(state, rack: int, trailer_id: int) -> bool:
    """Stack jig on rack from the right trailer (Factory)."""
    if rack >= len(state.racks):
        return False

    trailers = state.trailers_factory
    if trailer_id >= len(trailers) or trailers[trailer_id] is None:
        return False

    jig_id = trailers[trailer_id]
    jig = state.jigs[jig_id]
    jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded

    rack_obj = state.racks[rack]
    if rack_obj.get_free_space(state.jigs) < jig_size:
        return False

    # Effects: Clear trailer slot and stack jig onto the rack
    trailers[trailer_id] = None
    rack_obj.current_jigs.append(jig_id)
    return True


# 6
def left_unstack_rack(state, rack: int, trailer_id: int) -> bool:
    """Unstack jig from rack to the left trailer (Beluga)."""
    if rack >= len(state.racks):
        return False

    trailers = state.trailers_beluga
    if trailer_id >= len(trailers) or trailers[trailer_id] is not None or not state.racks[rack].current_jigs:
        return False

    # Effects: Remove jig from rack and place it into the trailer
    trailers[trailer_id] = state.racks[rack].current_jigs.pop(0)
    return True


# 7
def right_unstack_rack(state, rack: int, trailer_id: int) -> bool:
    """Unstack jig from rack to the right trailer (Factory)."""
    if rack >= len(state.racks):
        return False

    trailers = state.trailers_factory
    if trailer_id >= len(trailers) or trailers[trailer_id] is not None or not state.racks[rack].current_jigs:
        return False

    # Effects: Remove jig from rack and place it into the trailer
    trailers[trailer_id] = state.racks[rack].current_jigs.pop(-1)
    return True