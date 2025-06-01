from problem_state import *

def load_beluga(state: ProblemState, jig: Jig, jig_type: JigType, beluga: Beluga, trailer_beluga: int):
    #Preconditions
    if state.trailers_beluga[trailer_beluga] == None:
        return False
    if jig.empty == False:
        return False
    if jig.jig_type != jig_type:
        return False
    if beluga.processed == False:
        return False 
    if jig.jig_type != beluga.outgoing[0]:
        return False

    #Effekte
    beluga.outgoing.pop(0)
    state.trailers_beluga[trailer_beluga] = None
    return True
    

def unload_beluga(state: ProblemState, jig: Jig, beluga: Beluga, trailer_beluga: int):
    # Preconditions
    if state.trailers_beluga[trailer_beluga] != None:
        return False
    if jig.empty == True:
        return False
    if beluga.processed == False:
        return False
    if not beluga.current_jigs:
        return False
    if extract_id(jig) != beluga.current_jigs[-1]:
        return False

    # Effekte
    beluga.current_jigs.pop(-1)
    state.trailers_beluga[trailer_beluga] = extract_id(jig)
    return True


def get_from_hangar(state: ProblemState, jig: Jig, hangar: int, trailer_factory: int):
    # Preconditions
    if state.hangars[hangar] == None:
        return False
    if jig.empty == False:
        return False
    if state.trailers_factory[trailer_factory] != None:
        return False
    if extract_id(jig) != state.hangars[hangar]:
        return False

    # Effekte
    state.hangars[hangar] = None
    state.trailers_factory[trailer_factory] = extract_id(jig)
    return True


def deliver_to_hangar(state: ProblemState, jig: Jig, hangar: int, trailer_factory: int, size: int, empty_size: int, production_line: int):
    # Preconditions
    if state.hangars[hangar] != None:
        return False
    if jig.empty == True:
        return False
    if state.trailers_factory[trailer_factory] == None:
        return False
    if extract_id(jig) != state.trailers_factory[trailer_factory]:
        return False
    if jig.jig_type.size != size:
        return False
    if jig.jig_type.empty_size != empty_size:
        return False
    if extract_id(jig) != state.production_lines[production_line].scheduled_jigs[0]:  
        return False

    # Effekte
    state.hangars[hangar] = jig
    state.trailers_factory[trailer_factory] = None
    state.production_lines[production_line].scheduled_jigs.pop(0)
    jig.empty = True
    return True


def put_down_rack(state: ProblemState, jig: Jig, rack: int, trailer_id: int, side: int):

    if side == 0:
        trailer = state.trailers_beluga[trailer_id]
    elif side == 1:
        trailer = state.trailers_factory[trailer_id]

    if jig.empty: 
        current_jig_size = jig.jig_type.empty_size
    else:
        current_jig_size = jig.jig_type.size

    # Preconditions
    if trailer == None:
        return False
    if trailer != jig:
        return False
    if state.racks[rack].current_jigs != []:
        return False
    if current_jig_size > state.racks[rack].size:
        return False
    if state.racks[rack].get_free_space(state.jigs) < current_jig_size:
        return False

    # Effekte
    if side == 0:
        state.trailers_beluga[trailer_id] = None
        state.racks[rack].current_jigs.insert(0, extract_id(jig))
    elif side == 1:
        state.trailers_factory[trailer_id] = None
        state.racks[rack].current_jigs.append(extract_id(jig))
    return True


def stack_rack(state: ProblemState, jig: Jig, next_jig: Jig, rack: int, trailer_id: int, side: int):
    # Trailer abrufen
    if side == 0:
        trailer = state.trailers_beluga[trailer_id]
    elif side == 1:
        trailer = state.trailers_factory[trailer_id]
    else:
        return False

    # Jig-Größe berechnen
    if jig.empty:
        current_jig_size = jig.jig_type.empty_size
    else:
        current_jig_size = jig.jig_type.size

    rack_obj = state.racks[rack]

    # Preconditions
    if trailer != jig:
        return False
    if extract_id(next_jig) not in rack_obj.current_jigs:
        return False
    if side == 0 and extract_id(next_jig) != rack_obj.current_jigs[0]:
        return False
    if side == 1 and extract_id(next_jig) != rack_obj.current_jigs[-1]:
        return False
    if current_jig_size > rack_obj.size:
        return False
    if rack_obj.get_free_space(state.jigs) < current_jig_size:
        return False

    # Effekte
    if side == 0:
        state.trailers_beluga[trailer_id] = None
        rack_obj.current_jigs.insert(0, extract_id(jig))
    elif side == 1:
        state.trailers_factory[trailer_id] = None
        rack_obj.current_jigs.append(extract_id(jig))
    return True


def pick_up_rack(state: ProblemState, jig: Jig, rack: int, trailer_id: int, side: int):

    if side == 0:
        trailer = state.trailers_beluga[trailer_id]
    elif side == 1:
        trailer = state.trailers_factory[trailer_id]
    else:
        return False

    if jig.empty: 
        current_jig_size = jig.jig_type.empty_size
    else:
        current_jig_size = jig.jig_type.size

    # Preconditions
    if trailer != None:
        return False
    if extract_id(jig) not in state.racks[rack].current_jigs:
        return False
    if extract_id(jig) != state.racks[rack].current_jigs[0] and extract_id(jig) != state.racks[rack].current_jigs[-1]:
        return False
    if state.racks[rack].get_free_space(state.jigs) + current_jig_size > state.racks[rack].size:
        return False

    # Effekte
    state.racks[rack].current_jigs.remove(extract_id(jig))
    if side == 0:
        state.trailers_beluga[trailer_id] = extract_id(jig)
    elif side == 1:
        state.trailers_factory[trailer_id] = extract_id(jig)

    return True


def unstack_rack(state: ProblemState, jig: Jig, next_jig: Jig, rack: int, trailer_id: int, side: int):
    # Trailer abrufen
    if side == 0:
        trailer = state.trailers_beluga[trailer_id]
    elif side == 1:
        trailer = state.trailers_factory[trailer_id]
    else:
        return False

    # Jig-Größe berechnen
    if jig.empty:
        current_jig_size = jig.jig_type.empty_size
    else:
        current_jig_size = jig.jig_type.size

    rack_obj = state.racks[rack]

    # Preconditions
    if trailer != None:
        return False
    if extract_id(jig) not in rack_obj.current_jigs:
        return False
    if extract_id(next_jig) not in rack_obj.current_jigs:
        return False
    if side == 0 and extract_id(jig) != rack_obj.current_jigs[0]:
        return False
    if side == 1 and extract_id(jig) != rack_obj.current_jigs[-1]:
        return False
    if side == 0 and extract_id(next_jig) != rack_obj.current_jigs[1]:
        return False
    if side == 1 and extract_id(next_jig) != rack_obj.current_jigs[-2]:
        return False
    if rack_obj.get_free_space(state.jigs) + current_jig_size > rack_obj.size:
        return False

    # Effekte
    rack_obj.current_jigs.remove(extract_id(jig))
    if side == 0:
        state.trailers_beluga[trailer_id] = extract_id(jig)
    elif side == 1:
        state.trailers_factory[trailer_id] = extract_id(jig)
    return True

def beluga_complete(state: ProblemState, current_beluga_index: int, next_beluga_index: int):
    current_beluga = state.belugas[current_beluga_index]
    next_beluga = state.belugas[next_beluga_index]

    # Preconditions
    if current_beluga.processed == False:
        return False
    if current_beluga.outgoing != []:
        return False
    if current_beluga.current_jigs != []:
        return False

    # Effekte
    current_beluga.processed = False
    next_beluga.processed = True
    return True