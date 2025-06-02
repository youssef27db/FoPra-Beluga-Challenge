from problem_state import *

def load_beluga(state: ProblemState, trailer_beluga: int):
    jig_id = state.trailers_beluga[trailer_beluga]

    # Teste ob Beluga vorhanden ist
    if len(state.belugas) == 0:
        return False
    beluga = state.belugas[0]

    #Preconditions
    # Trailer darf nicht leer sein
    if jig_id == None:
        return False
    # Jig muss leer sein
    if state.jigs[jig_id].empty == False:
        return False
    # Beluga darf nicht beladen werden wenn alle die Outgoing-Types bereits beladen sind
    if beluga.outgoing == []:
        return False
    # JigType muss mit der Outgoing-Liste des Belugas übereinstimmen
    if state.jigs[jig_id].jig_type != beluga.outgoing[0]:
        return False

    #Effekte
    beluga.outgoing.pop(0)
    state.trailers_beluga[trailer_beluga] = None

    if beluga.current_jigs == []:
        if beluga.outgoing == []:
            state.belugas.pop(0)


    return True
    

def unload_beluga(state: ProblemState):
    # Leeren Trailer-Beluga finden
    trailer_beluga = 0
    for trailer in state.trailers_beluga:
        if trailer is None:
            break
        trailer_beluga += 1

    # Teste ob Beluga vorhanden ist
    if len(state.belugas) == 0:
        return False
    beluga = state.belugas[0]

    # Preconditions
    # Beihnaltet Beluga Jigs
    if beluga.current_jigs == []:
        return False
    # Kein leerer Trailer-Beluga gefunden
    if trailer_beluga >= len(state.trailers_beluga):
        return False

    # Effekte
    state.trailers_beluga[trailer_beluga] = beluga.current_jigs[-1]
    beluga.current_jigs.pop(-1)

    if beluga.current_jigs == []:
        if beluga.outgoing == []:
            state.belugas.pop(0)

    return True


def get_from_hangar(state: ProblemState, hangar: int, trailer_factory: int):
    # Preconditions
    # Hangar muss belegt sein
    if state.hangars[hangar] == None:
        return False
    # Jig im Hangar muss leer sein
    if state.jigs[state.hangars[hangar]].empty == False:
        return False
    # Trailer-Fabrik darf nicht belegt sein
    if state.trailers_factory[trailer_factory] != None:
        return False

    # Effekte
    state.trailers_factory[trailer_factory] = state.hangars[hangar]
    state.hangars[hangar] = None
    return True


def deliver_to_hangar(state: ProblemState, hangar: int, trailer_factory: int):
    # Preconditions
    # Hangar darf nicht belegt sein
    if state.hangars[hangar] != None:
        return False
    # Trailer-Fabrik muss belegt sein
    if state.trailers_factory[trailer_factory] == None:
        return False

    jig_id = state.trailers_factory[trailer_factory]
    #Jig darf nicht leer sein
    if state.jigs[jig_id].empty == True:
        return False
    
    # Suche nach dem Jig in der Produktionslinie
    production_line_id : int = 0
    for production_line in state.production_lines:
        if jig_id == production_line.scheduled_jigs[0]:
            break
        production_line_id += 1
    
    # falls Jig nicht in der Produktionslinie gefunden wurde
    if production_line_id >= len(state.production_lines):
        return False

    # Effekte
    state.production_lines[production_line_id].scheduled_jigs.pop(0)
    state.hangars[hangar] = jig_id
    state.jigs[jig_id].empty = True
    state.trailers_factory[trailer_factory] = None

    # Wenn die Produktionslinie keine Jigs mehr hat, entfernen wir sie
    if state.production_lines[production_line_id].scheduled_jigs == []:
        state.production_lines.pop(production_line_id)

    return True


def stack_rack(state: ProblemState, rack: int, trailer_id: int, side: int):
    # Preconditions
    jig_id = None
    current_jig_size = 0

    if side == 0:
        jig_id = state.trailers_beluga[trailer_id]
    elif side == 1:
        jig_id = state.trailers_factory[trailer_id]

    # Trailer darf nicht leer sein
    if jig_id == None:
        return False

    if state.jigs[jig_id].empty: 
        current_jig_size = state.jigs[jig_id].jig_type.size_empty
    else:
        current_jig_size = state.jigs[jig_id].jig_type.size_loaded

    # Rack muss genug Platz haben
    if state.racks[rack].get_free_space(state.jigs) < current_jig_size:
        return False

    # Effekte
    if side == 0:
        state.trailers_beluga[trailer_id] = None
        state.racks[rack].current_jigs.insert(0, jig_id)
    elif side == 1:
        state.trailers_factory[trailer_id] = None
        state.racks[rack].current_jigs.append(jig_id)
    return True


def unstack_rack(state: ProblemState, rack: int, trailer_id: int, side: int):
    # Preconditions
    jig_id = None

    if side == 0:
        jig_id = state.trailers_beluga[trailer_id]
    elif side == 1:
        jig_id = state.trailers_factory[trailer_id]

    # Trailer muss leer sein
    if jig_id != None:
        return False    

    # Rack darf nicht leer sein
    if state.racks[rack].current_jigs == []:
        return False


    # Effekte
    if side == 0:
        state.trailers_beluga[trailer_id] = state.racks[rack].current_jigs.pop(0)
    elif side == 1:
        state.trailers_factory[trailer_id] = state.racks[rack].current_jigs.pop(-1)
    return True

def beluga_complete(state: ProblemState):
    # Preconditions
    if len(state.belugas) == 0:
        return False
    current_beluga = state.belugas[0]

    # Beluga Outgoing-Types muss leer sein
    if current_beluga.outgoing != []:
        return False
    # Beluga muss keine Jigs haben
    if current_beluga.current_jigs != []:
        return False

    # Effekte
    state.belugas.pop(0)
    return True

def goal(state: ProblemState) -> bool:
    # Beluga liste muss leer sein
    if state.belugas != []:
        return False
    
    # produktion_lines muss leer sein
    if state.production_lines != []:
        return False

    return True