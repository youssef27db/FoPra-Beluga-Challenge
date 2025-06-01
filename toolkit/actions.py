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

