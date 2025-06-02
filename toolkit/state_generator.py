from problem_state import *
from actions import *
import hashlib
from collections import deque
from typing import Optional

def generate_following_states(state: ProblemState) -> list[ProblemState]:
  
    following_states : list[ProblemState] = []

    # # Beluga Complete
    # copy = state.deep_copy()
    # if beluga_complete(copy):
    #     following_states.append(copy)

    # Unload Beluga
    copy = state.deep_copy()
    if unload_beluga(copy):
        following_states.append(copy) 

    # Load Beluga
    for trailer_id in range(len(copy.trailers_beluga)):
        copy = state.deep_copy()
        if load_beluga(copy, trailer_id):
            following_states.append(copy)

    # Stack-Rack
    for side in [0, 1]:
        for rack in range(len(state.racks)):
            if side == 0: 
                for trailer_id in range(len(state.trailers_beluga)):
                    copy = state.deep_copy()
                    if stack_rack(copy, rack, trailer_id, side):
                        following_states.append(copy)
            else:
                for trailer_id in range(len(state.trailers_factory)):
                    copy = state.deep_copy()
                    if stack_rack(copy, rack, trailer_id, side):
                        following_states.append(copy)
    
    # Unstack-Rack
    for side in [0, 1]:
        for rack in range(len(state.racks)):
            if side == 0: 
                for trailer_id in range(len(state.trailers_beluga)):
                    copy = state.deep_copy()
                    if unstack_rack(copy, rack, trailer_id, side):
                        following_states.append(copy)
            else:
                for trailer_id in range(len(state.trailers_factory)):
                    copy = state.deep_copy()
                    if unstack_rack(copy, rack, trailer_id, side):
                        following_states.append(copy)
        
    # deliver to Hangar
    for hangar in range(len(state.hangars)):
        for trailer_id in range(len(state.trailers_factory)):
            copy = state.deep_copy()
            if deliver_to_hangar(copy, hangar, trailer_id):
                following_states.append(copy)
    
    # get from Hangar
    for hangar in range(len(state.hangars)):
        for trailer_id in range(len(state.trailers_factory)):
            copy = state.deep_copy()
            if get_from_hangar(copy, hangar, trailer_id):
                following_states.append(copy)

    return following_states


def get_state_hash(state: ProblemState) -> str:
    state_string = str(state)  # oder eine gezielte serialisierte Darstellung
    return hashlib.sha1(state_string.encode('utf-8')).hexdigest()


def breadth_first_search(start_state: ProblemState) -> Optional[list[ProblemState]]:
    visited = set()
    queue = deque()
    queue.append((start_state, [start_state]))  # (aktueller Zustand, Pfad bis hierher)

    while queue:
        current_state, path = queue.popleft()

        # Zieltest
        if goal(current_state):
            return path

        # Zustand als besucht markieren
        state_id = get_state_hash(current_state)  # funktioniert nur, wenn __hash__ implementiert ist
        if state_id in visited:
            continue
        visited.add(state_id)

        # Nachfolger erzeugen
        for next_state in generate_following_states(current_state):
            next_state_id = get_state_hash(next_state)
            if next_state_id not in visited:
                queue.append((next_state, path + [next_state]))

    return None

def main():
    problem_state = load_from_json(r"toolkit\out\problem2.json")
    print(problem_state)

    print(breadth_first_search(problem_state))

main()