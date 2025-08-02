'TODO: add docstrings to all classes and methods'
'TODO: viel zu viel code, vorschlag: trennung in base-classes, mcts-api, utility-functions, etc.'

import json
import copy
from .action import (
    left_stack_rack, right_stack_rack, left_unstack_rack, right_unstack_rack,
    load_beluga, unload_beluga, get_from_hangar, deliver_to_hangar
)
import numpy as np
from rl.agents.low_level.heuristics import decide_parameters

class JigType:
    def __init__(self, name: str, size_empty: int, size_loaded: int):
        self.name = name
        self.size_empty = size_empty
        self.size_loaded = size_loaded

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __ne__(self, other):
        return self.name != other.name
    
    def __eq__(self, other):
        return self.name == other.name


class Jig:
    def __init__(self, jig_type: JigType, empty: bool):
        self.jig_type = jig_type
        self.empty = empty

    def __str__(self):
        return str(self.jig_type) + " | " + str(self.empty)
    
    def copy(self):
        return Jig(self.jig_type, self.empty)


class Beluga:
    def __init__(self, current_jigs: list[int], outgoing: list[JigType]):
        self.current_jigs = current_jigs
        self.outgoing = outgoing

    def __str__(self):
        return "current_jigs = " + str(self.current_jigs) + " | outgoing = " + str(self.outgoing)
    
    def copy(self):
        # current_jigs und outgoing sind Listen von ints/Objekten → flache Kopie reicht
        return Beluga(self.current_jigs[:], self.outgoing[:])


class Rack:
    def __init__(self, size: int, current_jigs: list[int]):
        self.size = size
        self.current_jigs = current_jigs

    def __str__(self):
        return "size = " + str(self.size) + " | current_jigs = " + str(self.current_jigs)
    
    def get_free_space(self, all_jigs: list[Jig]) -> int:
        total_used_space = 0
        for jig_id in self.current_jigs:
            jig = all_jigs[jig_id - 1]
            jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded 
            total_used_space += jig_size
        
        remaining_space = self.size - total_used_space
        return remaining_space

    def copy(self):
        return Rack(self.size, self.current_jigs[:])

class ProductionLine:
    def __init__(self, scheduled_jigs: list[int]):
        self.scheduled_jigs = scheduled_jigs

    def __str__(self):
        return "scheduled_jigs = " + str(self.scheduled_jigs)
    
    def copy(self):
        return ProductionLine(self.scheduled_jigs[:])


class ProblemState:
    def __init__(self, jigs : list[Jig], belugas: list[Beluga], trailers_beluga: list[int | None], trailers_factory: list[int | None], racks: list[Rack], production_lines: list[ProductionLine], hangars: list[int | None]):
        self.jigs = jigs
        self.belugas = belugas
        self.trailers_beluga = trailers_beluga
        self.trailers_factory = trailers_factory
        self.racks = racks
        self.production_lines = production_lines
        self.hangars = hangars

        # Subgoals
        # for reward (High-Level) and evaluation (Low-Level-MCTS) 
        self.belugas_unloaded = 0 #counter
        self.belugas_finished = 0 #counter
        self.production_lines_finished = 0 #counter
        self.total_lines = len(self.production_lines) # total production lines, for evaluation
        self.total_belugas = len(self.belugas) # total belugas, for evaluation
        self.problem_solved = False
        


    def copy(self):
        new_state = ProblemState(
            jigs=[jig.copy() for jig in self.jigs],
            belugas=[beluga.copy() for beluga in self.belugas],
            trailers_beluga=self.trailers_beluga[:],
            trailers_factory=self.trailers_factory[:],
            racks=[rack.copy() for rack in self.racks],
            production_lines=[pl.copy() for pl in self.production_lines],
            hangars=self.hangars[:]  # Liste aus ints oder None
        )
        new_state.belugas_unloaded = self.belugas_unloaded
        new_state.belugas_finished = self.belugas_finished
        new_state.production_lines_finished = self.production_lines_finished
        new_state.total_lines = self.total_lines
        new_state.total_belugas = self.total_belugas  # Hinzugefügt!
        new_state.problem_solved = self.problem_solved
        return new_state
    
    '''
    ab hier API für MCTS:
    actions:
    -  clone() -> ProblemState, copies the current state
    -  is_terminal() -> bool, checks if the state is terminal e.g. goal state
    -  evaluate() -> float, evaluates the current state (e.g., score)
        - get_subgoals() Hilfsfunktion für evaluate(), gibt die Zwischenziele zurück
    -  def get_possible_parameter_actions(self, action: HighLevelAction) -> List[ParameterAction]
    '''

    def clone(self):
        return self.copy()
    
    def is_terminal(self):
        return len(self.belugas) == 0 and len(self.production_lines) == 0

    def evaluate(self, depth: int, mu = 0.05) -> float:
            score = 0.0
            subgoals = self.get_subgoals()
            score += sum(subgoals.values())
            # Abwertung je Tiefe des Pfades
            score -= mu * depth
            return score
    

    def get_subgoals(self) -> dict[str, float]:
        self.belugas_finished = self.total_belugas - len(self.belugas)
        self.production_lines_finished = self.total_lines - len(self.production_lines)

        
        if len(self.belugas) == 0 and len(self.production_lines) == 0:
            self.problem_solved = True
        return {
            "subgoal_1": self.belugas_unloaded * 15,
            "subgoal_2": self.belugas_finished * 60,
            "subgoal_3": self.production_lines_finished * 100,
            "goal": self.problem_solved * 1000    
        }
        
    def apply_action(self, action_name, params):
        params = list(params.values()) if isinstance(params, dict) else list(params)  # ensure params is a list
        #action_name, params = candidate
        if action_name == "left_stack_rack":
            return left_stack_rack(self, *params)
        elif action_name == "right_stack_rack":
            return right_stack_rack(self, *params)
        elif action_name == "left_unstack_rack":
            return left_unstack_rack(self, *params)
        elif action_name == "right_unstack_rack":
            return right_unstack_rack(self, *params)
        elif action_name == "load_beluga":
            return load_beluga(self, *params)
        elif action_name == "unload_beluga":
            return unload_beluga(self)
        elif action_name == "get_from_hangar":
            return get_from_hangar(self, *params)
        elif action_name == "deliver_to_hangar":
            return deliver_to_hangar(self, *params)
        else:
            raise NotImplementedError(f"Aktionsname nicht bekannt: {action_name}")
      

    def check_action_valid(self, action_name: str, params=None) -> bool:
        """
        Prüft, ob eine Aktion mit den angegebenen Parametern gültig ist,
        ohne den aktuellen State zu verändern.
        """
        state_copy = self.copy()
        
        try:
            if action_name == "left_stack_rack":
                return left_stack_rack(state_copy, *params)
            elif action_name == "right_stack_rack":
                return right_stack_rack(state_copy, *params)
            elif action_name == "left_unstack_rack":
                return left_unstack_rack(state_copy, *params)
            elif action_name == "right_unstack_rack":
                return right_unstack_rack(state_copy, *params)
            elif action_name == "load_beluga":
                return load_beluga(state_copy, *params)
            elif action_name == "unload_beluga":
                return unload_beluga(state_copy)
            elif action_name == "get_from_hangar":
                return get_from_hangar(state_copy, *params)
            elif action_name == "deliver_to_hangar":
                return deliver_to_hangar(state_copy, *params)
            else:
                return False
        except Exception as e:
            print(f"Fehler bei Aktion {action_name} mit Params {params}: {e}")
            return False

    def enumerate_valid_params(self, action):
        action_name = action
        params = []
        
        if action_name == "left_stack_rack":
            all_param = [(rack_id, trailer_id) 
                        for rack_id in range(len(self.racks)) 
                        for trailer_id in range(len(self.trailers_beluga))]
                        
            for t in all_param:
                if self.check_action_valid(action_name, t):
                    params.append(t)
        
        elif action_name == "right_stack_rack":
            all_param = [(rack_id, trailer_id) 
                        for rack_id in range(len(self.racks)) 
                        for trailer_id in range(len(self.trailers_factory))]
                        
            for t in all_param:
                if self.check_action_valid(action_name, t):
                    params.append(t)

        elif action_name == "left_unstack_rack":
            all_param = [(rack_id, trailer_id) 
                        for rack_id in range(len(self.racks)) 
                        for trailer_id in range(len(self.trailers_beluga))]
                        
            for t in all_param:
                if self.check_action_valid(action_name, t):
                    params.append(t)

        elif action_name == "right_unstack_rack":
            all_param = [(rack_id, trailer_id) 
                        for rack_id in range(len(self.racks)) 
                        for trailer_id in range(len(self.trailers_factory))]
                        
            for t in all_param:
                if self.check_action_valid(action_name, t):
                    params.append(t)
        
        elif action_name == "load_beluga":
            all_param = [trailer_id for trailer_id in range(len(self.trailers_beluga))]
            for t in all_param:
                if self.check_action_valid(action_name, (t, None)):
                    params.append((t, None))

        elif action_name == "deliver_to_hangar":
            all_param = [(hangar_id, trailer_id) 
                        for hangar_id in range(len(self.hangars)) 
                        for trailer_id in range(len(self.trailers_factory))]
                        
            for t in all_param:
                if self.check_action_valid(action_name, t):
                    params.append(t)
        
        elif action_name == "get_from_hangar":
            all_param = [(hangar_id, trailer_id) 
                        for hangar_id in range(len(self.hangars)) 
                        for trailer_id in range(len(self.trailers_factory))]
                        
            for t in all_param:
                if self.check_action_valid(action_name, t):
                    params.append(t)
        
        return params



    def get_possible_actions(self):
        """
        Gibt eine Liste aller möglichen Aktionen zurück.
        Eine Aktion gilt als möglich, wenn mindestens eine gültige Parameterkombination existiert.
        """
        # action = ("action_name", "params")
        possible_actions = []
        
        # Prüfe unload_beluga (keine Parameter)
        if self.check_action_valid("unload_beluga"):
            possible_actions.append(("unload_beluga", {}))
        
        # Prüfe Aktionen mit Parametern
        param_actions = [
            "left_stack_rack",
            "right_stack_rack",
            "left_unstack_rack",
            "right_unstack_rack",
            "load_beluga",
            "get_from_hangar",
            "deliver_to_hangar"
        ]
        for action in param_actions:
            # all actions with parameters, if there are no params, no legal actions
            params = self.enumerate_valid_params(action)
            possible_actions.extend([(action, param) for param in params])
        
        
        return possible_actions


    def beluga_complete(self) -> bool:
        """Mark beluga as complete."""
        if not self.belugas:
            return False
            
        beluga = self.belugas[0]
        if beluga.outgoing or beluga.current_jigs:
            return False
        
        # Effekte
        self.belugas.pop(0)
        return True 
    


    def get_observation_high_level(self):
        # Return the current state of the environment for a high-level agent as array
        # High-Level-Agents converts array into tensor

        ### CURRENTLY MAX 10 RACKS
        n_racks = 10

        out = np.zeros(10 + 3*n_racks)

        needed_outgoing_types = []
        needed_in_production_lines = []

        for pl in self.production_lines:
            if len(pl.scheduled_jigs) > 0:
                needed_in_production_lines.append(pl.scheduled_jigs[0])

        # First slot 0 beluga
        if len(self.belugas) > 0:
            out[0] = max(0, min(len(self.belugas[0].current_jigs), 1))
            if out[0] == 0:
                needed_outgoing_types = self.belugas[0].outgoing
        else:
            out[0] = -1

        # Slot 1-3 Beluga Trailer
        slot = 1
        for i in range(3):
            if i < len(self.trailers_beluga):
                if self.trailers_beluga[i] is None:
                    out[slot + i] = 0.5
                else:
                    if self.jigs[self.trailers_beluga[i]].empty and out[0] == 0:
                        if needed_outgoing_types.__contains__(
                                self.jigs[self.trailers_beluga[i]].jig_type):
                            out[slot + i] = 0
                        else:
                            out[slot + i] = 0.25
                    else:
                        out[slot + i] = 1
            else:
                out[slot + i] = -1

        # Slot 4-6 Factory Trailer
        slot = 4
        for i in range(3):
            if i < len(self.trailers_factory):
                if self.trailers_factory[i] is None:
                    out[slot + i] = 0.5
                else:
                    if not self.jigs[self.trailers_factory[i]].empty:
                        if needed_in_production_lines.__contains__(self.trailers_factory[i]):
                            out[slot + i] = 1
                        else:
                            out[slot + i] = 0.75
                    else:
                        out[slot + i] = 0
            else:
                out[slot + i] = -1

        # Slot 7-9 Hangars
        slot = 7
        for i in range(3):
            if i < len(self.hangars):
                if self.hangars[i] is None:
                    out[slot + i] = 0
                else:
                    out[slot + i] = 1
            else:
                out[slot + i] = -1

        # Slot 10-39 Racks
        slot = 10
        for i in range(n_racks):
            if i < len(self.racks):
                rack = self.racks[i]
                items = len(rack.current_jigs)
                if items == 0:
                    out[slot + i * 3] = 0
                    out[slot + i * 3 + 1] = 0
                    out[slot + i * 3 + 2] = 0

                else:
                    out[slot + i * 3] = 0
                    out[slot + i * 3 + 1] = 0
                    out[slot + i * 3 + 2] = rack.get_free_space(self.jigs)/rack.size
                    for k in range(items):
                        jig = self.jigs[rack.current_jigs[k]]
                        if jig.empty and needed_outgoing_types.__contains__(jig.jig_type):
                            out[slot + i * 3] = (items - k) / items
                            continue
                    for k in range(items):
                        if needed_in_production_lines.__contains__(rack.current_jigs[k]):
                            out[slot + i * 3 + 1] = (k + 1) / items
                            continue
            else:
                out[slot + i * 3] = -1
                out[slot + i * 3 + 1] = -1
                out[slot + i * 3 + 2] = -1


        return out



    def __str__(self):
        count = 0
        out = "jigs:\n"
        for jig in self.jigs:
            out += "\t" + str(count) + ": " + str(jig) + "\n"
            count += 1
        out += "belugas:\n"
        count = 0
        for beluga in self.belugas:
            out += "\t" + str(count) + ": " + str(beluga) + "\n"
            count += 1
        out += "trailers_beluga: " + str(self.trailers_beluga) + "\n"
        out += "trailers_factory: " + str(self.trailers_factory) + "\n"
        out += "racks:\n"
        count = 0
        for rack in self.racks:
            out += "\t" + str(count) + ": " + str(rack) + "\n"
            count += 1
        out += "production_lines:\n"
        count = 0
        for production_line in self.production_lines:
            out += "\t" + str(count) + ": " + str(production_line) + "\n"
            count += 1
        out += "hangars: " + str(self.hangars)
        return out

    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(str(self)) 

    def __eq__(self, other):    
        return str(self) == str(other)
    

def extract_id(name: str) -> int:
    name = name.replace("jig", "")
    return int(name) - 1

def get_name_from_id(id: int) -> str:
    return "jig" + "0" * (4-len(str(id))) + str(id)


def get_type(name: str) -> JigType | None:
    if name == "typeA":
        return JigType("typeA", 4, 4)
    elif name == "typeB":
        return JigType("typeB", 8, 11)
    elif name == "typeC":
        return JigType("typeC", 9, 18)
    elif name == "typeD":
        return JigType("typeD", 18, 25)
    elif name == "typeE":
        return JigType("typeE", 32, 32)
    return None

def load_from_json(path) -> ProblemState:

    data = open(path, "r")
    dictionary = json.loads(data.read())
    data.close()

    jig_data = dictionary["jigs"]

    jigs: list[Jig] = []
    for jig_n, jig in jig_data.items():
        jigs.append(Jig(get_type(jig["type"]), jig["empty"]))

    beluga_data = dictionary["flights"]
    belugas: list[Beluga] = []
    for beluga in beluga_data:
        incoming: list[int] = []
        outgoing: list[JigType] = []
        for entry in beluga["incoming"]:
            incoming.append(extract_id(entry))
        for entry in beluga["outgoing"]:
            outgoing.append(get_type(entry))
        belugas.append(Beluga(incoming, outgoing))

    production_lines_data = dictionary["production_lines"]
    production_lines: list[ProductionLine] = []
    for production_line in production_lines_data:
        schedule: list[int] = []
        for entry in production_line["schedule"]:
            schedule.append(extract_id(entry))
        production_lines.append(ProductionLine(schedule))

    racks_data = dictionary["racks"]
    racks: list[Rack] = []
    for rack in racks_data:
        storage: list[int] = []
        for entry in rack["jigs"]:
            storage.append(extract_id(entry))
        racks.append(Rack(rack["size"], storage))

    hangars: list[Jig | None] = [None] * len(dictionary["hangars"])
    trailers_beluga: list[Jig | None] = [None] * len(dictionary["trailers_beluga"])
    trailers_factory: list[Jig | None] = [None] * len(dictionary["trailers_factory"])

    
    return ProblemState(jigs, belugas, trailers_beluga, trailers_factory, racks, production_lines, hangars)
