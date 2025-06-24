import json
import copy

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


class Beluga:
    def __init__(self, current_jigs: list[int], outgoing: list[JigType]):
        self.current_jigs = current_jigs
        self.outgoing = outgoing

    def __str__(self):
        return "current_jigs = " + str(self.current_jigs) + " | outgoing = " + str(self.outgoing)


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
            total_used_space += jig.jig_type.size_loaded
        
        remaining_space = self.size - total_used_space
        return remaining_space


class ProductionLine:
    def __init__(self, scheduled_jigs: list[int]):
        self.scheduled_jigs = scheduled_jigs

    def __str__(self):
        return "scheduled_jigs = " + str(self.scheduled_jigs)


class ProblemState:
    def __init__(self, jigs : list[Jig], belugas: list[Beluga], trailers_beluga: list[int | None], trailers_factory: list[int | None], racks: list[Rack], production_lines: list[ProductionLine], hangars: list[int | None]):
        self.jigs = jigs
        self.belugas = belugas
        self.trailers_beluga = trailers_beluga
        self.trailers_factory = trailers_factory
        self.racks = racks
        self.production_lines = production_lines
        self.hangars = hangars


    def copy(self):
        return ProblemState(
            jigs=copy.deepcopy(self.jigs),
            belugas=copy.deepcopy(self.belugas),
            trailers_beluga=copy.deepcopy(self.trailers_beluga),
            trailers_factory=copy.deepcopy(self.trailers_factory),
            racks=copy.deepcopy(self.racks),
            production_lines=copy.deepcopy(self.production_lines),
            hangars=copy.deepcopy(self.hangars)
        )
    



    def is_terminal(self):
        return False  # oder nach einer Tiefe / Zielbedingung

    def apply_action(self, action_name, params):
      if action_name == "left_stack_rack":
          return self.left_stack_rack(*params)
      elif action_name == "right_stack_rack":
          return self.right_stack_rack(*params)
      elif action_name == "left_unstack_rack":
          return self.left_unstack_rack(*params)
      elif action_name == "right_unstack_rack":
          return self.right_unstack_rack(*params)
      elif action_name == "load_beluga":
          return self._load_beluga(*params)
      elif action_name == "unload_beluga":
          return self._unload_beluga()
      elif action_name == "get_from_hangar":
          return self._get_from_hangar(*params)
      elif action_name == "deliver_to_hangar":
          return self._deliver_to_hangar(*params)
      else:
          raise NotImplementedError(f"Aktionsname nicht bekannt: {action_name}")


    def enumerate_valid_params(self, action_type):
      params = []
      if action_type == "left_stack_rack":
          for rack_id, rack in enumerate(self.racks):
              for trailer_id, jig_id in enumerate(self.trailers_beluga):
                  if jig_id is None:
                      continue
                  jig = self.jigs[jig_id]
                  jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded
                  if rack.get_free_space(self.jigs) >= jig_size:
                      params.append((rack_id, trailer_id))
      elif action_type == "right_stack_rack":
          for rack_id, rack in enumerate(self.racks):
              for trailer_id, jig_id in enumerate(self.trailers_factory):
                  if jig_id is None:
                      continue
                  jig = self.jigs[jig_id]
                  jig_size = jig.jig_type.size_empty if jig.empty else jig.jig_type.size_loaded
                  if rack.get_free_space(self.jigs) >= jig_size:
                      params.append((rack_id, trailer_id))
      elif action_type == "left_unstack_rack":
          for rack_id, rack in enumerate(self.racks):
              if not rack.current_jigs:
                  continue
              for trailer_id, trailer in enumerate(self.trailers_beluga):
                  if trailer is None:
                      params.append((rack_id, trailer_id))
      elif action_type == "right_unstack_rack":
          for rack_id, rack in enumerate(self.racks):
              if not rack.current_jigs:
                  continue
              for trailer_id, trailer in enumerate(self.trailers_factory):
                  if trailer is None:
                      params.append((rack_id, trailer_id))
      return params


    def _is_action_legal(self, action_name: str, params: tuple) -> bool:
      state_copy = self.copy()
      try:
          return state_copy.apply_action(action_name, params)
      except:
          return False
    
    def enumerate_param_candidates(self, action_name):
      if action_name == "unload_beluga":
          return [()]

      elif action_name == "load_beluga":
          return [(i,) for i in range(len(self.trailers_beluga))]

      elif action_name in ["left_stack_rack", "left_unstack_rack"]:
          return [
              (rack_id, trailer_id)
              for rack_id in range(len(self.racks))
              for trailer_id in range(len(self.trailers_beluga))
          ]

      elif action_name in ["right_stack_rack", "right_unstack_rack"]:
          return [
              (rack_id, trailer_id)
              for rack_id in range(len(self.racks))
              for trailer_id in range(len(self.trailers_factory))
          ]

      elif action_name in ["get_from_hangar", "deliver_to_hangar"]:
          return [
              (hangar_id, trailer_id)
              for hangar_id in range(len(self.hangars))
              for trailer_id in range(len(self.trailers_factory))
          ]

      else:
          return []


    def get_possible_actions(self):
      all_actions = [
          "unload_beluga",
          "load_beluga",
          "get_from_hangar",
          "deliver_to_hangar",
          "left_stack_rack",
          "right_stack_rack",
          "left_unstack_rack",
          "right_unstack_rack"
      ]

      actions = []
      for action_name in all_actions:
          param_candidates = self.enumerate_param_candidates(action_name)
          for params in param_candidates:
              if self._is_action_legal(action_name, params):
                  actions.append((action_name, params))
      return actions


    def get_score(self):
        return 1 


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