import json
import copy
from json import JSONEncoder

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


    def deep_copy(self):
        return ProblemState(
            jigs=copy.deepcopy(self.jigs),
            belugas=copy.deepcopy(self.belugas),
            trailers_beluga=copy.deepcopy(self.trailers_beluga),
            trailers_factory=copy.deepcopy(self.trailers_factory),
            racks=copy.deepcopy(self.racks),
            production_lines=copy.deepcopy(self.production_lines),
            hangars=copy.deepcopy(self.hangars)
        )


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

def extract_id(name: str) -> int:
    name = name.replace("jig", "")
    return int(name) - 1

def get_name_from_id(id: int) -> str:
    return "jig" + "0" * (4-len(str(id))) + str(id)

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

class StateEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


def save_to_json(path: str, problem: ProblemState) -> None:
    file = open(path, "w")
    data : dict = {}


    #Trailers Beluga
    count = 1
    trailers_beluga = []
    for trailer in problem.trailers_beluga:
        dummy = {"name": "beluga_trailer_" + str(count), "slot": get_name_from_id(trailer)}
        trailers_beluga.append(dummy)
        count += 1

    data["trailers_beluga"] = trailers_beluga

    #Trailer Factory
    count = 1
    trailers_factory = []
    for trailer in problem.trailers_factory:
        dummy = {"name": "factory_trailer_" + str(count), "slot": get_name_from_id(trailer)}
        trailers_factory.append(dummy)
        count += 1

    data["trailers_factory"] = trailers_factory

    #Hangar
    count = 1
    hangars = []
    for hangar in problem.hangars:
        hangars.append("hangar" + str(count))
        count += 1

    data["hangars"] = hangars

    #Jig Types
    jig_types = {
        "typeA": {
            "name": "typeA",
            "size_empty": 4,
            "size_loaded": 4
        },
        "typeB": {
            "name": "typeB",
            "size_empty": 8,
            "size_loaded": 11
        },
        "typeC": {
            "name": "typeC",
            "size_empty": 9,
            "size_loaded": 18
        },
        "typeD": {
            "name": "typeD",
            "size_empty": 18,
            "size_loaded": 25
        },
        "typeE": {
            "name": "typeE",
            "size_empty": 32,
            "size_loaded": 32
        }
    }

    data["jig_types"] = jig_types

    #Racks
    count = 0
    racks = []
    for rack in problem.racks:
        dummy = {}
        dummy["name"] = "rack" + "0" * (2-len(str(count))) + str(count)
        dummy["size"] = rack.size
        jigs = []
        for jid in rack.current_jigs:
            jigs.append(get_name_from_id(jid))

        dummy["jigs"] = jigs
        racks.append(dummy)
        count += 1

    data["racks"] = racks

    #Jigs
    count = 1
    jigs = {}
    for jig in problem.jigs:
        dummy = {"name": get_name_from_id(count), "type": str(jig.jig_type.name), "empty": jig.empty}
        jigs[get_name_from_id(count)] = dummy
        count += 1

    data["jigs"] = jigs

    #Productionlines
    count = 0
    production_lines = []
    for production_line in problem.production_lines:
        dummy = {}
        dummy["name"] = "pl" + str(count)
        schedule = []
        for jid in production_line.scheduled_jigs:
            schedule.append(get_name_from_id(jid))
        dummy["schedule"] = schedule
        production_lines.append(dummy)
        count += 1

    data["production_lines"] = production_lines

    #Flights
    count = 1
    flights = []
    for beluga in problem.belugas:
        dummy = {}
        dummy["name"] = "beluga" + str(count)
        incoming = []
        for jid in beluga.current_jigs:
            incoming.append(get_name_from_id(jid))
        dummy["incoming"] = incoming
        dummy["outgoing"] = beluga.outgoing
        flights.append(dummy)
        count += 1

    data["flights"] = flights

    file.write(json.dumps(data, indent=4, cls=StateEncoder))
