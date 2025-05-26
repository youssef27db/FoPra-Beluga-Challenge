import json


class JigType:
    def __init__(self, name: str, size_empty: int, size_loaded: int):
        self.name = name
        self.size_empty = size_empty
        self.size_loaded = size_loaded

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


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


class ProductionLine:
    def __init__(self, scheduled_jigs: list[int]):
        self.scheduled_jigs = scheduled_jigs

    def __str__(self):
        return "scheduled_jigs = " + str(self.scheduled_jigs)


class ProblemState:

    def __init__(self, jigs : list[Jig], belugas: list[Beluga], trailers_beluga: list[Jig | None], trailers_factory: list[Jig | None], racks: list[Rack], production_lines: list[ProductionLine], hangars: list[JigType | None]):
        self.jigs = jigs
        self.belugas = belugas
        self.trailers_beluga = trailers_beluga
        self.trailers_factory = trailers_factory
        self.racks = racks
        self.production_lines = production_lines
        self.hangars = hangars


    def __str__(self):
        count = 1
        out = "jigs:\n"
        for jig in self.jigs:
            out += "\t" + str(count) + ": " + str(jig) + "\n"
            count += 1
        out += "belugas:\n"
        count = 1
        for beluga in self.belugas:
            out += "\t" + str(count) + ": " + str(beluga) + "\n"
            count += 1
        out += "trailers_beluga: " + str(self.trailers_beluga) + "\n"
        out += "trailers_factory: " + str(self.trailers_factory) + "\n"
        out += "racks:\n"
        count = 1
        for rack in self.racks:
            out += "\t" + str(count) + ": " + str(rack) + "\n"
            count += 1
        out += "production_lines:\n"
        count = 1
        for production_line in self.production_lines:
            out += "\t" + str(count) + ": " + str(production_line) + "\n"
            count += 1
        out += "hangars: " + str(self.hangars)
        return out

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
    return int(name)

def main():

    data = open("out/problem.json", "r")
    dictionary = json.loads(data.read())
    data.close()

    type_a = JigType("typeA", 4, 4)
    type_b = JigType("typeB", 8, 11)
    type_c = JigType("typeC", 9, 18)
    type_d = JigType("typeD", 18, 25)
    type_e = JigType("typeE", 32, 32)

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

    hangars: list[JigType | None] = [None] * len(dictionary["hangars"])
    trailers_beluga: list[Jig | None] = [None] * len(dictionary["trailers_beluga"])
    trailers_factory: list[Jig | None] = [None] * len(dictionary["trailers_factory"])

    print(ProblemState(jigs, belugas, trailers_beluga, trailers_factory, racks, production_lines, hangars))


if __name__ == '__main__':
    main()