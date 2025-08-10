"""!
@file problem_filter.py
@brief Problem filtering utilities for the Beluga Challenge

This module provides functions to filter and modify problem instances
by adjusting the number of jigs, belugas, production lines, and racks
to create problems of different complexity levels.
"""

import json
import random
import os
from pathlib import Path

JIG_TYPES = {
    "typeA": {"name": "typeA", "size_empty": 4, "size_loaded": 4},
    "typeB": {"name": "typeB", "size_empty": 8, "size_loaded": 11},
    "typeC": {"name": "typeC", "size_empty": 9, "size_loaded": 18},
    "typeD": {"name": "typeD", "size_empty": 18, "size_loaded": 25},
    "typeE": {"name": "typeE", "size_empty": 32, "size_loaded": 32}
}

def filter_problem(input_file, output_file, max_jigs, max_belugas, max_prod_lines, max_racks):
    """!
    @brief Filter a problem instance to reduce its complexity
    
    This function takes a problem JSON file and creates a simplified version
    by limiting the number of jigs, belugas, production lines, and racks.
    
    @param input_file Path to the input problem JSON file
    @param output_file Path for the output filtered problem JSON file
    @param max_jigs Maximum number of jigs to keep
    @param max_belugas Maximum number of belugas to keep
    @param max_prod_lines Maximum number of production lines to keep
    @param max_racks Maximum number of racks to keep
    """

    with open(input_file) as f:
        data = json.load(f)

    # Step 1: Trim jigs
    all_jig_keys = list(data["jigs"].keys())
    kept_jig_keys = set(all_jig_keys[:max_jigs])
    
    # Remove unnecessary jigs
    data["jigs"] = {k: v for k, v in data["jigs"].items() if k in kept_jig_keys}

    # Clean racks
    for rack in data["racks"]:
        rack["jigs"] = []

    # Clean production lines
    for pl in data["production_lines"]:
        pl["schedule"] = []

    # Clean flights
    for flight in data["flights"]:
        flight["incoming"] = []
        flight["outgoing"] = []

    # Schritt 2: Belugas, Racks und Produktionslinien trimmen
    data["flights"] = data["flights"][:max_belugas]
    data["racks"] = data["racks"][:max_racks]
    data["production_lines"] = data["production_lines"][:max_prod_lines]

    # Schritt 3: Wir nehmen ein sample von Jigs und packen es in die Produktionslinien
    non_empty_jigs = [k for k, v in data["jigs"].items() if not v.get("empty", False)]
    random.shuffle(non_empty_jigs)

    used_jigs = set()
    for pl in data["production_lines"]:
        if not non_empty_jigs:
            break

        max_count = min(15, len(non_empty_jigs))  # max 15 Jigs pro Linie oder weniger, je nach Verfügbarkeit
        count = random.randint(1, max_count)     # zufällige Anzahl Jigs (1 bis max_count)
    
        selected = non_empty_jigs[:count]
        pl["schedule"] = selected

        used_jigs.update(selected)
        non_empty_jigs = non_empty_jigs[count:]

    # Schritt 4: Verteile die verwendeten Jigs aus den Produktionslinien
    used_jigs = set(jig for pl in data["production_lines"] for jig in pl["schedule"])

    # Liste der verwendbaren Jigs mit Größenangabe
    jig_objects = []
    for jig_id in used_jigs:
        jig = data["jigs"][jig_id]
        jig_type = jig["type"]
        type_info = data["jig_types"][jig_type]
        size_loaded = type_info["size_loaded"]
        jig_objects.append((jig_id, size_loaded))

    # Zuerst ein paar wenige Jigs für Racks auswählen
    rack_fraction = max(1, int(0.1 * len(jig_objects)))  # 10% für jigs in Racks
    random.shuffle(jig_objects)

    rack_jigs = []
    beluga_jigs = []

    # Kopie der racks zum Füllen (mit current_size zum Tracken)
    racks_state = []
    for rack in data["racks"]:
        racks_state.append({
            "rack": rack,
            "remaining_size": rack["size"],
            "jigs": []
        })

    # 1. Packe ein paar wenige Jigs in passende Racks
    for jig_id, jig_size in jig_objects:
        if len(rack_jigs) >= rack_fraction:
            break

        # Suche einen Rack mit genug Platz
        for rack_info in racks_state:
            if jig_size <= rack_info["remaining_size"]:
                rack_info["jigs"].append(jig_id)
                rack_info["remaining_size"] -= jig_size
                rack_jigs.append(jig_id)
                break

    # Update racks mit den zugewiesenen Jigs
    for rack_info in racks_state:
        rack_info["rack"]["jigs"] = rack_info["jigs"]

    # 2. Der Rest kommt in die Belugas (gleichmäßig verteilen)
    beluga_jigs = [jig for jig in used_jigs if jig not in rack_jigs]
    num_belugas = len(data["flights"])
    for i, jig in enumerate(beluga_jigs):
        beluga = data["flights"][i % num_belugas]
        beluga["incoming"].append(jig)

    # Schritt 5: Paar random jigs in racks und belugas verteilen (Jigs die aber davor nicht benutzt worden sind)
    # Alle Jig-IDs, die bisher noch NICHT verwendet wurden
    all_jig_ids = set(data["jigs"].keys())
    unused_jigs = list(all_jig_ids - used_jigs)
    random.shuffle(unused_jigs)

    # Wie viele zusätzliche Jigs wollen wir zufällig verteilen?
    random_int = random.randint(3, 10) # (3, 10) für kleine Probleme, mehr für größere
    extra_count = min(random_int, len(unused_jigs)) 
    extra_jigs = unused_jigs[:extra_count]

    # Racks vorbereiten mit verbleibender Größe
    racks_state = []
    for rack in data["racks"]:
        current_jigs = rack.get("jigs", [])
        remaining = rack["size"]
        for jig_id in current_jigs:
            jig_type = data["jigs"][jig_id]["type"]
            is_empty = data["jigs"][jig_id].get("empty", False)
            size = data["jig_types"][jig_type]["size_empty" if is_empty else "size_loaded"]
            remaining -= size
        racks_state.append({
            "rack": rack,
            "remaining_size": remaining,
            "jigs": current_jigs
        })

    # Jetzt verteilen wir die zusätzlichen Jigs
    for jig_id in extra_jigs:
        jig = data["jigs"][jig_id]
        jig_type = jig["type"]
        is_empty = jig.get("empty", False)
        size = data["jig_types"][jig_type]["size_empty" if is_empty else "size_loaded"]

        # Zufällig entscheiden: Rack oder Beluga
        target = random.choice(["rack", "beluga"])

        if target == "rack":
            random.shuffle(racks_state)
            placed = False
            for rack_info in racks_state:
                if size <= rack_info["remaining_size"]:
                    rack_info["rack"]["jigs"].append(jig_id)
                    rack_info["remaining_size"] -= size
                    placed = True
                    break
            if not placed:
                # Kein Rack mit Platz gefunden → auf Beluga ausweichen
                target = "beluga"

        if target == "beluga" and not is_empty:
            beluga = random.choice(data["flights"])
            beluga["incoming"].append(jig_id)

    # Schritt 6: Von den ganzen Jigs nemmen wir ein random sample von Jig_Typen und verteile es auf die Outgoing der Belugas

    # Alle benutzten Jigs: Belugas und Racks
    all_used_jigs = set()

    for rack in data["racks"]:
        all_used_jigs.update(rack.get("jigs", []))
    for beluga in data["flights"]:
        all_used_jigs.update(beluga.get("incoming", []))

    # Alle Jig-Typen extrahieren
    jig_types_used = [data["jigs"][jig_id]["type"] for jig_id in all_used_jigs]

    random.shuffle(jig_types_used)
    max_types = min(8, len(jig_types_used)) # 8 bei kleinen Problemen, sonst mehr je schweriger das Problem
    
    if max_types >= 1:
        count_to_use = random.randint(0, max_types)
    else:
        count_to_use = 0
        
    types_to_distribute = jig_types_used[:count_to_use]

    # Verteile die Typen zufällig auf die Belugas
    for jig_type in types_to_distribute:
        beluga = random.choice(data["flights"])
        beluga["outgoing"].append(jig_type)

    # Schritt 7: Entferne belugas und produktionslinien, die leer sind
    # Production lines filtern: nur behalten, wenn schedule nicht leer ist
    data["production_lines"] = [
        pl for pl in data["production_lines"] if pl.get("schedule")
    ]

    # Belugas filtern: nur behalten, wenn incoming oder outgoing nicht leer sind
    data["flights"] = [
        fl for fl in data["flights"]
        if fl.get("incoming") or fl.get("outgoing")
    ]

    # Speichern
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

def generate_problems(
    num_problems=20,
    input_folder="problems",
    output_folder="problemset1",
    jig_range=(5, 30),
    beluga_range=(2, 8),
    prod_line_range=(2, 6),
    rack_range=(2, 10)
):
    os.makedirs(output_folder, exist_ok=True)
    problem_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

    for i in range(1, num_problems + 1):
        random_input_file = random.choice(problem_files)
        input_path = os.path.join(input_folder, random_input_file)

        # Zufällige Werte generieren und zwischenspeichern
        max_jigs_val = random.randint(*jig_range)
        max_belugas_val = random.randint(*beluga_range)
        max_prod_lines_val = random.randint(*prod_line_range)
        max_racks_val = random.randint(*rack_range)

        # Temporärer Dateiname
        temp_filename = f"problem{i}_tmp.json"
        temp_path = os.path.join(output_folder, temp_filename)

        filter_problem(
            input_file=input_path,
            output_file=temp_path,
            max_jigs=max_jigs_val,
            max_belugas=max_belugas_val,
            max_prod_lines=max_prod_lines_val,
            max_racks=max_racks_val
        )

         # JSON wieder einlesen
        with open(temp_path) as f:
            data = json.load(f)

        # Infos extrahieren
        num_jigs = len(data.get("jigs", []))
        num_racks = len(data.get("racks", []))
        num_belugas = len(data.get("flights", []))
        num_prod_lines = len(data.get("production_lines", []))

        # Neuer Dateiname
        new_filename = f"problem{i}_j{num_jigs}_r{num_racks}_b{num_belugas}_pl{num_prod_lines}.json"
        new_path = os.path.join(output_folder, new_filename)

        # Datei umbenennen
        os.rename(temp_path, new_path)

if __name__ == "__main__":
    generate_problems()
