import json
import random
import os
from pathlib import Path
from collections import Counter
import copy


def filter_problem(input_file, output_file, max_jigs, max_belugas, max_prod_lines, max_racks):
    random.seed(42)

    # Originaldaten laden
    with open(input_file) as f:
        data = json.load(f)

    # Erstes Setup
    all_jig_names = list(data["jigs"].keys())
    selected_jig_names = all_jig_names[:max_jigs]
    selected_jigs = {f"jig{str(i + 1).zfill(4)}": data["jigs"][name] for i, name in enumerate(selected_jig_names)}

    # Neue Jignamen anwenden
    for new_name, jig_data in selected_jigs.items():
        jig_data["name"] = new_name

    # Jig-Typ-Zähler für outgoing später
    jig_type_counts = Counter(jig["type"] for jig in selected_jigs.values())
    remaining_types = dict(jig_type_counts)

    # Zuweisung vorbereiten
    all_jig_keys = list(selected_jigs.keys())
    random.shuffle(all_jig_keys)

    jigs_in_belugas = set()
    jigs_in_racks = set()
    jigs_in_production = set()

    # Belugas (incoming)
    selected_flights = []
    used_jigs = set()
    for i in range(min(max_belugas, len(data["flights"]))):
        incoming = []
        num_incoming = random.randint(1, 3)
        for _ in range(num_incoming):
            available_jigs = list(set(all_jig_keys) - used_jigs)
            if not available_jigs:
                break
            jig = random.choice(available_jigs)
            incoming.append(jig)
            used_jigs.add(jig)
            jigs_in_belugas.add(jig)

            # empty = False setzen
            selected_jigs[jig]["empty"] = False

        # Erzeuge outgoing-Types
        outgoing = []
        available_types = [typ for typ, count in remaining_types.items() if count > 0]

        if available_types:
            num_types = random.randint(1, min(3, len(available_types)))
            selected_types = random.sample(available_types, num_types)

            for typ in selected_types:
                outgoing.append(typ)
                remaining_types[typ] -= 1

        selected_flights.append({
            "name": f"beluga{i+1}",
            "incoming": incoming,
            "outgoing": outgoing
        })

    # Racks
    selected_racks = []
    rack_pool = data["racks"][:max_racks]
    for rack in rack_pool:
        new_rack = copy.deepcopy(rack)
        new_rack["jigs"] = []
        for _ in range(random.randint(0, 2)):
            available = list(set(all_jig_keys) - used_jigs)
            if not available:
                break
            if random.random() < 0.2:  # niedrigere Wahrscheinlichkeit als Beluga
                jig = random.choice(available)
                new_rack["jigs"].append(jig)
                used_jigs.add(jig)
                jigs_in_racks.add(jig)
        selected_racks.append(new_rack)

    # Production Lines
    selected_lines = []
    available_jigs = set(selected_jigs)  # Kopie der ausgewählten Jigs

    while available_jigs and len(selected_lines) < max_prod_lines:
        max_possible = min(4, len(available_jigs))  # Maximal 4 Jigs pro Line
        num_jigs = random.randint(1, max_possible)

        line_jigs = random.sample(list(available_jigs), num_jigs)

        # empty = False setzen für alle Jigs in dieser Production Line
        for jig in line_jigs:
            selected_jigs[jig]["empty"] = False

        available_jigs.difference_update(line_jigs)
        jigs_in_production = jigs_in_production.union(set(line_jigs))

        selected_lines.append({
            "name": f"pl{len(selected_lines)}",
            "schedule": line_jigs
        })

    # JigTypes
    jig_types = data["jig_types"]

    # Gefilterte Beluga-Trailer
    filtered_trailers_beluga = data["trailers_beluga"][:max_belugas] if max_belugas else data["trailers_beluga"]

    # Neue Datenstruktur aufbauen
    filtered_data = {
        "trailers_beluga": filtered_trailers_beluga,
        "trailers_factory": data["trailers_factory"],
        "hangars": data["hangars"],
        "jig_types": jig_types,
        "racks": selected_racks,
        "jigs": selected_jigs,
        "production_lines": selected_lines,
        "flights": selected_flights,
    }

    # Speicherort vorbereiten
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)

    # Zusammenfassung
    print(f"Problem erstellt mit:")
    print(f"- {len(selected_jigs)} Jigs")
    print(f"- {len(selected_lines)} Production Lines")
    print(f"- {len(selected_racks)} Racks")
    print(f"- {len(selected_flights)} Belugas")
    print(f"- Jigs in Belugas: {len(jigs_in_belugas)}")
    print(f"- Jigs in Racks: {len(jigs_in_racks)}")
    print(f"- Jigs in ProdLines: {len(jigs_in_production)}")

if __name__ == "__main__":
    problem_folder = "problems"
    problem_files = [f for f in os.listdir(problem_folder) if f.endswith(".json")]

    # Zufällige Datei auswählen
    random_input_file = random.choice(problem_files)

    input_path = os.path.join(problem_folder, random_input_file)

    filter_problem(
        input_file = input_path,
        output_file = "problemset1/problem20.json",
        max_jigs = 20,
        max_belugas = 6,
        max_prod_lines = 6,
        max_racks = 10
    )
