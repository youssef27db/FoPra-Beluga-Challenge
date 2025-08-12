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

    # Step 2: Limit the number of belugas, racks, and production lines
    data["flights"] = data["flights"][:max_belugas]
    data["racks"] = data["racks"][:max_racks]
    data["production_lines"] = data["production_lines"][:max_prod_lines]

    # Step 3: Assign jigs to production lines
    non_empty_jigs = [k for k, v in data["jigs"].items() if not v.get("empty", False)]
    random.shuffle(non_empty_jigs)

    used_jigs = set()
    for pl in data["production_lines"]:
        if not non_empty_jigs:
            break

        max_count = min(15, len(non_empty_jigs))  # Limit to 15 jigs per production line
        count = random.randint(1, max_count)     # Randomly choose how many jigs to assign
    
        selected = non_empty_jigs[:count]
        pl["schedule"] = selected

        used_jigs.update(selected)
        non_empty_jigs = non_empty_jigs[count:]

    # Step 4: Assign remaining jigs to hangars and trailers
    used_jigs = set(jig for pl in data["production_lines"] for jig in pl["schedule"])

    # Create a list of jig objects with their sizes
    jig_objects = []
    for jig_id in used_jigs:
        jig = data["jigs"][jig_id]
        jig_type = jig["type"]
        type_info = data["jig_types"][jig_type]
        size_loaded = type_info["size_loaded"]
        jig_objects.append((jig_id, size_loaded))

    # Sort jigs by size (descending)
    rack_fraction = max(1, int(0.1 * len(jig_objects)))  # At least 1 jig per rack
    random.shuffle(jig_objects)

    rack_jigs = []
    beluga_jigs = []

    # Prepare racks with remaining size
    racks_state = []
    for rack in data["racks"]:
        racks_state.append({
            "rack": rack,
            "remaining_size": rack["size"],
            "jigs": []
        })

    # Assign jigs to racks based on their sizes
    for jig_id, jig_size in jig_objects:
        if len(rack_jigs) >= rack_fraction:
            break

        # Try to place the jig in a rack
        for rack_info in racks_state:
            if jig_size <= rack_info["remaining_size"]:
                rack_info["jigs"].append(jig_id)
                rack_info["remaining_size"] -= jig_size
                rack_jigs.append(jig_id)
                break

    # Update racks mit den zugewiesenen Jigs
    for rack_info in racks_state:
        rack_info["rack"]["jigs"] = rack_info["jigs"]

    # Assign remaining jigs to belugas
    beluga_jigs = [jig for jig in used_jigs if jig not in rack_jigs]
    num_belugas = len(data["flights"])
    for i, jig in enumerate(beluga_jigs):
        beluga = data["flights"][i % num_belugas]
        beluga["incoming"].append(jig)

    # Step 5: Randomly distribute unused jigs
    # Collect all used jigs
    all_jig_ids = set(data["jigs"].keys())
    unused_jigs = list(all_jig_ids - used_jigs)
    random.shuffle(unused_jigs)

    # Distribute unused jigs randomly
    random_int = random.randint(3, 10)
    extra_count = min(random_int, len(unused_jigs)) 
    extra_jigs = unused_jigs[:extra_count]

    # Prepare racks state for distribution
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

    #  Distribute extra jigs to racks or belugas
    for jig_id in extra_jigs:
        jig = data["jigs"][jig_id]
        jig_type = jig["type"]
        is_empty = jig.get("empty", False)
        size = data["jig_types"][jig_type]["size_empty" if is_empty else "size_loaded"]

        # Randomly choose a target: rack or beluga
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
                # If no rack can accommodate the jig, assign it to a beluga
                target = "beluga"

        if target == "beluga" and not is_empty:
            beluga = random.choice(data["flights"])
            beluga["incoming"].append(jig_id)

    # Step 6: Randomly distribute jig types to belugas

    # Collect all used jigs from racks and belugas
    all_used_jigs = set()

    for rack in data["racks"]:
        all_used_jigs.update(rack.get("jigs", []))
    for beluga in data["flights"]:
        all_used_jigs.update(beluga.get("incoming", []))

    # Get jig types from used jigs
    jig_types_used = [data["jigs"][jig_id]["type"] for jig_id in all_used_jigs]

    random.shuffle(jig_types_used)
    max_types = min(8, len(jig_types_used)) 
    
    if max_types >= 1:
        count_to_use = random.randint(0, max_types)
    else:
        count_to_use = 0
        
    types_to_distribute = jig_types_used[:count_to_use]

    # Randomly assign jig types to belugas
    for jig_type in types_to_distribute:
        beluga = random.choice(data["flights"])
        beluga["outgoing"].append(jig_type)

    # Step 7: Clean up empty jigs and ensure all racks have at least one jig
    # Remove empty jigs
    data["production_lines"] = [
        pl for pl in data["production_lines"] if pl.get("schedule")
    ]

    # Ensure all racks have at least one jig
    data["flights"] = [
        fl for fl in data["flights"]
        if fl.get("incoming") or fl.get("outgoing")
    ]

    # Step 8: Save the filtered problem to the output file
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

        # Randomly select limits for the problem
        max_jigs_val = random.randint(*jig_range)
        max_belugas_val = random.randint(*beluga_range)
        max_prod_lines_val = random.randint(*prod_line_range)
        max_racks_val = random.randint(*rack_range)

        # Temporary file to store the filtered problem
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

         # Load the filtered problem to extract information
        with open(temp_path) as f:
            data = json.load(f)

        # Extract the number of jigs, racks, belugas, and production lines
        num_jigs = len(data.get("jigs", []))
        num_racks = len(data.get("racks", []))
        num_belugas = len(data.get("flights", []))
        num_prod_lines = len(data.get("production_lines", []))

        # Construct the new filename based on the problem parameters
        new_filename = f"problem{i}_j{num_jigs}_r{num_racks}_b{num_belugas}_pl{num_prod_lines}.json"
        new_path = os.path.join(output_folder, new_filename)

        # Rename the temporary file to the new filename
        os.rename(temp_path, new_path)

if __name__ == "__main__":
    generate_problems()
