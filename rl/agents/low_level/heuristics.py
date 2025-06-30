from typing import Tuple, Optional


def decide_parameters(obs, high_level_action) -> Tuple[str, Optional[dict]]:
    """
    Entscheidet die nächste Aktion basierend auf dem High-Level-Plan.
    Gibt zurück: (action_name, params) oder (None, None) falls keine Aktion möglich.
    """
    # Switchcase für die Aktion des High-Level-Agents
    match high_level_action:
        # Falls unload_beluga, dann werden keine Parameter zurückgegeben
        case "unload_beluga":
            return "unload_beluga", None

        # Falls load_beluga, dann wird der Trailer-Index zurückgegeben
        case "load_beluga":
            for i in range(3):
                if obs[1 + i] == 0:  # Trailer hat passenden leeren Jig
                    return "load_beluga", {"trailer_beluga": i}

        # Falls right_unstack_rack, dann wird Rack-Index und Trailer-ID zurückgegeben
        case "right_unstack_rack":
            for rack_idx in range(10):
                slot = 10 + rack_idx * 2
                if obs[slot + 1] > 0:
                    for trailer_idx in range(3):
                        if obs[4 + trailer_idx] == 0.5:
                            return "right_unstack_rack", {"rack": rack_idx, "trailer_id": trailer_idx}

        # Falls get_from_hangar, dann wird Hangar-Index und Trailer-Fabrik-Index zurückgegeben
        case "get_from_hangar":
            for hangar_idx in range(3):
                if obs[7 + hangar_idx] == 1:
                    for trailer_idx in range(3):
                        if obs[4 + trailer_idx] == 0.5:
                            return "get_from_hangar", {"hangar": hangar_idx, "trailer_factory": trailer_idx}

        # Falls deliver_to_hangar, dann wird Hangar-Index und Trailer-Fabrik-Index zurückgegeben
        case "deliver_to_hangar":
            for trailer_idx in range(3):
                if obs[4 + trailer_idx] == 1:
                    for hangar_idx in range(3):
                        if obs[7 + hangar_idx] == 0:
                            return "deliver_to_hangar", {"hangar": hangar_idx, "trailer_factory": trailer_idx}

        # Falls left_stack_rack, dann wird Rack-Index und Trailer-ID zurückgegeben
        case "left_stack_rack":
            for trailer_idx in range(3):
                if obs[1 + trailer_idx] == 0.25:
                    for rack_idx in range(10):
                        slot = 10 + rack_idx * 2
                        if obs[slot] == 0:
                            return "left_stack_rack", {"rack": rack_idx, "trailer_id": trailer_idx}

        # Keine Aktion
        case _:
            return None, None