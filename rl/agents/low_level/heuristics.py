from typing import Tuple, Optional

class Heuristics:
    def __init__(self, env):
        self.env = env

    def decide_action(self) -> Tuple[str, Optional[dict]]:
        """
        Entscheidet die nächste Aktion basierend auf dem aktuellen Zustand.
        Gibt zurück: (action_name, params) oder (None, None) falls keine Aktion möglich.
        """
        obs = self.env.get_observation_high_level()

        # Priorität: Beluga entladen, wenn voll (Slot 0 == 1)
        if obs[0] == 1:
            return "unload_beluga", None

        # Priorität: Beluga beladen, wenn leer (Slot 0 == 0) und passender Jig verfügbar
        if obs[0] == 0:
            # Prüfe Beluga-Trailer (Slots 1-3)
            for i in range(3):
                if obs[1 + i] == 0:  # Trailer hat passenden leeren Jig
                    return "load_beluga", {"trailer_beluga": i}

        # Priorität: Jigs von Racks zu Factory-Trailern bewegen (wenn benötigt)
        # Suche nach Racks mit Jigs, die in Produktionslinien benötigt werden (rechter Slot > 0)
        n_racks = 10
        for rack_idx in range(n_racks):
            slot = 10 + rack_idx * 2
            if obs[slot + 1] > 0:  # Rechter Slot zeigt an, dass Jig benötigt wird
                # Suche freien Factory-Trailer (Slots 4-6)
                for trailer_idx in range(3):
                    if obs[4 + trailer_idx] == 0.5:  # Trailer ist frei
                        return "right_unstack_rack", {"rack": rack_idx, "trailer_id": trailer_idx}

        # Priorität: Jigs von Hangars zu Factory-Trailern bewegen
        for hangar_idx in range(3):
            if obs[7 + hangar_idx] == 1:  # Hangar hat Jig
                for trailer_idx in range(3):
                    if obs[4 + trailer_idx] == 0.5:  # Trailer ist frei
                        return "get_from_hangar", {"hangar": hangar_idx, "trailer_factory": trailer_idx}

        # Priorität: Jigs von Factory-Trailern zu Hangars liefern
        for trailer_idx in range(3):
            if obs[4 + trailer_idx] == 1:  # Trailer hat vollen Jig
                for hangar_idx in range(3):
                    if obs[7 + hangar_idx] == 0:  # Hangar ist frei
                        return "deliver_to_hangar", {"hangar": hangar_idx, "trailer_factory": trailer_idx}

        # Priorität: Jigs von Beluga-Trailern zu Racks bewegen
        for trailer_idx in range(3):
            if obs[1 + trailer_idx] == 0.25:  # Trailer hat unpassenden leeren Jig
                for rack_idx in range(n_racks):
                    slot = 10 + rack_idx * 2
                    if obs[slot] == 0:  # Rack hat Platz
                        return "left_stack_rack", {"rack": rack_idx, "trailer_id": trailer_idx}

        # Keine Aktion gefunden
        return None, None