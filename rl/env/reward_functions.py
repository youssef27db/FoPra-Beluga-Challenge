from toolkit.actions import *

#Berechnung vom Reward für die angewendeten Aktion.
def reward_function(state: ProblemState, action: tuple) -> float:

    executed, action_name = action

    # Strafe, wenn Aktion fehlschlägt
    if not executed: 
        return -20.0
    
    if action_name == "unload_beluga":
        return -1.0
    
    if action_name == "load_beluga":
        return 2.0

    if action_name in ["stack_rack", "unstack_rack"]:
        return -2.0
    
    if action_name == "deliver_to_hangar":
        return 10.0
    
    if action_name == "get_from_hangar":
        return 5.0

    # Großer Reward für Zielerreichung
    if action_name == "goal":
        return 100.0

    # Bonus, wenn alle Belugas abgearbeitet sind
    if state.belugas == []:
        return 30.0
    
    # Bonus, wenn alle Produktionslinien fertig sind
    if state.production_lines == []:
        return 30.0

    # Neutraler Reward für sonstige erfolgreiche Aktionen
    return 1.0