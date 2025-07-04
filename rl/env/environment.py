from .state import ProblemState, load_from_json
from .action import (
    load_beluga,
    unload_beluga,
    get_from_hangar,
    deliver_to_hangar,
    left_stack_rack,
    right_stack_rack,
    left_unstack_rack,
    right_unstack_rack
)
import numpy as np
from numpy.random import randint

class Env:
    def __init__(self, path: str):
        # Initialize environment variables here
        self.state : ProblemState = None  # Not initialized yet, will be set in reset()
        self.path = path 

        # Map action names to action functions
        self.action_map = {
            "load_beluga": load_beluga,
            "unload_beluga": unload_beluga,
            "get_from_hangar": get_from_hangar,
            "deliver_to_hangar": deliver_to_hangar,
            "left_stack_rack": left_stack_rack,
            "right_stack_rack": right_stack_rack,
            "left_unstack_rack": left_unstack_rack,
            "right_unstack_rack": right_unstack_rack
        }

        self.step_count = 0  # Counter for the number of steps taken, as termination condition in training/evaluation

    def step(self, action_name: str, params=None):
        """
        Executes the specified action on self.state.
        'params' is optional and could be None if the action doesn't need parameters.
        """
        if action_name not in self.action_map:
            raise ValueError(f"Unknown action '{action_name}'")

        action_func = self.action_map[action_name]

        # test ob beluga fertig ist, dann neuen laden

        # subgoal_variablen müssen inkrementiert werden, falls sie korrekt sind
        self.state.belugas_unloaded += int(not self.state.belugas[0].current_jigs)
        print(f"DEBUG - Belugas unloaded: {self.state.belugas_unloaded}")
        print(f"Condition: {int(not self.state.belugas[0].current_jigs)}")
        self.state.belugas_finished += int(not self.state.belugas[0].outgoing and not self.state.belugas[0].current_jigs)
        self.state.production_lines_finished = self.state.total_lines - len(self.state.production_lines)
        self.state.racks_with_empty_jigs = sum(
                                                1 for rack in self.state.racks
                                                if rack.current_jigs and all(self.state.jigs[jig_id].empty for jig_id in rack.current_jigs)
                                            )
        self.state.racks_with_loaded_jigs = sum(
                                                1 for rack in self.state.racks
                                                if rack.current_jigs and all(not self.state.jigs[jig_id].empty for jig_id in rack.current_jigs)
                                                )
        #self.state.beluga_complete()
        # If params is None or empty, call the function without arguments besides state
        # Otherwise, pass state + whatever we have in params
        
        n_production_lines = len(self.state.production_lines)
        could_execute = False

        if not params:
            could_execute = action_func(self.state)
        else:
            # Unpack params as needed. This example assumes params is a dictionary
            # containing the arguments to be passed (besides state).
            could_execute = action_func(self.state, **params)

        obs = self.get_observation_high_level()  # Get the current observation before executing the action
        reward = self.get_reward(could_execute, action_name, n_production_lines, obs)
        
        return obs, reward, self.state.is_terminal()



    def reset(self):
        """
        Resets the environment’s state from a JSON file,
        or any other method of your choice.
        """
        # Wähle random datei aus problemset
        number = randint(1, 21)

        self.state = load_from_json(self.path + f"problem_{number}.json")

    def get_reward(self, could_execute: bool, action_name: str, production_line_n_old, obs):
        """
        Returns the current reward of the environment.
        """
        # Strafe, wenn Aktion fehlschlägt
        if not could_execute: 
            return -1000.0
        
        if action_name == "unload_beluga":
            return -1.0
        
        if action_name == "load_beluga":
            if obs[0] == 1:
                return 100.0
            return 5.0

        if action_name in ["right_stack_rack", "left_stack_rack", "right_unstack_rack", "left_unstack_rack"]:
            return -5.0
        
        if action_name == "deliver_to_hangar":
            if production_line_n_old > len(self.state.production_lines):
                return 100.0
            return 10.0
        
        if action_name == "get_from_hangar":
            return 5.0
        
        return 0

    def get_observation_high_level(self):
        # Return the current state of the environment for a high-level agent as array
        # High-Level-Agents converts array into tensor

        ### CURRENTLY MAX 10 RACKS
        n_racks = 10

        out = np.zeros(10 + 2*n_racks)

        needed_outgoing_types = []
        needed_in_production_lines = []

        for pl in self.state.production_lines:
            needed_in_production_lines.append(pl.scheduled_jigs[0])

        # First slot 0 beluga
        if len(self.state.belugas) > 0:
            out[0] = max(0, min(len(self.state.belugas[0].current_jigs), 1))
            needed_outgoing_types = self.state.belugas[0].outgoing
        else:
            out[0] = -1

        # Slot 1-3 Beluga Trailer
        slot = 1
        for i in range(3):
            if i < len(self.state.trailers_beluga):
                if self.state.trailers_beluga[i] is None:
                    out[slot + i] = 0.5
                else:
                    if self.state.jigs[self.state.trailers_beluga[i]].empty and out[0] == 0:
                        if needed_outgoing_types.__contains__(
                                self.state.jigs[self.state.trailers_beluga[i]].jig_type):
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
            if i < len(self.state.trailers_factory):
                if self.state.trailers_factory[i] is None:
                    out[slot + i] = 0.5
                else:
                    if not self.state.jigs[self.state.trailers_factory[i]].empty:
                        if needed_in_production_lines.__contains__(self.state.trailers_factory[i]):
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
            if i < len(self.state.hangars):
                if self.state.hangars[i] is None:
                    out[slot + i] = 0
                else:
                    out[slot + i] = 1
            else:
                out[slot + i] = -1

        # Slot 10-29 Racks
        slot = 10
        for i in range(n_racks):
            if i < len(self.state.racks):
                rack = self.state.racks[i]
                items = len(rack.current_jigs)
                if items == 0:
                    out[slot + i * 2] = 0
                    out[slot + i * 2 + 1] = 0

                else:
                    out[slot + i * 2] = -1
                    out[slot + i * 2 + 1] = -1
                    for k in range(items):
                        jig = self.state.jigs[rack.current_jigs[k]]
                        if jig.empty and needed_outgoing_types.__contains__(jig.jig_type):
                            out[slot + i * 2] = (items - k) / items
                            continue
                    for k in range(items):
                        if needed_in_production_lines.__contains__(rack.current_jigs[k]):
                            out[slot + i * 2 + 1] = (k + 1) / items
                            continue


        return out

    def get_observation_low_level(self):
        # Return the current state of the environment for a low-level agent
        #TODO: Nils
        pass