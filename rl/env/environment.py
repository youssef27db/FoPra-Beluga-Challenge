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
        # If params is None or empty, call the function without arguments besides state
        # Otherwise, pass state + whatever we have in params
        
        n_production_lines = len(self.state.production_lines)
        could_execute = False

        if not params and action_name == "unload_beluga":
            could_execute = self.state.apply_action(action_name, {})
        else:
            # Unpack params as needed. This example assumes params is a dictionary
            # containing the arguments to be passed (besides state).
            if params is not None:
                could_execute = self.state.apply_action(action_name, params)
            else:
                could_execute = False

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

        return self.get_observation_high_level()

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
        return self.state.get_observation_high_level()

    def get_observation_low_level(self):
        # Return the current state of the environment for a low-level agent
        #TODO: Nils
        pass