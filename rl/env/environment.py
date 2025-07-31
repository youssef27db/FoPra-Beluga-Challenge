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
from .check_action import *
from numpy.random import randint

class Env:
    def __init__(self, path: str):
        # Initialize environment variables here
        self.state : ProblemState = None  # Not initialized yet, will be set in reset()
        self.path = path 
        self.step_count = 0  # Counter for the number of steps taken, as termination condition in training/evaluation
        self.problem_name = None


        # Map action names to action functions
        self.check_action_map = {
            "load_beluga": check_load_beluga,
            "unload_beluga": check_unload_beluga,
            "get_from_hangar": check_get_from_hangar,
            "deliver_to_hangar": check_deliver_to_hangar,
            "left_stack_rack": check_left_stack_rack,
            "right_stack_rack": check_right_stack_rack,
            "left_unstack_rack": check_left_unstack_rack,
            "right_unstack_rack": check_right_unstack_rack
        }

    def step(self, action_name: str, params=None):
        """
        Executes the specified action on self.state.
        'params' is optional and could be None if the action doesn't need parameters.
        """
    
        n_production_lines = len(self.state.production_lines)
        could_execute = False

        if params == [] and action_name == "unload_beluga":
            could_execute = self.state.apply_action(action_name, {})
        else:
            # Unpack params as needed. This example assumes params is a dictionary
            # containing the arguments to be passed (besides state).
            if params != []:
                could_execute = self.state.apply_action(action_name, params)
            else:
                could_execute = False

        obs = self.get_observation_high_level()  # Get the current observation before executing the action
        reward = self.get_reward(could_execute, action_name, n_production_lines)
        self.step_count += 1  # Increment the step count

        return obs, reward, self.state.is_terminal()



    def reset(self):
        """
        Resets the environment’s state from a random JSON file
        """
        # Wähle random datei aus problemset
        number = randint(1, 21)

        self.problem_name = self.path + f"problem_{number}.json"
        self.state = load_from_json(self.problem_name)
        self.step_count = 0  # Reset the step count

        return self.get_observation_high_level()
    
    def reset_specific_problem(self, problem):
        """
        Resets the environment’s state from a specific JSON file
        """
        self.problem_name = problem
        self.state = load_from_json(self.problem_name)
        self.step_count = 0

        return self.get_observation_high_level()

    def get_reward(self, could_execute: bool, action_name: str, production_line_n_old):
        """
        Returns the current reward of the environment.
        """
        # Goal completed
        if self.state.is_terminal():
            return 10000
        
        # Strafe, wenn Aktion fehlschlägt, aber weniger extrem
        if not could_execute: 
            return -1000 + min(20, self.step_count) * 10 # Mildere Bestrafung mit moderatem Anstieg
        
        if action_name == "unload_beluga":
            # Belohen wenn beluga komplett entladen
            if len(self.state.belugas[0].current_jigs) == 1:
                return 2000.0  # Erhöht für bessere positive Signale
            return 100.0  # Erhöht für bessere Zwischenschritte
        
        if action_name == "load_beluga":
            if len(self.state.belugas) > 0:
                if len(self.state.belugas[0].outgoing) == 1:
                    return 2000.0  # Erhöht für bessere positive Signale
                return 100.0  # Erhöht für bessere Zwischenschritte
            else: 
                return 5000.0  # Belassen, da dies ein großer Erfolg ist

        if action_name in ["right_stack_rack", "left_stack_rack", "right_unstack_rack", "left_unstack_rack"]:
            return 10.0  # Geändert zu positiv, um diese Aktionen nicht zu bestrafen
        
        if action_name == "deliver_to_hangar":
            if production_line_n_old > len(self.state.production_lines):
                return 2000.0  # Erhöht für bessere positive Signale
            return 200.0  # Erhöht für bessere Zwischenschritte
        
        if action_name == "get_from_hangar":
            return 50.0  # Erhöht für bessere Zwischenschritte
        
        return 0

    def get_observation_high_level(self):
        return self.state.get_observation_high_level()
    
    def get_max_steps(self):
        """
        Returns the maximum number of steps based on the problem size.
        """
        return len(self.state.jigs) * 12 + 50

    def get_observation_low_level(self):
        # Return the current state of the environment for a low-level agent
        #TODO: Nils
        pass

    def check_action_execution(self, action_name: str, obs):
        """
        Checks if the action can be executed without actually executing it.
        Returns True if the action can be executed, False otherwise.
        """
        if action_name in self.check_action_map:
            return self.check_action_map[action_name](self.state, obs)