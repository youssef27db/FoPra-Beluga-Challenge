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
import os, re
from .check_action import *
from numpy.random import randint

class Env:
    def __init__(self, path: str, base_index: int = -1):
        # Initialize environment variables here
        self.state : ProblemState = None  # Not initialized yet, will be set in reset()
        self.path = path 
        self.step_count = 0  # Counter for the number of steps taken, as termination condition in training/evaluation
        self.problem_name = None
        self.sorted_problems = []  # List to hold sorted problems by jig count
        self.problem_count = 0  # Counter for the number of problems solved
        self.base_index = base_index  # Base index for problem selection, used to select problems in ascending order of jig count
        self.problems_solved = 0  # Counter for the number of problems solved
        self.block_size = 6  # Number of problems to select in each block

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

        # Alle JSON-Dateien im problems-Ordner finden
        if os.path.exists(self.path):
            problem_files = [f for f in os.listdir(self.path) if f.endswith('.json')]
            
            # Extrahiere die Anzahl der Jigs aus jedem Dateinamen mit regulärem Ausdruck
            jig_counts = []
            for file in problem_files:
                match = re.search(r'_j(\d+)_', file)
                if match:
                    jig_count = int(match.group(1))
                    jig_counts.append((file, jig_count))
                     
            # Sortiere nach Anzahl der Jigs (aufsteigend)
            self.sorted_problems = sorted(jig_counts, key=lambda x: x[1])
            self.problem_count = len(self.sorted_problems)  # Set the problem count based on the sorted problems



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

        if self.state.is_terminal():
            self.problems_solved += 1

        return obs, reward, self.state.is_terminal()



    def reset(self):
        """
        Resets the environment's state from a JSON file in the problems folder,
        selecting problems in ascending order of jigs count (in blocks of 6 problems)
        """

        number = randint(1, self.block_size + 1)

        # Erhöhe base_index nur, wenn problems_solved > 0 ist (um Erhöhung bei Initialisierung zu verhindern)
        # und wenn es ein Vielfaches von block_size * 3 ist (alle 18 Probleme bei block_size=6)
        if self.problems_solved > 0 and (self.problems_solved % (self.block_size * 3)) == 0:
            self.base_index += 1
            self.problems_solved = 0

        # If we have reached the end of the sorted problems, choose a random problem
        if self.base_index + number >= self.problem_count:
            self.problem_name = os.path.join(self.path, self.sorted_problems[randint(0, self.problem_count)][0])
        else:    
            self.problem_name = os.path.join(self.path, self.sorted_problems[self.base_index + number][0])

        self.state = load_from_json(self.problem_name)
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
            if len(self.state.belugas) > 0:
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
        return len(self.state.jigs) * 20 + 100

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