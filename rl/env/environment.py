"""!
@file environment.py
@brief Main environment class for the Beluga Challenge

This module implements the reinforcement learning environment for the
Beluga Challenge container optimization problem.
"""

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
    """!
    @brief Beluga Challenge environment for reinforcement learning
    
    This class implements the main environment for the Beluga Challenge shipping
    container optimization problem. It manages problem states, action execution,
    reward calculation, and episode management.
    """
    
    def __init__(self, path: str, base_index: int = -1):
        """!
        @brief Initialize the Beluga Challenge environment
        @param path Path to the directory containing problem JSON files
        @param base_index Base index for problem selection
        """
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

        # Find all JSON files in the problems folder
        if os.path.exists(self.path):
            problem_files = [f for f in os.listdir(self.path) if f.endswith('.json')]
            
            # Extract the number of jigs from each filename using regular expression
            jig_counts = []
            for file in problem_files:
                match = re.search(r'_j(\d+)_', file)
                if match:
                    jig_count = int(match.group(1))
                    jig_counts.append((file, jig_count))
                     
            # Sort by number of jigs (ascending)
            self.sorted_problems = sorted(jig_counts, key=lambda x: x[1])
            self.problem_count = len(self.sorted_problems)  # Set the problem count based on the sorted problems



    def step(self, action_name: str, params=None):
        """!
        @brief Execute a single environment step with the given action
        @param action_name Name of the action to execute
        @param params Parameters for the action (optional)
        @return Tuple of (observation, reward, done_flag)
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
        """!
        @brief Reset the environment with a new problem instance
        
        Resets the environment's state from a JSON file in the problems folder,
        selecting problems in ascending order of jigs count (in blocks of 6 problems)
        
        @return Initial observation of the new episode
        """

        number = randint(1, self.block_size + 1)

        # Increase base_index only if problems_solved > 0 (to prevent increase at initialization)
        # and if it's a multiple of block_size * 3 (every 18 problems when block_size=6)
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
        """!
        @brief Reset the environment with a specific problem instance
        @param problem Path to the specific problem JSON file
        @return Initial observation of the new episode
        """
        self.problem_name = problem
        self.state = load_from_json(self.problem_name)
        self.step_count = 0

        return self.get_observation_high_level()

    def get_reward(self, could_execute: bool, action_name: str, production_line_n_old):
        """!
        @brief Calculate the reward for the current action
        @param could_execute Boolean indicating if the action was successfully executed
        @param action_name Name of the action taken
        @param production_line_n_old Number of production lines before the action
        @return Reward value based on the action and state
        """

        # Goal completed
        if self.state.is_terminal():
            return 10000
        
        # Penalty if action fails, but less severe
        if not could_execute: 
            return -1000 + min(20, self.step_count) * 10 # Mild penalty with moderate increase
        
        if action_name == "unload_beluga":
            # Reward if beluga is completely unloaded
            if len(self.state.belugas) > 0:
                if len(self.state.belugas[0].current_jigs) == 1:
                    return 2000.0  
            return 100.0  
        
        if action_name == "load_beluga":
            if len(self.state.belugas) > 0:
                if len(self.state.belugas[0].outgoing) == 1:
                    return 2000.0  
                return 100.0  
            else: 
                return 5000.0  

        # Actions that stack or unstack racks
        if action_name in ["right_stack_rack", "left_stack_rack", "right_unstack_rack", "left_unstack_rack"]:
            return 10.0 
        
        if action_name == "deliver_to_hangar":
            if production_line_n_old > len(self.state.production_lines):
                return 2000.0 
            return 200.0 
        
        if action_name == "get_from_hangar":
            return 50.0 
        
        return 0

    def get_observation_high_level(self):
        """!
        @brief Get the high-level observation of the current state
        @return High-level observation array
        """
        return self.state.get_observation_high_level()
    
    def get_max_steps(self):
        """!
        @brief Get the maximum number of steps to solve the current problem
        @return Maximum steps based on the problem size
        """
    
        return len(self.state.jigs) * 20 + 100

    def check_action_execution(self, action_name: str, obs):
        """!
        @brief Check if the action can be executed without actually executing it
        @param action_name Name of the action to check
        @param obs Current observation of the environment
        @return True if the action can be executed, False otherwise
        """
        
        if action_name in self.check_action_map:
            return self.check_action_map[action_name](self.state, obs)