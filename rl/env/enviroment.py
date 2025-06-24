from state import ProblemState, load_from_json
from action import (
    load_beluga,
    unload_beluga,
    get_from_hangar,
    deliver_to_hangar,
    left_stack_rack,
    right_stack_rack,
    left_unstack_rack,
    right_unstack_rack
)


class Env:
    def __init__(self):
        # Initialize environment variables here
        self.state = None  # Not initialized yet, will be set in reset()

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

        # If params is None or empty, call the function without arguments besides state
        # Otherwise, pass state + whatever we have in params
        if not params:
            return action_func(self.state)
        else:
            # Unpack params as needed. This example assumes params is a dictionary
            # containing the arguments to be passed (besides state).
            return action_func(self.state, **params)

    def reset(self):
        """
        Resets the environment’s state from a JSON file,
        or any other method of your choice.
        """
        self.state = load_from_json("data/problem.json")

    def get_observation_high_level(self):
        # Return the current state of the environment for a high-level agent
        #TODO: Jan
        pass

    def get_observation_low_level(self):
        # Return the current state of the environment for a low-level agent
        #TODO: Nils
        pass