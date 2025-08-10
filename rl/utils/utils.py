"""!
@file utils.py
@brief Utility functions for the Beluga Challenge RL solution

This module provides various utility functions including debug logging
and observation permutation for data augmentation during training.
"""

import numpy as np

DEBUG = False

def debuglog(param):
    """!
    @brief Print debug information if DEBUG flag is enabled
    @param param Parameter to print for debugging
    """
    if DEBUG:
        print(param)

def permute_high_level_observation(permutation: np.array, obs: np.array) -> np.array:
    """!
    @brief Permute high-level observation based on given permutation
    
    This function applies a permutation to the rack-related parts of the observation
    while keeping other parts unchanged. Used for data augmentation during training
    to improve generalization.
    
    @param permutation The permutation array to apply (size 10 for racks)
    @param obs The observation array to permute (size 40)
    @return The permuted observation array
    """
    
    permuted_obs = np.zeros(40)
    for i in range(10):
        pos = permutation[i] * 3
        permuted_obs[i] = obs[i]
        permuted_obs[10 + i*3] = obs[10 + pos]
        permuted_obs[11 + i*3] = obs[11 + pos]
        permuted_obs[12 + i*3] = obs[12 + pos]

    return permuted_obs
