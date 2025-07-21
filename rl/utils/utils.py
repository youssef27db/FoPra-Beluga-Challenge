import numpy as np


DEBUG = False


def debuglog(param):
    if DEBUG:
        print(param)


def permute_high_level_observation(permutation: np.array, obs: np.array) -> np.array:
    '''
    ========================
    Permutes the high level observation based on the given permutation.
    ========================
    :param permutation: The permutation array to apply.
    :param obs: The observation to permute.
    :return: The permuted observation.
    '''
    
    permuted_obs = np.zeros(40)
    for i in range(10):
        pos = permutation[i] * 3
        permuted_obs[i] = obs[i]
        permuted_obs[10 + i*3] = obs[10 + pos]
        permuted_obs[11 + i*3] = obs[11 + pos]
        permuted_obs[12 + i*3] = obs[12 + pos]

    return permuted_obs
