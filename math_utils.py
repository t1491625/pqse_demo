import numpy as np


def normalise_vector(v):
    """
    Returns vector such that |v|**2 = 1

    Parameters:
        v : ndarray : input vector

    Returns:
        ndarray
    """
    norm_factor = np.sqrt(sum([np.abs(v_i) ** 2 for v_i in v]))

    if norm_factor == 0:
        raise Exception("Tried to normalise a zero vector")

    return v / norm_factor


def compute_var(exp_H_sq, exp_H, exp_H_0):
    return (exp_H_sq / exp_H_0) - (exp_H / exp_H_0) ** 2
