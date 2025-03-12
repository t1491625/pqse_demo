import numpy as np
import itertools


def construct_partitioned_qse_mat_cached(
    size_basis, unique_H_elements_noisy, coeff_cache, matrix_type=np.float64, cast=True
):
    H = np.zeros((size_basis, size_basis), dtype=matrix_type)
    S = np.zeros((size_basis, size_basis), dtype=matrix_type)

    for idx in range(size_basis):
        for idy in range(size_basis):
            new_H_element = compute_partitioned_state_power_H_cached(
                unique_H_elements_noisy, coeff_cache, idx + idy + 1
            )
            new_S_element = compute_partitioned_state_power_H_cached(
                unique_H_elements_noisy, coeff_cache, idx + idy
            )

            if cast:
                H[idx][idy] = new_H_element.real
                S[idx][idy] = new_S_element.real
            else:
                H[idx][idy] = new_H_element
                S[idx][idy] = new_S_element

    return H, S


def compute_partitioned_state_power_H_cached(unique_H_elements, coeffs, power):
    """
    Unlike compute_partitioned_state_power_H, which figures out how to map from current iteration inner product
    power to unique H elements inner product power, this function takes the unique H elements power as input.
    """
    return sum(coeffs[_] * unique_H_elements[power + _] for _ in coeffs)


def update_coeff_cache(coeff_cache, new_opt_coeffs, kbo_current):
    new_coeffs_prods = {}

    for new_coeff_idxs in itertools.product(
        *[range(len(new_opt_coeffs)), range(len(new_opt_coeffs))]
    ):
        coeff_idx_sum = sum(new_coeff_idxs)

        if coeff_idx_sum in new_coeffs_prods:
            new_coeffs_prods[coeff_idx_sum] += (
                np.conj(new_opt_coeffs[new_coeff_idxs[0]])
                * new_opt_coeffs[new_coeff_idxs[1]]
            )
        else:
            new_coeffs_prods[coeff_idx_sum] = (
                np.conj(new_opt_coeffs[new_coeff_idxs[0]])
                * new_opt_coeffs[new_coeff_idxs[1]]
            )

    return {
        m: compute_recombined_coeff(coeff_cache, new_coeffs_prods, m)
        for m in range(0, (2 * kbo_current) + 1)
    }


def compute_recombined_coeff(coeff_cache, new_coeffs_prods, m):
    new_coeff = 0

    for j in range(m + 1):
        if m - j in coeff_cache and j in new_coeffs_prods:
            new_coeff += new_coeffs_prods[j] * coeff_cache[m - j]

    return new_coeff


def min_variance(
    curr_best_soln,
    curr_min_abs_var,
    coeff_cache,
    unique_H_elements,
    kbo_current,
    trial_optimal_solns,
):
    vars = []

    for coeffs, energy, stored_H, stored_S in trial_optimal_solns:
        temp_coeff_cache = dict(coeff_cache)

        temp_coeff_cache = update_coeff_cache(
            temp_coeff_cache, coeffs, kbo_current + (len(coeffs) - 1)
        )

        H_10 = compute_partitioned_state_power_H_cached(
            unique_H_elements, temp_coeff_cache, 2
        )

        H_00 = compute_partitioned_state_power_H_cached(
            unique_H_elements, temp_coeff_cache, 1
        )
        S_00 = compute_partitioned_state_power_H_cached(
            unique_H_elements, temp_coeff_cache, 0
        )

        var = compute_var(H_10, H_00, S_00)

        vars.append(var)

        if curr_min_abs_var is None or abs(var) < curr_min_abs_var:
            curr_min_abs_var = abs(var)
            curr_best_soln = (coeffs, energy, stored_H, stored_S)

    return curr_best_soln, curr_min_abs_var, vars


def compute_var(exp_H_sq, exp_H, exp_H_0):
    return (exp_H_sq / exp_H_0) - (exp_H / exp_H_0) ** 2
