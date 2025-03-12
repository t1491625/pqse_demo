from scipy.linalg import eig
import numpy as np
import pandas as pd
import pickle as pkl

from qse_utils import (
    construct_unique_H_elements,
    compute_qse_matrices_krylov_basis,
    apply_thresholding,
)
from pqse_utils import (
    construct_partitioned_qse_mat_cached,
    compute_partitioned_state_power_H_cached,
    min_variance,
    update_coeff_cache,
)
from noise_utils import calc_noise_rate
from hamiltonian_utils import get_initial_state
from math_utils import normalise_vector
from file_utils import convert_script_params_to_filename


def run_pqse_experiment(experiment_params):
    exp_data = {
        "size_state": [],
        "seed": [],
        "J": [],
        "h_bound": [],
        "reference_update": [],
        "noise_strength": [],
        "noise_type": [],
        "noise_seed": [],
        "init_reference_state": [],
        "size_basis_current": [],
        "iteration_current": [],
        "final_iteration": [],
        "kbo_final": [],
        "kbo_current": [],
        "exact_energy": [],
        "energy": [],
        "energy_err_rel": [],
    }

    rng = np.random.default_rng(experiment_params["seed"])
    noise_rng = np.random.default_rng(experiment_params["n_seed"])

    reference_state, rng = get_initial_state(
        experiment_params["ref_st_name"],
        rng,
        params={
            "J": experiment_params["J"],
            "h_bnd": experiment_params["h_bnd"],
            "seed": experiment_params["seed"],
            "n_qubit": experiment_params["s_s"],
        },
    )

    reference_state = normalise_vector(reference_state)

    ham_fname = convert_script_params_to_filename(
        {
            "s_s": experiment_params["s_s"],
            "h_bnd": experiment_params["h_bnd"],
            "J": experiment_params["J"],
            "seed": experiment_params["seed"],
        }
    )

    with open("saved_hamiltonians/" + ham_fname + ".pkl", "rb") as f:
        ham_data = pkl.load(f)
        target_H = ham_data["H"]
        exact_evals = ham_data["evals"]

    exact_gs_energy = min(exact_evals)

    size_basis_final = (
        experiment_params["kbo_final"] + 1 + 1
    )  # extra + 1 for gradient norm calculation

    matrix_params = {
        "sb_final": size_basis_final,
        "ref_st_name": experiment_params["ref_st_name"],
        "seed": experiment_params["seed"],
        "J": experiment_params["J"],
        "h_bnd": experiment_params["h_bnd"],
    }

    exp_data = run_pqse(
        target_H,
        reference_state,
        size_basis_final,
        exact_gs_energy,
        experiment_params,
        exp_data,
        matrix_params,
        noise_rng,
    )

    # populate constant data points
    for _ in range(
        len(exp_data["energy"])
    ):  # take an arbitrary parameter to learn number of data points
        exp_data["size_state"].append(experiment_params["s_s"])
        exp_data["seed"].append(experiment_params["seed"])
        exp_data["J"].append(experiment_params["J"])
        exp_data["h_bound"].append(experiment_params["h_bnd"])
        exp_data["reference_update"].append(experiment_params["ref_upd"])
        exp_data["noise_strength"].append(experiment_params["noise_str"])
        exp_data["noise_type"].append(experiment_params["noise_type"])
        exp_data["noise_seed"].append(experiment_params["n_seed"])
        exp_data["init_reference_state"].append(experiment_params["ref_st_name"])
        exp_data["kbo_final"].append(experiment_params["kbo_final"])
        exp_data["exact_energy"].append(exact_gs_energy)

    return pd.DataFrame(exp_data)


def run_pqse(
    target_H,
    reference_state,
    size_basis_final,
    exact_gs_energy,
    experiment_params,
    exp_data,
    matrix_params,
    noise_rng,
):
    unique_H_elements = construct_unique_H_elements(
        matrix_params, size_basis_final, reference_state, target_H
    )

    _, _, unique_H_elements_noisy = compute_qse_matrices_krylov_basis(
        size_basis_final,
        unique_H_elements,
        noise_rng,
        noise_type=experiment_params["noise_type"],
        noise_strength=experiment_params["noise_str"],
    )

    kbo_current = 0

    optimal_coeffs = []
    counter = 0
    best_soln = None
    curr_min_abs_var = None

    coeff_cache = {0: 1.0}

    while kbo_current < experiment_params["kbo_final"]:
        trial_optimal_solns = []  # (coeffs, energy, H, S)

        max_size_basis = (experiment_params["kbo_final"] - kbo_current) + 1

        if len(optimal_coeffs) > 0:
            coeff_cache = update_coeff_cache(
                coeff_cache, optimal_coeffs[-1], kbo_current
            )

        for size_basis in range(2, max_size_basis + 1):
            H, S = construct_partitioned_qse_mat_cached(
                size_basis, unique_H_elements_noisy, coeff_cache
            )

            evals, evecs = eig(H, b=S, right=True)

            sorted_evals = np.sort(evals)
            evals_coeffs = {evals[i]: evecs[:, i] for i in range(len(evals))}
            gs_evec = evals_coeffs[sorted_evals[0]]

            if sorted_evals[0].imag < 1e-10:
                #  we can remove imaginary part of energy here as we check if it is negligible above
                trial_optimal_solns.append((gs_evec, sorted_evals[0].real, H, S))

        #  update dependent variables
        if len(trial_optimal_solns) > 0:
            new_best_soln, new_min_abs_var, all_vars = min_variance(
                best_soln,
                curr_min_abs_var,
                coeff_cache,
                unique_H_elements_noisy,
                kbo_current,
                trial_optimal_solns,
            )

            if (
                best_soln is None or new_best_soln[1] != best_soln[1]
            ):  # compare energy estimates
                best_soln = new_best_soln
                curr_min_abs_var = new_min_abs_var

                optimal_coeffs.append(best_soln[0])
                kbo_current = kbo_current + (len(optimal_coeffs[-1]) - 1)

                if kbo_current == experiment_params["kbo_final"]:
                    exp_data["final_iteration"].append(True)
                else:
                    exp_data["final_iteration"].append(False)
            else:  # no update, terminate
                exp_data["final_iteration"].append(True)

            exp_data["size_basis_current"].append(len(optimal_coeffs[-1]))
            exp_data["iteration_current"].append(counter)
            exp_data["kbo_current"].append(kbo_current)

            temp_coeff_cache = dict(coeff_cache)
            temp_coeff_cache = update_coeff_cache(
                temp_coeff_cache, optimal_coeffs[-1], kbo_current
            )

            state_norm = compute_partitioned_state_power_H_cached(
                unique_H_elements_noisy, temp_coeff_cache, 0
            )

            energy = best_soln[1]

            exp_data["energy"].append(energy)
            exp_data["energy_err_rel"].append(
                np.abs(energy - exact_gs_energy) / np.abs(exact_gs_energy)
            )

            if exp_data["final_iteration"][-1]:
                break

            counter += 1
        else:
            (
                exp_data["size_basis_current"].append(
                    exp_data["size_basis_current"][-1]
                )
                if len(exp_data["size_basis_current"]) > 0
                else size_basis
            )

            exp_data["iteration_current"].append(counter)
            exp_data["final_iteration"].append(True)
            exp_data["kbo_current"].append(kbo_current)
            exp_data["energy"].append(exp_data["energy"][-1])
            exp_data["energy_target_H"].append(exp_data["energy_target_H"][-1])
            exp_data["energy_err_rel"].append(exp_data["energy_err_rel"][-1])

            print("\nFAILED")
            print("Could not continue iteration")
            print("Experiment parameters:")
            print(experiment_params)
            print("FAILED\n")
            break

    return exp_data


def run_qse_experiment(experiment_params):
    exp_data = {
        "size_state": [],
        "noise_strength": [],
        "noise_type": [],
        "noise_seed": [],
        "thresholding": [],
        "init_reference_state": [],
        "kbo_final": [],
        "exact_energy": [],
        "energy": [],
        "energy_err_rel": [],
    }

    if experiment_params["thresh"]:
        exp_data["thresholding_scale"] = []

    N = 2 ** experiment_params["s_s"]
    rng = np.random.default_rng(experiment_params["seed"])
    noise_rng = np.random.default_rng(experiment_params["n_seed"])

    reference_state, rng = get_initial_state(
        experiment_params["ref_st_name"],
        rng,
        params={
            "J": experiment_params["J"],
            "h_bnd": experiment_params["h_bnd"],
            "seed": experiment_params["seed"],
            "n_qubit": experiment_params["s_s"],
        },
    )

    reference_state = normalise_vector(reference_state)

    ham_fname = convert_script_params_to_filename(
        {
            "s_s": experiment_params["s_s"],
            "h_bnd": experiment_params["h_bnd"],
            "J": experiment_params["J"],
            "seed": experiment_params["seed"],
        }
    )

    with open("saved_hamiltonians/" + ham_fname + ".pkl", "rb") as f:
        ham_data = pkl.load(f)
        target_H = ham_data["H"]
        exact_evals = ham_data["evals"]

    tr_H = sum([target_H[i][i] for i in range(len(target_H))])

    exact_gs_energy = min(exact_evals)

    size_basis_final = experiment_params["kbo_final"] + 1

    matrix_params = {
        "sb_final": size_basis_final,
        "ref_st_name": experiment_params["ref_st_name"],
        "seed": experiment_params["seed"],
        "J": experiment_params["J"],
        "h_bnd": experiment_params["h_bnd"],
    }

    exp_data = run_qse(
        target_H,
        reference_state,
        size_basis_final,
        exact_gs_energy,
        experiment_params,
        exp_data,
        matrix_params,
        noise_rng,
    )

    # populate constant data points
    n_qubits = int(np.log2(len(reference_state)))
    for _ in range(
        len(exp_data["energy"])
    ):  # take an arbitrary parameter to learn number of data points
        exp_data["size_state"].append(experiment_params["s_s"])
        exp_data["noise_strength"].append(experiment_params["noise_str"])
        exp_data["noise_type"].append(experiment_params["noise_type"])
        exp_data["noise_seed"].append(experiment_params["n_seed"])
        exp_data["init_reference_state"].append(experiment_params["ref_st_name"])
        exp_data["thresholding"].append(experiment_params["thresh"])
        if experiment_params["thresh"]:
            exp_data["thresholding_scale"].append(experiment_params["thresh_scale"])
        exp_data["kbo_final"].append(experiment_params["kbo_final"])
        exp_data["exact_energy"].append(exact_gs_energy)

    return pd.DataFrame(exp_data)


def run_qse(
    target_H,
    reference_state,
    size_basis_final,
    exact_gs_energy,
    experiment_params,
    exp_data,
    matrix_params,
    noise_rng,
):
    unique_H_elements = construct_unique_H_elements(
        matrix_params, size_basis_final, reference_state, target_H
    )

    H_final, S_final, unique_H_elements_noisy = compute_qse_matrices_krylov_basis(
        size_basis_final,
        unique_H_elements,
        noise_rng,
        noise_type=experiment_params["noise_type"],
        noise_strength=experiment_params["noise_str"],
    )

    assert (
        len(unique_H_elements_noisy) == (2 * size_basis_final - 1) + 1
    )  # extra <state|H^0|state> = 1 element at 0th idx

    H_final_no_noise, S_final_no_noise, _ = compute_qse_matrices_krylov_basis(
        size_basis_final, unique_H_elements, noise_rng, noise_strength=0
    )

    assert (
        H_final_no_noise.shape == S_final_no_noise.shape
        and H_final_no_noise.shape == (size_basis_final, size_basis_final)
    )

    # thresholding
    if experiment_params["thresh"]:
        if experiment_params["noise_str"] == 0:
            threshold = 1e-13
        else:
            threshold = (10 ** (-experiment_params["thresh_scale"])) * calc_noise_rate(
                H_final, S_final, H_final_no_noise, S_final_no_noise
            )

        H, S, kept_s_evecs = apply_thresholding(H_final, S_final, threshold)
    else:
        H, S = H_final, S_final

    evals, evecs = eig(H, b=S, right=True)
    sorted_evals = np.sort(evals)
    evals_coeffs = {evals[i]: evecs[:, i] for i in range(len(evals))}

    eval_0 = sorted_evals[0]

    exp_data["energy"].append(eval_0)
    exp_data["energy_err_rel"].append(
        np.abs(eval_0 - exact_gs_energy) / np.abs(exact_gs_energy)
    )

    return exp_data
