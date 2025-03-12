import pickle as pkl
import numpy as np
from file_utils import convert_script_params_to_filename


def get_initial_state(name, rng, params=None):
    if name == "J_zero":
        ham_fname = convert_script_params_to_filename(
            {
                "s_s": params["n_qubit"],
                "h_bnd": params["h_bnd"],
                "J": 0,
                "seed": params["seed"],
            }
        )

        with open("saved_hamiltonians/" + ham_fname + ".pkl", "rb") as f:
            ham_data = pkl.load(f)
            exact_evals_no_cpl_H = ham_data["evals"]
            exact_evecs_no_cpl_H = ham_data["evecs"]

        exact_evals_evecs_no_cpl_H = {
            exact_evals_no_cpl_H[i]: exact_evecs_no_cpl_H[:, i]
            for i in range(len(exact_evals_no_cpl_H))
        }

        exact_gs_energy_no_cpl_H = min(exact_evals_no_cpl_H)
        reference_state = exact_evals_evecs_no_cpl_H[exact_gs_energy_no_cpl_H]
        # re initialise rng, so when it is called next actual H has same values for hs.
        rng = np.random.default_rng(params["seed"])
    else:
        raise Exception("Initial state: {} not recognised".format(name))

    return reference_state, rng
