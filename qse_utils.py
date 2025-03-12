import numpy as np
from file_utils import convert_script_params_to_filename
from noise_utils import add_shot_noise_unique_H_elements
from scipy.linalg import eig


def construct_unique_H_elements(
    matrix_params, size_basis, reference_state, target_H, saved=True
):
    shot_noise_matrix_params = dict(matrix_params)
    shot_noise_matrix_params["sb_final"] = 2 * matrix_params["sb_final"]

    if saved:
        matrix_string = convert_script_params_to_filename(shot_noise_matrix_params)

        fname = "qse_matrices/" + matrix_string

        try:
            unique_H_elements = np.load(fname + ".npy", allow_pickle=True)
            return unique_H_elements.real
        except FileNotFoundError:
            pass

    unique_H_elements = []

    for power in range(0, 2 * size_basis * 2):
        unique_H_elements.append(
            np.vdot(
                reference_state,
                np.linalg.matrix_power(target_H, power).dot(reference_state),
            )
        )

    if saved:
        matrix_string = convert_script_params_to_filename(shot_noise_matrix_params)
        fname = "qse_matrices/" + matrix_string

        np.save(fname, unique_H_elements, allow_pickle=True)

    return np.array(unique_H_elements).real


def compute_qse_matrices_krylov_basis(
    size_basis, unique_H_elements, noise_rng, noise_type="none", noise_strength=0
):
    # populate these matrices and solve with krylov basis
    H = np.zeros((size_basis, size_basis), dtype=np.float64)

    if (
        noise_strength > 0
    ):  # we don't add noise to 0th element as we know this is 1 for a normalised ref st.
        if noise_type == "finite_shot_noise":
            unique_H_elements_noisy = add_shot_noise_unique_H_elements(
                unique_H_elements, size_basis, 1 / (noise_strength**2), noise_rng
            )
        else:
            raise Exception("Noise type {} not recognised".format(noise_type))
    else:
        unique_H_elements_noisy = list(unique_H_elements[: 2 * size_basis])

    for i in range(
        size_basis
    ):  # + 1 index offset to ignore <state|H^0|state> element at i=0
        for j in range(i, size_basis):
            H[i][j] = unique_H_elements_noisy[i + j + 1]

            if i != j:
                H[j][i] = H[i][j]

    S = derive_S_from_H(
        H, size_basis, normalised_input=False, S_00=unique_H_elements_noisy[0]
    )

    return H, S, unique_H_elements_noisy


def derive_S_from_H(H, size_basis, normalised_input, S_00=None):
    """
    IMPORTANT: for our purposes we expect S and H to be real, regardless of noise. This may not hold for other
    choices for QSE basis i.e. not a Krylov basis.
    """
    S = np.zeros((size_basis, size_basis), dtype=np.float64)

    if normalised_input:
        S[0][0] = 1
    else:
        if S_00 is None:
            raise Exception("Norm must be defined if normalised input is False")
        S[0][0] = S_00

    for i in range(size_basis - 1):
        for j in range(size_basis):
            S[i + 1][j] = H[i][j]

    for i in range(size_basis - 1):
        S[0][i + 1] = H[0][i]

    return S


def apply_thresholding(H, S, threshold):
    s_evals, s_evecs = eig(S)
    kept_s_evecs = []

    for eval_idx in range(len(s_evals)):
        if s_evals[eval_idx] > threshold:
            kept_s_evecs.append(s_evecs[:, eval_idx])

    if len(kept_s_evecs) == 0:
        raise Exception(
            "No eigenvectors left after thresholding, for KBO {}".format(len(H) - 1)
        )

    kept_s_evecs = np.transpose(kept_s_evecs)  # get back into eigenvector column format
    thresholded_H = np.dot(np.transpose(np.conj(kept_s_evecs)), np.dot(H, kept_s_evecs))
    thresholded_S = np.dot(np.transpose(np.conj(kept_s_evecs)), np.dot(S, kept_s_evecs))

    return thresholded_H, thresholded_S, kept_s_evecs
