import numpy as np


def add_shot_noise_unique_H_elements(unique_H_elements, basis_size, shots, noise_rng):
    if len(unique_H_elements) < 4 * basis_size:
        raise Exception(
            "Need to estimate additional powers of H for shot noise, expected: {}, got: {}".format(
                4 * basis_size, len(unique_H_elements)
            )
        )

    noisy_unique_H_elements = [unique_H_elements[0]]

    for i in range(1, int(len(unique_H_elements) / 2)):
        var = unique_H_elements[2 * i] - unique_H_elements[i] ** 2
        std = np.sqrt(var / shots)

        noisy_unique_H_elements.append(unique_H_elements[i] + noise_rng.normal(0, std))

    return noisy_unique_H_elements


def calc_noise_rate(H_noisy, S_noisy, H_no_noise, S_no_noise):
    H_perturbation = H_noisy - H_no_noise
    S_perturbation = S_noisy - S_no_noise

    return np.sqrt(
        np.linalg.norm(H_perturbation, ord=2) ** 2
        + np.linalg.norm(S_perturbation, ord=2) ** 2
    )
