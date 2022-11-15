import typing

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.stats as scst
import seaborn as sns

rng = np.random.RandomState(42)


def sample_states(
    n_time_steps: int,
    initial_state_dist: np.ndarray,
    transition_matrix: np.ndarray,
) -> np.ndarray:
    states = [rng.multinomial(1, initial_state_dist)]
    for i in range(n_time_steps - 1):
        dist = transition_matrix @ states[-1]
        states.append(rng.multinomial(1, dist))

    return np.stack(states, 0)


def sample_emisson(
    states: np.ndarray, obs_scales: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    scales = obs_scales[states]
    return rng.normal(loc=0, scale=(obs_scales[np.newaxis] * states).sum(-1))


def sample(
    n_time_steps: int,
    initial_state_dist: np.ndarray,
    transition_matrix: np.ndarray,
    obs_scales: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    states = sample_states(n_time_steps, initial_state_dist, transition_matrix)
    emissions = sample_emisson(states, obs_scales)
    return states, emissions


def emit_log_probs(obs: np.ndarray, obs_scales: np.ndarray):
    dist_0 = scst.norm(loc=0, scale=obs_scales[0])
    dist_1 = scst.norm(loc=0, scale=obs_scales[1])
    return np.stack([dist_0.logpdf(obs), dist_1.logpdf(obs)], -1)


def infer_states(
    initial_state_dist: np.ndarray,
    transition_matrix: np.ndarray,
    obs_scales: np.ndarray,
    obs: np.ndarray,
) -> np.ndarray:
    log_priors = [np.log(initial_state_dist + 1e-8)]

    unn_log_filter = log_priors[0] * emit_log_probs(
        obs=obs[0], obs_scales=obs_scales
    )
    filtr = np.exp(unn_log_filter)
    filtr /= filtr.sum()
    filtrs = [filtr]

    for i in range(1, n_time_steps):
        prior = transition_matrix @ filtrs[-1]
        log_priors.append(np.log(prior + 1e-8))
        unn_log_filter = log_priors[-1] + emit_log_probs(
            obs=obs[i], obs_scales=obs_scales
        )
        filtr = np.exp(unn_log_filter)
        filtr /= filtr.sum()
        filtrs.append(filtr)

    return np.stack(filtrs, 0)


if __name__ == "__main__":
    n_time_steps = 255
    initial_state_dist = np.array([0.5, 0.5])
    transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    obs_scales = np.array([0.2, 1.0])

    states = sample_states(n_time_steps, initial_state_dist, transition_matrix)
    emissions = sample_emisson(states, obs_scales=obs_scales)
    inferred = infer_states(
        obs=emissions,
        initial_state_dist=initial_state_dist,
        obs_scales=obs_scales,
        transition_matrix=transition_matrix,
    )

    fig, axs = plt.subplots(3, 1, figsize=(6, 6))

    axs[0].imshow(states.T, cmap="binary")
    axs[0].set_aspect("auto")
    axs[0].set_title("states")
    axs[0].set_yticks([])
    axs[0].set_xticks([])

    axs[1].set_title("observations")
    axs[1].plot(emissions)
    axs[0].set_xticks([])

    axs[2].imshow(inferred.T, cmap='binary')
    axs[2].set_aspect("auto")
    axs[2].set_title("inferred states")
    axs[2].set_yticks([])

    plt.tight_layout()
    sns.despine(fig)

    plt.show()
