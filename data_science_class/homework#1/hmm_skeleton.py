import numpy as np


def viterbi(pis: np.ndarray, trans_probs: np.ndarray, emission_probs: np.ndarray, x: np.ndarray) -> np.ndarray:
    # pis: initial probability in a one-dimensional array with m elements
    # trans_probs: transition probability (i.e., trans_probs[i,j] is P(Z_{t+1}=j | Z_t=i)), 2d-array with shape (m, m)
    # emission_probs: emission probability, emission_probs[i,k] is P(X_t=k | Z_t=i), 2d-array with shape (m, size of X's domain)
    # x: an observational sequence in a 1d array with n elements
    T, E, n, m = trans_probs, emission_probs, len(x), len(pis)
    mu = np.zeros((n, m))  # to fill in
    max_seq = [None] * n  # to return

    # TODO

    return max_seq
