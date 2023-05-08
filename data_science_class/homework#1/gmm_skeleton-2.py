from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import dirichlet, rand, multinomial
from scipy.stats import multivariate_normal


# Multivariate Normal distribution
class Normal:
    def __init__(self, mu: np.ndarray, cov: np.ndarray):
        self.mu = mu  # R^p
        self.cov = cov  # p x p matrix

    def pdf(self, data: np.array) -> np.ndarray:
        # data is an n x p array
        # returns densities for n data points
        return multivariate_normal.pdf(data, self.mu, self.cov)

    def rvs(self, n: int) -> np.ndarray:
        # returns n random samples (n x p matrix)
        return multivariate_normal.rvs(self.mu, self.cov, n)

    def __repr__(self):
        return f'Normal({self.mu}, {self.cov})'


def as_col(x):
    return x.reshape((-1, 1))


class GMM:
    def __init__(self, p: int, pis, mus, covs):
        self.p = p  # x is in p-dimensional space.
        self.pis = pis  # initial distribution
        # K components initialized with mus (K means) and covs (K covariances)
        self.comps = [Normal(mu, cov) for mu, cov in zip(mus, covs)]

    def logpdf(self, data: np.array):
        pdf_per_comp = [pi * comp.pdf(data) for pi, comp in zip(self.pis, self.comps)]
        return np.log(np.sum(np.array(pdf_per_comp), axis=0))

    def sampling(self, n: int) -> np.array:
        # samples n data points
        counts = multinomial(n, self.pis)
        samples = [self.comps[z].rvs(cnt) for z, cnt in enumerate(counts)]
        return np.concatenate(samples, axis=0)

    @staticmethod
    def random_GMM(K: int, p: int) -> 'GMM':
        pis = dirichlet([1.] * K, 1).flatten()
        mus = rand(K, p) * 4.0
        covs = [None] * K
        for _ in range(K):
            A = rand(p, p) * 2. - 1.
            covs[_] = A @ A.transpose()  # one way to create a covariance matrix
        covs = np.array(covs)
        return GMM(p, pis, mus, covs)

    @staticmethod
    def train(K: int, p: int, data: np.ndarray) -> Tuple['GMM', float]:
        # random initialization
        pis = ...
        mus = ...
        covs = ...

        ll = [-np.inf]
        while True:
            curr_comps = [Normal(mus[k], covs[k]) for k in range(K)]
            # expectation-step
            ...

            # maximization-step
            for k in range(K):
                # maximize parameters for each k-th component
                ...

            new_ll = ...  # compute new log likelihood
            if np.isclose(ll[-1], new_ll):
                break
            ll.append(new_ll)

        return GMM(p, pis, mus, covs), new_ll


if __name__ == '__main__':
    K, p, n = 2, 2, 500
    gmm = GMM.random_GMM(K, p)
    data = gmm.sampling(n)
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

    gmm, _ = GMM.train(K, p, data)
    print(gmm.comps)
