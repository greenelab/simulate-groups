import contextlib

import numpy as np

def simulate_cov_groups(p, num_groups, pcov_value=0.9):
    """Simulate covariance matrix"""
    # precision matrix
    theta = np.zeros((p, p))

    # correlation groups
    network_groups = np.array_split(np.arange(p), num_groups)
    for group in network_groups:
        i = group[0]
        for j in group[1:]:
            theta[i, j] = pcov_value
            theta[j, i] = pcov_value

    # make matrix positive definite (invertible) by adding to diagonal
    # https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
    min_eig = np.linalg.eig(theta)[0].min()
    # the +1 to the diagonal makes matrix inversion a bit more numerically
    # stable (since the minimum eigenvalue is the smallest possible
    # diagonal perturbation)
    theta = theta + ((-min_eig+1) * np.eye(p))

    # then invert to get covariance matrix
    # this ensures sigma is PSD
    sigma = np.linalg.inv(theta)

    return theta, sigma


def simulate_ll(n, p, uncorr_frac, num_groups, seed=1, verbose=False):
    """TODO: document"""

    with _temp_seed(seed):

        # calculate numbers of correlated and uncorrelated features
        p_uncorr = int(uncorr_frac * p)
        p_corr = p - p_uncorr

        if verbose:
            print('Number of informative features: {}'.format(p_corr))
            print('Number of uninformative features: {}'.format(p_uncorr))

        # start by generating a covariance matrix for correlated features
        _, sigma = simulate_cov_groups(p_corr, num_groups)
        # then generate data from a MVN distribution with that covariance
        X_corr = np.random.multivariate_normal(mean=np.zeros(p_corr),
                                               cov=sigma,
                                               size=(n,),
                                               check_valid='warn')

        # generate uncorrelated data from standard normal
        X_uncorr = np.random.randn(n, p_uncorr)
        X = np.concatenate((X_corr, X_uncorr), axis=1)

        # create a bool vector to remember which features are correlated
        # this will be useful when we shuffle features
        is_correlated = np.zeros(X.shape[1]).astype('bool')
        is_correlated[:p_corr] = True

        # shuffle data and is_correlated indicators in same order, so we know
        # which features are correlated/not correlated with outcome
        X, is_correlated = _shuffle_same(X, is_correlated)

        # draw regression coefficients (betas) from N(0, 1), plus a bias
        # TODO: add sparsity
        B = np.random.randn(p_corr+1)

        # calculate Bernoulli parameter pi(x_i) for each sample x_i
        linsum = B[0] + (X_corr @ B[1:, np.newaxis])
        pis = 1 / (1 + np.exp(-linsum))

        # then sample labels y_i from a Bernoulli(pi_i)
        y = np.random.binomial(1, pis.flatten())

    return (X, y, pis, is_correlated)


def _shuffle_same(X, y):
    """Shuffle a data matrix and a label matrix in the same order."""
    assert X.shape[1] == y.shape[0]
    p = np.random.permutation(X.shape[1])
    return X[:, p], y[p]


@contextlib.contextmanager
def _temp_seed(cntxt_seed):
    """Set a temporary np.random seed in the resulting context.

    This saves the global random number state and puts it back once the context
    is closed. See https://stackoverflow.com/a/49557127 for more detail.
    """
    state = np.random.get_state()
    np.random.seed(cntxt_seed)
    try:
        yield
    finally:
        np.random.set_state(state)

