import contextlib

import numpy as np

def simulate_ll(n,
                p,
                uncorr_frac,
                num_groups,
                group_sparsity=0.5,
                seed=1):
    """TODO: document"""

    with _temp_seed(seed):

        # calculate numbers of correlated and uncorrelated features
        p_uncorr = int(uncorr_frac * p)
        p_corr = p - p_uncorr

        # start by generating a covariance matrix for correlated features
        _, sigma, groups = simulate_cov_groups(p_corr, num_groups)
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
        # X, is_correlated = _shuffle_same(X, is_correlated)

        # decide which feature groups to keep and which to zero out
        B = np.zeros((p_corr+1,))
        num_groups = len(groups)
        if group_sparsity < 1.0:
            num_groups_to_keep = int(num_groups * group_sparsity)
            groups_to_keep = np.random.choice(num_groups, num_groups_to_keep)
        else:
            groups_to_keep = np.arange(num_groups)

        for g_ix, group in enumerate(groups):
            if g_ix in groups_to_keep:
                # all variables in same group have same coefficient
                # (0 if we drop out that group)
                # draw from N(0, 1)
                B[group] = np.random.randn()

        # sample bias from N(0, 1)
        B[-1] = np.random.randn()

        # calculate Bernoulli parameter pi(x_i) for each sample x_i
        linsum = B[-1] + (X_corr @ B[:-1, np.newaxis])
        pis = 1 / (1 + np.exp(-linsum))

        # then sample labels y_i from a Bernoulli(pi_i)
        y = np.random.binomial(1, pis.flatten())

        info_dict = {
            'sigma': sigma,
            'betas': B,
            'pis': pis,
            'groups': groups,
            'is_correlated': is_correlated
        }

    return (X, y, info_dict)


def simulate_cov_groups(p, num_groups, pcov_value=0.9):
    """Simulate covariance matrix.

    TODO document
    """
    # precision matrix
    theta = np.zeros((p, p))

    # correlation groups
    groups = np.array_split(np.arange(p), num_groups)
    for group in groups:
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

    return theta, sigma, groups


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

