import contextlib
import itertools as it

import numpy as np

def simulate_ll(n,
                p,
                uncorr_frac,
                num_groups,
                group_sparsity=0.5,
                seed=1):
    """Simulate data (features and labels) from a log-linear model.

    For background on log-linear data simulation, see:
    https://stats.stackexchange.com/a/46525

    Labels are generated from a subset of the features (so some of
    the features are uncorrelated with the label). Correlated features
    have a block covariance structure (that is, there are num_groups
    covariance blocks; samples within each group are highly correlated and
    samples between groups should be uncorrelated).

    Arguments
    ----------
    n (int): number of samples
    p (int): number of features
    uncorr_frac (float): fraction of features to be uncorrelated with outcome
                         (must be between 0 and 1)
    num_groups (int): number of groups/covariance blocks in the data
                      (must be >0 and <=n)
    group_sparsity (float): proportion of groups that will have nonzero
                            coefficients (beta values) used to generate labels
    seed (int): seed for random number generation

    Returns
    -------
    X (array_like [n, p]): simulated features/samples
    y (array_like [n, 1]): simulated labels in {0, 1}
    info_dict (dict): dict containing parameters used to generate data
                      (e.g. covariance matrix, feature groups, linear sum
                       coefficients)
    """
    with _temp_seed(seed):

        # calculate numbers of correlated and uncorrelated features
        p_uncorr = int(uncorr_frac * p)
        p_corr = p - p_uncorr

        # start by generating a covariance matrix for correlated features
        # _, sigma, groups = simulate_cov_groups(p_corr, num_groups)
        sigma, groups = simulate_groups(p_corr, num_groups)
        # then generate data from a MVN distribution with that covariance
        X_corr = np.random.multivariate_normal(mean=np.zeros(p_corr),
                                               cov=sigma,
                                               size=(n,),
                                               check_valid='warn')

        # generate uncorrelated data from standard normal
        X_uncorr = np.random.randn(n, p_uncorr)
        X = np.concatenate((X_corr, X_uncorr), axis=1)

        # create a bool vector to remember which features are correlated
        # this will be useful if we shuffle features
        is_correlated = np.zeros(X.shape[1]).astype('bool')
        is_correlated[:p_corr] = True

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
    """Simulate a positive definite (invertible) covariance matrix.

    The easiest way to do this is to set the precision matrix (inverse
    of covariance matrix) to the desired correlations between variables, then
    add to the diagonal to make the matrix positive definite. This can always
    be done for a symmetric matrix, see:
    https://nhigham.com/2021/02/16/diagonally-perturbing-a-symmetric-matrix-to-make-it-positive-definite/
    """
    # create and fill precision matrix
    # the entries of the precision matrix specify partial correlations between
    # variables
    # https://en.wikipedia.org/wiki/Partial_correlation
    theta = np.zeros((p, p))

    # set correlation groups
    # this makes sure all variables in the group are correlated
    # (specifically, they will all have partial correlation pcov_value)
    groups = np.array_split(np.arange(p), num_groups)
    for group in groups:
        i = group[0]
        for j in group[1:]:
            theta[i, j] = pcov_value
            theta[j, i] = pcov_value

    # make theta positive definite (invertible) by adding to diagonal
    min_eig = np.linalg.eig(theta)[0].min()
    # adding +1 to the diagonal makes matrix inversion a bit more numerically
    # stable (since the minimum eigenvalue is the smallest possible
    # diagonal perturbation)
    theta = theta + ((-min_eig+1) * np.eye(p))

    # then invert to get covariance matrix
    # sigma is by definition positive definite, since we just inverted it
    sigma = np.linalg.inv(theta)

    return theta, sigma, groups


def simulate_groups(p, num_groups, cov_value=0.5, eps=0.1):
    """Specify covariance matrix directly and simulate correlated blocks.

    Directly specifying the covariance matrix works fine in the case
    where we want groups of correlated features, although specifying
    the precision matrix would allow more complex correlation patterns.
    """
    # create and fill covariance matrix directly
    sigma = np.zeros((p, p))

    # set correlation groups
    # this makes sure all variables in the group are correlated
    # (specifically, they will all have correlation cov_value)
    groups = np.array_split(np.arange(p), num_groups)
    for group in groups:
        for i, j in it.combinations(group, 2):
            sigma[i, j] = cov_value
            sigma[j, i] = cov_value

    # make theta positive definite (invertible) by adding to diagonal
    min_eig = np.linalg.eig(sigma)[0].min()
    # adding small perturbation to the diagonal makes matrix inversion a
    # bit more numerically stable (since the minimum eigenvalue is the
    # smallest possible diagonal perturbation)
    sigma = sigma + ((-min_eig + eps) * np.eye(p))
    return sigma, groups


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

