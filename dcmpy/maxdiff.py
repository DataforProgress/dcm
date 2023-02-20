import numpy as np
import cvxpy as cp

from dcmpy.utils import vectorized_dot_product


def maxchoice(
        choice_set_features, respondent_features, choice,
        lambd=None, alpha=None, solver="SCS"
):
    """
    Fit a maxdiff using choice features and respondent features.

    If a feature matrix is sparse, as it would be if it were individual or choice fixed effects, there will be
    significant speed benefits to passing sparse matrices instead of dense matrices.

    N is the number of profiles presented
    C is the number of choices in each choice set
    K is the number of choice features
    D is the number of respondent features
    :param choice_set_features: [N, C, K] array of features of choices presented to each respondent
    :type choice_set_features: np.array
    :param respondent_features: [N, D] array of respondent features
    :type respondent_features: np.array
    :param choice: [N] array of the index of the choice made by each respondent in each choice set
    :type choice: np.array
    :param lambd: parameter tuning amount of regularization
    :type lambd: float
    :param alpha: glmnet style convex combination between L1 and L2 regularization, alpha=0 is L2, alpha=1 is L1
    :type alpha: float
    :return: tuple of optimizer status and [K, D] array of parameters
    :rtype: tuple(str, array)
    """
    N, C, K = choice_set_features.shape
    D = respondent_features.shape[1]
    W = cp.Variable((K, D))

    W_respondent_features = respondent_features @ W.T
    logits = cp.vstack([
        vectorized_dot_product(choice_set_features[:, i, :], W_respondent_features)
        for i in range(C)
    ]).T
    choice_logit = logits[np.arange(N), choice]
    nll = cp.sum(cp.log_sum_exp(logits, axis=1) - choice_logit)
    if lambd is not None and alpha is not None:
        nll += lambd * (alpha * cp.sum(cp.abs(W)) + (1 - alpha) * cp.sum_squares(W))

    prob = cp.Problem(cp.Minimize(nll))
    prob.solve(solver=solver)
    return prob.status, W.value


def maxdiff(
        choice_set_features, respondent_features, best, worst,
        lambd=None, alpha=None, solver="SCS"
):
    """
    Fit a maxdiff using choice features and respondent features.

    If a feature matrix is sparse, as it would be if it were individual or choice fixed effects, there will be
    significant speed benefits to passing sparse matrices instead of dense matrices.

    N is the number of profiles presented
    C is the number of choices in each choice set
    K is the number of choice features
    D is the number of respondent features
    :param choice_set_features: [N, C, K] array of features of choices presented to each respondent
    :type choice_set_features: np.array
    :param respondent_features: [N, D] array of respondent features
    :type respondent_features: np.array
    :param choice: [N] array of the index of the choice made by each respondent in each choice set
    :type choice: np.array
    :param lambd: parameter tuning amount of regularization
    :type lambd: float
    :param alpha: glmnet style convex combination between L1 and L2 regularization, alpha=0 is L2, alpha=1 is L1
    :type alpha: float
    :return: tuple of optimizer status and [K, D] array of parameters
    :rtype: tuple(str, array)
    """
    N, C, K = choice_set_features.shape
    D = respondent_features.shape[1]
    t = cp.Variable(N)
    W = cp.Variable((K, D))

    W_respondent_features = respondent_features @ W.T
    logits = cp.vstack([
        vectorized_dot_product(choice_set_features[:, i, :], W_respondent_features)
        for i in range(C)
    ]).T
    best_logit = logits[np.arange(N), best]
    not_worst_logit = cp.multiply(1 - worst_choice_position_stacked, logits)
    # the above actually doesnt work lmao we need to actually do the proper indexing i think
    const = t == cp.log_sum_exp(not_worst_logit, axis=1)
    nll = cp.sum(cp.multiply(2, cp.log_sum_exp(logits, axis=1)) - best_logit - t)
    if lambd is not None and alpha is not None:
        nll += lambd * (alpha * cp.sum(cp.abs(W)) + (1 - alpha) * cp.sum_squares(W))

    prob = cp.Problem(cp.Minimize(nll), constraints=[const])
    prob.solve(solver=solver)
    return prob.status, W.value