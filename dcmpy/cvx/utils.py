import cvxpy as cp


def vectorized_dot_product(a, b):
    return cp.sum(cp.multiply(a, b), axis=1)
