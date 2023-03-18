import jax

from dcmpy.models import linear, init_linear
from dgp import normal_dgp, categorical_dgp


def linear_sim(rng, N, C, D, K, dgp="normal"):
    # parameter we're estimating
    rng, next_rng = jax.random.split(rng)
    theta = jax.random.normal(next_rng, shape=(K, D))
    true_model_params = {
        "theta": theta
    }

    if dgp == "categorical":
        raise NotImplementedError
        # there is something wrong with the categorical dgp right now
        dgp = categorical_dgp
    elif dgp == "normal":
        dgp = normal_dgp
    else:
        raise NotImplementedError

    rng, next_rng = jax.random.split(rng)
    data = dgp(next_rng, linear, true_model_params, N, C, D, K)

    model_params = init_linear(*data)

    return linear, true_model_params, model_params, data



