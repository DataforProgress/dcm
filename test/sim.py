import jax

from dcmpy.models import linear, init_linear
from dgp import normal_dgp


def linear_sim(rng, N, C, D, K):
    # parameter we're estimating
    rng, next_rng = jax.random.split(rng)
    theta = jax.random.normal(next_rng, shape=(K, D))
    true_model_params = {
        "theta": theta
    }

    data = normal_dgp(rng, linear, true_model_params, N, C, D, K)

    model_params = init_linear(*data)

    return linear, true_model_params, model_params, data



