import jax

from sim import linear_sim, normal_dgp
from dcmpy.fit import fit
from dcmpy.losses import *


def param_diff(true_model_params, model_params):
    return np.mean(np.array([np.sum((true_model_params[k] - model_params[k])**2) for k in true_model_params]))


def util_diff(model_fn, true_model_params, model_params, data):
    return np.mean((model_fn(true_model_params, *data) - model_fn(model_params, *data))**2)


D, K = 5, 5
n_seeds = 10

results = []

for N in [1000, 5000, 10000, 20000, 50000]:
    for C in [4, 6, 8]:
        for seed in range(n_seeds):
            rng = jax.random.PRNGKey(seed)

            model_fn, true_model_params, init_model_params, data = linear_sim(rng, N, C, D, K)

            first_model_params = fit(first_loss, model_fn, init_model_params, data)
            first_last_model_params = fit(first_last_loss, model_fn, init_model_params, data)
            full_model_params = fit(full_loss, model_fn, init_model_params, data)

            val_data = normal_dgp(rng, model_fn, true_model_params, 100_000, C, D, K)

            results.append([
                N, C,
                param_diff(true_model_params, first_model_params),
                param_diff(true_model_params, first_last_model_params),
                param_diff(true_model_params, full_model_params),
                util_diff(model_fn, true_model_params, first_model_params, data),
                util_diff(model_fn, true_model_params, first_last_model_params, data),
                util_diff(model_fn, true_model_params, full_model_params, data)
            ])
            print("\t".join(map(str, results[-1])))

np.savez("results", results=results)
