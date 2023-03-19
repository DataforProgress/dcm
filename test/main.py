import jax
import pandas as pd

from sim import linear_sim, dgp
from dcmpy.fit import fit
from dcmpy.losses import *


def probs(logits):
    return np.exp(logits - sp.logsumexp(logits, axis=-1)[:, None])


def param_diff(true_model_params, model_params):
    return np.mean(np.array([np.sum((true_model_params[k] - model_params[k])**2) for k in true_model_params]))


def util_diff(model_fn, true_model_params, model_params, data):
    return np.mean(np.max(probs(model_fn(true_model_params, *data)) - probs(model_fn(model_params, *data)), axis=-1))


D, K = 10, 30
n_seeds = 10

results = []

for choice_type in ["categorical", "normal"]:
    for N in [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
        for C in [4, 6, 8]:
            for seed in range(n_seeds):
                rng = jax.random.PRNGKey(seed)

                rng, next_rng = jax.random.split(rng)
                model_fn, true_model_params, init_model_params = linear_sim(next_rng, N, C, D, K)
                data = dgp(rng, model_fn, true_model_params, N, C, D, K, choice_type=choice_type)

                first_model_params = fit(first_loss, model_fn, init_model_params, data)
                first_last_model_params = fit(first_last_loss, model_fn, init_model_params, data)
                sawtooth_model_params = fit(sawtooth_loss, model_fn, init_model_params, data)
                full_model_params = fit(full_loss, model_fn, init_model_params, data)

                val_data = dgp(rng, model_fn, true_model_params, 1_000_000, K, D, K, choice_type=choice_type)

                results.append([
                    seed, choice_type, N, C,
                    util_diff(model_fn, true_model_params, first_model_params, val_data),
                    util_diff(model_fn, true_model_params, first_last_model_params, val_data),
                    util_diff(model_fn, true_model_params, sawtooth_model_params, val_data),
                    util_diff(model_fn, true_model_params, full_model_params, val_data)
                ])
                print("\t".join(map(str, results[-1])))

            pd.DataFrame(
                results, columns=["seed", "choice_type", "N", "C", "first", "first_last", "sawtooth", "full"]
            ).to_csv("results.csv", index=False)
