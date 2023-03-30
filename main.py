import jax
import pandas as pd

from dcmpy.test.dgp import dgp
from dcmpy.models import linear_sim
from dcmpy.fit import fit, visualize_loss_landscape
from dcmpy.losses import *
from dcmpy.uncertainty import robust_confidence_intervals as cis
from dcmpy.utils import probs

def param_diff(true_model_params, model_params):
    return np.mean(np.array([np.sum((true_model_params[k] - model_params[k])**2) for k in true_model_params]))


def util_diff(model_fn, true_model_params, model_params, data):
    return np.mean(np.max(probs(model_fn(true_model_params, *data)) - probs(model_fn(model_params, *data)), axis=-1))


n_seeds = 10

results = []

for choice_type in ["categorical", "normal"]:
    for D in [10]:
        for K in [10, 20, 50]:
            for N in [100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
                for C in [8, 6, 4]:
                    for seed in range(n_seeds):
                        rng = jax.random.PRNGKey(seed)

                        rng, next_rng = jax.random.split(rng)
                        model_fn, true_model_params, init_model_params = linear_sim(next_rng, N, C, D, K)
                        data = dgp(rng, model_fn, true_model_params, N, C, D, K, choice_type=choice_type)

                        first_model_params, _ = fit(first_loglik, model_fn, init_model_params, data)

                        next_rng, _ = jax.random.split(next_rng)
                        # visualize_loss_landscape(next_rng, first_loss, model_fn, first_model_params, data, suffix="first")
                        # cis(first_loss, model_fn, first_model_params, data)
                        first_last_model_params, _ = fit(first_last_loglik, model_fn, init_model_params, data)
                        # visualize_loss_landscape(next_rng, first_last_loss, model_fn, first_last_model_params, data, suffix="first_last")
                        # sawtooth_model_params, _ = fit(sawtooth_loss, model_fn, init_model_params, data)
                        full_model_params, _ = fit(full_loglik, model_fn, init_model_params, data)
                        # visualize_loss_landscape(next_rng, full_loss, model_fn, full_model_params, data, suffix="full")

                        val_data = dgp(rng, model_fn, true_model_params, 100_000, K, D, K, choice_type=choice_type)

                        results.append([
                            seed, choice_type, D, K, N, C,
                            param_diff( true_model_params, first_model_params),
                            param_diff( true_model_params, first_last_model_params),
                            # param_diff( true_model_params, sawtooth_model_params),
                            param_diff( true_model_params, full_model_params),
                            util_diff(model_fn, true_model_params, first_model_params, val_data),
                            util_diff(model_fn, true_model_params, first_last_model_params, val_data),
                            # util_diff(model_fn, true_model_params, sawtooth_model_params, val_data),
                            util_diff(model_fn, true_model_params, full_model_params, val_data)
                        ])
                        print("\t".join(map(str, results[-1])))

                    pd.DataFrame(
                        results, columns=["seed", "choice_type", "D", "K", "N", "C", "first_param", "first_last_param", "full_param", "first", "first_last", "full"]
                    ).to_csv("results2.csv", index=False)
