import jax
import pandas as pd

from dcmpy.test.dgp import dgp
from dcmpy.models import linear_sim
from dcmpy.fit import fit, visualize_loss_landscape, print_hess_landscape, visualize_grad_landscape, visualize_grad_hess_quad_landscape
from dcmpy.losses import *
from dcmpy.uncertainty import robust_confidence_intervals as cis
from dcmpy.utils import probs

def param_diff(true_model_params, model_params):
    return np.mean(np.array([np.sum((true_model_params[k] - model_params[k])**2) for k in true_model_params]))


def util_diff(model_fn, true_model_params, model_params, data):
    return np.mean(np.max(probs(model_fn(true_model_params, *data)) - probs(model_fn(model_params, *data)), axis=-1))


n_seeds = 32

results = []
first_loss_fn, first_dir_fn = None, None
first_last_loss_fn, first_last_dir_fn = None, None
first_two_loss_fn, first_two_dir_fn = None, None
full_loss_fn, full_dir_fn = None, None

for choice_type in ["categorical", "normal"]:
    for D in [4, 8]:
        for K in [8, 16, 32]:
            for N in [1_000, 2_000, 4_000, 8_000]:
                for C in [8, 6, 4]:
                    for seed in range(n_seeds):
                        rng = jax.random.PRNGKey(seed)

                        rng, next_rng = jax.random.split(rng)
                        model_fn, true_model_params, init_model_params = linear_sim(next_rng, N, C, D, K)
                        rng, next_rng = jax.random.split(rng)
                        data, all_choices = dgp(next_rng, model_fn, true_model_params, N, C, D, K, choice_type=choice_type)

                        first_model_params, first_grad_norm, (first_loss_fn, first_dir_fn) = fit(
                            first_loglik, model_fn, true_model_params, data, loss_fn=first_loss_fn, dir_fn=first_dir_fn
                        )
                        first_last_model_params, first_last_grad_norm, (first_last_loss_fn, first_last_dir_fn) = fit(
                            first_last_loglik, model_fn, true_model_params, data, loss_fn=first_last_loss_fn, dir_fn=first_last_dir_fn
                        )
                        first_two_model_params, first_two_grad_norm, (first_two_loss_fn, first_two_dir_fn) = fit(
                            lambda logits: full_loglik(logits, first_k=2), model_fn, true_model_params, data, loss_fn=first_two_loss_fn, dir_fn=first_two_dir_fn
                        )
                        full_model_params, full_grad_norm, (full_loss_fn, full_dir_fn) = fit(
                            full_loglik, model_fn, true_model_params, data, loss_fn=full_loss_fn, dir_fn=full_dir_fn
                        )

                        val_data, _ = dgp(rng, model_fn, true_model_params, 100_000, K, D, K, all_choices=all_choices)

                        results.append([
                            seed, choice_type, D, K, N, C, 
                            param_diff( true_model_params, first_model_params),
                            param_diff( true_model_params, first_last_model_params),
                            param_diff( true_model_params, first_two_model_params),
                            param_diff( true_model_params, full_model_params),
                            util_diff(model_fn, true_model_params, first_model_params, val_data),
                            util_diff(model_fn, true_model_params, first_last_model_params, val_data),
                            util_diff(model_fn, true_model_params, first_two_model_params, val_data),
                            util_diff(model_fn, true_model_params, full_model_params, val_data),
                            first_grad_norm, first_last_grad_norm, first_two_grad_norm, full_grad_norm
                        ])
                        print("\t".join(map(str, results[-1][:-4])))

                    pd.DataFrame(
                        results, 
                        columns=[
                            "seed", "choice_type",  
                            "D", "K", "N", "C", 
                            "first_param", "first_last_param", "first_two_param", "full_param", 
                            "first", "first_last", "first_two", "full",
                            "first_grad_norm", "first_last_grad_norm", "first_two_grad_norm", "full_grad_norm"
                        ]
                    ).to_csv("results.csv", index=False)
