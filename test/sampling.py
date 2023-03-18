import jax

from sim import linear_sim, sim
from dcmpy.fit import fit
from dcmpy.losses import *

rng = jax.random.PRNGKey(3)
N = 5_000
C = 6
D, K = 4, 4


def param_diff(true_model_params, model_params):
    return np.mean(np.array([np.sum((true_model_params[k] - model_params[k])**2) for k in true_model_params]))

model_fn, true_model_params, init_model_params, data = linear_sim(rng, N, C, D, K)


first_model_params = fit(first_loss, model_fn, init_model_params, data)
first_last_model_params = fit(first_last_loss, model_fn, init_model_params, data)
full_model_params = fit(full_loss, model_fn, init_model_params, data)

print(param_diff(true_model_params, first_model_params))
print(param_diff(true_model_params, first_last_model_params))
print(param_diff(true_model_params, full_model_params))

val_data = sim(rng, model_fn, true_model_params, 100_000, C, D, K)
true_utils = model_fn(true_model_params, *val_data)
print(np.mean((true_utils - model_fn(first_model_params, *val_data))**2))
print(np.mean((true_utils - model_fn(first_last_model_params, *val_data))**2))
print(np.mean((true_utils - model_fn(full_model_params, *val_data))**2))
