import jax.numpy as np
import jax

def normal_dgp(rng, model_fn, model_params, N, C, D, K):
    # simulated covariates
    rng, next_rng = jax.random.split(rng)
    respondent_features = jax.random.normal(next_rng, shape=(N, D))
    rng, next_rng = jax.random.split(rng)
    choice_set_features = jax.random.normal(next_rng, shape=(N, C, K))

    logits = model_fn(model_params, choice_set_features, respondent_features)

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1)[:, None]
    exp_samples = probs * jax.random.exponential(next_rng, shape=(N, C))
    choices = np.argsort(-exp_samples)

    ordered_choice_set_features = choice_set_features[np.arange(N)[:, None], choices]
    return ordered_choice_set_features, respondent_features
