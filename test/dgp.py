import jax.numpy as np
import jax


def ordered_samples_exponential(rng, logits):
    expits = np.exp(logits)
    probs = expits / np.sum(expits, axis=-1)[:, None]
    exp_samples = 1/probs * jax.random.exponential(rng, shape=logits.shape)
    return np.argsort(exp_samples, axis=-1)


def ordered_samples_sequential(rng, logits):
    N, C = logits.shape
    order = np.zeros((N, C), dtype=int)
    for c in range(C):
        rng, next_rng = jax.random.split(rng)
        selected = jax.random.categorical(next_rng, logits)
        logits = logits.at[np.arange(N), selected].set(-np.inf)
        order = order.at[np.arange(N), c].set(selected)
    return order


ordered_samples = ordered_samples_exponential


def dgp(
        rng, model_fn, model_params, N, C, D, K, total_choices=None,
        respondent_intercept=True, choice_intercept=False, choice_type="discrete"
):
    if total_choices is None:
        total_choices = K

    if choice_type == "categorical" and total_choices == K:
        all_choices = np.eye(K)
    elif choice_type == "normal":
        rng, next_rng = jax.random.split(rng)
        all_choices = jax.random.normal(next_rng, shape=(total_choices, K)) / K
    else:
        raise NotImplementedError

    # simulated covariates
    rng, next_rng = jax.random.split(rng)
    respondent_features = jax.random.normal(next_rng, shape=(N, D))
    if respondent_intercept:
        respondent_features = respondent_features.at[:, 0].set(1)
    rng, next_rng = jax.random.split(rng)

    choice_set_features = all_choices[
        jax.vmap(
            lambda rng: jax.random.choice(rng, K, shape=(C,), replace=False)
        )(jax.random.split(rng, N))
    ]

    if choice_intercept:
        choice_set_features = choice_set_features.at[:, :, 0].set(1)

    logits = model_fn(model_params, choice_set_features, respondent_features)
    rng, next_rng = jax.random.split(rng)
    order = ordered_samples(next_rng, logits)
    ordered_choice_set_features = choice_set_features[np.arange(N)[:, None], order]

    return ordered_choice_set_features, respondent_features
