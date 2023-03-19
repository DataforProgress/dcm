import jax.numpy as np


def init_linear(N, C, D, K):
    model_params = {
        "theta": np.zeros((K, D))
    }
    return model_params


def linear(model_params, choice_set_features, respondent_features):
    return np.einsum('nck,kd,nd->nc', choice_set_features, model_params["theta"], respondent_features)
