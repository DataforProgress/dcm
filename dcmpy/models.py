import jax.numpy as np


def init_linear(choice_set_features, respondent_features):
    N, D = respondent_features.shape
    _, C, K = choice_set_features.shape
    model_params = {
        "theta": np.zeros((K, D))
    }
    return model_params


def linear(model_params, choice_set_features, respondent_features):
    return np.einsum('nck,kd,nd->nc', choice_set_features, model_params["theta"], respondent_features)
