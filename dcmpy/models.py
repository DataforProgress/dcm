import jax
import jax.numpy as np

def linear_sim(rng, N, C, D, K):
    # parameter we're estimating
    rng, next_rng = jax.random.split(rng)
    theta = jax.random.normal(next_rng, shape=((K - 1) * D,)) 
    true_model_params = {
        "theta": theta
    }

    model_params = init_linear(N, C, D, K)

    return linear, true_model_params, model_params


def init_linear(N: int, C: int, D: int, K: int):
    """Initialize the parameters for the linear model.

    Args:
        N: number of respondents.
        C: number of choices.
        D: number of respondent features.
        K: number of choice set features.

    Returns:
        dictionary of model parameters.
    """
    model_params = {
        "theta": np.zeros((K - 1) * D)
    }
    return model_params


def linear(
        model_params: dict, 
        choice_set_features: np.ndarray, 
        respondent_features: np.ndarray
):
    """Linear model with interaction features.
    
    Args:
        model_params: dictionary of model parameters.
        choice_set_features: array of shape (N, C, K) containing the choice set features.
        respondent_features: array of shape (N, D) containing the respondent features.
        
    Returns:
        array of shape (N, C) containing the logits.
    """
    K = choice_set_features.shape[-1]
    D = respondent_features.shape[-1]
    # interaction_features = (choice_set_features[:, :, :, None] * respondent_features[:, None, None, :]).reshape(N, C, -1)
    # interaction_features @ model_params["theta"].flatten()
    theta = model_params["theta"].reshape((K - 1, D))
    theta = np.vstack([theta, np.zeros(D)])
    return np.einsum('nck,kd,nd->nc', choice_set_features, theta, respondent_features)
