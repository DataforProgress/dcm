from typing import Any, Optional, Tuple, Callable

import jax.numpy as np
import jax


def robust_confidence_intervals(
        loss: Callable,
        model_fn: Callable,
        model_params: dict,
        data: Tuple[Any, ...]
):
    """Compute robust confidence intervals using the sandwich estimator.

    This assumes asymptotic normality of the maximum likelihood estimator.
    
    Args:
        loss: loss function
        model_fn: model function taking model parameters and data as arguments
        model_params: model parameters
        data: tuple of data, will be a tuple with respondent-specific data first, then item-specific data
        
    Returns:
        robust confidence intervals
    """
    grad_fn = jax.vmap(
        jax.grad(
            lambda model_params, data: np.mean(loss(model_fn(model_params, data[0][None, :, :], data[1][None, :]))),
            argnums=(0,)
        ), in_axes=(None, (0, 0))
    )

    def outer_grads(data):
        grads, = grad_fn(model_params, data)
        return jax.vmap(np.outer)(grads["theta"], grads["theta"])


    hess_fn = jax.jit(
        jax.hessian(
            lambda model_params: np.mean(np.mean(loss(model_fn(model_params, *data))))
        )
    )
    outer_grad = np.mean(outer_grads(data), axis=0)
    hess = hess_fn(model_params)["theta"]["theta"]
    sandwhich_sigma = outer_grad @ np.linalg.inv(hess) @ outer_grad
    return sandwhich_sigma
