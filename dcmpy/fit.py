from typing import Any, Optional, Tuple, Callable

import jax.numpy as np
import jax

import matplotlib.pyplot as plt


def grad_update(params: dict, grads: dict, lr: Optional[float]=1.0):
    """Update parameters using gradient descent.
    
    Args:
        params: dict of parameters
        grads: dict of gradients
        lr: learning rate
        
    Returns:
        updated parameters
    """
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def fit(
        loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        data: Tuple[Any, ...],
        maxit: Optional[int] = 500_000, 
        eps: Optional[float] = 1e-8, 
        verbose: Optional[bool] = False, 
        lr: Optional[float] = 0.5
):
    """Fit a model to data using gradient descent.

    Args:
        loss_fn: loss function
        model_fn: model function taking model parameters and data as arguments
        model_params: model parameters
        data: tuple of data, will be a tuple with respondent-specific data first, then item-specific data
        maxit: maximum number of iterations
        eps: convergence threshold
        verbose: whether to print progress
        lr: learning rate
    
    Returns:
        fitted model parameters
        gradient norm
    """
    loss_grad = jax.jit(
        jax.grad(
            lambda model_params: np.mean(loss_fn(model_fn(model_params, *data)))
        )
    )

    for i in range(maxit):
        grads = loss_grad(model_params)
        grad_norm = sum(np.sum(grads[k]**2) for k in grads)
        if i % 500 == 0 and verbose:
            print(grads)
            print(i, loss_fn(model_fn(model_params, *data)), grad_norm)
        if any([np.any(np.isnan(grads[k])) for k in grads]):
            print("NaN gradient update, aborting...")
            break
        model_params = grad_update(model_params, grads, lr=lr)
        if grad_norm < eps:
            print("Converged!")
            break

    if grad_norm > eps:
        print(f"Failed to converge, gradient norm is {grad_norm}.")

    return model_params, grad_norm


def visualize_loss_landscape(
        rng, loss_fn, model_fn, model_params, data, 
        scale_width=3, n_grid=100, suffix=""
):
    """Visualize the loss landscape around the current model parameters.

    Args:
        rng: random number generator
        loss_fn: loss function
        model_fn: model function
        model_params: model parameters
        data: data
        scale_width: width of the loss landscape
        n_grid: number of grid points in each direction
        suffix: suffix for the saved file
    """
    rng, next_rng = jax.random.split(rng)
    a = jax.random.normal(next_rng, shape=model_params["theta"].shape)
    b = jax.random.normal(rng, shape=model_params["theta"].shape)

    scale = np.linspace(-scale_width, scale_width, n_grid)
    losses = np.zeros(n_grid ** 2)
    j = 0
    for s_a in scale:
        for s_b in scale:
            test_params = {"theta": model_params["theta"] + s_a * a + s_b * b}
            loss = np.mean(loss_fn(model_fn(test_params, *data)))
            losses = losses.at[j].set(loss)
            # print(s_a, s_b, loss)
            j += 1

    x, y = np.meshgrid(scale, scale)
    fig, ax = plt.subplots(ncols=1, figsize=(12, 10))
    ax = plt.tricontourf(x.flatten(), y.flatten(), losses)
    plt.plot(0, 0, marker='o', color="white")
    cbar = fig.colorbar(ax)
    plt.savefig(f"loss_landscape.{suffix}.png")