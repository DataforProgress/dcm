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
        _loss_fn: Callable, 
        model_fn: Callable,
        model_params: dict, 
        data: Tuple[Any, ...],
        maxit: Optional[int] = 500, 
        eps: Optional[float] = 1e-6, 
        verbose: Optional[bool] = False, 
        lr: Optional[float] = 1.,
        loss_fn: Optional[Callable] = None,
        dir_fn: Optional[Callable] = None
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
    if loss_fn is None:
        @jax.jit
        def loss_fn(model_params, data):
            return np.mean(_loss_fn(model_fn(model_params, *data)))

    if dir_fn is None:
        @jax.jit
        def dir_fn(model_params, data):
            hess = jax.hessian(loss_fn)(model_params, data)
            hess = hess["theta"]["theta"]
            grad = jax.grad(loss_fn)(model_params, data)
            grad = grad["theta"]
            return {"theta": np.linalg.solve(hess, grad)}

    # loss_grad = jax.jit(cg_fn)
    # loss_grad = jax.jit(
    #     jax.grad(
    #        lambda model_params: np.mean(loss_fn(model_fn(model_params, *data)))
    #     )
    # )

    for i in range(maxit):
        grads = dir_fn(model_params, data)
        grad_norm = sum(np.sum(grads[k]**2) for k in grads)
        if i % 10 == 0 and verbose:
            print(i, loss_fn(model_params, data), grad_norm)
        if any([np.any(np.isnan(grads[k])) for k in grads]):
            print("NaN gradient update, aborting...")
            break
        model_params = grad_update(model_params, grads, lr=lr)
        if grad_norm < eps:
            print("Converged!")
            break

    if grad_norm > eps:
        print(f"Failed to converge, gradient norm is {grad_norm}.")

    return model_params, grad_norm, (loss_fn, dir_fn)


def visualize_loss_landscape(
        rng, loss_fn, model_fn, model_params, data, 
        scale_width=6, n_grid=100, suffix=""
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
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

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


def visualize_grad_landscape(
        rng, loss_fn, model_fn, model_params, data, 
        scale_width=6, n_grid=100, suffix=""
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
    grad_fn = jax.jit(
        jax.grad(
            lambda model_params: np.mean(loss_fn(model_fn(model_params, *data)))
        )
    )


    rng, next_rng = jax.random.split(rng)
    a = jax.random.normal(next_rng, shape=model_params["theta"].shape)
    b = jax.random.normal(rng, shape=model_params["theta"].shape)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    scale = np.linspace(-scale_width, scale_width, n_grid)
    grads = np.zeros(n_grid ** 2)
    j = 0
    for s_a in scale:
        for s_b in scale:
            test_params = {"theta": model_params["theta"] + s_a * a + s_b * b}
            grad = grad_fn(test_params)
            grads = grads.at[j].set(np.linalg.norm(grad["theta"]))
            # print(s_a, s_b, loss)
            j += 1

    x, y = np.meshgrid(scale, scale)
    fig, ax = plt.subplots(ncols=1, figsize=(12, 10))
    ax = plt.tricontourf(x.flatten(), y.flatten(), grads)
    plt.plot(0, 0, marker='o', color="white")
    cbar = fig.colorbar(ax)
    plt.savefig(f"grad_landscape.{suffix}.png")

def visualize_grad_hess_quad_landscape(
        rng, loss_fn, model_fn, model_params, data, 
        scale_width=6, n_grid=100, suffix=""
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
    grad_fn = jax.jit(
        jax.grad(
            lambda model_params: np.mean(loss_fn(model_fn(model_params, *data)))
        )
    )

    hess_fn = jax.jit(
        jax.hessian(
            lambda model_params: np.mean(loss_fn(model_fn(model_params, *data)))
        )
    )


    rng, next_rng = jax.random.split(rng)
    a = jax.random.normal(next_rng, shape=model_params["theta"].shape)
    b = jax.random.normal(rng, shape=model_params["theta"].shape)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    scale = np.linspace(-scale_width, scale_width, n_grid)
    quad = np.zeros(n_grid ** 2)
    j = 0
    for s_a in scale:
        for s_b in scale:
            test_params = {"theta": model_params["theta"] + s_a * a + s_b * b}
            grad = grad_fn(test_params)
            hess = hess_fn(test_params)

            quad = quad.at[j].set(grad["theta"].T @ hess["theta"]["theta"] @ grad["theta"])
            # print(s_a, s_b, loss)
            j += 1

    x, y = np.meshgrid(scale, scale)
    fig, ax = plt.subplots(ncols=1, figsize=(12, 10))
    ax = plt.tricontourf(x.flatten(), y.flatten(), quad)
    plt.plot(0, 0, marker='o', color="white")
    cbar = fig.colorbar(ax)
    plt.savefig(f"grad_hess_quad_landscape.{suffix}.png")


def print_hess_landscape(
        rng, loss_fn, model_fn, model_params, data, 
        scale_width=3, n_grid=5, suffix=""
):
    hess_fn = jax.jit(
        jax.hessian(
            lambda model_params: np.mean(loss_fn(model_fn(model_params, *data)))
        )
    )
        
    rng, next_rng = jax.random.split(rng)
    a = jax.random.normal(next_rng, shape=model_params["theta"].shape)
    b = jax.random.normal(rng, shape=model_params["theta"].shape)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    scale = np.linspace(-scale_width, scale_width, n_grid)
    losses = np.zeros(n_grid ** 2)
    j = 0
    for s_a in scale:
        for s_b in scale:
            test_params = {"theta": model_params["theta"] + s_a * a + s_b * b}
            hess = hess_fn(test_params)
            print(s_a, s_b, np.linalg.eigvalsh(hess["theta"]["theta"]))
            j += 1