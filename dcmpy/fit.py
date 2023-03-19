import jax.numpy as np
import jax


def grad_update(params, grads, lr=0.5):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def fit(loss, model_fn, model_params, data, maxit=100_000, eps=1e-8, verbose=False):
    loss_grad = jax.jit(
        jax.grad(
            lambda model_params: loss(model_fn(model_params, *data)) + 0.001 * np.sum(model_params["theta"]**2)
        )
    )

    for i in range(maxit):
        grads = loss_grad(model_params)
        if any([np.any(np.isnan(grads[k])) for k in grads]):
            print("NaN gradient update, aborting...")
            break
        model_params = grad_update(model_params, grads)
        grad_norm = sum(np.sum(grads[k]**2) for k in grads)
        if i % 500 == 0 and verbose:
            print(i, loss(model_fn(model_params, *data)), grad_norm)
        if grad_norm < eps:
            break
    return model_params
