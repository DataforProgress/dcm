import jax
import jax.numpy as np
from jax.scipy import special as sp

def probs(logits):
    return np.exp(logits - sp.logsumexp(logits, axis=-1)[:, None])


def factorial(n, start=2):
    return jax.lax.fori_loop(start, n + 1, lambda carry, i: carry * i, 1)

def comb(N,k):
    return (factorial(N, N - k + 1) / factorial(k)).astype(int)

comb = jax.vmap(comb, in_axes=(None, 0))


import jax.numpy as jnp


def log1m_exp(x):
    """
    Numerically stable calculation
    of the quantity log(1 - exp(x)),
    following the algorithm of
    Machler [1]. This is
    the algorithm used in TensorFlow Probability,
    PyMC, and Stan, but it is not provided
    yet with Numpyro.

    Currently returns NaN for x > 0,
    but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    # return 0. rather than -0. if
    # we get a negative exponent that exceeds
    # the floating point representation
    arr_x = 1.0 * jnp.array(x)
    oob = arr_x < jnp.log(jnp.finfo(
        arr_x.dtype).smallest_normal)
    mask = arr_x > -0.6931472  # appox -log(2)
    more_val = jnp.log(-jnp.expm1(arr_x))
    less_val = jnp.log1p(-jnp.exp(arr_x))

    return jnp.where(
        oob,
        0.,
        jnp.where(
            mask,
            more_val,
            less_val))
