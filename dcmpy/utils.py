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