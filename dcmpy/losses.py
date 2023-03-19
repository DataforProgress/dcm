import itertools
import jax.numpy as np
from jax.scipy import special as sp


def last_prob(weights, eps=1e-16):
    N = weights.shape[1]
    p = 1
    for n in range(1, N):
        for A in itertools.combinations(range(N - 1), n):
            p += (-1) ** n * weights[:, N - 1] / (weights[:, N - 1] + np.sum(weights[:, A], axis=-1))

    p = np.maximum(p, eps)
    # print(p)
    return p


def first_last_loss(logits):
    loglik_first = -logits[:, 0] + sp.logsumexp(logits, axis=-1)
    loglik_last = -np.log(last_prob(np.exp(logits[:, 1:])))
    return np.mean(loglik_first + loglik_last)


def first_loss(logits):
    loglik_first = -logits[:, 0] + sp.logsumexp(logits, axis=-1)
    return np.mean(loglik_first)


def full_loss(logits):
    N = logits.shape[1]
    loglik_full = 0
    for n in range(N - 1):
        loglik_full += sp.logsumexp(logits[:, n:], axis=-1) - logits[:, n]
    return np.mean(loglik_full)


def sawtooth_loss(logits):
    loglik_sawtooth = -logits[:, 0] + logits[:, -1] + 2 * sp.logsumexp(logits, axis=-1)
    return np.mean(loglik_sawtooth)
