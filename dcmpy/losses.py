from typing import Any, Optional, Tuple, Union

import math
import itertools
import jax.numpy as np
from jax.scipy import special as sp

from dcmpy.utils import comb

def last_loglik(logits: np.ndarray, eps: Optional[float]=1e-16):
    """Log-likelihood of last item being chosen, assuming only the most preferred and least preferred are observed.

    This relies on the symmetry between sequential sampling without replacement and exponential random variables
    to define a more computationally efficient expression for the probability of the last item being chosen.

    NOTE: This is numerically unstable. The way to make it stable is to rewrite it to store each log-probability
    in a matrix, without taking the exp, and then logsumexp over the matrix. This is not implemented here.
    
    Args:
        weights: weights of choice model, shape (N, K), ordered from most to least preferred
        eps: small value to add to avoid log(0)
    
    Returns:
        probability of last item being chosen, shape (N,)
    """
    N = logits.shape[1]
    p = 1
    for n in range(1, N):
        for A in itertools.combinations(range(N - 1), n):
            A = (N - 1,) + A
            p += (-1) ** n * np.exp(logits[:, N - 1] - sp.logsumexp(logits[:, A], axis=-1)) 

    p = np.maximum(p, eps)
    return np.log(p)

def last_loglik_stable(logits: np.ndarray, eps: Optional[float]=1e-16):
    """Probability of last item being chosen, assuming only the most preferred and least preferred are observed.

    This relies on the symmetry between sequential sampling without replacement and exponential random variables
    to define a more computationally efficient expression for the probability of the last item being chosen.

    NOTE: This is an attempt to make the above function numerically stable. It is not yet working.
    
    Args:
        weights: weights of choice model, shape (N, K), ordered from most to least preferred
    
    Returns:
        probability of last item being chosen, shape (N,)
    """
    N = logits.shape[1]
    total_probs = sum(math.comb(N-1, i) for i in range(1, N)) 

    logp = np.zeros((logits.shape[0], 1 + total_probs))
    signp = np.ones(1 + total_probs)

    ind = 1
    for n in range(1, N):
        for A in itertools.combinations(range(N - 1), n):
            A = (N - 1,) + A
            signp = signp.at[ind].set((-1) ** n)
            logp = logp.at[:, ind].set(logits[:, N - 1] - sp.logsumexp(logits[:, A], axis=-1))
            ind += 1

    last_loglik, _ = sp.logsumexp(logp, b=signp, axis=-1, return_sign=True)
    return np.maximum(last_loglik, np.log(eps))


def first_last_loglik(logits: np.ndarray):
    """Log-likelihood of choice model, assuming only the most preferred and least preferred are observed.
    
    Args:
        logits: logits of choice model, shape (N, K), ordered from most to least preferred
    
    Returns:
        log-likelihood of choice model, shape (N,)
    """
    loglik_first = -logits[:, 0] + sp.logsumexp(logits, axis=-1)
    loglik_last = -last_loglik(logits[:, 1:])
    return loglik_first + loglik_last


def first_loglik(logits: np.ndarray):
    """Log-likelihood of choice model, assuming only the most preferred is observed.
    
    Args:
        logits: logits of choice model, shape (N, K), ordered from most to least preferred
        
    Returns:
        log-likelihood of choice model, shape (N,)
    """
    loglik_first = -logits[:, 0] + sp.logsumexp(logits, axis=-1)
    return loglik_first


def full_loglik(logits: np.ndarray):
    """Full log-likelihood of choice model, assuming full ordering is observed.
    
    Args:
        logits: logits of choice model, shape (N, K), ordered from most to least preferred
        
    Returns:
        log-likelihood of choice model, shape (N,)
    """
    N = logits.shape[1]
    loglik_full = 0
    for n in range(N - 1):
        loglik_full += sp.logsumexp(logits[:, n:], axis=-1) - logits[:, n]
    return loglik_full


def sawtooth_loglik(logits: np.ndarray):
    """Log-likelihood of choice model, assuming only the most preferred and least preferred are observed, using sawtooth method.

    WARNING: This does not converge without regularization!

    Args:
        logits: logits of choice model, shape (N, K), ordered from most to least preferred

    Returns:
        log-likelihood of choice model, shape (N,)
    """
    loglik_sawtooth = -logits[:, 0] + logits[:, -1] + 2 * sp.logsumexp(logits, axis=-1)
    return np.mean(loglik_sawtooth)
