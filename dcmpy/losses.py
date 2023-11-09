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
    return np.log(p).mean()


def approx_last_loglik(logits: np.ndarray, eps: Optional[float]=1e-5, T: Optional[float]=10_000):
    """Log-likelihood of last item being chosen, assuming only the most preferred and least preferred are observed.

    This relies on the symmetry between sequential sampling without replacement and exponential random variables
    to define a more computationally efficient expression for the probability of the last item being chosen.
    
    Args:
        weights: weights of choice model, shape (N, K), ordered from most to least preferred
        eps: small value to add to avoid log(0)
    
    Returns:
        probability of last item being chosen, shape (N,)
    """
    u = np.linspace(0, 1 - eps, T)
    z = np.log1p(-u[:, None, None] ** np.exp(logits[:, :-1] - logits[:, -1][:, None])[None, :, :]).sum(axis=-1) 
    loglik = sp.logsumexp(z - np.log(T), axis=0)
    return loglik.mean()


def first_last_loglik(logits: np.ndarray):
    """Log-likelihood of choice model, assuming only the most preferred and least preferred are observed.
    
    Args:
        logits: logits of choice model, shape (N, K), ordered from most to least preferred
    
    Returns:
        log-likelihood of choice model, shape (N,)
    """
    loglik_first = -logits[:, 0] + sp.logsumexp(logits, axis=-1)
    loglik_last = -approx_last_loglik(logits[:, 1:])
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


def full_loglik(logits: np.ndarray, first_k: Optional[int]=None):
    """Full log-likelihood of choice model, assuming full ordering is observed.
    
    Args:
        logits: logits of choice model, shape (N, K), ordered from most to least preferred
        first_k: if not None, only use the first k logits to compute the log-likelihood
        
    Returns:
        log-likelihood of choice model, shape (N,)
    """
    N = logits.shape[1] if first_k is None else first_k + 1
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
