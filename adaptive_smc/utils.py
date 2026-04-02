from typing import Tuple, Optional, Callable, Union

import jax
import jax.random
from jax import numpy as jnp, Array
from jax.typing import ArrayLike


def apply_vmap_batch(fun: Callable[[ArrayLike], ArrayLike], arg: ArrayLike, batch: Union[int, jnp.inf],
                     output_shape=()):
    """
    Given a vmappable function, apply it to arg in batch of size batch.
    """
    output_shape = output_shape or fun(arg[:1]).shape[1:]
    if arg.ndim == 1:
        shape = (arg.shape[0], *output_shape)
    else:
        shape = (*arg.shape[:-1], *output_shape)
    res = jnp.empty(shape)

    if batch > res.shape[0]:
        return fun(arg)

    def iter(i, res):
        my_arg = jax.lax.dynamic_slice(arg, (i * batch, *(0,) * (arg.ndim - 1)), (batch, *arg.shape[1:]))
        _res = fun(my_arg)
        res = jax.lax.dynamic_update_slice(res, _res, (i * batch, *(0,) * (res.ndim - 1)))
        return res

    res = jax.lax.cond(res.shape[0] // batch > 0,
                       lambda _: jax.lax.fori_loop(0, res.shape[0] // batch, iter, res),
                       lambda _: res, None)
    res = res.at[res.shape[0] // batch * batch:].set(fun(arg.at[res.shape[0] // batch * batch:].get()))
    return res


def normalize_log_weights(log_weights: ArrayLike, N_particles: Optional[int]=None) -> Tuple[ArrayLike, float]:
    r"""
    Normalize the log weights \exp \log w_i and return the log normalization constant:
        \log N^{-1}\sum_{i=1}^N \exp(\log w_i)
    """
    if N_particles is None:
        N_particles = log_weights.shape[0] * log_weights.shape[1]
    log_normalization = jax.scipy.special.logsumexp(log_weights)
    return log_weights - log_normalization, log_normalization - jnp.log(N_particles)


def log_ess(delta: float, log_weights: Array, N_particles: Optional[float] = None) -> float:
    """
    See Algorithm 17.3,
    Introduction to Sequential Monte Carlo, Chopin, Papaspiliopoulos
    """
    if N_particles is None:
        N_particles = jnp.prod(jnp.array(log_weights.shape))
    log_ess = 2 * jax.scipy.special.logsumexp(delta * log_weights) - jax.scipy.special.logsumexp(
        2 * delta * log_weights)
    log_ess_scaled = log_ess - jnp.log(N_particles)
    return log_ess_scaled


def vec(X: ArrayLike) -> Array:
    """
    Vectorization of a matrix.
    """
    return X.reshape(-1, order='F')


def unvec(vecX: ArrayLike, shape=Optional[Tuple[int, int]]) -> Array:
    """
    Invert the previous operation
    """
    if shape is None:
        shape = (int(vecX.shape[0] ** 0.5), int(vecX.shape[0] ** 0.5))
    return vecX.reshape(shape, order='F')


def dichotomy(fun, min_delta, max_delta, eps=1e-4, max_iter=100):
    """
    Taken from blackjax/smc/solver.py,
    Modified to remove the jnp.nan

    Credits to BlackJAX authors
    """
    """Solves for delta by dichotomy.

    If max_delta is such that fun(max_delta) > 0, then we assume that max_delta
    can be used as an increment in the tempering.

    Parameters
    ----------
    fun: Callable
        The decreasing function to solve, we must have fun(min_delta) > 0, fun(max_delta) < 0
    min_delta: float
        Starting point of the interval search
    max_delta: float
        End point of the interval search
    eps: float
        Tolerance for :math:`|f(a) - f(b)|`
    max_iter: int
        Maximum of iterations in the dichotomy search

    Returns
    -------
    delta: Array, shape (,)
        The root of `fun`

    """

    def body(carry):
        i, a, b, f_a, f_b = carry

        mid = 0.5 * (a + b)
        f_mid = fun(mid)
        a, b, f_a, f_b = jax.lax.cond(
            f_mid < 0,
            lambda _: (a, mid, f_a, f_mid),
            lambda _: (mid, b, f_mid, f_b),
            None,
        )
        return i + 1, a, b, f_a, f_b

    def cond(carry):
        i, a, b, f_a, f_b = carry
        return jnp.logical_and(i < max_iter, f_a - f_b > eps)

    f_min_delta, f_max_delta = fun(min_delta), fun(max_delta)

    if_no_opt = lambda _: max_delta

    def if_opt(_):
        _, res_a, res_b, fun_res_a, fun_res_b = jax.lax.while_loop(
            cond, body, (0, min_delta, max_delta, f_min_delta, f_max_delta)
        )
        return res_a

    # if the upper end of the interval returns positive already, just return it,
    # otherwise search the optimum as long as the start of the interval is positive.
    return jax.lax.cond(
        f_max_delta > 0,
        if_no_opt,
        if_opt,
        None,
    )
