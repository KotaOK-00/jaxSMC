from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import SMCStatebis


def square_distance(x: ArrayLike, y: ArrayLike, _: SMCStatebis, __: int, ___=None) -> ArrayLike:
    """
    Expected square jumping distance criterion: square distance between x and y.
    """
    return jnp.sum(jnp.square(x - y), axis=-1)


def mahalanobis(x: ArrayLike, y: ArrayLike, state: SMCStatebis, i: int, j: Optional[int] = None) -> ArrayLike:
    r"""
    At iteration i,
        for particles x_i ~ \pi_{i-1},
        compute the Mahalanobis distances between x and y.
        The scaling matrix is the estimated covariance under \pi_i (using weights w_{i}, and particles x_i).

            Example of application:
            maximising mahalanobis ESJD at iteration i, to compute \theta_{i+1} parameterizing q_{i+1}, leaving
            \pi_{i} invariant.

    """

    j = j or i

    def _mahalanobis(x, y):
        r"""
        Compute the MH distance between x and y, using the covariance matrix estimated from x ~ \pi_{i-1}, and weights \pi_{i}/\pi_i
        """
        dim = x.shape[-1]
        _x = x.reshape(-1, dim)  # ~ \pi_{i-1}
        _w = jnp.exp(state.log_weights.at[j].get().reshape(-1))
        cov, _ = cov_estimate(_x, _w)
        chol = jnp.linalg.cholesky(cov)
        diff = (x - y).reshape(-1, dim)
        z = jax.scipy.linalg.solve_triangular(chol, diff.T, lower=True)
        return jnp.sum(jnp.square(z), axis=0).reshape(x.shape[:-1])

    return jax.lax.select(i == 0, jnp.sum(jnp.square(x - y), axis=-1), _mahalanobis(x, y))
