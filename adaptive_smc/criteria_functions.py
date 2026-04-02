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
        _x = x.reshape(x.shape[0] * x.shape[1], 1)  # ~ \pi_{i-1}
        _w = jnp.exp(state.log_weights.at[j].get().squeeze())
        cov, _ = cov_estimate(_x, _w)
        return jnp.einsum('...j,...k,...jk->...', x - y, x - y, jnp.linalg.inv(cov))

    return jax.lax.select(i == 0, jnp.sum(jnp.square(x - y), axis=-1), _mahalanobis(x, y))
