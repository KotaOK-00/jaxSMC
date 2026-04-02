from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc_types import LogDensity
from adaptive_smc.utils import unvec, vec


def cov_estimate(particles: ArrayLike, weights: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    r"""
    Given particles and weights at iteration t, of shapes resp. (N, dim), and (N, ),
    the weighted covariance estimate of \pi_t is
    \sum_{1\leq i\leq N} W_{t}^{i} (X^i_t - \hat{\mu}_t) (X^i_t - \hat{\mu}_t)^{\top},
    where \hat{\mu}_t = \sum_{1\leq i\leq N} X^i_t w^i_t
    """
    mu_hat = jnp.sum(weights[:, jnp.newaxis] * particles, axis=0)
    to_sum = (particles - mu_hat)
    cov_hat = jnp.einsum('ij,ik,i->jk', to_sum, to_sum, weights)
    return cov_hat, mu_hat
