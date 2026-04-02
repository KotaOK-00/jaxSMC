import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc_types import LogDensity, SMCStatebis

__all__ = [
    "build_build_pmala_proposal",
    "build_pmala_proposal",
]


def build_build_pmala_proposal(C: ArrayLike):
    r"""
    Build the builder (...).
    Preconditioned MALA proposal,
    Auxiliary gradient-based sampling algorithms,
    q(y\mid x) = N(y, (1-\delta/2)x+\delta/2 C \grad f, \delta C),
    where C is a given matrix
    """

    def _build(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int, j=None):
        delta = state.mh_proposal_parameters.at[i - 1].get()

        def log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, (1 - delta / 2) * x + delta / 2 * C @ jax.jacfwd(
                log_tgt_density_fn)(x), delta * C)

        def sampler(key, x):
            return jax.random.multivariate_normal(key, (1 - delta / 2) * x + delta / 2 * C @ jax.jacfwd(
                log_tgt_density_fn)(x), delta * C)

        return log_proposal, sampler, jnp.empty(1)

    return _build


def build_pmala_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, log_likelihood_fn: LogDensity, i: int,
                         j=None):
    r"""
    Preconditioned MALA proposal,
    Auxiliary gradient-based sampling algorithms,
    q(y\mid x) = N(y, (1-\delta/2)x+\delta/2 C \grad f, \delta C),
    where C is a fixed matrix, here I_n.
    """
    dim = state.particles.shape[-1]
    C = jnp.eye(dim)
    return build_build_pmala_proposal(C)(state, log_tgt_density_fn, log_likelihood_fn, i, j)
