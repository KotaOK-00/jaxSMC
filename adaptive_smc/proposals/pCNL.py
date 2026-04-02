import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc_types import LogDensity
from adaptive_smc.smc_types import SMCStatebis

__all__ = ["build_build_pCNL_proposal",
           "build_build_ARLW",
           ]


def build_build_pCNL_proposal(mu: ArrayLike, C: ArrayLike):
    r"""
    The target distribution is \pi(x) \propto \mathcal{N}(0, C)e^{-f}, with f convex and \grad f (0) = 0
    Construct the build function for pCNL proposal
    q(y\mid x) = N(mu + \rho (x-mu) - (1-\rho) C @ \grad f(x), (1-\rho^2)C),
    where C is a given matrix, and mu given vector
    """

    def _build(state: SMCStatebis, _: LogDensity, f: LogDensity, i: int, j=None):
        rho = state.mh_proposal_parameters.at[i - 1].get()

        def log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, mu + rho * (x - mu) - (1 - rho) * C @ jax.jacfwd(f)(x),
                                                              (1 - rho ** 2) * C)

        def sampler(key, x):
            return jax.random.multivariate_normal(key, mu + rho * (x - mu) - (1 - rho) * C @ jax.jacfwd(f)(x), (1 - rho ** 2) * C)

        return log_proposal, sampler, jnp.empty(1)

    return _build


def build_build_ARLW(mu: ArrayLike, C: ArrayLike):
    r"""
    Construct the build function for the uncoupled version of pCNL
    q(y\mid x) = N(mu + \rho (x-mu) - (1-\rho) C @ \grad f(x), \tau^2C),
    where C is a given matrix, and mu given vector.
    """

    def _build(state: SMCStatebis, _: LogDensity, f: LogDensity, i: int, j=None):
        rho = state.mh_proposal_parameters.at[i - 1, 0].get()
        tau = state.mh_proposal_parameters.at[i - 1, 1].get()

        def log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y,  mu + rho * (x - mu) - (1 - rho) * C @ jax.jacfwd(f)(x), (tau ** 2) * C)

        def sampler(key, x):
            return jax.random.multivariate_normal(key, mu + rho * (x - mu), (tau ** 2) * C)

        return log_proposal, sampler, jnp.empty(1)

    return _build

