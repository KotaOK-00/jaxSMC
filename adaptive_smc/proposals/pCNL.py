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
    pCNL proposal for the tempered target \pi_{i-1} = \nu e^{\lambda_{i-1} \ell} with
    reference \nu = N(\mu, C). The Crank-Nicolson part handles \nu exactly through \rho,
    while the Langevin drift corrects the tempered log-likelihood \lambda_{i-1} \ell:
        q(y\mid x) = N(\mu + \rho (x-\mu) + (1-\rho) \lambda_{i-1} C \grad \ell(x), (1-\rho^2)C).
    \lambda_{i-1} is read from state.tempering_sequence, so the temperature rides with the
    kernel and the log-likelihood \ell is taken from the third argument.
    """

    def _build(state: SMCStatebis, _: LogDensity, log_likelihood_fn: LogDensity, i: int, j=None):
        rho = state.mh_proposal_parameters.at[i - 1].get()
        lmbda = state.tempering_sequence.at[i - 1].get()
        grad_log_likelihood_fn = jax.jacfwd(log_likelihood_fn)

        def mean_fn(x):
            return mu + rho * (x - mu) + (1 - rho) * lmbda * C @ grad_log_likelihood_fn(x)

        def log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, mean_fn(x), (1 - rho ** 2) * C)

        def sampler(key, x):
            return jax.random.multivariate_normal(key, mean_fn(x), (1 - rho ** 2) * C)

        return log_proposal, sampler, jnp.empty(1)

    return _build


def build_build_ARLW(mu: ArrayLike, C: ArrayLike):
    r"""
    Uncoupled version of pCNL (a mix between AR/pCN and Langevin), with parameters
    (\rho, \tau) read from state.mh_proposal_parameters[i - 1]:
        q(y\mid x) = N(\mu + \rho (x-\mu) + (1-\rho) \lambda_{i-1} C \grad \ell(x), \tau^2 C).
    See build_build_pCNL_proposal for the tempering convention.
    """

    def _build(state: SMCStatebis, _: LogDensity, log_likelihood_fn: LogDensity, i: int, j=None):
        rho = state.mh_proposal_parameters.at[i - 1, 0].get()
        tau = state.mh_proposal_parameters.at[i - 1, 1].get()
        lmbda = state.tempering_sequence.at[i - 1].get()
        grad_log_likelihood_fn = jax.jacfwd(log_likelihood_fn)

        def mean_fn(x):
            return mu + rho * (x - mu) + (1 - rho) * lmbda * C @ grad_log_likelihood_fn(x)

        def log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, mean_fn(x), (tau ** 2) * C)

        def sampler(key, x):
            return jax.random.multivariate_normal(key, mean_fn(x), (tau ** 2) * C)

        return log_proposal, sampler, jnp.empty(1)

    return _build

