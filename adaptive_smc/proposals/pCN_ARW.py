import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import LogDensity
from adaptive_smc.smc_types import SMCStatebis

__all__ = ["build_build_autoregressive_gaussian_proposal",
           "build_autoregressive_gaussian_proposal",
           "build_build_uncoupled_autoregressive_gaussian_proposal",
           ]

__experimental__ = ["build_autoregressive_gaussian_proposal_with_cov_estimate"]


def build_build_autoregressive_gaussian_proposal(mu: ArrayLike, C: ArrayLike):
    r"""
    Construct the build function for autoregressive proposal:
    q(y\mid x) = N(mu + \rho (x-mu), (1-\rho^2)C),
    where C is a given matrix, and mu given vector.
    """

    def _build(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int, j=None):
        rho = state.mh_proposal_parameters.at[i - 1].get()

        def gaussian_ar_log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, mu + rho * (x - mu), (1 - rho ** 2) * C)

        def gaussian_ar_sampler(key, x):
            return jax.random.multivariate_normal(key, mu + rho * (x - mu), (1 - rho ** 2) * C)

        return gaussian_ar_log_proposal, gaussian_ar_sampler, jnp.empty(1)

    return _build


def build_build_uncoupled_autoregressive_gaussian_proposal(mu: ArrayLike, C: ArrayLike):
    r"""
    Construct the build function for uncoupled AR proposal (mix between random-walk and AR):
    q(y\mid x) = N(mu + \rho (x-mu), \tau^2C),
    where C is a given matrix, and mu given vector.
    """

    def _build(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int, j=None):
        rho = state.mh_proposal_parameters.at[i - 1, 0].get()
        tau = state.mh_proposal_parameters.at[i - 1, 1].get()

        def gaussian_ar_log_proposal(x, y):
            return jax.scipy.stats.multivariate_normal.logpdf(y, mu + rho * (x - mu), (tau ** 2) * C)

        def gaussian_ar_sampler(key, x):
            return jax.random.multivariate_normal(key, mu + rho * (x - mu), (tau ** 2) * C)

        return gaussian_ar_log_proposal, gaussian_ar_sampler, jnp.empty(1)

    return _build


def build_autoregressive_gaussian_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity,
                                           log_likelihood_fn: LogDensity, i: int, j=None):
    r"""
    Autoregressive proposal:
    q(y\mid x) = N(\rho x, (1-\rho^2)C),
    where C is a fixed matrix, here I_n.
    """
    dim = state.particles.shape[-1]
    C = jnp.eye(dim)
    mu = jnp.zeros(dim)
    return build_build_autoregressive_gaussian_proposal(mu, C)(state, log_tgt_density_fn, log_likelihood_fn, i, j)


def build_autoregressive_gaussian_proposal_with_cov_estimate(state: SMCStatebis, log_tgt_density_fn: LogDensity,
                                                             log_likelihood_fn: LogDensity, i: int, j=None):
    particles = state.particles
    log_weights = state.log_weights

    def fun_to_be_called_if_i_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_i_minus_one = particles.at[i - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_i_minus_one = log_weights.at[i - 1].get().reshape(-1, )
        weights_at_i_minus_one = jnp.exp(log_weights_at_i_minus_one)
        cov_hat, mu_hat = cov_estimate(particles_at_i_minus_one, weights_at_i_minus_one)
        return cov_hat, mu_hat

    C, mu = fun_to_be_called_if_i_greater_than_one()
    return build_build_autoregressive_gaussian_proposal(mu, C)(state, log_tgt_density_fn, log_likelihood_fn,
                                                               i, j)
