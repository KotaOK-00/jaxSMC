from typing import Optional, Tuple
from adaptive_smc.smc_types import LogDensity, LogDensity, LogProposal, ProposalSampler

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import LogDensity, SMCStatebis

__all__ = [
    "build_MALA_proposal_gamma_cov",
    "build_build_MALA_proposal_gamma"
]


def MALA_proposal(Sigma, log_tgt_density_fn: LogDensity) -> Tuple[LogProposal, ProposalSampler, ArrayLike]:
    r"""
    MALA proposal and sampler for a certain conditioning matrix
    \Sigma

    y = x + 0.5 \Sigma @ grad \log \pi (x) + C^{1/2} zeta, zeta\sim \mathcal{N}(0, I).
    \Sigma is factorised once here, so that neither the sampler nor the log density
    refactorises it at every MH step.
    """
    Sigma = jnp.atleast_2d(Sigma)
    chol = jnp.linalg.cholesky(Sigma)
    dim = chol.shape[-1]
    log_norm_const = -0.5 * dim * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diagonal(chol)))
    grad_log_tgt_fn = jax.jacfwd(log_tgt_density_fn)

    def drift(x):
        return x + 0.5 * Sigma @ grad_log_tgt_fn(x)

    def gaussian_mala_log_proposal(x, y):
        z = jax.scipy.linalg.solve_triangular(chol, y - drift(x), lower=True)
        return log_norm_const - 0.5 * jnp.sum(jnp.square(z))

    def gaussian_mala_sampler(key, x):
        return drift(x) + chol @ jax.random.normal(key, (dim,))

    return gaussian_mala_log_proposal, gaussian_mala_sampler, jnp.empty(1)


def build_MALA_proposal_gamma_cov(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                                  j: Optional[int] = None):
    """
    Langevin proposal with a gamma parameter and adaptive covariance matrix
    """
    particles = state.particles
    log_weights = state.log_weights
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    dim = particles.shape[-1]

    j = j or i

    def fun_to_be_called_if_j_greater_than_one():
        r"""
        Compute the covariance estimate of \pi_{t-1} given t\geq 1
        """
        particles_at_j_minus_one = particles.at[j - 1].get().reshape(-1, particles.shape[-1])
        log_weights_at_j_minus_one = log_weights.at[j - 1].get().reshape(-1, )
        weights_at_j_minus_one = jnp.exp(log_weights_at_j_minus_one)
        cov_hat, _ = cov_estimate(particles_at_j_minus_one, weights_at_j_minus_one)
        return cov_hat

    cov_hat = fun_to_be_called_if_j_greater_than_one()
    scaled_cov_hat = cov_hat * gamma ** 2 / dim ** (1 / 3)
    return MALA_proposal(scaled_cov_hat, log_tgt_density_fn)


def build_build_MALA_proposal_gamma(C):
    """
    Fixed covariance matrix (up to the scaling parameter)
    """

    def build_MALA_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                            j: Optional[int] = None):
        gamma = state.mh_proposal_parameters.at[i - 1].get()
        particles = state.particles
        dim = particles.shape[-1]
        optimal_scale = gamma ** 2 / dim ** (1 / 3)
        _C = optimal_scale * C
        return MALA_proposal(_C, log_tgt_density_fn)

    return build_MALA_proposal
