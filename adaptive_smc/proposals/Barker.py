from typing import Optional, Tuple
from adaptive_smc.smc_types import LogDensity, LogDensity, LogProposal, ProposalSampler

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import LogDensity, SMCStatebis

__all__ = [
    "build_barker_proposal_gamma_cov",
    "build_build_barker_proposal_gamma"
]


def barker_proposal(Sigma, log_tgt_density_fn: LogDensity) -> Tuple[LogProposal, ProposalSampler, ArrayLike]:
    r"""
    Preconditioned Barker proposal and sampler for a certain conditioning matrix
    \Sigma (Algorithms 4-5 of Livingstone and Zanella, 2022).

    The coordinate-wise Barker proposal is applied in the whitened coordinates
    associated with L L^T = \Sigma (the paper writes C = chol(\Sigma) with
    C^T C = \Sigma, i.e. C = L^T in the jnp.linalg.cholesky convention):
        z \sim \mathcal{N}(0, I), c(x) = L^T grad \log \pi (x),
        b_i = +1 with probability sigmoid(z_i c_i(x)), else -1,
        y = x + L (b \odot z).
    By change of variables the transition density is
        q(x, y) = |det L|^{-1} \prod_i 2 \varphi(z_i) sigmoid(z_i c_i(x)),
    with z = L^{-1}(y - x), whose ratio q(y, x) / q(x, y) reproduces the
    acceptance of Algorithm 5.
    \Sigma is factorised once here, so that neither the sampler nor the log density
    refactorises it at every MH step.
    """
    Sigma = jnp.atleast_2d(Sigma)
    chol = jnp.linalg.cholesky(Sigma)
    dim = chol.shape[-1]
    log_norm_const = -0.5 * dim * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diagonal(chol))) + dim * jnp.log(2.0)
    grad_log_tgt_fn = jax.grad(log_tgt_density_fn)

    def whitened_grad(x):
        return chol.T @ grad_log_tgt_fn(x)

    def barker_log_proposal(x, y):
        z = jax.scipy.linalg.solve_triangular(chol, y - x, lower=True)
        return log_norm_const - 0.5 * jnp.sum(jnp.square(z)) + jnp.sum(jax.nn.log_sigmoid(z * whitened_grad(x)))

    def barker_sampler(key, x):
        key_z, key_b = jax.random.split(key)
        z = jax.random.normal(key_z, (dim,))
        u = jax.random.uniform(key_b, (dim,))
        b = jnp.where(u < jax.nn.sigmoid(z * whitened_grad(x)), 1.0, -1.0)
        return x + chol @ (b * z)

    return barker_log_proposal, barker_sampler, jnp.empty(1)


def build_barker_proposal_gamma_cov(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                                    j: Optional[int] = None):
    """
    Barker proposal with a gamma parameter and adaptive covariance matrix
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
    return barker_proposal(scaled_cov_hat, log_tgt_density_fn)


def build_build_barker_proposal_gamma(C):
    """
    Fixed covariance matrix (up to the scaling parameter). Pass C = I for the
    plain Barker kernel where only gamma is adapted.
    """

    def build_barker_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                              j: Optional[int] = None):
        gamma = state.mh_proposal_parameters.at[i - 1].get()
        particles = state.particles
        dim = particles.shape[-1]
        optimal_scale = gamma ** 2 / dim ** (1 / 3)
        _C = optimal_scale * C
        return barker_proposal(_C, log_tgt_density_fn)

    return build_barker_proposal
