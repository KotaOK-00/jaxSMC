from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import LogDensity, SMCStatebis, ProposalBuilder

__all__ = [
    "build_gaussian_rw_proposal",
    "build_gaussian_rwmh_cov_proposal",
    "build_gaussian_rwmh_cov_proposal_gamma",
    "build_build_gaussian_rw_proposal",
    "build_gaussian_rw_diag_proposal", # new
    "build_gaussian_rwmh_diag_cov_proposal_gamma", # new
    "build_gaussian_rwmh_regular_cov_proposal_gamma", # new
    "build_gaussian_rwmh_regular_prec_proposal_gamma", # new
]

__experimental__ = []


def build_gaussian_rw_proposal(C: ArrayLike):
    """
    Gaussian RW with fixed covariance matrix C.
    C is factorised once here, so that neither the sampler nor the log density
    refactorises it at every MH step.
    """
    C = jnp.atleast_2d(C)
    chol = jnp.linalg.cholesky(C)
    dim = chol.shape[-1]
    log_norm_const = -0.5 * dim * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diagonal(chol)))

    def gaussian_rwmh_cov_log_proposal(x, y):
        z = jax.scipy.linalg.solve_triangular(chol, y - x, lower=True)
        return log_norm_const - 0.5 * jnp.sum(jnp.square(z))

    # q(x, y) = q(y, x): the proposal terms cancel in the MH ratio.
    gaussian_rwmh_cov_log_proposal.is_symmetric = True

    def gaussian_rwmh_sampler(key, x):
        return x + chol @ jax.random.normal(key, (dim,))

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)


def build_gaussian_rwmh_cov_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, log_likelihood_fn: LogDensity,
                                     i: int, j: Optional[int] = None):
    """
    Adaptative RWMH kernel with scaling set to the optimal asymptotic scaling, i.e. 2.38^2/dim.
    See Optimal scaling for various Metropolis-Hastings algorithms, Gareth O. Roberts and Jeffrey S. Rosenthal
    """
    state = state._replace(mh_proposal_parameters=state.mh_proposal_parameters.at[i - 1].set(2.38)
                          )
    return build_gaussian_rwmh_cov_proposal_gamma(state, log_tgt_density_fn, log_likelihood_fn, i, j)


def build_gaussian_rwmh_cov_proposal_gamma(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                           j: Optional[int] = None):
    """
    Same as build_gaussian_rwmh_cov_proposal with gamma**2/dim in front of the covariance matrix
    """
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    particles = state.particles
    dim = particles.shape[-1]
    log_weights = state.log_weights
    optimal_scale = gamma ** 2 / dim

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

    C = optimal_scale * fun_to_be_called_if_j_greater_than_one()

    gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, _ = build_gaussian_rw_proposal(C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)


def build_build_gaussian_rw_proposal(C: ArrayLike) -> ProposalBuilder:
    """
    Fixed covariance matrix (up to the scaling parameter)
    """

    def build_gaussian_rw_proposal_gamma(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                         j: Optional[int] = None):
        gamma = state.mh_proposal_parameters.at[i - 1].get()
        particles = state.particles
        dim = particles.shape[-1]
        optimal_scale = gamma ** 2 / dim
        _C = optimal_scale * C

        return build_gaussian_rw_proposal(_C)

    return build_gaussian_rw_proposal_gamma


def build_gaussian_rw_diag_proposal(variances: ArrayLike):
    """
    Gaussian RW with fixed *diagonal* covariance diag(variances).
    O(dim) per evaluation: no Cholesky factorisation, no triangular solve.
    """
    variances = jnp.atleast_1d(variances)
    scale = jnp.sqrt(variances)
    dim = scale.shape[-1]
    log_norm_const = -0.5 * dim * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(scale))

    def gaussian_rw_diag_log_proposal(x, y):
        z = (y - x) / scale
        return log_norm_const - 0.5 * jnp.sum(jnp.square(z))

    # q(x, y) = q(y, x): the proposal terms cancel in the MH ratio.
    gaussian_rw_diag_log_proposal.is_symmetric = True

    def gaussian_rw_diag_sampler(key, x):
        return x + scale * jax.random.normal(key, (dim,))

    return gaussian_rw_diag_log_proposal, gaussian_rw_diag_sampler, jnp.empty(1)


def build_gaussian_rwmh_diag_cov_proposal_gamma(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                                j: Optional[int] = None):
    r"""
    Same as build_gaussian_rwmh_cov_proposal_gamma but keeping only the diagonal of the
    covariance estimate, i.e. C = gamma**2/dim * diag(diag(\hat\Sigma))
    """
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    particles = state.particles
    dim = particles.shape[-1]
    log_weights = state.log_weights
    optimal_scale = gamma ** 2 / dim

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

    variances = optimal_scale * jnp.diagonal(fun_to_be_called_if_j_greater_than_one())

    gaussian_rw_diag_log_proposal, gaussian_rw_diag_sampler, _ = build_gaussian_rw_diag_proposal(variances)

    return gaussian_rw_diag_log_proposal, gaussian_rw_diag_sampler, jnp.empty(1)


def build_gaussian_rwmh_regular_cov_proposal_gamma(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                                   j: Optional[int] = None):
    r"""
    Regularised adaptive RWMH: C = gamma**2/dim * I + \hat\Sigma, with
    gamma = mh_proposal_parameters[i - 1]. The gamma**2/dim * I term keeps C
    positive definite even when \hat\Sigma is singular (weight degeneracy,
    duplicated particles after resampling, N < dim, ...). Note that only the
    identity component is tuned: C >= \hat\Sigma in the Loewner order, so the
    proposal never shrinks below the spread of \pi_{t-1}.
    """
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    particles = state.particles
    dim = particles.shape[-1]
    log_weights = state.log_weights
    optimal_scale = gamma ** 2 / dim

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

    C = optimal_scale * jnp.eye(dim) + 2.38 ** 2 / dim * fun_to_be_called_if_j_greater_than_one()

    gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, _ = build_gaussian_rw_proposal(C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)


def build_gaussian_rwmh_regular_prec_proposal_gamma(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                                    j: Optional[int] = None):
    r"""
    Precision-regularised adaptive RWMH: C = (\hat\Sigma^{-1} + gamma I)^{-1},
    with gamma = mh_proposal_parameters[i - 1]. 
    N.B. (\hat\Sigma^{-1} + gamma I)^{-1} = (I + \gamma \hat{\Sigma}) \hat{\Sigma}^{-1})^{-1}
                                          = (\hat{\Sigma}^{-1})^{-1} (I + \gamma \hat{\Sigma})^{-1}
                                          = \hat{\Sigma} (I + \gamma \hat{\Sigma})^{-1}
                                          = (I + \gamma \hat{\Sigma})^{-1} \hat{\Sigma}
    """
    gamma = state.mh_proposal_parameters.at[i - 1].get()
    particles = state.particles
    dim = particles.shape[-1]
    log_weights = state.log_weights

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
    C = jnp.linalg.solve(jnp.eye(dim) + gamma * cov_hat, cov_hat)
    C = 0.5 * (C + C.T)

    gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, _ = build_gaussian_rw_proposal(C)

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)
