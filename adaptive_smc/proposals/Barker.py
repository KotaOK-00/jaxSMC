from typing import Optional, Tuple
from adaptive_smc.smc_types import LogDensity, LogDensity, LogProposal, ProposalSampler

import jax
import jax.numpy as jnp

from jax.typing import ArrayLike

from adaptive_smc.estimates import cov_estimate
from adaptive_smc.smc_types import LogDensity, SMCStatebis

__all__ = [
    "build_barker_proposal_gamma_cov",
    "build_build_barker_proposal_gamma",
    "build_build_bimodal_barker_proposal_gamma",
    "build_build_bimodal_barker_proposal_gamma_cov",
    "build_build_bimodal_barker_proposal_gamma_m"
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


def bimodal_barker_proposal(Sigma, log_tgt_density_fn: LogDensity, m: ArrayLike = 0.1,
                            mode: Optional[ArrayLike] = None) -> Tuple[LogProposal, ProposalSampler, ArrayLike]:
    r"""
    Preconditioned Barker proposal driven by a *bimodal* Gaussian base kernel
    instead of the standard normal one used in :func:`barker_proposal`.

    The first-order (locally balanced) construction only requires the base
    kernel \mu to be symmetric about the origin: nothing in Algorithms 4-5 of
    Livingstone and Zanella (2022) uses Gaussianity, and the normalisation
        \int 2 \mu(z) sigmoid(z c) dz = \int \mu(z) [sigmoid(z c) + sigmoid(-z c)] dz = 1
    holds for any symmetric \mu, so the sign trick carries over verbatim. Here
    the whitened noise is the two-component mixture centred at \pm\sqrt{1 - m^2},
    each component having standard deviation m,
        \mu_m(z) = 1/2 N(z; -\sqrt{1 - m^2}, m^2) + 1/2 N(z; +\sqrt{1 - m^2}, m^2)
                 = (m\sqrt{2\pi})^{-1} \exp(-(z^2 + 1 - m^2) / (2 m^2)) \cosh(z \sqrt{1 - m^2} / m^2),
    applied coordinate-wise in the whitened coordinates of L L^T = \Sigma:
        \xi \sim N(0, I), \varepsilon_i \sim Unif\{-1, +1\},
        z_i = \varepsilon_i \sqrt{1 - m^2} + m \xi_i,
        c(x) = L^T grad \log \pi (x),
        b_i = +1 with probability sigmoid(z_i c_i(x)), else -1,
        y = x + L (b \odot z),
    with transition density
        q(x, y) = |det L|^{-1} \prod_i 2 \mu_m(z_i) sigmoid(z_i c_i(x)), z = L^{-1}(y - x).

    Splitting the unit variance as (1 - m^2) + m^2, exactly as pCN splits it
    between \rho and \sqrt{1 - \rho^2}, keeps Var(z_i) = 1 for every
    m \in (0, 1]: \Sigma is scaled as in the Gaussian case and gamma keeps its
    meaning. m is the width of each component, so m = 1 puts both modes at the
    origin and reproduces :func:`barker_proposal` coefficient for coefficient,
    while a small m sharpens the two modes towards the two-point kernel
    z_i = \pm 1 -- the default m = 0.1 is firmly bimodal, modes at \pm 0.995 of
    width 0.1. Moving the noise mass away from zero is the point of the bimodal
    base kernel: the Gaussian one wastes a sizeable share of its proposals on
    near-zero moves. m must stay positive (m = 0 divides by zero); the (gamma, m)
    builder clips it away from zero, on the statistical grounds that the kernel
    degenerates -- float32 acceptance ratios were measured against float64 on
    identical (x, y) pairs and stay accurate to 1e-5 down to m = 0.02 despite
    the 1 / m^2 factors, which cancel between q(x, y) and q(y, x). Pass ``mode``
    explicitly to place the modes somewhere other than \pm\sqrt{1 - m^2}, the
    noise variance then being mode^2 + m^2.
    """
    Sigma = jnp.atleast_2d(Sigma)
    chol = jnp.linalg.cholesky(Sigma)
    dim = chol.shape[-1]
    s = jnp.reshape(jnp.asarray(m, dtype=chol.dtype), ())
    mode = jnp.sqrt(jnp.maximum(1.0 - s ** 2, 0.0)) if mode is None else jnp.reshape(
        jnp.asarray(mode, dtype=chol.dtype), ())
    inv_var = 1.0 / s ** 2
    log_norm_const = (-0.5 * dim * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diagonal(chol)))
                      + dim * jnp.log(2.0) - dim * jnp.log(s) - 0.5 * dim * mode ** 2 * inv_var)
    grad_log_tgt_fn = jax.grad(log_tgt_density_fn)

    def whitened_grad(x):
        return chol.T @ grad_log_tgt_fn(x)

    def log_cosh(t):
        r"""\log \cosh, written so that the large |t| regime does not overflow"""
        abs_t = jnp.abs(t)
        return abs_t + jnp.log1p(jnp.exp(-2.0 * abs_t)) - jnp.log(2.0)

    def bimodal_barker_log_proposal(x, y):
        z = jax.scipy.linalg.solve_triangular(chol, y - x, lower=True)
        return (log_norm_const - 0.5 * inv_var * jnp.sum(jnp.square(z))
                + jnp.sum(log_cosh(z * mode * inv_var))
                + jnp.sum(jax.nn.log_sigmoid(z * whitened_grad(x))))

    def bimodal_barker_sampler(key, x):
        key_xi, key_eps, key_b = jax.random.split(key, 3)
        eps = jnp.where(jax.random.uniform(key_eps, (dim,)) < 0.5, -1.0, 1.0)
        z = mode * eps + s * jax.random.normal(key_xi, (dim,))
        u = jax.random.uniform(key_b, (dim,))
        b = jnp.where(u < jax.nn.sigmoid(z * whitened_grad(x)), 1.0, -1.0)
        return x + chol @ (b * z)

    return bimodal_barker_log_proposal, bimodal_barker_sampler, jnp.empty(1)


def build_build_bimodal_barker_proposal_gamma(C, m: ArrayLike = 0.1):
    """
    Fixed covariance matrix (up to the scaling parameter) and fixed component
    width m (default 0.1, i.e. modes at +-0.995), only gamma being adapted.
    Pass C = I for the plain bimodal Barker kernel.
    """

    def build_bimodal_barker_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                                      j: Optional[int] = None):
        gamma = state.mh_proposal_parameters.at[i - 1].get()
        dim = state.particles.shape[-1]
        optimal_scale = gamma ** 2 / dim ** (1 / 3)
        return bimodal_barker_proposal(optimal_scale * C, log_tgt_density_fn, m)

    return build_bimodal_barker_proposal


def build_build_bimodal_barker_proposal_gamma_cov(m: ArrayLike = 0.1):
    """
    Bimodal Barker proposal with a gamma parameter, an adaptive covariance
    matrix and a fixed component width m (default 0.1, i.e. modes at +-0.995).
    """

    def build_bimodal_barker_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                                      j: Optional[int] = None):
        particles = state.particles
        log_weights = state.log_weights
        gamma = state.mh_proposal_parameters.at[i - 1].get()
        dim = particles.shape[-1]

        j_ = j or i

        def fun_to_be_called_if_j_greater_than_one():
            r"""
            Compute the covariance estimate of \pi_{t-1} given t\geq 1
            """
            particles_at_j_minus_one = particles.at[j_ - 1].get().reshape(-1, particles.shape[-1])
            log_weights_at_j_minus_one = log_weights.at[j_ - 1].get().reshape(-1, )
            weights_at_j_minus_one = jnp.exp(log_weights_at_j_minus_one)
            cov_hat, _ = cov_estimate(particles_at_j_minus_one, weights_at_j_minus_one)
            return cov_hat

        cov_hat = fun_to_be_called_if_j_greater_than_one()
        scaled_cov_hat = cov_hat * gamma ** 2 / dim ** (1 / 3)
        return bimodal_barker_proposal(scaled_cov_hat, log_tgt_density_fn, m)

    return build_bimodal_barker_proposal


def build_build_bimodal_barker_proposal_gamma_m(C, m_min: float = 0.05):
    r"""
    Fixed covariance matrix (up to the scaling parameter), with both the scale
    and the shape of the base kernel tuned: \theta = (\gamma, m) is read from
    state.mh_proposal_parameters[i - 1] (shape (..., 2)), exactly as for the
    (\gamma, L) version of HMC or the (\rho, \tau) proposals. m is clipped to
    [m_min, 1] because the modes +-\sqrt{1 - m^2} degenerate to a two-point
    kernel of vanishing width as m -> 0 (m_min = 0.05 leaves 1 / m^2 = 400).
    """

    def build_bimodal_barker_proposal(state: SMCStatebis, log_tgt_density_fn: LogDensity, _: LogDensity, i: int,
                                      j: Optional[int] = None):
        gamma = jnp.reshape(state.mh_proposal_parameters.at[i - 1, 0].get(), ())
        m = jnp.clip(jnp.reshape(state.mh_proposal_parameters.at[i - 1, 1].get(), ()), m_min, 1.0)
        dim = state.particles.shape[-1]
        optimal_scale = gamma ** 2 / dim ** (1 / 3)
        return bimodal_barker_proposal(optimal_scale * C, log_tgt_density_fn, m)

    return build_bimodal_barker_proposal
