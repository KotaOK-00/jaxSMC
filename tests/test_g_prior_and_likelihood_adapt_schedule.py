import jax
import jax.numpy as jnp
import jax.random

from adaptive_smc.problems.gaussian import create_problem
from adaptive_smc.proposals import build_gaussian_rwmh_cov_proposal_gamma
from adaptive_smc.SMC import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.optimise import make_optimize_within_a_fixed_grid, make_constant

jax.config.update("jax_enable_x64", False)

def test():
    """
    Checking that the samples from the posterior distribution given by a
        - Gaussian prior N(m_P, C_P)
        - Likelihood N(m_L, C_L),
    obtained via SMC, have empirical mean m and covariance C approximately satisfying
        C^{-1} = C_P^{-1} + C_L^{-1},
        C^{-1}m = C_P^{-1}m_P + C_L^{-1}m_L.
    To check the first equality, we multiply C_P^{-1} + C_L^{-1} by C, and compute the distance to the identity,
        absolute tolerance of 5 % * sqrt(dim).
    Component-wise relative tolerance up to 5% for the second inequality.
    """
    OP_key = jax.random.PRNGKey(0)
    dim = 2

    mean_likelihood = jnp.ones(dim) * 20
    cov_likelihood = jnp.eye(dim) * 0.5 ** 2
    loglikelihood_fn = create_problem(dim, mean=mean_likelihood, cov=cov_likelihood)
    mean_prior = jnp.ones(dim) * 5
    cov_prior = 5 * jnp.eye(dim)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mean_prior, cov_prior)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mean_prior, cov=cov_prior)

    length_of_the_tempering_sequence = 25
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    num_parallel_chain = 10
    num_mcmc_steps = 2000
    init_param = jnp.array([2.38])
    n_chains = 2

    optimisation_method = make_constant()
    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal_gamma, optimisation_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            res = wrapper_smc(keys)

    res = res[0][:,:,0] #keeping only accepted states
    n_particles = res.shape[2] * res.shape[3]
    cov = jax.vmap(lambda X: jnp.cov(X, rowvar=False))(
        res[:, -1].reshape((*res[:, -1].shape[:1], n_particles, res.shape[-1])))
    mean = res[:, -1].mean(axis=[1, 2])

    target_1 = jnp.linalg.inv(cov_prior) @ mean_prior + jnp.linalg.inv(
        cov_likelihood) @ mean_likelihood
    target_2 = jnp.linalg.inv(cov_prior) + jnp.linalg.inv(cov_likelihood)
    rtol = 10 * 1e-2
    assert jnp.all(jax.vmap(lambda X: jnp.allclose(X, target_1, rtol=rtol))(
        jax.vmap(lambda X, Y: X @ Y)(jnp.linalg.inv(cov), mean)))
    atol_accounted_for_the_dimension = 10 * 1e-2
    assert jnp.all(jax.vmap(
        lambda C: jnp.linalg.norm(target_2 @ C - jnp.eye(dim)) <= jnp.sqrt(dim) * atol_accounted_for_the_dimension)(
        cov))
