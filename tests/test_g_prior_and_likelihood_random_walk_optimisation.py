import jax
import jax.numpy as jnp
import jax.random

from adaptive_smc.optimise import make_optimize_within_a_fixed_grid
from adaptive_smc.problems.gaussian import create_problem
from adaptive_smc.proposals import build_gaussian_rwmh_cov_proposal_gamma
from adaptive_smc.SMC import GenericAdaptiveWasteFreeTemperingSMC

jax.config.update("jax_enable_x64", False)


def test():
    r"""
    Checking the optimisation procedure for the ESJD in the RWM case.
    Should be near 2.38 up to 2%
    """
    OP_key = jax.random.PRNGKey(0)
    dim = 2

    mean_likelihood = jnp.ones(dim) * 0
    cov_likelihood = jnp.eye(dim) * 0.5 ** 2
    loglikelihood_fn = create_problem(dim, mean=mean_likelihood, cov=cov_likelihood)
    mean_prior = jnp.ones(dim) * 2.0
    cov_prior = 2 * jnp.eye(dim)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mean_prior, cov_prior)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=mean_prior, cov=cov_prior)

    length_of_the_tempering_sequence = 5
    my_tempering_sequence = jnp.linspace(0, 1, length_of_the_tempering_sequence)

    num_parallel_chain = 10
    num_mcmc_steps = 2000
    init_param = jnp.array([2.38])
    n_chains = 1

    optimization_method = make_optimize_within_a_fixed_grid(jnp.linspace(0.01, 5, 200))

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               build_gaussian_rwmh_cov_proposal_gamma, optimization_method)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess=0.5)

    keys = jax.random.split(OP_key, n_chains)
    with jax.disable_jit(False):
        with jax.default_device(jax.devices("cpu")[0]):
            res = wrapper_smc(keys)

    temperatures = res[6]
    temperatures = jnp.insert(temperatures, 0, 0., -1)
    assert jnp.all(temperatures[:, -1] == 1.0)  # assert all temperatures at the end are 1.0

    rtol = 5e-2
    r"""
    When the current distribution is the target (T = -1 or t>=inf T : \lambda_{T} = 1.), 
    We are sure the optimal parameter is 2.38.
    """
    max_min_idx_temp_equal_1 = jnp.argwhere(temperatures==1.0)[:,1]
    max_min_idx_temp_equal_1 = max_min_idx_temp_equal_1.reshape((n_chains, max_min_idx_temp_equal_1.shape[0]//n_chains))
    max_min_idx_temp_equal_1 = jnp.min(max_min_idx_temp_equal_1[:,0])
    assert jnp.all(jnp.allclose(res[3][:, max_min_idx_temp_equal_1+1:], jnp.array([2.38]), rtol=rtol))
