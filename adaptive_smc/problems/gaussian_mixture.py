import jax.numpy as jnp
import jax.random

from adaptive_smc.problems.gaussian import create_sparse_problem


def construct_my_prior_and_target(config):
    r"""
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution 0.1N(0, C) + 0.9N(0, C'),
    where C is scaled identity on the latent space otherwise scaled down to tau**2,
    same for C' with a different scaling factor. Orthogonal directions.
    if latent_dim is set to 0, the target is N(1, tau**2 * I) \beta + (1-\beta) N(0, tau'**2 I)
    """

    problem = config.get('problem')
    latent_dim = problem.get('latent_dim', 0)
    tau = problem.get('tau')
    tau2 = problem.get('tau2')
    dim = config.get('dim')
    weight = problem.get('weight', 0.1)

    logpdf1 = create_sparse_problem(dim, latent_dim=latent_dim, mean=-jnp.ones(dim),
                                    scale=1 / (1 / tau ** 2))
    logpdf2 = create_sparse_problem(dim, latent_dim=latent_dim, mean=jnp.ones(dim),
                                    scale=1 / (1 / tau2 ** 2))

    def loglikelihood_fn(x):
        log_tgt_distrib = jnp.log(weight * jnp.exp(logpdf1(x)) + (1 - weight) * jnp.exp(logpdf2(x)))
        log_prior = jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim),
                                                               cov=jnp.eye(dim))
        ll = log_tgt_distrib - log_prior
        return ll

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn
