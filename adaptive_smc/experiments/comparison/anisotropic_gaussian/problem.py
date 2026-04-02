from typing import Tuple

import jax.numpy as jnp
import jax

from adaptive_smc.problems.gaussian import create_sparse_problem


def construct_my_prior_and_target(config):
    r"""
    The prior is a standard Gaussian distribution.
    The target is a Gaussian distribution N(1, C),
    where C is scaled identity on the latent space otherwise scaled down to tau**2.
    """

    r"""
    Take the log-likelihood function such that the target is N((1, ..., 1) (or zero?), C),
    with C = (1, ..., 1, \tau^2, ..., \tau^2)
    """

    config_problem = config.get('problem')
    tau = config_problem.get('tau')
    dim = config.get('dim')
    latent_dim = config_problem.get('latent_dim')

    loglikelihood_fn = create_sparse_problem(dim, latent_dim=latent_dim, mean=jnp.zeros(dim),
                                             scale=1 / (1 / tau ** 2 - 1))

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), jnp.eye(dim))

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=jnp.zeros(dim), cov=jnp.eye(dim))

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn


def sample_from_wishart(key, degree, scale):
    p = scale.shape[0]

    def body_fun(_, vals: Tuple[jnp.ndarray, jax.Array]):
        val, key = vals
        g = jax.random.multivariate_normal(key, jnp.zeros(p), scale).reshape((p, 1))
        _, key = jax.random.split(key)
        return val + g @ g.T, key

    S, _ = jax.lax.fori_loop(0, degree, body_fun, (jnp.zeros((p, p)), key))

    return S
