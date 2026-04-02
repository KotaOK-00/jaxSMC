from adaptive_smc.problems.logistic import get_dataset, get_log_likelihood
import jax
import jax.numpy as jnp
from adaptive_smc.laplace import laplace_approximation


def construct_my_prior_and_target(config):
    r"""
    The prior is a standard Gaussian distribution.
    The target is the posterior of the logistic regression model, with a Gaussian prior.
    """
    config_problem = config["problem"]
    base_measure_type = config_problem.get('base_measure_type', 'prior')

    flipped_predictors = get_dataset(dataset="Sonar")
    dim = flipped_predictors.shape[1]
    loglikelihood_fn = get_log_likelihood(flipped_predictors)

    # Definition of the prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_mean = jnp.zeros(dim)

    my_tgt_log_density = lambda x: loglikelihood_fn(x) + jax.scipy.stats.multivariate_normal.logpdf(x,
                                                                                                    mean=my_prior_mean,
                                                                                                    cov=my_prior_covariance)

    if base_measure_type == "laplace":  # i.e., Laplace

        # Perform Laplace approximation to find the base measure mean and covariance
        _, m, C = laplace_approximation(my_tgt_log_density, my_prior_mean)
        base_measure_mean = m
        base_measure_cov = C
        _loglikelihood_fn = lambda x: my_tgt_log_density(x) - jax.scipy.stats.multivariate_normal.logpdf(x,
                                                                                                         mean=base_measure_mean,
                                                                                                         cov=base_measure_cov)
    else:
        base_measure_mean = my_prior_mean
        base_measure_cov = my_prior_covariance
        _loglikelihood_fn = loglikelihood_fn

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, mean=base_measure_mean,
                                              cov=base_measure_cov)

    def logbase_density_fn(x):
        return jax.scipy.stats.multivariate_normal.logpdf(x, mean=base_measure_mean,
                                                          cov=base_measure_cov)

    return _loglikelihood_fn, base_measure_sampler, logbase_density_fn, base_measure_mean, base_measure_cov
