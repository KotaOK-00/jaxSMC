import jax.numpy as jnp
import jax.scipy.stats.norm
from particles import datasets

logistic = jax.scipy.special.expit

def normal_cdf(x):
    return 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))


def get_log_likelihood(flipped_predictors, cdf=logistic):
    """
    Define the log target density of the posterior distribution of the logistic regression model,
    assuming a Gaussian prior.
    """
    if cdf == logistic:
        def log_likelihood_fn(beta):
            logcdf = -jnp.log1p(jnp.exp(-flipped_predictors @ beta.T))
            logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
            log_likelihood = jnp.sum(logcdf, axis=-1)
            return log_likelihood
    else:
        def log_likelihood_fn(beta):
            logcdf = jnp.log(cdf(flipped_predictors @ beta.T))
            logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
            log_likelihood = jnp.sum(logcdf, axis=-1)
            return log_likelihood

    return log_likelihood_fn

def get_tgt_log_density(flipped_predictors, log_prior_fn, cdf=logistic):
    return lambda beta: get_log_likelihood(flipped_predictors, cdf)(beta) + log_prior_fn(beta)

def get_dataset(flip=True, dataset="Sonar"):
    dataset = getattr(datasets, dataset)()
    data = dataset.preprocess(dataset.raw_data, return_y=not flip)
    return data
