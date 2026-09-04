import jax
import jax.numpy as jnp
import jax.scipy.stats.norm
from jax.scipy.special import betainc, gammaln

from particles import datasets

logistic = jax.scipy.special.expit

def normal_cdf(x):
    return 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))

def make_robit_logcdf(df=7.0, scale=1.5484):
    """Student-t(df) CDF in closed form, df = odd integer:
        F(t) = 1/2 + (theta + sin(theta) cos(theta) P(cos^2 theta)) / pi,  theta = arctan(t / sqrt(df))
        P(w) = sum_{j<(df-1)/2} c_j w^j,  c_0 = 1,  c_j = c_{j-1} * 2j / (2j+1)
    """
    m = (int(df) - 1) // 2
    coef, c = [], 1.0
    for j in range(m):
        coef.append(c); c *= (2*j + 2) / (2*j + 3)
    def logcdf(x):
        th = jnp.arctan(x / (scale * jnp.sqrt(df)))
        c, s = jnp.cos(th), jnp.sin(th)
        w = c * c
        return jnp.log(0.5 + (th + s * c * sum(cj * w**j for j, cj in enumerate(coef))) / jnp.pi)
    return logcdf

def get_log_likelihood(flipped_predictors, cdf=logistic, logcdf=None):
    """
    Define the log target density of the posterior distribution of the logistic regression model,
    assuming a Gaussian prior. 
    The logcdf can define any link function.
    """
    if logcdf is not None:
        def log_likelihood_fn(beta):
            lc = logcdf(flipped_predictors @ beta.T)
            return jnp.sum(lc, axis=-1)
 
    elif cdf == logistic:
        def log_likelihood_fn(beta):
            lc = -jnp.log1p(jnp.exp(-flipped_predictors @ beta.T))
            lc = jnp.nan_to_num(lc, False, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.sum(lc, axis=-1)
 
    else:
        def log_likelihood_fn(beta):
            lc = jnp.log(cdf(flipped_predictors @ beta.T))
            lc = jnp.nan_to_num(lc, False, nan=0.0, posinf=0.0, neginf=0.0)
            return jnp.sum(lc, axis=-1)
 
    return log_likelihood_fn

def get_tgt_log_density(flipped_predictors, log_prior_fn, cdf=logistic):
    return lambda beta: get_log_likelihood(flipped_predictors, cdf)(beta) + log_prior_fn(beta)

def get_dataset(flip=True, dataset="Sonar"):
    dataset = getattr(datasets, dataset)()
    data = dataset.preprocess(dataset.raw_data, return_y=not flip)
    return data
