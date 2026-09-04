import jax
import jax.numpy as jnp
import jax.scipy.stats.norm
from jax.scipy.special import betainc, gammaln

from particles import datasets

logistic = jax.scipy.special.expit

def normal_cdf(x):
    return 0.5 * (1 + jax.scipy.special.erf(x / jnp.sqrt(2)))

def make_robit_logcdf(df: float = 7.0, scale: float = 1.5484):
    """log F_nu(x / scale), F_nu = Student-t(df) CDF.
 
    z_i ~ t_nu(x_i'beta, scale), y_i = 1{z_i > 0}
    P(y_i = 1 | beta) = F_nu(x_i'beta / scale)。
 
    df=7, scale=1.5484 (Liu 2004): beta has the same scale as logistic
 
      - jax.scipy.stats.t does not have cdf nor logcdf (only logpdf and pdf);
            we define F(t) = 0.5 * I_z(nu/2, 1/2), z = nu/(nu+t^2) if t<=0.
    """
    nu, s = float(df), float(scale)
    a, b = 0.5 * nu, 0.5
    logC = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * jnp.log(nu * jnp.pi)
 
    def _val(u):
        z = nu / (nu + u * u)
        Iz = betainc(a, b, z) # = 2 * F_nu(-|u|)
        return jnp.where(
            u <= 0.0,
            jnp.log(jnp.clip(Iz, 1e-300, 1.0)) - jnp.log(2.0),   # log F(u)
            jnp.log1p(-0.5 * Iz),                                # log(1 - F(-u))
        )
 
    def _logpdf(u):
        return logC - 0.5 * (nu + 1.0) * jnp.log1p(u * u / nu)
 
    @jax.custom_jvp
    def robit_logcdf(x):
        return _val(x / s)
 
    @robit_logcdf.defjvp
    def _robit_logcdf_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        v = robit_logcdf(x)
        return v, jnp.exp(_logpdf(x / s) - v) / s * dx
 
    return robit_logcdf

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
