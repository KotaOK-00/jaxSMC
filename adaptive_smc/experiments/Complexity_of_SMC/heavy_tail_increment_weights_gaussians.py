import jax
import jax.numpy as jnp
import jax.scipy as jsp


def random_well_conditioned_cov(key, dim, eig_min=0.5, eig_max=2.0):
    """
    Generate a random SPD matrix with bounded condition number.
    The condition number <= eig_max / eig_min, independent of dim.
    """
    key_q, key_eig = jax.random.split(key)

    # Random orthogonal matrix via QR
    A = jax.random.normal(key_q, (dim, dim))
    Q, R = jnp.linalg.qr(A)

    # Fix sign to make QR deterministic
    Q = Q * jnp.sign(jnp.diag(R))

    # Eigenvalues in bounded range
    eigvals = jax.random.uniform(
        key_eig, (dim,), minval=eig_min, maxval=eig_max
    )

    return Q @ jnp.diag(eigvals) @ Q.T


def make_model(dim, heavy_factor, key, mean=None):
    """
    Returns:
        loglikelihood_fn(x)
        base_measure_sampler(key)
        logbase_density_fn(x)
    with light or heavy-tailed weights.
    """

    key_cov, key_cov_base = jax.random.split(key, 2)
    # -----------------------
    # Base: keep well-conditioned
    # -----------------------
    cov_base = random_well_conditioned_cov(key_cov_base, dim)
    prec_base = jnp.linalg.inv(cov_base)

    # Take base covariance and multiply eigenvalues by a factor 
    base_eigvals, base_vecs = jnp.linalg.eigh(cov_base)
    cov = (base_vecs @ jnp.diag(base_eigvals * heavy_factor) @ base_vecs.T)
    precision = jnp.linalg.inv(cov)
    if not mean:
        mean = jnp.zeros((dim,))
    if isinstance(mean, float):
        mean = jnp.ones((dim, ))*mean
    # -----------------------
    # Log-likelihood (unnormalized)
    # -----------------------
    def loglikelihood_fn(x):
        return -0.5 * jnp.einsum('i,ij,j->', x-mean, precision-prec_base, x-mean)

    # -----------------------
    # Base sampler and log-density
    # -----------------------
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), cov_base)

    def logbase_density_fn(x):
        return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), cov_base)

    _, logdet = jnp.linalg.slogdet(precision)
    logdet = -logdet
    _, logdetbase = jnp.linalg.slogdet(cov_base)
    posterior_mean = cov @ (precision - prec_base) @ mean
    true_log_normalising_constant = 0.5 * logdet - 0.5 * logdetbase + (-0.5 * mean.T@(precision-prec_base)@mean + 0.5 * posterior_mean.T @ precision @ posterior_mean)

    return loglikelihood_fn, base_measure_sampler, logbase_density_fn, true_log_normalising_constant, cov, posterior_mean

def make_model_unaligned(dim, heavy_factor, key, mean=None):
    """
    Same as make_model, but the target covariance is sampled independently
    from the base covariance (no shared eigenvectors).

    heavy_factor still controls how different (heavier/lighter) the target is.
    """

    key_cov_base, key_cov_target = jax.random.split(key, 2)

    # -----------------------
    # Base: well-conditioned
    # -----------------------
    cov_base = random_well_conditioned_cov(key_cov_base, dim)
    prec_base = jnp.linalg.inv(cov_base)

    # -----------------------
    # Target: independent covariance
    # -----------------------
    cov_target_raw = random_well_conditioned_cov(key_cov_target, dim)

    # Scale eigenvalues to control tail behavior
    eigvals, eigvecs = jnp.linalg.eigh(cov_target_raw)
    cov = eigvecs @ jnp.diag(eigvals * heavy_factor) @ eigvecs.T
    precision = jnp.linalg.inv(cov)

    # -----------------------
    # Mean handling
    # -----------------------
    if mean is None:
        mean = jnp.zeros((dim,))
    elif isinstance(mean, float):
        mean = jnp.ones((dim,)) * mean

    # -----------------------
    # Log-likelihood (unnormalized)
    # -----------------------
    def loglikelihood_fn(x):
        return -0.5 * jnp.einsum('i,ij,j->', x - mean, precision - prec_base, x - mean)

    # -----------------------
    # Base sampler and log-density
    # -----------------------
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), cov_base)

    def logbase_density_fn(x):
        return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), cov_base)

    # -----------------------
    # True normalizing constant
    # -----------------------
    _, logdet = jnp.linalg.slogdet(precision)
    logdet = -logdet
    _, logdetbase = jnp.linalg.slogdet(cov_base)

    posterior_mean = cov @ (precision - prec_base) @ mean

    true_log_normalising_constant = (
        0.5 * logdet
        - 0.5 * logdetbase
        + (
            -0.5 * mean.T @ (precision - prec_base) @ mean
            + 0.5 * posterior_mean.T @ precision @ posterior_mean
        )
    )

    return (
        loglikelihood_fn,
        base_measure_sampler,
        logbase_density_fn,
        true_log_normalising_constant,
        cov,
        posterior_mean,
    )


def make_model_bimodal(dim: int, heavy_factor: float, key, mean=None, delta=50.):
    """
    Model with:
        - Gaussian base: N(0, I)
        - Bimodal target: mixture of two Gaussians
            N(mean, heavy_factor * I) and N(mean + delta, heavy_factor * I)
        - Returns:
            loglikelihood_fn(x), base_sampler(key), logbase_density_fn(x),
            true_log_normalising_constant (None),
            target covariance placeholder (None),
            mean (used in the target)
    """
    if mean is None:
        mean = jnp.zeros((dim,))
    elif isinstance(mean, float):
        mean = jnp.ones((dim,)) * mean

    # -----------------------
    # Base: standard Gaussian
    # -----------------------
    cov_base = jnp.eye(dim)

    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), cov_base)

    def logbase_density_fn(x):
        return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), cov_base)

    # -----------------------
    # Target: bimodal Gaussian
    # -----------------------
    cov_target = jnp.eye(dim) * heavy_factor

    def loglikelihood_fn(x):
        # Evaluate both modes
        logpdf1 = jsp.stats.multivariate_normal.logpdf(x, mean, 2*cov_target)
        logpdf2 = jsp.stats.multivariate_normal.logpdf(x, mean + delta*jnp.ones((dim,)), 1.2*cov_target)
        # Mixture log-density with equal weights
        max_log = jnp.maximum(logpdf1, logpdf2)
        # log-sum-exp trick for numerical stability
        log_mix = max_log + jnp.log(0.25 * jnp.exp(logpdf1 - max_log) + 0.75 * jnp.exp(logpdf2 - max_log))
        return log_mix

    return lambda x: loglikelihood_fn(x)-logbase_density_fn(x), base_measure_sampler, logbase_density_fn, None, None, mean


def make_model_bimodal_simple(dim: int, heavy_factor: None, key, mean=None):
    """
    Multidimensional bimodal Gaussian mixture:
        0.5 N(-2 * 1_d, I) + 0.5 N(2 * 1_d, I)

    Base: standard Gaussian N(0, I)
    """

    mean1 = -2 * jnp.ones((dim,))
    mean2 =  3 * jnp.ones((dim,))
    cov = jnp.eye(dim)

    # -----------------------
    # Base: standard Gaussian
    # -----------------------
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), 1 * cov)

    def logbase_density_fn(x):
        return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), 1 * cov)

    # -----------------------
    # Target: bimodal Gaussian
    # -----------------------
    def loglikelihood_fn(x):
        logpdf1 = jsp.stats.multivariate_normal.logpdf(x, mean1, cov)
        logpdf2 = jsp.stats.multivariate_normal.logpdf(x, mean2, cov)

        # Stable log-sum-exp
        max_log = jnp.maximum(logpdf1, logpdf2)
        log_mix = max_log + jnp.log(
            0.5 * jnp.exp(logpdf1 - max_log) +
            0.5 * jnp.exp(logpdf2 - max_log)
        )
        return log_mix

    return (
        lambda x: loglikelihood_fn(x) - logbase_density_fn(x),
        base_measure_sampler,
        logbase_density_fn,
        None,
        None,
        jnp.zeros((dim,)),
    )




def make_model_bimodal_cov(dim: int, heavy_factor: None, key, mean=None):
    """
    Multidimensional bimodal Gaussian mixture:
        0.5 N(-2 * 1_d, I) + 0.5 N(2 * 1_d, I)

    Base: standard Gaussian N(0, I)
    """

    mean1 = -2 * jnp.ones((dim,))
    mean2 =  3 * jnp.ones((dim,))
    
    # -----------------------
    key1, key2 = jax.random.split(key)
    cov = jnp.eye(dim) + 0.0 * random_well_conditioned_cov(key1, dim)

    # -----------------------
    # Base: standard Gaussian
    # -----------------------
    def base_measure_sampler(key):
        return jax.random.multivariate_normal(key, jnp.zeros(dim), 1 * jnp.eye(dim))

    def logbase_density_fn(x):
        return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), 1 * jnp.eye(dim))

    # -----------------------
    # Target: bimodal Gaussian
    # -----------------------
    def loglikelihood_fn(x):
        logpdf1 = jsp.stats.multivariate_normal.logpdf(x, mean1, cov)
        logpdf2 = jsp.stats.multivariate_normal.logpdf(x, mean2, cov)

        # Stable log-sum-exp
        max_log = jnp.maximum(logpdf1, logpdf2)
        log_mix = max_log + jnp.log(
            0.5 * jnp.exp(logpdf1 - max_log) +
            0.5 * jnp.exp(logpdf2 - max_log)
        )
        return log_mix

    return (
        lambda x: loglikelihood_fn(x) - logbase_density_fn(x),
        base_measure_sampler,
        logbase_density_fn,
        None,
        None,
        jnp.zeros((dim,)),
    )