"""
Least-Squares Variational Inference (LSVI), dense Gaussian scheme.

    Le Fay Y., Chopin N., Barthelme S., "Least Squares Variational Inference", NeurIPS 2025, arXiv:2502.18475 -- Section 4, Alg. 3.

Matrix-form rewrite of `LSVI/variational/gaussian_lsvi.py::gaussian_lsvi` from https://github.com/ylefay/LSVI (Apache-2.0), vendored rather than imported because the
package pulls in pymc / blackjax / particles>=0.4. 
Requires float64 (`jax.config.update("jax_enable_x64", True)`).
"""
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["gaussian_lsvi", "lsvi_gaussian_approximation", "get_theta", "get_mean_cov"]


def get_theta(mean: ArrayLike, cov: ArrayLike):
    """`GenericNormalDistribution.get_theta`: (mean, cov) -> (eta1, invcov)."""
    invcov = jnp.linalg.pinv(0.5 * (cov + cov.T))
    return invcov @ mean, invcov


def get_mean_cov(eta1: ArrayLike, invcov: ArrayLike):
    """`GenericNormalDistribution.get_mean_cov`: (eta1, invcov) -> (mean, cov)."""
    cov = jnp.linalg.pinv(0.5 * (invcov + invcov.T))
    return cov @ eta1, cov


def gaussian_lsvi(OP_key: jax.Array,
                  tgt_log_density: Callable,
                  init_mean: ArrayLike,
                  init_cov: ArrayLike,
                  n_iter: int = 30,
                  n_samples: int = 200_000,
                  lr_schedule=1.0,
                  target_residual_schedule=None,
                  batch_size=jnp.inf,
                  return_all: bool = False):
    """
    Dense Gaussian scheme, Alg. 3.

    :param OP_key: PRNGKey
    :param tgt_log_density: log-density of the target distribution
    :param init_mean, init_cov: initial variational distribution (Laplace >> prior)
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: samples per iteration, replacing the exact expectations
    :param lr_schedule: float or (n_iter,) array, learning rate schedule
    :param target_residual_schedule: None (disabled), float or (n_iter,) array
    :param batch_size: samples per chunk; must divide n_samples.  Memory only, but it changes the RNG stream.
    :param return_all: return the (n_iter, 4) diagnostics
        [residual, eta0, rel_d_eta1, rel_d_invcov]

    Returns ((eta0, eta1, invcov), all_results).
    """
    mean = jnp.asarray(init_mean, dtype=jnp.result_type(float))
    cov = jnp.asarray(init_cov, dtype=jnp.result_type(float))
    dimension = mean.shape[0]
    eye = jnp.eye(dimension)

    if batch_size >= n_samples:
        n_batches, batch = 1, n_samples
    elif n_samples % batch_size:
        raise ValueError(f"batch_size={batch_size} must divide n_samples={n_samples}")
    else:
        batch = int(batch_size)
        n_batches = n_samples // batch

    lr_schedule = jnp.broadcast_to(jnp.asarray(lr_schedule, float), (n_iter,))
    use_residual = target_residual_schedule is not None
    target_residual_schedule = jnp.broadcast_to(
        jnp.asarray(target_residual_schedule if use_residual else jnp.inf, float), (n_iter,))
    use_residual = use_residual and bool(jnp.isfinite(target_residual_schedule).any())

    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)
    batch_keys = lambda key: key[None] if n_batches == 1 else jax.random.split(key, n_batches)

    def modified_statistic(key, current_mean, sqrtm):
        """
        Eq. 16, contracted against y: returns `X.T @ y / n_samples` directly rather than
        the (n_samples, d(d+1)/2 + d + 1) matrix X, so nothing of that size is ever built.
        gamma0tilde = E[y], gamma1tilde = E[y z], and E[y z z'] carries the triu /
        sqrt(2)-scaled entries of gamma2tildetilde.  E[y^2] is what `get_residual` needs.
        """

        def body(acc, key_batch):
            sum_y, sum_yy, sum_yz, sum_yzz = acc
            z = jax.random.normal(key_batch, (batch, dimension))
            y = vmapped_tgt_log_density(current_mean + z @ sqrtm)
            return (sum_y + y.sum(), sum_yy + y @ y,
                    sum_yz + z.T @ y, sum_yzz + z.T @ (y[:, None] * z)), None

        init = (jnp.zeros(()), jnp.zeros(()), jnp.zeros(dimension), jnp.zeros((dimension, dimension)))
        acc, _ = jax.lax.scan(body, init, batch_keys(key))
        s, sum_yy, v, M = (a / n_samples for a in acc)
        return s, sum_yy, v, 0.5 * (M + M.T)

    def from_gammatildetilde_to_gamma(s, M):
        """gammatildetilde -> gammatilde -> gamma, collapsed (Eqs. 63-64)."""
        return 0.5 * (M - s * eye)

    def from_gamma_to_eta(current_mean, inv_chol, s, v, gamma2):
        """Eqs. 56, 58, 59.  eta2 = -0.5 * vec(invcov)."""
        invcov = -2 * inv_chol @ gamma2 @ inv_chol
        eta1 = invcov @ current_mean + inv_chol @ v
        gamma0 = s - jnp.trace(gamma2)
        eta0 = gamma0 - eta1 @ current_mean + 0.5 * current_mean @ invcov @ current_mean
        return eta0, eta1, invcov

    def get_residual(s, sum_yy, v, M, eta, next_eta):
        """
        Returned as a closed-form quadratic in lr.  With r(lr) = a + lr b, a = y +
        z'invcov z / 2 - eta1 . z and b the same for the step, every term is either a
        y-moment already computed above or an exact Gaussian moment of z (E[zz'] = I,
        Var[z'Az] = 2 tr A^2, Cov[z'Az, g.z] = 0).
        """
        _, eta1, invcov = eta
        _, next_eta1, next_invcov = next_eta
        d_eta1, d_invcov = next_eta1 - eta1, next_invcov - invcov
        cov_y_q = lambda A, g: 0.5 * (jnp.sum(A * M) - s * jnp.trace(A)) - g @ v
        cov_q_q = lambda A, g, B, f: 0.5 * jnp.sum(A * B) + g @ f
        var_a = (sum_yy - s ** 2) + 2 * cov_y_q(invcov, eta1) + cov_q_q(invcov, eta1, invcov, eta1)
        cov_ab = cov_y_q(d_invcov, d_eta1) + cov_q_q(invcov, eta1, d_invcov, d_eta1)
        var_b = cov_q_q(d_invcov, d_eta1, d_invcov, d_eta1)
        return lambda lr: var_a + 2 * lr * cov_ab + lr ** 2 * var_b

    def sanity(invcov):
        """True if the parameter is *not* a proper Gaussian.  Checked on invcov rather
        than on pinv(invcov): same predicate, no SVD inside the backtracking loop."""
        return jnp.isnan(jnp.linalg.cholesky(invcov)).any()

    def momentum_backtracking(lr, eta, next_eta, residual, target_residual):
        """
        Halve lr until the natural parameter defines a valid distribution, then shrink it
        so the residual variance meets `target_residual`, and take the minimum of the two.
        Bounded: upstream's loop never exits on a NaN next_eta (0 * NaN = NaN).
        """
        invcov, next_invcov = eta[2], next_eta[2]
        not_pd = lambda _lr: sanity(_lr * next_invcov + (1 - _lr) * invcov)
        lr, i = jax.lax.while_loop(lambda c: jnp.logical_and(not_pd(c[0]), c[1] < 30),
                                   lambda c: (c[0] / 2, c[1] + 1), (lr, 0))
        lr = jnp.where(i >= 30, 0.0, lr)
        if not use_residual:
            return lr, jnp.nan
        current_residual = residual(lr)
        lr_tempering = jnp.where(current_residual <= target_residual, lr,
                                 jnp.sqrt(target_residual / current_residual))
        lr = jnp.minimum(lr, lr_tempering)
        return lr, residual(lr)

    def iter_routine(eta, inps):
        """See Alg. 3."""
        key, lr, target_residual = inps
        eta0, eta1, invcov = eta

        # current_mean / sqrtm / inv_chol from ONE eigh; upstream does get_mean_cov
        # (a pinv), then eigh(current_cov), then inv(sqrtm).  Eigenvalues below pinv's
        # rcond are dropped, so the pseudo-inverse convention is preserved.
        lam, V = jax.scipy.linalg.eigh(invcov)
        keep = lam > jnp.max(lam) * dimension * jnp.finfo(lam.dtype).eps
        lam = jnp.where(keep, lam, 1.0)
        current_cov = (V * jnp.where(keep, 1.0 / lam, 0.0)) @ V.T
        sqrtm = (V * jnp.where(keep, lam ** -0.5, 0.0)) @ V.T
        inv_chol = (V * jnp.where(keep, lam ** 0.5, 0.0)) @ V.T
        current_mean = current_cov @ eta1

        s, sum_yy, v, M = modified_statistic(key, current_mean, sqrtm)
        next_gamma = from_gammatildetilde_to_gamma(s, M)
        next_eta = from_gamma_to_eta(current_mean, inv_chol, s, v, next_gamma)

        lr, residual = momentum_backtracking(
            lr, eta, next_eta, get_residual(s, sum_yy, v, M, eta, next_eta), target_residual)
        next_eta = tuple(lr * new + (1 - lr) * old for new, old in zip(next_eta, eta))

        rel = lambda new, old: jnp.linalg.norm(new - old) / (jnp.linalg.norm(old) + 1e-300)
        return next_eta, jnp.array([residual, next_eta[0],
                                    rel(next_eta[1], eta1), rel(next_eta[2], invcov)])

    eta_init = (jnp.zeros(()),) + get_theta(mean, cov)
    eta, all_results = jax.lax.scan(iter_routine, eta_init,
                                    (iter_keys, lr_schedule, target_residual_schedule))
    return eta, (all_results if return_all else None)


def lsvi_gaussian_approximation(log_density: Callable,
                                init_mean: ArrayLike,
                                init_cov: ArrayLike,
                                key: jax.Array = None,
                                n_iter: int = 30,
                                n_samples: int = 200_000,
                                lr_schedule=1.0,
                                target_residual_schedule=None,
                                batch_size=jnp.inf,
                                return_all: bool = False):
    """
    Drop-in replacement for `adaptive_smc.laplace.laplace_approximation`: same `(-log_density(m), m, C)` return signature, so it can be swapped in wherever a Gaussian reference / base measure is built.  
    With `return_all`, a 4th element:
    the (n_iter, 4) array of [residual, eta0, rel_d_eta1, rel_d_invcov] per sweep.

    Tuning, in order of what actually binds:

    n_samples -- sets the noise floor of the answer, and nothing else does: each sweep *replaces* the state rather than averaging into it, so the noise does not wash
        out over sweeps.  The failure mode is over-dispersion, which looks exactly like LSVI correctly widening a too-narrow Laplace.  Start at 2e5, expect 1e6 at
        d >~ 100, and settle it by rerunning with a different `key`: if sd(C) moves, the run is still noise-dominated and more sweeps will not help.

    lr_schedule -- how fast the state travels, and simultaneously how much the MC noise is averaged out (at lr the state is an EWMA of ~1/lr sweeps).  
    A decaying schedule, e.g. jnp.minimum(1, 6 / jnp.arange(1, n_iter + 1)), travels fast early and averages late; it beat constant lr = 1 on every problem tried here.

    n_iter -- only as many sweeps as it takes to arrive.  Watch the residual trace: while it falls monotonically the run is still travelling, and once it starts fluctuating instead the travel is over.  After that, extra sweeps still help, but only by widening the averaging window of a decaying lr.

    target_residual_schedule -- NOT a tolerance.  It caps lr at sqrt(target_residual / residual), so a target well below the residual acts as a
        hidden constant learning rate.  The residual has a floor that has nothing to do with fit quality -- for a Gaussian target at the fixed point it is exactly

            0.5 * sum((eigvalsh(inv(C)) - 1) ** 2) + sum((inv(C) @ m) ** 2)

        because upstream measures the x-space parameter against standardised z -- so it
        is problem-specific and does not transfer (upstream's 10. was set on Sonar).
        Set it near the observed plateau to keep it as a guard on the early transient, or None with a decaying lr_schedule, which is how upstream pairs them.

    eta0 is the intercept of the fit, so it gives the normalising constant for free:

        log_Z ~= eta0 + m @ inv(C) @ m / 2 + (d * log(2 pi) + slogdet(C)[1]) / 2

    (checked against a Gaussian target with known Z: 5.007 at n_samples 5e5, 5.002 at 2e6.)
    """
    key = jax.random.PRNGKey(0) if key is None else key
    (eta0, eta1, invcov), all_results = gaussian_lsvi(
        key, log_density, init_mean, init_cov, n_iter=n_iter, n_samples=n_samples,
        lr_schedule=lr_schedule, target_residual_schedule=target_residual_schedule,
        batch_size=batch_size, return_all=return_all)

    m, C = get_mean_cov(eta1, invcov)
    if not bool(jnp.all(jnp.isfinite(m)) & ~jnp.isnan(jnp.linalg.cholesky(C)).any()):
        raise RuntimeError(
            "LSVI did not return a proper Gaussian (non-PD covariance or NaNs). "
            "Try more samples, a damped lr_schedule such as "
            "jnp.minimum(1, 6 / jnp.arange(1, n_iter + 1)), a smaller "
            "target_residual_schedule, or a wider init_cov.")

    if return_all:
        return -log_density(m), m, C, all_results
    return -log_density(m), m, C
