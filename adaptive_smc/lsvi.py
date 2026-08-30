"""
Least-Squares Variational Inference (LSVI), dense Gaussian scheme.

    Le Fay Y., Chopin N., Barthelme S., "Least Squares Variational Inference",
    NeurIPS 2025, arXiv:2502.18475 -- Section 4, Alg. 3.

Derived from the reference implementation at https://github.com/ylefay/LSVI
(Copyright the LSVI authors, Apache-2.0); rewritten in matrix form and reduced
to the Gaussian case.  Vendored rather than imported because the package pulls
in pymc / blackjax / particles>=0.4 as hard dependencies.

`lsvi_gaussian_approximation` is a drop-in replacement for
`adaptive_smc.laplace.laplace_approximation`: same `(-log_density(m), m, C)`
return signature, so it can be swapped in wherever a Gaussian reference /
base measure is built.

The sweep, in one line: with the current q = N(mu, Sigma) and Sigma = S S,
project y(z) = log pi~(mu + S z) onto the quadratics in L2(N(0, I)).  The basis
1, z_i, (z_i^2 - 1)/sqrt(2), z_i z_j is orthonormal there, so the projection is
just moments -- with s = E[y], v = E[y z], M = E[y z z^T],

    quadratic coefficient   C2     = (M - s I) / 2
    new precision           Lambda = -2 S^-1 C2 S^-1
    new precision * mean    h      = Lambda mu + S^-1 v

which is what upstream's Appendix-D.4 gamma -> eta chain computes with vec/kron
plumbing (verified equal analytically and numerically).  The intercept of the
regression never enters, so it is dropped.

Requires float64 (`jax.config.update("jax_enable_x64", True)`).
"""
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

__all__ = ["lsvi_gaussian_approximation"]


def lsvi_gaussian_approximation(log_density: Callable,
                                init_mean: ArrayLike,
                                init_cov: ArrayLike,
                                key: jax.Array = None,
                                n_iter: int = 50,
                                n_samples: int = 20_000,
                                lr_schedule=1.0,
                                target_residual_schedule=10.0,
                                batch_size=jnp.inf,
                                return_residuals: bool = False):
    """
    Gaussian LSVI reference N(m, C) for an unnormalised `log_density`.

    Parameters
    ----------
    init_mean, init_cov : seed of the variational family.  Seeding with the
        Laplace approximation converges in far fewer sweeps than with the prior;
        both land in the same place.
    lr_schedule : float or (n_iter,) array, the damping factor.
    target_residual_schedule : None (disabled), float or (n_iter,) array.  Caps the
        variance of the regression residual; a step exceeding it is shrunk by
        sqrt(target / residual).  This is what keeps the scheme stable at lr = 1:
        with `None` the iterates random-walk around the optimum instead of settling.
    batch_size : samples per chunk of the MC sums (`jnp.inf` = one shot, the house
        default); peak memory ~ batch_size * d.  Must divide `n_samples`.  Unlike
        `utils.apply_vmap_batch`, the chunks are drawn inside the loop rather than
        sliced out of a materialised array -- at d = 167, n_samples = 1e6 the array
        of standardised samples alone would be 1.3 GB.

    Choosing `n_samples`: the dense family has d(d+1)/2 free parameters and the
    sweep uses moments rather than OLS, so the fit degrades when n_samples is only
    a small multiple of that.  Measured on logistic posteriors (ELBO, 12-30 sweeps):

        d    d(d+1)/2   n_samples   ELBO(lsvi)   ELBO(laplace)
        61      1 891      20 000      -471.5        -330.0   diverged
        61      1 891     100 000      -313.1        -330.0   ok
        61      1 891     400 000      -297.5        -330.0   good
       167     14 028     100 000      -790.0        -666.5   diverged
       167     14 028   1 000 000      -580.8        -666.5   ok

    Rule of thumb: n_samples >~ 50 * d(d+1)/2 (~25 d^2).  Below ~10x, LSVI is
    *worse* than the Laplace approximation it started from.  lr = 1 with
    target_residual = 10 beat a decaying lr at every (d, n_samples) tested.

    Returns
    -------
    (-log_density(m), m, C), plus the (n_iter,) residual trace if requested.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    mean = jnp.asarray(init_mean, dtype=jnp.result_type(float))
    cov = jnp.asarray(init_cov, dtype=jnp.result_type(float))
    d = mean.shape[0]
    eye = jnp.eye(d)

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
        jnp.asarray(jnp.inf if not use_residual else target_residual_schedule, float), (n_iter,))
    use_residual = use_residual and bool(jnp.isfinite(target_residual_schedule).any())

    log_density_v = jax.vmap(log_density)
    batch_keys = lambda k: k[None] if n_batches == 1 else jax.random.split(k, n_batches)
    zeros = (jnp.zeros(()), jnp.zeros(()))

    def moments(k, mu, S):
        """One MC pass: cached y, and E[y], E[y z], E[y z z^T]."""
        def body(acc, kb):
            sy, v, M = acc
            z = jax.random.normal(kb, (batch, d))
            y = log_density_v(mu + z @ S)
            return (sy + y.sum(), v + z.T @ y, M + z.T @ (y[:, None] * z)), y
        acc, ys = jax.lax.scan(body, (jnp.zeros(()), jnp.zeros(d), jnp.zeros((d, d))), batch_keys(k))
        sy, v, M = (a / n_samples for a in acc)
        return ys, sy, v, M

    def residual(k, ys, h, P):
        """
        Damping statistic of Alg. 3, reproduced exactly as upstream computes it:
        Var_z[ y(z) - eta . s(z) ], with eta the *x-space* natural parameter
        (h, -vec(P)/2) applied to the standardised statistic s(z) = (z, vec(zz'), 1).
        Not the z-space regression residual -- but `target_residual` is a tuned
        constant on this scale, so keep the two consistent.  No target evals.
        """
        def body(acc, ky):
            kb, y = ky
            z = jax.random.normal(kb, (batch, d))
            r = y + 0.5 * ((z @ P) * z).sum(-1) - z @ h
            return (acc[0] + r.sum(), acc[1] + (r ** 2).sum()), None
        (s1, s2), _ = jax.lax.scan(body, zeros, (batch_keys(k), ys))
        return s2 / n_samples - (s1 / n_samples) ** 2

    def backtrack(P, P_new, lr):
        """Halve lr until the interpolated precision is PD (bounded loop)."""
        not_pd = lambda _lr: jnp.isnan(jnp.linalg.cholesky(_lr * P_new + (1 - _lr) * P)).any()
        lr, i = jax.lax.while_loop(lambda c: jnp.logical_and(not_pd(c[0]), c[1] < 50),
                                   lambda c: (c[0] / 2, c[1] + 1), (lr, 0))
        return jnp.where(i >= 50, 0.0, lr)          # gave up -> keep the current state

    def sweep(state, inps):
        h, P = state
        k, lr, target = inps
        cov = jnp.linalg.pinv(P)
        mu = cov @ h
        D, V = jax.scipy.linalg.eigh(cov)
        S = (V * jnp.sqrt(D)) @ V.T                 # symmetric square root
        S_inv = jax.scipy.linalg.inv(S)

        ys, s, v, M = moments(k, mu, S)
        C2 = 0.5 * (M - s * eye)                    # L2(N(0,I)) projection onto quadratics
        P_new = -2 * S_inv @ C2 @ S_inv
        h_new = P_new @ mu + S_inv @ v

        lr = backtrack(P, P_new, lr)
        res = jnp.nan
        if use_residual:
            damp = lambda _lr: (_lr * h_new + (1 - _lr) * h, _lr * P_new + (1 - _lr) * P)
            res = residual(k, ys, *damp(lr))
            lr = jnp.minimum(lr, jnp.where(res <= target, lr, jnp.sqrt(target / res)))
            res = residual(k, ys, *damp(lr))
        return (lr * h_new + (1 - lr) * h, lr * P_new + (1 - lr) * P), res

    P0 = jnp.linalg.pinv(0.5 * (cov + cov.T))
    (h, P), residuals = jax.lax.scan(
        sweep, (P0 @ mean, P0),
        (jax.random.split(key, n_iter), lr_schedule, target_residual_schedule))

    C = jnp.linalg.pinv(P)
    C = 0.5 * (C + C.T)
    m = C @ h
    if not bool(jnp.all(jnp.isfinite(m)) & ~jnp.isnan(jnp.linalg.cholesky(C)).any()):
        raise RuntimeError(
            "LSVI did not return a proper Gaussian (non-PD covariance or NaNs). "
            "Try more samples (see the n_samples table above), a damped lr_schedule "
            "such as jnp.minimum(1, 3 / jnp.arange(1, n_iter + 1)), a smaller "
            "target_residual_schedule, or a wider init_cov.")

    if return_residuals:
        return -log_density(m), m, C, residuals
    return -log_density(m), m, C
