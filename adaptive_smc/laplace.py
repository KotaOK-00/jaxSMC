from typing import Callable, Optional

import jax
import jax.numpy as jnp
import scipy
from jax.typing import ArrayLike


def newton_descent(loss: Callable, init: ArrayLike, step_size: float = 1):
    """
    Newton descent algorithm.
    Constant step size.
    """

    def update(x):
        grad = jax.grad(loss)(x)
        hess = jax.hessian(loss)(x)
        x = x - step_size * jax.scipy.linalg.solve(hess, grad)
        return x

    x = jax.lax.fori_loop(0, 100, lambda i, x: update(x), init)
    return x, jnp.linalg.pinv(jax.hessian(loss)(x))


def bfgs_wrapper(loss: Callable, init: ArrayLike, jac: Optional[Callable] = None):
    res = scipy.optimize.minimize(loss, init, method="BFGS", jac=jac)
    return res.x, res.hess_inv


def laplace_approximation(log_density: Callable, init: ArrayLike, optimization_method=bfgs_wrapper, log_density_jac: Optional[Callable] = None):
    """
    Compute the Laplace approximation of a density.
    See
        Approximate Bayesian inference for latent
        Gaussian models by using integrated nested Laplace approximations,
        Rue, Martino, Chopin (2009)
    """

    def loss(theta):
        return -log_density(theta)

    if log_density_jac:
        def jac_loss(theta):
            return -log_density_jac(theta)
    else:
        jac_loss = None

    x, hess_inv = optimization_method(loss, init, jac=jac_loss)
    return -log_density(x), x, hess_inv
