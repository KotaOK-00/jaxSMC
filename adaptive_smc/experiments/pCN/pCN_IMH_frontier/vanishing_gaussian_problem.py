import jax.numpy as jnp


def construct_my_prior_and_target(config):
    r"""
    The prior is N(0, C), with C = \textup{diag}(1/k^{\alpha})
    The target is a Gaussian distribution N(0, \tau^2 \textup{diag}(1/k^{\beta})),
    The objective of this problem is to demonstrate that
    the pCN critical temperature depends on the trace of the proposal covariance (i.e., C).
    """

    config_problem = config.get('problem')
    tau = config_problem.get('tau')
    dim = config.get('dim')
    vanishing_order = config_problem.get('vanishing_order_V', 1)

    C = jnp.diag(1 / jnp.arange(1, dim + 1) ** vanishing_order)

    loglikelihood_fn = lambda x: - 0.5 * x.T @ jnp.linalg.inv(C) @ x * (1 / tau ** 2 - 1)

    return loglikelihood_fn
