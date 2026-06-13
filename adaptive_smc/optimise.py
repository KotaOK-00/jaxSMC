from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.smc_types import OptimisingProcedure
from adaptive_smc.utils import apply_vmap_batch


def make_optimize_within_a_fixed_grid(grid: ArrayLike, batch_size=jnp.inf) -> OptimisingProcedure:
    """
    Constructing a maximisation procedure of a function over a grid.
    We use apply_vmap_batch to apply the function to the grid in batches of fixed sizes (by default inf)
    """

    def optimize_within_a_grid(func: Callable[[ArrayLike], ArrayLike], x: ArrayLike) -> ArrayLike:
        output_shape = func(grid.at[0].get()).shape
        fun_applied_to_grid = apply_vmap_batch(jax.vmap(func), grid, batch_size, output_shape)
        best = grid.at[jnp.argmax(fun_applied_to_grid, keepdims=True).at[0].get()].get()
        # If every grid point is masked (-inf, e.g. by the ESS guard), keep the previous parameter.
        return jnp.where(jnp.any(jnp.isfinite(fun_applied_to_grid)), best, x)

    # Exposed so that the SMC samplers can detect when this grid coincides with their
    # criteria grid and reuse the already-computed criteria values instead of
    # evaluating the objective on the grid a second time.
    optimize_within_a_grid.grid = grid

    return optimize_within_a_grid


def make_optimize_within_a_grid(minmax: Tuple[float, float], interval: Tuple[float, float],
                                n_steps: int) -> OptimisingProcedure:
    """
    Constructing a maximisation procedure of a function over a unidimensional grid.
    The grid is centered around the initialisation point of the maximisation procedure, is made of
    n_steps regularly distributed points. All the points outside the interval defined by the minmax tuple are flattened.
    """
    my_min, my_max = minmax
    a, b = interval

    def optimize_within_a_grid(func: Callable[[ArrayLike], ArrayLike], x: ArrayLike) -> ArrayLike:
        grid = x + jnp.linspace(a, b, n_steps)
        grid = jnp.minimum(jnp.maximum(grid, my_min), my_max)
        fun_applied_to_grid = jax.vmap(func)(grid)
        best = grid.at[jnp.argmax(fun_applied_to_grid, keepdims=True).at[0].get()].get()
        # If every grid point is masked (-inf, e.g. by the ESS guard), keep the previous parameter.
        return jnp.where(jnp.any(jnp.isfinite(fun_applied_to_grid)), best, x)

    return optimize_within_a_grid


def make_constant() -> OptimisingProcedure:
    def solve(_, x):
        """
        Trivial optimisation procedure returning the input
        """
        return x

    return solve
