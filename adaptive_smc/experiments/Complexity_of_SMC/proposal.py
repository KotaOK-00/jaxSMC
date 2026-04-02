from adaptive_smc.smc_types import SMCStatebis, LogDensity
from typing import Optional
import jax.numpy as jnp
import jax


def build_gaussian_rw_proposal_fixed_scaling(state: SMCStatebis, _: LogDensity, __: LogDensity, i: int,
                                             j: Optional[int] = None):
    particles = state.particles
    dim = particles.shape[-1]
    optimal_scale = 2.38 / jnp.sqrt(dim)

    def gaussian_rwmh_cov_log_proposal(x, y):  # No need to implement it, it is symmetric.
        return 0.

    def gaussian_rwmh_sampler(key, x):
        return x + jax.random.normal(key, shape=(dim,)) * optimal_scale

    return gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, jnp.empty(1)
