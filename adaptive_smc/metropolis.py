from typing import Tuple

import jax
from jax.typing import ArrayLike

from adaptive_smc.smc_types import PRNGKey


def accept_reject_mh_step(key: PRNGKey, log_density_for_proposed: ArrayLike, log_density_for_current: ArrayLike,
                          log_proposal_for_current: ArrayLike, log_proposal_for_proposed: ArrayLike) -> Tuple[
    ArrayLike, ArrayLike]:
    """
    Given the log densities and log proposals for the proposed and current particles, return a boolean array
    """

    logU = -jax.random.exponential(key)
    log_mh_ratio = jax.lax.min(0.,
                               log_density_for_proposed - log_density_for_current + log_proposal_for_current - log_proposal_for_proposed)
    return logU <= log_mh_ratio, log_mh_ratio
