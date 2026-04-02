from typing import Callable, NamedTuple, Optional, Tuple

import jax
from jax.typing import ArrayLike

PRNGKey = jax.Array

"""
Either used for (un-normalised) log-likelihood or target log density
"""
type LogDensity = Callable[[ArrayLike], ArrayLike]
type LogProposal = Callable[[ArrayLike, ArrayLike], ArrayLike]
type Sampler = Callable[[PRNGKey], ArrayLike]
type ProposalSampler = Callable[[PRNGKey, ArrayLike], ArrayLike]
type KernelSampler = Callable[[PRNGKey, ArrayLike], ArrayLike] # Unused
type Density = Callable[[ArrayLike], ArrayLike] # Unused

class SMCStatebis(NamedTuple):
    particles: ArrayLike  # of shape (iteration + 1, num_parallel_chain, P, dim)
    proposed_particles: ArrayLike  # of shape (iteration + 1, num_parallel_chain, P, dim)
    log_weights: ArrayLike  # of shape (iteration + 1, num_parallel_chain, P)
    mh_proposal_parameters: ArrayLike  # of shape (iteration, *initial_shape_of_mh_proposal_parameter)
    tempering_sequence: ArrayLike  # of shape (iteration + 1, )
    others: Optional[ArrayLike] = None  # of shape (iteration, *initial_shape_of_other)


# The first integer is the index used for accessing the MH parameters, the second integer, to access the particles, and related, arrays.
type ProposalBuilder = Callable[
    [SMCStatebis, LogDensity, LogDensity, int, Optional[int]], Tuple[LogProposal, ProposalSampler, ArrayLike]]

type OptimisingProcedure = Callable[
    [Callable[[ArrayLike], ArrayLike], ArrayLike], ArrayLike]

# The second integer is exactly there for the same reason as the second one in ProposalBuilder
type CriteriaFunction = Callable[[ArrayLike, ArrayLike, SMCStatebis, int, int], ArrayLike]

