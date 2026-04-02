from typing import Union, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from adaptive_smc.proposals import build_build_autoregressive_gaussian_proposal
from adaptive_smc.proposals import build_gaussian_rwmh_cov_proposal_gamma
from adaptive_smc.smc_types import ProposalBuilder, LogDensity, SMCStatebis, LogProposal

__all__ = [
    "build_build_mixture_ar_rwm",
]


def mixture_log_proposal(log_proposal1: LogProposal, log_proposal2: LogProposal,
                         beta: Union[float, ArrayLike]) -> LogProposal:
    def log_mixture(x, y):
        log_density = jnp.log(
            beta * jnp.exp(log_proposal1(x, y)) +
            (1 - beta) * jnp.exp(log_proposal2(x, y)))
        return log_density

    return log_mixture


def build_build_mixture_ar_rwm(mu, C) -> ProposalBuilder:
    r"""
    Assuming the first component of mh_proposal_parameters is \beta, the second is \gamma, and the third is \rho.
    The proposal is a mixture of RWMH and AR with weights (\beta 1 - \beta).
    """

    _build_ar = build_build_autoregressive_gaussian_proposal(mu, C)

    def _build(state: SMCStatebis, log_tgt_density_fn: LogDensity, log_likelihood_fn: LogDensity, i: int,
               j: Optional[int] = None):
        beta = state.mh_proposal_parameters.at[i - 1, 0].get()
        # Construct the RWM proposal.
        gammas = state.mh_proposal_parameters.at[:, 1].get()
        _state_reduced_for_gaussian_rwmh = SMCStatebis(
            state.particles,
            state.proposed_particles,
            state.log_weights,
            gammas,
            state.tempering_sequence,
            state.others
        )
        gaussian_rwmh_cov_log_proposal, gaussian_rwmh_sampler, _ = build_gaussian_rwmh_cov_proposal_gamma(
            _state_reduced_for_gaussian_rwmh, log_tgt_density_fn, log_likelihood_fn, i, j)
        # Construct the AR proposal.
        rhos = state.mh_proposal_parameters.at[:, 2].get()
        _state_reduced_for_ar = SMCStatebis(
            state.particles,
            state.proposed_particles,
            state.log_weights,
            rhos,
            state.tempering_sequence,
            state.others
        )
        gaussian_ar_log_proposal, gaussian_ar_sampler, _ = _build_ar(_state_reduced_for_ar, log_tgt_density_fn,
                                                                     log_likelihood_fn, i, None)

        mixture_ar_rwm_log_proposal = mixture_log_proposal(gaussian_rwmh_cov_log_proposal, gaussian_ar_log_proposal,
                                                           beta)

        def mixture_ar_rwm_sampler(key, x):
            """
            Toss a coin with probability \beta
            """
            toss = jax.random.bernoulli(key, beta)
            sample = jax.lax.cond(toss, lambda _: gaussian_rwmh_sampler(*_), lambda _: gaussian_ar_sampler(*_),
                                  (key, x))
            return sample

        return mixture_ar_rwm_log_proposal, mixture_ar_rwm_sampler, jnp.empty(1)

    return _build
