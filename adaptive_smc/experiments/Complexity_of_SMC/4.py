r"""
The initial distribution is N(0, C0), the target distribution is
N(0, C1)N(0, C0) (product of densities),
where C0, C1 are two random scaled positive covariance matrices
so that they are well-conditioned (e.g., with conditioning number independent on d)

The sequence of distributions is the tempered interpolation with temperature
obtained by automatically setting ESS >= N/2.

The Markov kernels are Metropolis Hastings kernel with RWM set to 2.38^2/d.

The number of particles is either set to
    i) M = O(1), P = C log(T/eta) T^2/(eps^2 gamma), where
        T is an upper bound on the number of required tempering steps,
        e.g., set to
        T = sqrt(d) x constant
        gamma = d
    ii) M = O(1), P = C T^2/(eps^2 gamma), and run J = log(T/eta)
    independent samplers.

The saved output is the weights.
"""

from adaptive_smc.SMC import GenericWasteFreeTemperingSMC
from adaptive_smc.experiments_bis.paper_complexity.heavy_tail_increment_weights_gaussians import make_model
from adaptive_smc.experiments_bis.paper_complexity.proposal import \
    build_gaussian_rw_proposal_fixed_scaling
from adaptive_smc.experiments_bis.paper_complexity.save import save

import os
from datetime import datetime
import jax
import jax.numpy as jnp
import yaml

jax.config.update("jax_enable_x64", False)

OP_key = jax.random.PRNGKey(12)
_, key = jax.random.split(OP_key)


def default_title(prefix='', eps=None):
    now = datetime.now()
    filename = (
        f"{prefix}_eps{eps:.4f}_"
        f"{os.path.basename(__file__)}_"
        f"{now.strftime('%m%d%H%M%S')}.pkl"
    )
    return filename


def xp(config, eps, keys):
    # Copy config so we never mutate the original
    run_config = dict(config)
    
    dim = run_config["dim"]
    target_ess = run_config.get("ess", None)
    

    T = int(jnp.sqrt(dim))
    gamma = 1 / dim

    num_parallel_chain = 20
    num_mcmc_steps = int(run_config["num_mcmc_steps"] * dim ** 2.0 // 2) #int(T ** 2 / (eps ** 2 * gamma) * jnp.log(T * 5)/500)
    tempering_length = T * 2

    tempering_sequence = jnp.linspace(0, 1, tempering_length)

    # Generate model
    model_key = jax.random.PRNGKey(0)
    loglikelihood_fn, base_measure_sampler, logbase_density_fn, true_log_Z, _, _ = make_model(dim, config.get('heavy_factor', 1.0), model_key, config.get('mean', None))

    run_config.update({
        "eps": eps,
        "P": num_mcmc_steps,
        "M": num_parallel_chain,
        "logZ": true_log_Z,
        "tempering_sequence": tempering_sequence
    })

    smc = GenericWasteFreeTemperingSMC(
        logbase_density_fn,
        base_measure_sampler,
        loglikelihood_fn,
        build_gaussian_rw_proposal_fixed_scaling
    )

    @jax.jit
    @jax.vmap
    def wrapper_smc(key):
        return smc.low_memory_sample(
            key,
            num_parallel_chain,
            num_mcmc_steps,
            tempering_sequence,
            target_ess
        )

    res = wrapper_smc(keys)

    output_file = (
            run_config["OUTPUT_PATH"]
            + default_title(run_config.get("prefix", ""), eps)
    )

    save(res, run_config, output_file)

if __name__ == "__main__":

    yaml_file = "4.yaml"

    with open(yaml_file, "r") as file:
        y_config = yaml.safe_load(file)

    for name_of_my_config, config in y_config.items():

        if config.get("run", True):

            # Fix epsilon for this scaling experiment
            eps = config.get("eps", 0.1)

            # Set a list of dimensions to vary
            dim_list = config.get("dim_list", [5])

            sequential_repetitions = config.pop("sequential_repetitions", 1)
            parallel_repetitions = config["parallel_repetitions"]

            seq_keys = jax.random.split(key, sequential_repetitions)

            all_keys = jax.vmap(
                lambda k: jax.random.split(k, parallel_repetitions)
            )(seq_keys)

            _, key = jax.random.split(seq_keys[-1])

            # Loop over dimensions
            for dim in dim_list:
                # update config with current dim
                config["dim"] = dim

                for keys in all_keys:
                    xp(config, eps, keys)