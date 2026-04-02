import os
from datetime import datetime

import jax.numpy as jnp
import jax.profiler
import jax.random
import yaml

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.SMC import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.experiments.comparison.anisotropic_gaussian.problem import construct_my_prior_and_target
from adaptive_smc.save_and_read_and_postprocess import save

"""
This methodology fails to detect the small temperature regime, why?
"""

OP_key = jax.random.PRNGKey(2)
_, key = jax.random.split(OP_key)

jax.config.update("jax_enable_x64", True)


def default_title(prefix=''):
    now = datetime.now()

    output_path = f"{prefix}_{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def experiment_ar(config, keys, dim):
    rho_grid = jnp.linspace(0, 0.99, 100)
    config.update({'dim': dim})
    CovProposal = jnp.eye(dim)

    target_ess = config.get('target_ess', None)
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    critical_temperature = 0.5 * (config.get('problem').get('tau') ** (-2) - 1) ** (-1) * 1 / jnp.linalg.trace(
        CovProposal)
    # my_tempering_sequence = jnp.linspace(0, min(1, 5*critical_temperature), tempering_length)

    params_optimization_method = {"grid": rho_grid, "batch_size": 10}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(config)
    tempering_length = config.get('tempering_length', dim + 5)
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    init_param = jnp.array([0])
    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_autoregressive_gaussian_proposal",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param, "dim": dim})

    my_proposal = getattr(proposals, config['proposal'])(jnp.zeros(dim), CovProposal)
    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method,
                                               grid_criteria=params_optimization_method['grid'],
                                               batch_size_criteria=10)

    if config.get('low_memory', False):
        @jax.vmap
        def wrapper_smc(key):
            return smc.low_memory_sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence,
                                         target_ess,
                                         b16=config.get('b16', False))
    else:
        @jax.vmap
        def wrapper_smc(key):
            return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                              save_disk_mem=True)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')))


if __name__ == "__main__":
    yaml_file = "g_pcn_regime.yaml"
    with open(yaml_file, "r") as file:
        y_config = yaml.safe_load(file)
    for name_of_my_config, config in y_config.items():
        if config.get('run', True):
            sequential_repetitions = config.pop('sequential_repetitions', 1)
            parallel_repetitions = config.get('parallel_repetitions')
            seq_keys = jax.random.split(key, sequential_repetitions)
            all_keys = jax.vmap(lambda k: jax.random.split(k, parallel_repetitions))(seq_keys)
            _, key = jax.random.split(seq_keys.at[-1].get())
            for keys in all_keys:
                if name_of_my_config == 'config_lm':
                    for dim in [1, 2, 3, 5, 10, 15, 30]:
                        experiment_ar(config, keys, dim)
                else:
                    for dim in [50, 75, 100]:
                        experiment_ar(config, keys, dim)
