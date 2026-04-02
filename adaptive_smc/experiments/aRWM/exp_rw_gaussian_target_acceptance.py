import os
from datetime import datetime

import jax.numpy as jnp
import jax.random
import yaml

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.SMC import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.experiments.aRWM.problem import construct_my_prior_and_target
from adaptive_smc.save_and_read_and_postprocess import save

OP_key = jax.random.PRNGKey(0)
_, key = jax.random.split(OP_key)


def default_title(prefix=''):
    now = datetime.now()

    output_path = f"{prefix}_{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def experiment_rwmh(config, keys):
    fun_to_be_applied_to_the_mh_ratio_in_the_criteria = lambda x: (x - 0.234)
    fun_to_be_applied_to_the_criteria = lambda x: -jnp.abs(x)
    dim = config.get('dim')

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length', 10 + dim)
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    params_optimization_method = {"grid": jnp.linspace(0.01, 5, 500)}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_my_prior_and_target(config)

    init_param = jnp.array([2.38])
    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_gaussian_rwmh_cov_proposal_gamma",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])
    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method,
                                               criteria_function=lambda w, x, y, z: 1.,
                                               fun_to_be_applied_to_the_mh_ratio_in_the_criteria=fun_to_be_applied_to_the_mh_ratio_in_the_criteria,
                                               fun_to_be_applied_to_the_criteria=fun_to_be_applied_to_the_criteria)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')))


if __name__ == "__main__":

    yaml_file = "exp_rw_gaussian_target_acceptance.yaml"
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
                experiment_rwmh(config, keys)
