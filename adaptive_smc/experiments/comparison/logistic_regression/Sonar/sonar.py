import os
from datetime import datetime

import jax.numpy as jnp
import jax.random
import yaml

from adaptive_smc import optimise
from adaptive_smc import proposals
from adaptive_smc.SMC import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.experiments.comparison.logistic_regression.Sonar.problem import construct_my_prior_and_target
from adaptive_smc.save_and_read_and_postprocess import save

OP_key = jax.random.PRNGKey(0)
_, key = jax.random.split(OP_key)

jax.config.update("jax_enable_x64", True)


def default_title(prefix=''):
    now = datetime.now()

    output_path = f"{prefix}_{os.path.basename(__file__)}_{now.strftime("%m%D%H%M%S").replace("/", "")}.pkl"
    return output_path


def experiment_pCN(config, keys):
    rho_grid = jnp.linspace(0, 0.95, 100)

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length')
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    params_optimization_method = {"grid": rho_grid, "batch_size": 10}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn, base_measure_mean, base_measure_cov = construct_my_prior_and_target(
        config)

    init_param = jnp.array([0])
    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_autoregressive_gaussian_proposal",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])(base_measure_mean, base_measure_cov)
    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method,
                                               grid_criteria=params_optimization_method['grid'],
                                               batch_size_criteria=10)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          save_disk_mem=False)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')),
         config.get('compress_output', False))


def experiment_adaptive_rw(config, keys):
    tau_grid = jnp.linspace(0.05, 4, 100)

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length')
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    params_optimization_method = {"grid": tau_grid, "batch_size": 10}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn, _, __ = construct_my_prior_and_target(config)
    tempering_length = config.get('tempering_length')
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

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
                                               grid_criteria=params_optimization_method['grid'],
                                               batch_size_criteria=10)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          save_disk_mem=False)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')),
         config.get('compress_output', False))


def experiment_arw(config, keys):
    tau_grid = jnp.linspace(0.05, 4, 100)
    rho_grid = jnp.linspace(0, 0.99, 100)

    tempering_length = config.get('tempering_length')
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    loglikelihood_fn, base_measure_sampler, logbase_density_fn, base_measure_mean, base_measure_cov = construct_my_prior_and_target(
        config)

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_grid = jnp.array([[x, y] for x in rho_grid for y in tau_grid])
    params_optimization_method = {"grid": params_grid, "batch_size": 10}

    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')
    target_ess = config.get('target_ess')

    init_param = jnp.array([0., 1.])

    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_uncoupled_autoregressive_gaussian_proposal",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])(base_measure_mean, base_measure_cov)

    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method, grid_criteria=params_grid,
                                               batch_size_criteria=10)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          save_disk_mem=False)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config['prefix']), config.get('compress_output', False))


def experiment_pCNL(config, keys):
    rho_grid = jnp.linspace(0, 0.95, 100)

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length')
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    params_optimization_method = {"grid": rho_grid, "batch_size": 10}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn, base_measure_mean, base_measure_cov = construct_my_prior_and_target(
        config)

    init_param = jnp.array([0])
    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_pCNL_proposal",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])(base_measure_mean, base_measure_cov)
    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method,
                                               grid_criteria=params_optimization_method['grid'],
                                               batch_size_criteria=10)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          save_disk_mem=False)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')),
         config.get('compress_output', False))



def experiment_MALA(config, keys):
    rho_grid = jnp.linspace(0, 0.95, 100)

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')

    optimization_method_str = "make_optimize_within_a_fixed_grid"

    tempering_length = config.get('tempering_length')
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    params_optimization_method = {"grid": rho_grid, "batch_size": 10}
    # params_optimization_method = {"minmax": [0.1, 10.], "interval": [-5., 5.], "n_iter":4}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn, base_measure_mean, base_measure_cov = construct_my_prior_and_target(
        config)

    init_param = jnp.array([0])
    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_MALA_proposal_gamma",
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = getattr(proposals, config['proposal'])(base_measure_cov)
    if config['optimization_method']:
        optimization_method = getattr(optimise, config['optimization_method'])(**params_optimization_method)
    else:
        optimization_method = None

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method,
                                               grid_criteria=params_optimization_method['grid'],
                                               batch_size_criteria=10)

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          save_disk_mem=False)

    res = wrapper_smc(keys)
    save(res, config, config.get('OUTPUT_PATH') + default_title(config.get('prefix')),
         config.get('compress_output', False))


if __name__ == "__main__":
    yaml_file = "sonar.yaml"
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
                experiment_MALA(config, keys)
                #experiment_pCNL(config, keys)
                #experiment_pCN(config, keys)
                #experiment_adaptive_rw(config, keys)
                #experiment_arw(config, keys)
