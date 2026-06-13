r"""
Adaptive waste-free SMC with an uncoupled pCN proposal on a Gaussian target,
at fixed dimension. The MH proposal parameters are (\rho, \tau):

    q(y | x) = N(\mu + \rho (x - \mu), \tau^2 C),

with \rho and \tau uncoupled. The ESJD criteria is computed on a 2d
(\rho, \tau) grid; \tau^2 + \rho^2 = 1 is the pCN frontier (on it the
proposal preserves N(\mu, C)).

The output pkl is read by heatmap_esjd.ipynb.
"""
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import yaml

from adaptive_smc import optimise
from adaptive_smc.SMC import GenericAdaptiveWasteFreeTemperingSMC
from adaptive_smc.experiments_bis.problems import construct_gaussian_problem
from adaptive_smc.proposals.pCN_ARW import build_build_uncoupled_autoregressive_gaussian_proposal
from adaptive_smc.save_and_read_and_postprocess import save

OP_key = jax.random.PRNGKey(0)
_, key = jax.random.split(OP_key)

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))


def default_title(prefix=''):
    now = datetime.now()
    stamp = now.strftime('%m%d%H%M%S')
    return f"{prefix}_{os.path.basename(__file__)}_{stamp}.pkl"


def experiment_uncoupled_pcn(config, keys):
    dim = config.get('dim')

    grid_config = config.get('grid')
    rho_grid = jnp.linspace(grid_config['rho_min'], grid_config['rho_max'], grid_config['n_rho'])
    tau_grid = jnp.linspace(grid_config['tau_min'], grid_config['tau_max'], grid_config['n_tau'])
    params_grid = jnp.array([[r, t] for r in rho_grid for t in tau_grid])

    target_ess = config.get('target_ess')
    num_parallel_chain = config.get('num_parallel_chain')
    num_mcmc_steps = config.get('num_mcmc_steps')
    batch_size_criteria = config.get('batch_size_criteria', 100)

    tempering_length = config.get('tempering_length', 10 + dim)
    my_tempering_sequence = jnp.linspace(0, 1, tempering_length)

    optimization_method_str = "make_optimize_within_a_fixed_grid"
    params_optimization_method = {"grid": params_grid, "batch_size": batch_size_criteria}

    loglikelihood_fn, base_measure_sampler, logbase_density_fn = construct_gaussian_problem(config)

    init_param = jnp.array([0., 1.])  # (rho, tau): independent N(mu, C) proposal

    config.update(
        {"optimization_method": optimization_method_str, "params_optimization_method": params_optimization_method,
         "proposal": "build_build_uncoupled_autoregressive_gaussian_proposal",
         "rho_grid": rho_grid, "tau_grid": tau_grid,
         "tempering_sequence": my_tempering_sequence,
         "init_param": init_param})

    my_proposal = build_build_uncoupled_autoregressive_gaussian_proposal(jnp.zeros(dim), jnp.eye(dim))
    optimization_method = getattr(optimise, optimization_method_str)(**params_optimization_method)

    smc = GenericAdaptiveWasteFreeTemperingSMC(logbase_density_fn, base_measure_sampler, loglikelihood_fn,
                                               my_proposal, optimization_method,
                                               grid_criteria=params_grid,
                                               batch_size_criteria=batch_size_criteria,
                                               min_ess_fraction_criteria=config.get('min_ess_fraction_criteria', 0.05))

    @jax.vmap
    def wrapper_smc(key):
        return smc.sample(key, num_parallel_chain, num_mcmc_steps, init_param, my_tempering_sequence, target_ess,
                          save_disk_mem=False)

    res = wrapper_smc(keys)
    output_dir = os.path.join(HERE, config.get('OUTPUT_PATH'))
    save(res, config, os.path.join(output_dir, default_title(config.get('prefix'))),
         config.get('compress_output', False))


if __name__ == "__main__":
    yaml_file = os.path.join(HERE, "exp_pcn_gaussian.yaml")
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
                experiment_uncoupled_pcn(config, keys)
