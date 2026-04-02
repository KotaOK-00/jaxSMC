import os
import pickle

import jax
import jax.lax
import jax.numpy as jnp

def save(res, config, output_path="", compress=False, rapid_pkl=False):
    r"""
    Saving in a PKL file the config dictionary and the output of the SMC sampler
    as given by GenericAdaptiveWasteFreeTemperingSMC.
    .

    if `compress` is True, the samples, weights and criterion are converted to float16
    to save disk space (about a 75\% storage saving).

    if `rapid_pkl` is True, there is a secondary pickle file created
    containing only the sequence of MH parameters, of temperatures, and log normalisation constants.
    """
    # Extract directory from output_path
    directory = os.path.dirname(output_path)

    # Create directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if compress:
        fields_to_compress = [0, 1, 2, 3, 5, 9]
        res_generator = (
            jax.tree_util.tree_map(lambda x: x.astype(jnp.float16), field) if idx in fields_to_compress else field
            for
        idx, field in enumerate(res))
        res = []
        for el in res_generator:
            res.append(el)
        res = tuple(res)

    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path+'_small', 'wb') as handle:
        pickle.dump({'config': config, 'res': (
            None, None, None, res[3], None, res[5], res[6], None, res[8], None
        )}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def acf(samples, max_order=20):
    """
    Agarwal, M., & Vats, D. (2022). Globally Centered Autocovariances in MCMC. Journal of Computational and Graphical Statistics, 31(3), 629–638. https://doi.org/10.1080/10618600.2022.2037433

    Make ACF given set of samples (either particles or
    function evaluated particles) of shape (n_parallel_run, n_iterations, n_chain, n_length_of_chain, dim).
    It transforms the samples into a shape (n_iterations, n_parallel_run*n_chain, n_length_of_chain, dim)
    And compute the AC for order 1 to max_order,
    using the global mean as the stationary mean of the chain, and
    averaging over n_parallel_run*n_chain chains the computed autocorrelations.
    """
    # Cast into float32 to avoid nan due to numerical imprecision
    # (in particular when dividing vark by var0)
    samples = samples.astype(jnp.float32)

    n_iter = samples.shape[1]
    dim = samples.shape[-1]
    samples = jnp.swapaxes(samples, 0, 1)
    samples = samples.reshape(
        (samples.shape[0], samples.shape[1] * samples.shape[2], samples.shape[3], samples.shape[4]))
    global_mean = samples.mean(axis=[1, 2])
    global_mean = global_mean.reshape((samples.shape[0], 1, 1, samples.shape[3]))
    diff0 = samples - global_mean
    diff0 = diff0.reshape((diff0.shape[0], diff0.shape[1], diff0.shape[2], 1, diff0.shape[3]))
    var0 = jnp.mean(diff0 @ jnp.swapaxes(diff0, -1, -2), axis=[2])

    def fcorr(k):
        diff = samples[:, :, :-k] - global_mean
        diffk = samples[:, :, k:] - global_mean
        diff = diff.reshape((diff.shape[0], diff.shape[1], diff.shape[2], 1, diff.shape[3]))
        diffk = diffk.reshape((diffk.shape[0], diffk.shape[1], diffk.shape[2], diffk.shape[3], 1))
        prod = diff @ diffk
        vark = jnp.mean(prod, axis=[2])
        corr = jnp.mean(vark / var0, axis=[-3])
        return corr

    acf_result = jnp.zeros((max_order, n_iter, dim, dim), dtype=jnp.float16)
    for order in range(1, max_order + 1):
        acf_result = acf_result.at[order - 1].set(fcorr(order))
    return acf_result
