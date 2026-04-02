import os
import pickle


def save(res, config, output_path=""):
    r"""
    Saving in a PKL file the config dictionnary and the output of the SMC sampler
    as given by GenericWasteFreeTemperingSMC.
    .

    """
    # Extract directory from output_path
    directory = os.path.dirname(output_path)

    # Create directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_path, 'wb') as handle:
        pickle.dump({'config': config, 'res': res}, handle, protocol=pickle.HIGHEST_PROTOCOL)
