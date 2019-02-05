from __future__ import division, print_function
import pandas as pd
import numpy as np
"""
All misc functions must have the following parameters and return values:

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
        Misc function specific settings can be passed via the config file.
    misc_names : None, optional
        The names of the misc variables. This defines which variables to
        include as well as the ordering.
        If misc_names is None (e.g. first call to initate name list), then
        a list of misc names needs to be created and returned.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    np.ndarray
        The numpy array containing the misc variables.
        Shape: [batch_size] + misc_shape
    list of str
        The names of the misc variables.
"""


def dummy_misc_loader(input_data, config, misc_names=None, *args, **kwargs):
    """Dummy misc loader.

    Dummy function that does not do anything.

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    misc_names : None, optional
        The names of the misc variables. This defines which variables to
        include as well as the ordering.
        If misc_names is None (e.g. first call to initate name list), then
        a list of misc names needs to be created and returned.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    None
        The misc variables.
    empty list
        The names of the misc variables.
    """
    return None, []
