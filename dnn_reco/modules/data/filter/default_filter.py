import numpy as np
"""
All filter functions must have the following parameters and return values:

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    X_IC79 : numpy.ndarray
        DOM input data of main IceCube array.
        shape: [batch_size, 10, 10, 60, num_bins]
    X_DeepCore : numpy.ndarray
        DOM input data of DeepCore array.
        shape: [batch_size, 8, 60, num_bins]
    labels : numpy.ndarray
        Labels.
        shape: [batch_size] + label_shape
    misc : numpy.ndarray
        Misc variables.
        shape: [batch_size] + misc_shape
    time_range_start : numpy.ndarray
        Time offset for relative timing for each event.
        shape: [batch_size, 1]
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    numpy.ndarray of type bool
        The boolean mask to filter out events.
        shape: [batch_size]
"""


def dummy_filter(input_data, config,  X_IC79, X_DeepCore, labels, misc,
                 time_range_start, *args, **kwargs):
    """Dummy filter that accepts all events.

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    X_IC79 : numpy.ndarray
        DOM input data of main IceCube array.
        shape: [batch_size, 10, 10, 60, num_bins]
    X_DeepCore : numpy.ndarray
        DOM input data of DeepCore array.
        shape: [batch_size, 8, 60, num_bins]
    labels : numpy.ndarray
        Labels.
        shape: [batch_size] + label_shape
    misc : numpy.ndarray
        Misc variables.
        shape: [batch_size] + misc_shape
    time_range_start : numpy.ndarray
        Time offset for relative timing for each event.
        shape: [batch_size, 1]
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    numpy.ndarray of type bool
        The boolean mask to filter out events.
        shape: [batch_size]

    """
    return np.ones(len(time_range_start), dtype=bool)
