"""
All filter functions must have the following parameters and return values:

    Parameters
    ----------
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    x_IC78 : numpy.ndarray
        DOM input data of main IceCube array.
        shape: [batch_size, 10, 10, 60, num_bins]
    x_deepcore : numpy.ndarray
        DOM input data of DeepCore array.
        shape: [batch_size, 8, 60, num_bins]
    labels : numpy.ndarray
        Labels.
        shape: [batch_size] + label_shape
    misc_data : numpy.ndarray
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

from __future__ import division, print_function
import numpy as np


def dummy_filter(
    data_handler,
    input_data,
    config,
    x_ic70,
    x_deepcore,
    labels,
    misc_data,
    time_range_start,
    *args,
    **kwargs
):
    """Dummy filter that accepts all events.

    Parameters
    ----------
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    x_ic70 : numpy.ndarray
        DOM input data of main IceCube array.
        shape: [batch_size, 10, 10, 60, num_bins]
    x_deepcore : numpy.ndarray
        DOM input data of DeepCore array.
        shape: [batch_size, 8, 60, num_bins]
    labels : numpy.ndarray
        Labels.
        shape: [batch_size] + label_shape
    misc_data : numpy.ndarray
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


def general_filter(
    data_handler,
    input_data,
    config,
    x_ic70,
    x_deepcore,
    labels,
    misc_data,
    time_range_start,
    *args,
    **kwargs
):
    """A general filter.

    Filters events according to the key value pairs defined in the config:

        filter_equal:
            For events to pass this filter, the following must be True:
                    misc_data[key] == value
        filter_greater_than:
            For events to pass this filter, the following must be True:
                    misc_data[key] > value
        filter_less_than:
            For events to pass this filter, the following must be True:
                    misc_data[key] < value

    Optionally, a list of MC Primary PDG encodings may be provided to limit
    the application of the filter to events with the provided PDG encodings.
    The PDG encodings are provided via config['filter_apply_on_pdg_encodings'].

    Parameters
    ----------
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    x_ic70 : numpy.ndarray
        DOM input data of main IceCube array.
        shape: [batch_size, 10, 10, 60, num_bins]
    x_deepcore : numpy.ndarray
        DOM input data of DeepCore array.
        shape: [batch_size, 8, 60, num_bins]
    labels : numpy.ndarray
        Labels.
        shape: [batch_size] + label_shape
    misc_data : numpy.ndarray
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

    Raises
    ------
    NotImplementedError
        Description

    """
    mask_true = np.ones(len(time_range_start), dtype=bool)

    # go through equal mask
    if "filter_equal" in config:
        for key, value in config["filter_equal"].items():
            mask_true = np.logical_and(
                mask_true,
                misc_data[:, data_handler.misc_name_dict[key]] == value,
            )

    # go through greater than mask
    if "filter_greater_than" in config:
        for key, value in config["filter_greater_than"].items():
            mask_true = np.logical_and(
                mask_true,
                misc_data[:, data_handler.misc_name_dict[key]] > value,
            )

    # go through less than mask
    if "filter_less_than" in config:
        for key, value in config["filter_less_than"].items():
            mask_true = np.logical_and(
                mask_true,
                misc_data[:, data_handler.misc_name_dict[key]] < value,
            )

    # Only run the filter on events with MC Primaries of specified PDG encoding
    if "filter_apply_on_pdg_encodings" in config:
        pdg_encodings = config["filter_apply_on_pdg_encodings"]

        if pdg_encodings:
            pdg_key = "MCPrimary_pdg_encoding"

            pdg_values = misc_data[:, data_handler.misc_name_dict[pdg_key]]
            mask_pdg = np.array([p not in pdg_encodings for p in pdg_values])

            # undo filtering of events that are not of specifided PDG encoding
            mask_true[mask_pdg] = True

    return mask_true
