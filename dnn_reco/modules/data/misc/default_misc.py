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
        If misc_names is None (e.g. first call to initiate name list), then
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

import h5py
import pandas as pd
import numpy as np


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
        If misc_names is None (e.g. first call to initiate name list), then
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


def general_misc_loader(input_data, config, misc_names=None, *args, **kwargs):
    """A general misc variable loader.

    Loads values as specified in config.

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
    misc_names : None, optional
        The names of the misc variables. This defines which variables to
        include as well as the ordering.
        If misc_names is None (e.g. first call to initiate name list), then
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

    # create list of misc names
    if misc_names is None:
        misc_names = []
        for key, col_list in config["misc_load_dict"].items():
            if not isinstance(col_list, list):
                col_list = [col_list]
            for col in col_list:
                misc_names.append(key + "_" + col)
        misc_names = sorted(misc_names)

    if len(misc_names) == 0:
        misc_values = None
    else:
        with pd.HDFStore(input_data, mode="r") as f:
            num_events = len(f[config["data_handler_time_offset_name"]])

        misc_dict = {}
        for key, col_list in config["misc_load_dict"].items():
            if not isinstance(col_list, list):
                col_list = [col_list]
            for col in col_list:

                # special handling for FilterMask keys
                # FilterMask columns often have the year appended, e.g. _12
                # Here, we map a filter_name_12 to filter_name
                if key == "FilterMask" or key == "QFilterMask":
                    with h5py.File(input_data, "r") as f:
                        _mask = f[key][:]
                    for filter_name in _mask.dtype.fields.keys():
                        if col in filter_name:
                            misc_dict[key + "_" + col] = _mask[filter_name][
                                :, 1
                            ]

                # Standard hdf5 keys
                else:
                    try:
                        with pd.HDFStore(input_data, mode="r") as f:
                            misc_dict[key + "_" + col] = f[key][col].values

                    except KeyError as e:
                        # check if a fill value is defined
                        if key + "_" + col in config["misc_fill_values"]:
                            misc_dict[key + "_" + col] = np.tile(
                                config["misc_fill_values"][key + "_" + col],
                                num_events,
                            )
                        else:
                            raise e

        misc_values = [misc_dict[k] for k in misc_names]
        misc_values = np.array(
            misc_values, dtype=config["np_float_precision"]
        ).T

    return misc_values, misc_names
