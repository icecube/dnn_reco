from __future__ import division, print_function
import pandas as pd
import numpy as np
"""
All label functions must have the following parameters and return values:

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
        Label function specific settings can be passed via the config file.
    label_names : None, optional
        The names of the labels. This defines which labels to include as well
        as the ordering.
        If label_names is None (e.g. first call to initate name list), then
        a list of label names needs to be created and returned.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    np.ndarray
        The numpy array containing the labels.
        Shape: [batch_size] + label_shape
    list of str
        The names of the labels
"""


def upgoing_tracks(input_data, config, label_names=None, *args, **kwargs):
    """Upgoing Tracks and general neutrino selection (upgoing or starting)

    Will create the following labels:
        - is_upgoing_track:
            upgoing track (zenith > 85 degree, length in p60 > 100)
        - is_neutrino_selection:
            is_upgoing_track or starting event within 300m of convex hull

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
        Must contain:
            'data_handler_label_key': str
                The hdf5 key from which the labels will be loaded.
            'label_add_dir_vec': bool
                If true, the direction vector components will be calculated
                on the fly and added to the labels. For this, the keys
                'label_azimuth_key' and 'label_zenith_key' have to be provided.
    label_names : None, optional
        The names of the labels. This defines which labels to include as well
        as the ordering.
        If label_names is None, then all keys except event specificers will
        be used.

    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    list of str
        The names of the labels
    """

    with pd.HDFStore(input_data,  mode='r') as f:
        _labels = f[config['data_handler_label_key']]
        _primary = f['MCPrimary']

    mask = _labels['PrimaryZenith'] > np.deg2rad(85)
    mask = np.logical_and(mask, _labels['LengthInDetector'] > 100)
    is_numu = np.logical_or(_primary['pdg_encoding'] == -14,
                            _primary['pdg_encoding'] == 14)
    is_upgoing_track = np.logical_and(mask, is_numu)
    is_starting = _labels['p_starting_300m'] == 1
    is_neutrino_selection = np.logical_or(is_upgoing_track, is_starting)

    label_names = ['is_upgoing_track', 'is_neutrino_selection']
    labels = [is_upgoing_track, is_neutrino_selection]

    labels = np.array(labels, dtype=config['np_float_precision']).T

    return labels, label_names
