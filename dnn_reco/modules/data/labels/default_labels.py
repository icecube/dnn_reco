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


def simple_label_loader(input_data, config, label_names=None, *args, **kwargs):
    """Simple Label Loader.

    Will load variables contained in the hdf5 field specified via
    config['data_handler_label_key'].

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
        time_offset = f[config['data_handler_time_offset_name']]['value']

    ignore_columns = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists']

    if config['label_add_dir_vec']:
        ignore_columns.extend(['direction_x', 'direction_y', 'direction_z'])

    if 'label_position_at_rel_time' in config and \
            config['label_position_at_rel_time'] is not None:
        ignore_columns.extend(['rel_pos_x', 'rel_pos_y', 'rel_pos_z'])

    if label_names is None:
        if 'label_keys_to_load' in config:
            name_list = config['label_keys_to_load']
        else:
            name_list = _labels.keys().tolist()
        label_names = [n for n in name_list if n not in ignore_columns]

    labels = [_labels[name] for name in label_names
              if name not in ignore_columns]

    # calculate direction vector components on the fly
    if config['label_add_dir_vec']:

        # get azimuth and zenith
        azimuth = _labels[config['label_azimuth_key']]
        zenith = _labels[config['label_zenith_key']]

        # calculate direction vector components
        # We need the negative values here, since (azimuth, zenith) points
        # towards the source, whereas the direction vector points in the
        # direction of the moving particle
        dir_x = -np.sin(zenith)*np.cos(azimuth)
        dir_y = -np.sin(zenith)*np.sin(azimuth)
        dir_z = -np.cos(zenith)

        # add direction vector components to labels
        label_names.extend(['direction_x', 'direction_y', 'direction_z'])
        labels.extend([dir_x, dir_y, dir_z])

    # calculate position at relative time t (only makes sense for tracks)
    if 'label_position_at_rel_time' in config and \
            config['label_position_at_rel_time'] is not None:
        dir_x = _labels[config['label_dir_x_key']]
        dir_y = _labels[config['label_dir_y_key']]
        dir_z = _labels[config['label_dir_z_key']]

        delta_t = (time_offset + config['label_position_at_rel_time']) - \
            _labels['VertexTime']
        c = 0.299792458  # m / ns
        length = c * delta_t
        x_at_t = _labels['VertexX'] + length * dir_x
        y_at_t = _labels['VertexY'] + length * dir_y
        z_at_t = _labels['VertexZ'] + length * dir_z

        # add position at relative time to labels
        label_names.extend(['rel_pos_x', 'rel_pos_y', 'rel_pos_z'])
        labels.extend([x_at_t, y_at_t, z_at_t])

    labels = np.array(labels, dtype=config['np_float_precision']).T

    return labels, label_names
