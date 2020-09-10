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


def astroness(input_data, config, label_names=None, *args, **kwargs):
    """Astroness of an event.

    Will create the label 'astroness' which is defined as:

        Neutrinos:
            astroness = (weight_astro) / (weight_astro + weight_conv)
        Other:
            astroness = 0.

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
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

    if 'labels_starting_cascades_lengths' in config:
        lengths = [int(l) for l in config['labels_starting_cascades_lengths']]
    else:
        lengths = [25, 50, 100]

    with pd.HDFStore(input_data,  mode='r') as f:
        _primary = f['MCPrimary']
        try:
            _weights = f['weights_mese']
        except KeyError as error:
            # load dummy weights, this should only happen with Corsika files,
            # e.g. astroness should be all zero, which is checked by assert
            _weights = f['weights']

    # check if event is a neutrino
    abs_pdg_code = np.abs(_primary['pdg_encoding'])
    mask = np.logical_or(abs_pdg_code == 12, abs_pdg_code == 14)
    is_neutrino = np.logical_or(mask, abs_pdg_code == 16)

    if 'weight_E250' in _weights:
        astroness = (_weights['weight_E250'] /
                     (_weights['weight_E250'] + _weights['weight_conv']))
        assert (is_neutrino).all(), 'Expected only Neutrinos!'
    elif 'weights_mese_flux' in _weights:
        astroness = (
            _weights['weights_mese_flux'] /
            (_weights['weights_mese_flux'] +
             _weights['weights_honda2006_gaisserH4a_elbert_conv_NNFlux']))
        assert (is_neutrino).all(), 'Expected only Neutrinos!'
    else:
        astroness = np.zeros_like(is_neutrino)
        assert np.sum(is_neutrino) == 0, 'Expected no Neutrinos!'

    if label_names is None:
        label_names = ['astroness']

    if label_names != ['astroness']:
        raise ValueError('Labels {!r} != [astroness]'.format(label_names))

    labels = [astroness]

    labels = np.array(labels, dtype=config['np_float_precision']).T

    return labels, label_names


def starting_cascades(input_data, config, label_names=None, *args, **kwargs):
    """Starting cascades at various distanes to IceCube convex hull.

    Will create the following labels:
        - p_starting_cascade_L{l}_Dm60:
            Neutrino event with hull distance < -60 meter and Length < l meter
        - p_starting_cascade_L{l}_D0:
            Neutrino event with hull distance < 0 meter and Length < l meter
        - p_starting_cascade_L{l}_D60:
            Neutrino event with hull distance < 60 meter and Length < l meter
        - p_starting_cascade_L{l}_D300:
            Neutrino event with hull distance < 300 meter and Length < l meter
        - p_starting_cascade_L{l}_Dinf:
            Neutrino event with hull distance < inf meter and Length < l meter

    for each length l in config['labels_starting_cascades_lengths'].
    If 'labels_starting_cascades_lengths' is not provided in config, the
    default values: [25, 50, 100] will be chosen.
    To change the distances which should be added,
    config['labels_starting_cascades_distances'] may be added. Possible
    elements of this list are [-60., 0., 60, 150., 300., float('inf')]

    Parameters
    ----------
    input_data : str
            Path to input data file.
    config : dict
        Dictionary containing all settings as read in from config file.
        Must contain:
            'data_handler_label_key': str
                The hdf5 key from which the labels will be loaded.
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

    if 'labels_starting_cascades_lengths' in config:
        lengths = [int(l) for l in config['labels_starting_cascades_lengths']]
    else:
        lengths = [25, 50, 100]

    if 'labels_starting_cascades_distances' in config:
        distances = [
            float(l) for l in config['labels_starting_cascades_distances']]
    else:
        distances = [-60., 0., 60, 300., float('inf')]

    # make sure the provdied distances are allowed
    allowed_distances = [-60., 0., 60, 150., 300., float('inf')]
    assert np.all([d in allowed_distances for d in distances]), distances

    with pd.HDFStore(input_data,  mode='r') as f:
        if 60. in distances:
            _labels_p60 = f['LabelsDeepLearning_p60']
        if -60. in distances:
            _labels_m60 = f['LabelsDeepLearning_m60']
        if 150. in distances:
            _labels_p150 = f['LabelsDeepLearning_p150']
        if 0. in distances or 300. in distances or float('inf') in distances:
            _labels = f['LabelsDeepLearning']
        _primary = f['MCPrimary']

    # check if event is a neutrino
    abs_pdg_code = np.abs(_primary['pdg_encoding'])
    mask = np.logical_or(abs_pdg_code == 12, abs_pdg_code == 14)
    is_neutrino = np.logical_or(mask, abs_pdg_code == 16)
    cascade_lengths = _labels['Length']

    label_name_base = 'p_starting_cascade_L{}_D{}'
    label_dict = {}
    label_names_list = []
    for l in lengths:

        mask = np.logical_and(is_neutrino, cascade_lengths <= l)

        # get hull distance -60m
        if -60. in distances:
            name = label_name_base.format(l, 'm60')
            label_names_list.append(name)
            label_dict[name] = np.logical_and(mask, _labels_m60['p_starting'])

        # get hull distance 0m
        if 0. in distances:
            name = label_name_base.format(l, '0')
            label_names_list.append(name)
            label_dict[name] = np.logical_and(mask, _labels['p_starting'])

        # get hull distance 60m
        if 60. in distances:
            name = label_name_base.format(l, '60')
            label_names_list.append(name)
            label_dict[name] = np.logical_and(mask, _labels_p60['p_starting'])

        # get hull distance 150m
        if 150. in distances:
            name = label_name_base.format(l, '150')
            label_names_list.append(name)
            label_dict[name] = np.logical_and(mask, _labels_p150['p_starting'])

        # get hull distance 300m
        if 300. in distances:
            name = label_name_base.format(l, '300')
            label_names_list.append(name)
            label_dict[name] = np.logical_and(mask, _labels['p_starting_300m'])

        # get inf hull distance
        if float('inf') in distances:
            name = label_name_base.format(l, 'inf')
            label_names_list.append(name)
            label_dict[name] = mask

    if label_names is None:
        label_names = label_names_list

    labels = []
    for name in label_names:
        if name in label_dict:
            labels.append(label_dict[name])
        else:
            raise KeyError('Label {!r} does not exist!'.format(name))

    labels = np.array(labels, dtype=config['np_float_precision']).T

    return labels, label_names


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

    if label_names is None:
        label_names = ['is_upgoing_track', 'is_neutrino_selection']
        labels = [is_upgoing_track, is_neutrino_selection]
    else:
        labels = []
        for name in label_names:
            if name == 'is_upgoing_track':
                labels.append(is_upgoing_track)
            elif name == 'is_neutrino_selection':
                labels.append(is_neutrino_selection)
            else:
                raise KeyError('Label {!r} does not exist!'.format(name))

    labels = np.array(labels, dtype=config['np_float_precision']).T

    return labels, label_names
