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
        If label_names is None (e.g. first call to initiate name list), then
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

import numpy as np

from dnn_reco.modules.data.labels.default_labels import simple_label_loader


def muon_scattering(input_data, config, label_names=None, *args, **kwargs):
    """Default muon scattering labels in addition to relaxed thresholds for
    classification.

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
    labels, label_names_def = simple_label_loader(
        input_data, config, label_names=None, *args, **kwargs
    )

    if "labels_muon_scattering_defs" not in config:
        config["labels_muon_scattering_defs"] = {}

    label_dict = {}
    for i, name in enumerate(label_names_def):
        label_dict[name] = labels[:, i]

    for def_name, scattering_def in config[
        "labels_muon_scattering_defs"
    ].items():
        passed_cuts = np.ones_like(labels[:, 0], dtype=bool)
        for key, value in scattering_def.items():
            passed_cuts = np.logical_and(passed_cuts, label_dict[key] >= value)
        label_dict[def_name] = passed_cuts

    if label_names is None:
        label_names = sorted(label_dict.keys())

    labels = []
    for name in label_names:
        labels.append(label_dict[name])

    labels = np.array(labels, dtype=config["np_float_precision"]).T

    return labels, label_names
