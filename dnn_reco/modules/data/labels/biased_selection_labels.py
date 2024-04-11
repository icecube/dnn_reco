from __future__ import division, print_function
import h5py
import pandas as pd
import numpy as np

from dnn_reco.modules.data.labels.default_labels import simple_label_loader

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


def biased_muongun(input_data, config, label_names=None, *args, **kwargs):
    """Default muon scattering labels in addtion to relaxed thresholds for
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

    # define keys to load
    if "labels_biased_selection_filter_keys" in config:
        filter_mask_keys = config["labels_biased_selection_filter_keys"]
    else:
        filter_mask_keys = ["CascadeFilter", "MuonFilter"]

    if "labels_biased_selection_dnn_label_names" in config:
        dnn_labels_names = config["labels_biased_selection_dnn_label_names"]
    else:
        base = "DeepLearningReco_"
        dnn_labels_names = {
            base
            + "event_selection_cscdl3_300m_01": [
                ("p_starting_300m", 0.8),
            ],
            base
            + "event_selection_dnn_cscd_l3a_starting_events_03": [
                ("p_starting_300m", 0.5),
            ],
        }

    # create empty label dict
    label_dict = {}

    try:
        # special handling for FilterMask keys
        # FilterMask columns often have the year appended, e.g. _12
        # Here, we map a filter_name_12 to filter_name
        with h5py.File(input_data, "r") as f:
            _mask = f["FilterMask"][:]
        for filter_mask_key in filter_mask_keys:
            for filter_name in _mask.dtype.fields.keys():
                if filter_mask_key in filter_name:
                    label_dict["p_passed_" + filter_mask_key] = _mask[
                        filter_name
                    ][:, 1]

        # load CscdL3 and CscdL3_Cont_Tag and all defined dnn_labels
        with pd.HDFStore(input_data, mode="r") as f:

            _cscd_l3 = f["CscdL3"]["value"]
            _cscd_l3_cont_tag = f["CscdL3_Cont_Tag"]["value"]
            label_dict["p_passed_cscdl3"] = _cscd_l3
            label_dict["p_passed_cscdl3_cont"] = np.logical_and(
                _cscd_l3_cont_tag > 0, _cscd_l3
            )

            for model_name, values in dnn_labels_names.items():
                _dnn = f[model_name]
                for name, cut in values:
                    label_name = "{}_{}".format(
                        model_name.replace("DeepLearningReco_", ""), name
                    )
                    score = _dnn[name]
                    label_dict["p_passed_" + label_name] = score > cut
                    label_dict[label_name] = score

    except KeyError as k:
        print(k)
        print("Skipping file")

    if label_names is None:
        label_names = sorted(label_dict.keys())

    labels = []
    for name in label_names:
        labels.append(label_dict[name])

    labels = np.array(labels, dtype=config["np_float_precision"]).T

    return labels, label_names
