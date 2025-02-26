"""
This file contains commonly used utility functions to compute loss.
"""

import numpy as np
import tensorflow as tf

from dnn_reco import misc


def correct_azimuth_residual(
    y_diff_trafo, config, data_handler, data_transformer, name_pattern
):
    """Correct azimuth residuals for two pi periodicity.

    Parameters
    ----------
    y_diff_trafo : tf.Tensor
        The residuals between the transformed prediction and true values.
    config : dict
        Dictionary containing all settings as read in from config file.
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    data_transformer : :obj: of class DataTransformer
        An instance of the DataTransformer class. The object is used to
        transform data.
    name_pattern : str
        A name pattern specifying the labels that should be corrected.

    Returns
    -------
    tf.Tensor
        The residual tensor with corrected azimuth residual.
        Same shape and type as y_diff_trafo
    """
    misc.print_warning(
        "Correcting azimuth residuals for name pattern: {!r}".format(
            name_pattern
        )
    )

    # Assumes labels to be a vector, e.g. 1-dimensional
    assert len(data_handler.label_shape) == 1

    pi = (
        np.ones(
            [1] + data_handler.label_shape, dtype=config["np_float_precision"]
        )
        * np.pi
    )
    pi_trafo = tf.squeeze(
        data_transformer.transform(
            pi, data_type="label", bias_correction=False
        )
    )
    two_pi_trafo = tf.squeeze(
        data_transformer.transform(
            2 * pi, data_type="label", bias_correction=False
        )
    )

    # Get correct y_diff_trafo that accounts for 2pi periodicity
    y_diff_list = tf.unstack(y_diff_trafo, axis=-1)
    for i, name in enumerate(data_handler.label_names):
        if name_pattern in name:
            # sanity check: this correction does not work if log is applied
            assert (
                bool(data_transformer.trafo_model["log_label_bins"][i])
                is False
            )

            # Found an azimuth label: correct for 2 pi periodicity
            abs_diff = tf.abs(y_diff_list[i])
            y_diff_list[i] = tf.where(
                abs_diff < pi_trafo[i], abs_diff, two_pi_trafo[i] - abs_diff
            )

    return tf.stack(y_diff_list, axis=-1)


def get_y_diff_trafo(config, data_handler, data_transformer, shared_objects):
    """Get corrected transformed residuals.

    Parameters
    ----------
    config : dict
        Dictionary containing all settings as read in from config file.
    data_handler : :obj: of class DataHandler
        An instance of the DataHandler class. The object is used to obtain
        meta data.
    data_transformer : :obj: of class DataTransformer
        An instance of the DataTransformer class. The object is used to
        transform data.
    shared_objects : dict
        A dictionary containing settings and objects that are shared and passed
        on to sub modules.

    Returns
    -------
    tf.Tensor
        A tensorflow tensor containing the loss for each label.
        Shape: label_shape (same shape as labels)
    """
    y_diff_trafo = (
        shared_objects["y_pred_trafo"] - shared_objects["y_true_trafo"]
    )

    # correct azimuth residual for 2pi periodicity
    if config["label_azimuth_key"]:
        y_diff_trafo = correct_azimuth_residual(
            config=config,
            y_diff_trafo=y_diff_trafo,
            data_handler=data_handler,
            data_transformer=data_transformer,
            name_pattern=config["label_azimuth_key"],
        )
    return y_diff_trafo
