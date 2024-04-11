from __future__ import division, print_function
import tensorflow as tf

# Check and allow for newer TFScripts versions
try:
    from tfscripts.compat.v1 import layers as tfs
except ImportError:
    from tfscripts import layers as tfs
"""
This file contains commonly used utility functions to build icecube nn models.
"""


def preprocess_icecube_data(is_training, shared_objects):
    """Performs some basic preprocessing of IceCube input data.

    Applies drop out for whole DOMs.
    Reshapes and splits DeepCore input into two tensors with string dimension
    moved to channel dimension.

    Parameters
    ----------
    is_training : bool
        True if model is in training mode, false if in inference mode.
    shared_objects : dict
        A dictionary containg settings and objects that are shared and passed
        on to sub modules.

    Returns
    -------
    tf.Tensor
        Main IceCube array: IC78
        shape: [None, 10, 10, 60, num_bins]
    tf.Tensor
        Upper DeepCore array
        shape: [none, 1, 10, 8 * num_bins]
    tf.Tensor
        Lower DeepCore array
        shape: [none, 1, 50, 8 * num_bins]
    """
    X_IC78 = shared_objects["x_ic78_trafo"]
    X_DeepCore = shared_objects["x_deepcore_trafo"]
    keep_prob_list = shared_objects["keep_prob_list"]

    # -----------------------------------
    # DropOut on whole DOMs
    # -----------------------------------
    if is_training:
        noise_shape_IC78 = (
            [tf.shape(input=X_IC78)[0]]
            + X_IC78.get_shape().as_list()[1:-1]
            + [1]
        )
        noise_shape_DeepCore = (
            [tf.shape(input=X_DeepCore)[0]]
            + X_DeepCore.get_shape().as_list()[1:-1]
            + [1]
        )

        X_IC78 = tf.nn.dropout(
            X_IC78,
            rate=1 - (keep_prob_list[0]),
            noise_shape=noise_shape_IC78,
        )

        X_DeepCore = tf.nn.dropout(
            X_DeepCore,
            rate=1 - (keep_prob_list[0]),
            noise_shape=noise_shape_DeepCore,
        )

    # -----------------------------------
    # Reshape DeepCore:
    # Move strings in channel dimension:
    # only convolve over DOMs dimension
    # -----------------------------------
    X_DeepCore_upper = tf.transpose(
        a=X_DeepCore[:, :, 0:10, :], perm=[0, 2, 3, 1]
    )
    X_DeepCore_lower = tf.transpose(
        a=X_DeepCore[:, :, 10:, :], perm=[0, 2, 3, 1]
    )

    input_ch_size = X_DeepCore.get_shape().as_list()[-1]
    X_DeepCore_upper = tf.reshape(
        X_DeepCore_upper, [-1, 1, 10, input_ch_size * 8]
    )
    X_DeepCore_lower = tf.reshape(
        X_DeepCore_lower, [-1, 1, 50, input_ch_size * 8]
    )

    return X_IC78, X_DeepCore_upper, X_DeepCore_lower
