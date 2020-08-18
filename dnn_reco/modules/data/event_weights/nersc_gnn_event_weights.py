from __future__ import division, print_function
import tensorflow as tf


"""
All event weighting functions must have the following signature:

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
        A dictionary containg settings and objects that are shared and passed
        on to sub modules.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    tf.Tensor
        A tensorflow tensor containing the event weights for each event.
        The loss for each event will be multiplied by this event weight.
        Shape: [batch_size, 1]
"""


def nersc_gnn_weight(config, data_handler, data_transformer, shared_objects,
                     *args, **kwargs):
    """Event weight as calculated for nersc gnn comparison.

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
        A dictionary containg settings and objects that are shared and passed
        on to sub modules.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    tf.Tensor
        A tensorflow tensor containing the loss for each label.
        Shape: label_shape (same shape as labels)

    """
    # get indices
    index_weight = data_handler.get_misc_index('nersc_gnn_info_weight')
    index_signal = data_handler.get_label_index('is_signal')

    # weight for 10 years of livetime
    weight = shared_objects['x_misc'][:, index_weight] * 86400 * 365 * 10
    is_nugen = shared_objects['y_true'][:, index_signal] > .5

    event_weights = tf.compat.v1.where(
        is_nugen,
        weight / config['nersc_gnn_weight_num_nugen_files'],
        weight / config['nersc_gnn_weight_num_corsika_files'],
        )

    return tf.expand_dims(event_weights, axis=-1)
