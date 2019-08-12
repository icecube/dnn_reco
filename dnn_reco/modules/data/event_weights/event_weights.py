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


def event_selection_weight(config, data_handler, data_transformer,
                           shared_objects, *args, **kwargs):
    """Event weights for event selection models.

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
    event_weights = tf.zeros_like(shared_objects['x_misc'][:, 0])

    # get Corsika weights
    for key in config['event_weights_corsika_keys']:
        weights = shared_objects['x_misc'][:, data_handler.get_misc_index(key)]
        weights /= config['event_weights_num_corsika_files']
        event_weights += weights

    # get MuonGun weights
    for key in config['event_weights_muongun_keys']:
        weights = shared_objects['x_misc'][:, data_handler.get_misc_index(key)]
        weights /= config['event_weights_num_muongun_files']
        event_weights += weights

    # get NuGen weights
    for key in config['event_weights_nugen_keys']:
        weights = shared_objects['x_misc'][:, data_handler.get_misc_index(key)]
        weights /= config['event_weights_num_nugen_files']
        event_weights += weights

    # weight for 1 year of livetime
    event_weights *= 86400 * 365

    return tf.expand_dims(event_weights, axis=-1)
