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
        A dictionary containing settings and objects that are shared and passed
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

import tensorflow as tf


def event_selection_weight(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
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
        A dictionary containing settings and objects that are shared and passed
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
    event_weights = tf.zeros_like(shared_objects["x_misc"][:, 0])

    # get Corsika weights
    for key in config["event_weights_corsika_keys"]:
        weights = shared_objects["x_misc"][:, data_handler.get_misc_index(key)]
        weights /= config["event_weights_num_corsika_files"]
        event_weights += weights

    # get MuonGun weights
    for key in config["event_weights_muongun_keys"]:
        weights = shared_objects["x_misc"][:, data_handler.get_misc_index(key)]
        weights /= config["event_weights_num_muongun_files"]
        event_weights += weights

    # get NuGen weights
    for key in config["event_weights_nugen_keys"]:
        weights = shared_objects["x_misc"][:, data_handler.get_misc_index(key)]
        weights /= config["event_weights_num_nugen_files"]
        event_weights += weights

    # weight for 1 year of livetime
    event_weights *= 86400 * 365

    return tf.expand_dims(event_weights, axis=-1)


def clipped_astroness_weights(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Clipped "Astroness" Weights

    All events are assigned an event weight of 1 plus a possible addition, if
    the event is likely of astrophysikal origin.

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
    event_weights = tf.ones_like(shared_objects["x_misc"][:, 0])

    # get astro and conv weights
    astro = shared_objects["x_misc"][
        :,
        data_handler.get_misc_index(
            config["event_weights_nugen_astro_weight"]
        ),
    ]
    conv = shared_objects["x_misc"][
        :,
        data_handler.get_misc_index(config["event_weights_nugen_conv_weight"]),
    ]

    eps = 1e-32

    # event_weights += 2 * tf.clip_by_value(
    #     tf.math.sqrt(astro / (conv + eps)), 0., 10)

    # offset the Point Source energy llh term by 7.65
    # --> events will obtain maximum boosting factor of 7.65
    llh_energy = tf.math.log(astro / (astro + conv + eps) + eps)
    event_weights += tf.clip_by_value(7.65 + llh_energy, 0.0, 10)

    return tf.expand_dims(event_weights, axis=-1)
