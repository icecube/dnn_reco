from __future__ import division, print_function
import tensorflow as tf
import numpy as np

from dnn_reco import misc
from dnn_reco.utils import angles as angle_utils
from dnn_reco.modules.loss.utils import loss_utils

"""
All defined models must have the following signature:

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
        A weighted sum with weights as defined in the config will be performed
        over these loss terms to obtain a scalar loss.
        Shape: label_shape (same shape as labels)
"""


def weighted_mse(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Weighted mean squared error of transformed prediction and true values.

    The MSE is weighted by the per event uncertainty estimate.

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

    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    unc_trafo = tf.stop_gradient(shared_objects["y_unc_trafo"])
    unc_trafo = tf.clip_by_value(unc_trafo, 1e-3, float("inf"))

    loss_event = tf.square(y_diff_trafo / unc_trafo)

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        mse_values_trafo = tf.reduce_sum(
            input_tensor=loss_event * weights, axis=0
        ) / tf.reduce_sum(input_tensor=weights, axis=0)
    else:
        mse_values_trafo = tf.reduce_mean(input_tensor=loss_event, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return mse_values_trafo


def mse(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Mean squared error of transformed prediction and true values.

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

    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    loss_event = tf.square(y_diff_trafo)
    unc_diff = shared_objects["y_unc_trafo"] - tf.stop_gradient(
        tf.abs(y_diff_trafo)
    )

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        mse_values_trafo = (
            tf.reduce_sum(input_tensor=loss_event * weights, axis=0)
            / weight_sum
        )
        mse_unc_values_trafo = (
            tf.reduce_sum(input_tensor=unc_diff**2 * weights, axis=0)
            / weight_sum
        )
    else:
        mse_values_trafo = tf.reduce_mean(input_tensor=loss_event, axis=0)
        mse_unc_values_trafo = tf.reduce_mean(input_tensor=unc_diff**2, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return mse_values_trafo + mse_unc_values_trafo


def abs(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Absolute error of transformed prediction and true values.

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
    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    loss_event = tf.abs(y_diff_trafo)
    unc_diff = shared_objects["y_unc_trafo"] - tf.stop_gradient(
        tf.abs(y_diff_trafo)
    )

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        abs_values_trafo = (
            tf.reduce_sum(input_tensor=loss_event * weights, axis=0)
            / weight_sum
        )
        abs_unc_values_trafo = (
            tf.reduce_sum(input_tensor=tf.abs(unc_diff) * weights, axis=0)
            / weight_sum
        )
    else:
        abs_values_trafo = tf.reduce_mean(input_tensor=loss_event, axis=0)
        abs_unc_values_trafo = tf.reduce_mean(
            input_tensor=tf.abs(unc_diff), axis=0
        )

    loss_utils.add_logging_info(data_handler, shared_objects)

    return abs_values_trafo + abs_unc_values_trafo


def gaussian_likelihood(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Gaussian likelhood of transformed prediction and true values.

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
    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    # small float to prevent division by zero
    eps = 1e-6

    # uncertainty estimate on prediction
    unc = tf.clip_by_value(shared_objects["y_unc_trafo"], eps, float("inf"))

    loss_event = 2 * tf.math.log(unc) + (y_diff_trafo / unc) ** 2

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        loss = (
            tf.reduce_sum(input_tensor=loss_event * weights, axis=0)
            / weight_sum
        )
    else:
        loss = tf.reduce_mean(input_tensor=loss_event, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return loss


def pull_distribution_scale(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """This loss penalized the standard deviation of the pull distribution.

    This is meant to run for a few steps with very high batch size at the very
    end of the training procedure of a model in order to correct the scale
    of the uncertainty estimates such that the pull distribution has a
    standard deviation of 1.

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
    if "event_weights" in shared_objects:
        misc.print_warning(
            "Event weights will be ignored for loss function "
            "'pull_distribution_scale'"
        )

    y_diff_trafo = tf.stop_gradient(
        loss_utils.get_y_diff_trafo(
            config=config,
            data_handler=data_handler,
            data_transformer=data_transformer,
            shared_objects=shared_objects,
        )
    )

    # small float to prevent division by zero
    eps = 1e-6

    # uncertainty estimate on prediction
    unc = tf.clip_by_value(shared_objects["y_unc_trafo"], eps, float("inf"))

    pull = y_diff_trafo / unc

    # get variance
    mean, var = tf.nn.moments(x=pull, axes=[0])

    loss = (var - 1.0) ** 2

    loss_utils.add_logging_info(data_handler, shared_objects)

    return loss


def mse_and_cross_entropy(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Mean squared error of transformed prediction and true values.
    Cross entropy loss will be applied to labels for which logit tensors
    are defined in shared_objects[logit_tensors]. These logit tensors must be
    added to the shared_objects during building of the NN model.
    This is necessary since using the logits directly is more numerically
    stable than reverting the sigmoid function on the output of the model.

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

    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    loss_event = tf.square(y_diff_trafo)

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        mse_values_trafo = (
            tf.reduce_sum(input_tensor=loss_event * weights, axis=0)
            / weight_sum
        )
    else:
        mse_values_trafo = tf.reduce_mean(input_tensor=loss_event, axis=0)

    logit_tensors = shared_objects["logit_tensors"]

    label_loss = []
    for i, name in enumerate(data_handler.label_names):

        # sanity check for correct ordering of labels
        index = data_handler.get_label_index(name)
        assert i == index, "{!r} != {!r}".format(i, index)

        # apply cross entropy if logits are provided
        if name in logit_tensors:
            loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=shared_objects["y_true"][:, i],
                logits=logit_tensors[name],
            )
            if "event_weights" in shared_objects:
                label_loss.append(
                    tf.reduce_sum(input_tensor=loss_i * weights[:, 0], axis=0)
                    / weight_sum[0]
                )
            else:
                label_loss.append(tf.reduce_mean(input_tensor=loss_i))
        else:
            label_loss.append(mse_values_trafo[i])

    label_loss = tf.stack(label_loss)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return label_loss


def mse_and_weighted_cross_entropy(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Mean squared error of transformed prediction and true values.
    Weighted cross entroy loss will be applied to labels for which logit
    tensors are defined in shared_objects[logit_tensors].
    These logit tensors must be added to the shared_objects during building
    of the NN model. This is necessary since using the logits directly is
    more numerically stable than reverting the sigmoid function on the
    output of the model.

    Events will be weighted according to how many signal events
    [pid value == 1] have a lower classification score. This aims to ignore
    abundant background events in a highly imbalanced classification task.
    Generally we only care about correctly modeling the signal region near 1.
    Hence, here we downweight the background region which is defined to be
    at smaller classification scores than the signal events.

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

    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    loss_event = tf.square(y_diff_trafo)

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        mse_values_trafo = (
            tf.reduce_sum(input_tensor=loss_event * weights, axis=0)
            / weight_sum
        )
    else:
        mse_values_trafo = tf.reduce_mean(input_tensor=loss_event, axis=0)
        weights = tf.expand_dims(tf.ones_like(loss_event[:, 0]), axis=-1)

    logit_tensors = shared_objects["logit_tensors"]

    label_loss = []
    for i, name in enumerate(data_handler.label_names):

        # sanity check for correct ordering of labels
        index = data_handler.get_label_index(name)
        assert i == index, "{!r} != {!r}".format(i, index)

        # calculate average precision if logits are provided
        if name in logit_tensors:

            labels_i = shared_objects["y_true"][:, i]
            predictions_i = logit_tensors[name]
            loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_i, logits=predictions_i
            )

            # ------------------------------
            # compute weights for each event
            # ------------------------------
            # Here we assume that background class has value 0 and signal 1
            signal_weights = tf.where(
                labels_i > 0.5, weights[:, 0], tf.zeros_like(weights[:, 0])
            )

            # sort events according to the classification score
            sorted_indices = tf.argsort(predictions_i)
            signal_weights_sorted = tf.gather(signal_weights, sorted_indices)
            weights_sorted = tf.gather(weights[:, 0], sorted_indices)
            loss_i_sorted = tf.gather(loss_i, sorted_indices)

            loss_weight = tf.stop_gradient(tf.cumsum(signal_weights_sorted))

            # and now multiply event weight on top
            loss_weight *= weights_sorted

            eps = 1e-6
            label_loss.append(
                tf.reduce_sum(input_tensor=loss_i_sorted * loss_weight, axis=0)
                / (tf.reduce_sum(input_tensor=loss_weight) + eps)
            )
            # ------------------------------
        else:
            label_loss.append(mse_values_trafo[i])

    label_loss = tf.stack(label_loss)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return label_loss


def tukey(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Tukey loss of transformed prediction and true values.
    A robust loss measure that is equivalent to MSE for small residuals, but
    has constant loss for very large residuals. This reduces the effect of
    outliers.

    From Paper: 'Robust Optimization for Deep Regression'

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
    y_diff_trafo = loss_utils.get_y_diff_trafo(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        shared_objects=shared_objects,
    )

    y_diff_trafo_scaled = y_diff_trafo / (
        1.4826 * shared_objects["median_abs_dev"]
    )

    c = 4.6851
    loss_event = tf.where(
        tf.less(tf.abs(y_diff_trafo_scaled), c),
        (c**2 / 6) * (1 - (1 - (y_diff_trafo_scaled / c) ** 2) ** 3),
        tf.zeros_like(y_diff_trafo_scaled) + (c**2 / 6),
        name="tukey_loss",
    )

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        tukey_loss = (
            tf.reduce_sum(input_tensor=loss_event * weights, axis=0)
            / weight_sum
        )
    else:
        tukey_loss = tf.reduce_mean(input_tensor=loss_event, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return tukey_loss


def opening_angle(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Error of opening angle between true and predicted angle.

    The MSE of the opening angle will be predicted as uncertainty.
    This loss only applies to label_azimuth_key and label_zenith_key!

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
    index_azimuth = data_handler.get_label_index(config["label_azimuth_key"])
    index_zenith = data_handler.get_label_index(config["label_zenith_key"])

    azimuth_true = shared_objects["y_true"][:, index_azimuth]
    zenith_true = shared_objects["y_true"][:, index_zenith]

    azimuth_pred = shared_objects["y_pred"][:, index_azimuth]
    zenith_pred = shared_objects["y_pred"][:, index_zenith]

    azimuth_unc = shared_objects["y_unc"][:, index_azimuth]
    zenith_unc = shared_objects["y_unc"][:, index_zenith]

    angle = angle_utils.tf_get_angle_deviation(
        azimuth1=azimuth_true,
        zenith1=zenith_true,
        azimuth2=azimuth_pred,
        zenith2=zenith_pred,
    )

    # use zenith true here, even though it will hae to be predicted value
    sigma = tf.sqrt(zenith_unc**2 + azimuth_unc**2 * tf.sin(zenith_true) ** 2)
    sigma /= np.sqrt(2)

    unc_diff = tf.stop_gradient(angle) - sigma

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        loss_angle = (
            tf.reduce_sum(input_tensor=angle * weights, axis=0) / weight_sum
        )
        loss_unc = (
            tf.reduce_sum(input_tensor=unc_diff**2 * weights, axis=0)
            / weight_sum
        )
    else:
        loss_angle = tf.reduce_mean(input_tensor=angle, axis=0)
        loss_unc = tf.reduce_mean(input_tensor=unc_diff**2, axis=0)

    loss = loss_angle + loss_unc
    zeros = tf.zeros_like(loss)

    loss_all_list = []
    for label in data_handler.label_names:
        if label in [config["label_azimuth_key"], config["label_zenith_key"]]:
            loss_all_list.append(loss)
        else:
            loss_all_list.append(zeros)

    loss_all = tf.stack(loss_all_list, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return loss_all


def opening_angle_raleigh(
    config, data_handler, data_transformer, shared_objects, *args, **kwargs
):
    """Raleigh loss of opening angle between true and predicted angle.

    This loss only applies to label_azimuth_key and label_zenith_key!

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

    index_azimuth = data_handler.get_label_index(config["label_azimuth_key"])
    index_zenith = data_handler.get_label_index(config["label_zenith_key"])

    azimuth_true = shared_objects["y_true"][:, index_azimuth]
    zenith_true = shared_objects["y_true"][:, index_zenith]

    azimuth_pred = shared_objects["y_pred"][:, index_azimuth]
    zenith_pred = shared_objects["y_pred"][:, index_zenith]

    azimuth_unc = shared_objects["y_unc"][:, index_azimuth]
    zenith_unc = shared_objects["y_unc"][:, index_zenith]

    angle = angle_utils.tf_get_angle_deviation(
        azimuth1=azimuth_true,
        zenith1=zenith_true,
        azimuth2=azimuth_pred,
        zenith2=zenith_pred,
    )

    # use zenith true here, even though it will hae to be predicted value
    sigma = tf.sqrt(zenith_unc**2 + azimuth_unc**2 * tf.sin(zenith_true) ** 2)
    sigma /= np.sqrt(2)

    # small float to prevent division by zero
    eps = 1e-6
    sigma = tf.clip_by_value(sigma, eps, float("inf"))

    raleigh = (
        (angle / sigma) ** 2 + 4 * tf.math.log(sigma) - 2 * tf.math.log(angle)
    )

    if "event_weights" in shared_objects:
        weights = shared_objects["event_weights"]
        weight_sum = tf.reduce_sum(input_tensor=weights, axis=0)
        loss_angle = (
            tf.reduce_sum(input_tensor=angle * weights, axis=0) / weight_sum
        )
        raleigh_loss = (
            tf.reduce_sum(input_tensor=raleigh * weights, axis=0) / weight_sum
        )
    else:
        loss_angle = tf.reduce_mean(input_tensor=angle, axis=0)
        raleigh_loss = tf.reduce_mean(input_tensor=raleigh, axis=0)

    loss = loss_angle + raleigh_loss
    zeros = tf.zeros_like(loss)

    loss_all_list = []
    for label in data_handler.label_names:
        if label in [config["label_azimuth_key"], config["label_zenith_key"]]:
            loss_all_list.append(loss)
        else:
            loss_all_list.append(zeros)

    loss_all = tf.stack(loss_all_list, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return loss_all
