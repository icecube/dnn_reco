from __future__ import division, print_function
import tensorflow as tf

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


def weighted_mse(config, data_handler, data_transformer, shared_objects,
                 *args, **kwargs):
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
                                    shared_objects=shared_objects)

    unc_trafo = tf.stop_gradient(shared_objects['y_unc_trafo'])
    unc_trafo = tf.clip_by_value(unc_trafo, 1e-3, float('inf'))
    mse_values_trafo = tf.reduce_mean(tf.square(y_diff_trafo / unc_trafo), 0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return mse_values_trafo


def mse(config, data_handler, data_transformer, shared_objects,
        *args, **kwargs):
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
                                    shared_objects=shared_objects)

    mse_values_trafo = tf.reduce_mean(tf.square(y_diff_trafo), 0)

    unc_diff = shared_objects['y_unc_trafo'] - \
        tf.stop_gradient(tf.abs(y_diff_trafo))
    mse_unc_values_trafo = tf.reduce_mean(unc_diff**2, 0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return mse_values_trafo + mse_unc_values_trafo


def abs(config, data_handler, data_transformer, shared_objects,
        *args, **kwargs):
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
                                    shared_objects=shared_objects)

    abs_values_trafo = tf.reduce_mean(tf.abs(y_diff_trafo), 0)

    unc_diff = shared_objects['y_unc_trafo'] - \
        tf.stop_gradient(tf.abs(y_diff_trafo))
    abs_unc_values_trafo = tf.reduce_mean(tf.abs(unc_diff), 0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return abs_values_trafo + abs_unc_values_trafo


def gaussian_likelihood(config, data_handler, data_transformer, shared_objects,
                        *args, **kwargs):
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
                                    shared_objects=shared_objects)

    # small float to prevent division by zero
    eps = 1e-6

    # uncertainty estimate on prediction
    unc = tf.clip_by_value(shared_objects['y_unc_trafo'], eps, float('inf'))

    loss = tf.reduce_mean(2*tf.log(unc) + (y_diff_trafo / unc)**2, axis=0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return loss


def pull_distribution_scale(config, data_handler, data_transformer,
                            shared_objects, *args, **kwargs):
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
    y_diff_trafo = tf.stop_gradient(loss_utils.get_y_diff_trafo(
                                    config=config,
                                    data_handler=data_handler,
                                    data_transformer=data_transformer,
                                    shared_objects=shared_objects))

    # small float to prevent division by zero
    eps = 1e-6

    # uncertainty estimate on prediction
    unc = tf.clip_by_value(shared_objects['y_unc_trafo'], eps, float('inf'))

    pull = y_diff_trafo / unc

    # get variance
    mean, var = tf.nn.moments(pull, axes=[0])

    loss = (var - 1.)**2

    loss_utils.add_logging_info(data_handler, shared_objects)

    return loss


def mse_and_cross_entropy(config, data_handler, data_transformer,
                          shared_objects, *args, **kwargs):
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
                                    shared_objects=shared_objects)

    mse_values_trafo = tf.reduce_mean(tf.square(y_diff_trafo), 0)

    logit_tensors = shared_objects['logit_tensors']

    label_loss = []
    for i, name in enumerate(data_handler.label_names):

        # sanity check for correct ordering of labels
        index = data_handler.get_label_index(name)
        assert i == index, '{!r} != {!r}'.format(i, index)

        # apply cross entropy if logits are provided
        if name in logit_tensors:
            label_loss.append(tf.reduce_mean(
                                tf.nn.sigmoid_cross_entropy_with_logits(
                                        labels=shared_objects['y_true'][:, i],
                                        logits=logit_tensors[name])))
        else:
            label_loss.append(mse_values_trafo[i])

    label_loss = tf.stack(label_loss)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return label_loss


def tukey(config, data_handler, data_transformer, shared_objects,
          *args, **kwargs):
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
                                    shared_objects=shared_objects)

    y_diff_trafo_scaled = \
        y_diff_trafo / (1.4826 * shared_objects['median_abs_dev'])

    c = 4.6851
    tukey_loss = tf.reduce_mean(tf.where(
        tf.less(tf.abs(y_diff_trafo_scaled), c),
        (c**2/6) * (1 - (1 - (y_diff_trafo_scaled/c)**2)**3),
        tf.zeros_like(y_diff_trafo_scaled) + (c**2/6),
        name='tukey_loss'), 0)

    loss_utils.add_logging_info(data_handler, shared_objects)

    return tukey_loss
