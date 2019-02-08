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

    loss_utils.add_logging_info(data_handler, shared_objects)

    return mse_values_trafo


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

    loss_utils.add_logging_info(data_handler, shared_objects)

    return abs_values_trafo


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


def mse_and_cross_entropy(config, data_handler, data_transformer,
                          shared_objects, *args, **kwargs):
    """Mean squared error of transformed prediction and true values.
    Cross entropy loss for pid label variables (all labels starting with 'p_').

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
    raise NotImplementedError()


def tukey(config, data_handler, data_transformer, shared_objects,
          *args, **kwargs):
    """Tukey loss of transformed prediction and true values.
    A robust loss measure that is equivalent to MSE for small residuals, but
    has constant loss for very large residuals. This reduces the effect of
    outliers.

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
    raise NotImplementedError()
