"""
All defined models must have the following signature:

    Parameters
    ----------
    is_training : bool
        True if model is in training mode, false if in inference mode.
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
        The label prediction tensor y_pred.
    tf.Tensor
        The uncertainty estimate for each label.
    list of tf.Tensor
        The trainable parameters of the prediction network.
    list of tf.Tensor
        The trainable parameters of the uncertainty sub network.
        Can optionally be an empty list.
"""

import numpy as np
import tensorflow as tf

# Check and allow for newer TFScripts versions
try:
    from tfscripts.compat.v1 import layers as tfs
except ImportError:
    from tfscripts import layers as tfs

from dnn_reco.modules.models.utils.model_utils import preprocess_icecube_data


def general_model_IC86(
    is_training,
    config,
    data_handler,
    data_transformer,
    shared_objects,
    *args,
    **kwargs
):
    """A general NN model for the IceCube IC86 configuration.

    Very similar to general_model_IC86_opt4, but uncertainty fc layers get
    network prediction as input as well.

    Parameters
    ----------
    is_training : bool
        True if model is in training mode, false if in inference mode.
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
        The label prediction tensor y_pred.
    tf.Tensor
        The uncertainty estimate for each label.
    list of tf.Tensor
        The trainable parameters of the prediction network.
    list of tf.Tensor
        The trainable parameters of the uncertainty sub network.
    """
    keep_prob_list = shared_objects["keep_prob_list"]
    num_labels = data_handler.label_shape[-1]

    with tf.compat.v1.variable_scope("model_pred"):

        # apply DOM dropout, split and reshape DeepCore input
        X_IC78, X_DeepCore_upper, X_DeepCore_lower = preprocess_icecube_data(
            is_training,
            shared_objects,
            seed=config["tf_random_seed"],
        )

        # -----------------------------------
        # convolutional layers over DeepCore_1 : upper part
        # -----------------------------------
        conv2d_1_layers, kernels, biases = tfs.new_conv_nd_layers(
            X_DeepCore_upper,
            is_training=is_training,
            name="Upper DeepCore",
            method_list="convolution",
            keep_prob=keep_prob_list[1],
            seed=config["tf_random_seed"],
            **config["conv_upper_DeepCore_settings"]
        )

        # -----------------------------------
        # convolutional layers over DeepCore_2 : lower part
        # -----------------------------------
        conv2d_2_layers, kernels, biases = tfs.new_conv_nd_layers(
            X_DeepCore_lower,
            is_training=is_training,
            name="Lower DeepCore",
            method_list="convolution",
            keep_prob=keep_prob_list[1],
            seed=config["tf_random_seed"],
            **config["conv_lower_DeepCore_settings"]
        )

        # -----------------------------------
        # convolutional hex3d layers over X_IC78 data
        # -----------------------------------
        conv_hex3d_layers, kernels, biases = tfs.new_conv_nd_layers(
            X_IC78,
            name="IC78",
            is_training=is_training,
            method_list="hex_convolution",
            keep_prob=keep_prob_list[1],
            seed=config["tf_random_seed"],
            **config["conv_IC78_settings"]
        )

        # -----------------------------------
        # combine results of convolution and flatten
        # -----------------------------------
        # flatten layer
        layer_flat_IC78, num_features_IC78 = tfs.flatten_layer(
            conv_hex3d_layers[-1]
        )
        layer_flat_DeepCore_1, num_features_DeepCore_1 = tfs.flatten_layer(
            conv2d_1_layers[-1]
        )
        layer_flat_DeepCore_2, num_features_DeepCore_2 = tfs.flatten_layer(
            conv2d_2_layers[-1]
        )

        # combine layers
        layer_flat = tf.concat(
            [layer_flat_IC78, layer_flat_DeepCore_1, layer_flat_DeepCore_2],
            axis=1,
        )

        # dropout
        layer_flat = tf.nn.dropout(
            layer_flat,
            rate=1 - (keep_prob_list[2]),
            seed=config["tf_random_seed"],
        )

        # -----------------------------------
        # fully connected layers
        # -----------------------------------
        fc_settings = dict(config["fc_settings"])
        fc_settings["fc_sizes"][-1] = num_labels

        layers, weights, biases = tfs.new_fc_layers(
            input=layer_flat,
            keep_prob=keep_prob_list[3],
            is_training=is_training,
            seed=config["tf_random_seed"],
            **fc_settings
        )

        y_pred_trafo_orig = layers[-1]

        # -----------------------------------
        # Enforce Normalisation
        # -----------------------------------
        assert len(y_pred_trafo_orig.get_shape().as_list()) == 2

        # transform back
        y_pred = data_transformer.inverse_transform(
            y_pred_trafo_orig, data_type="label"
        )
        y_pred_list = tf.unstack(y_pred, axis=1)

        if "model_enforce_direction_norm" not in config:
            # backward compatibility
            enforce_direction_norm = True
        else:
            enforce_direction_norm = config["model_enforce_direction_norm"]

        if enforce_direction_norm:

            index_dir_x = data_handler.get_label_index(
                config["label_dir_x_key"]
            )
            index_dir_y = data_handler.get_label_index(
                config["label_dir_y_key"]
            )
            index_dir_z = data_handler.get_label_index(
                config["label_dir_z_key"]
            )
            index_zenith = data_handler.get_label_index(
                config["label_zenith_key"]
            )
            index_azimuth = data_handler.get_label_index(
                config["label_azimuth_key"]
            )

            trafo_indices = [
                index_dir_x,
                index_dir_y,
                index_dir_z,
                index_azimuth,
                index_zenith,
            ]

            norm = tf.sqrt(
                y_pred_list[index_dir_x] ** 2
                + y_pred_list[index_dir_y] ** 2
                + y_pred_list[index_dir_z] ** 2
            )

            y_pred_list[index_dir_x] /= norm
            y_pred_list[index_dir_y] /= norm
            y_pred_list[index_dir_z] /= norm

            # calculate zenith
            y_pred_list[index_zenith] = tf.acos(
                tf.clip_by_value(-y_pred_list[index_dir_z], -1, 1)
            )

            # calculate azimuth
            y_pred_list[index_azimuth] = (
                tf.atan2(-y_pred_list[index_dir_y], -y_pred_list[index_dir_x])
                + 2 * np.pi
            ) % (2 * np.pi)
        else:
            trafo_indices = []

        # limit PID variables to range 0 to 1

        # safety check
        for k in data_handler.label_names:
            if k[0:2] == "p_" and k not in config["label_pid_keys"]:
                raise ValueError("Did you forget about {!r}?".format(k))

        logit_tensors = {}
        for pid_key in config["label_pid_keys"]:
            if pid_key in data_handler.label_names:
                index_pid = data_handler.get_label_index(pid_key)
                trafo_indices.append(index_pid)
                logit_tensors[pid_key] = y_pred_list[index_pid]
                y_pred_list[index_pid] = tf.sigmoid(y_pred_list[index_pid])

        # save logit tensors. These will be needed if we want to use
        # cross entropy as a loss function. Tensorflow's cross entropy
        # functions use logits as input as opposed to sigmoid(logits) due
        # to numerical stability.
        shared_objects["logit_tensors"] = logit_tensors

        # put it back together
        y_pred = tf.stack(y_pred_list, axis=1)

        # transform
        y_pred_trafo = data_transformer.transform(y_pred, data_type="label")

        # Only do inv_trafo(trafo(y)) if necessary (numerical problems ...)
        y_pred_trafo_orig_list = tf.unstack(y_pred_trafo_orig, axis=1)
        y_pred_trafo_list = tf.unstack(y_pred_trafo, axis=1)
        y_pred_trafo_final_list = []
        for i in range(len(y_pred_trafo_orig_list)):
            if i in trafo_indices:
                y_pred_trafo_final_list.append(y_pred_trafo_list[i])
            else:
                y_pred_trafo_final_list.append(y_pred_trafo_orig_list[i])

        # # zero out labels with weights == 0 if they are nan
        # for i, non_zero in enumerate(shared_objects['non_zero_mask']):
        #     if not non_zero:
        #         y_pred_trafo_final_list[i] = tf.where(
        #                         tf.is_finite(y_pred_trafo_final_list[i]),
        #                         y_pred_trafo_final_list[i],
        #                         tf.zeros_like(y_pred_trafo_final_list[i]))

        # put it back together
        y_pred_trafo = tf.stack(y_pred_trafo_final_list, axis=1)

    with tf.compat.v1.variable_scope("model_unc"):

        # -----------------------------------
        # Uncertainty estimate
        # -----------------------------------
        fc_unc_settings = dict(config["fc_unc_settings"])
        fc_unc_settings["fc_sizes"][-1] = num_labels

        unc_input = tf.stop_gradient(
            tf.concat((y_pred_trafo, layer_flat), axis=-1)
        )

        uncertainty_layers, weights, biases = tfs.new_fc_layers(
            input=unc_input,
            is_training=is_training,
            keep_prob=keep_prob_list[3],
            seed=config["tf_random_seed"],
            **fc_unc_settings
        )
        y_unc_pred_trafo = uncertainty_layers[-1]
        y_unc_pred_trafo = tf.abs(y_unc_pred_trafo) + 1e-3

    # -----------------------------------
    # print architecture
    # -----------------------------------
    print("flat IC78:", layer_flat_IC78)
    print("layer_flat:", layer_flat)
    print("unc_input:", unc_input)
    print("y_pred_trafo:", y_pred_trafo)
    print("y_unc_pred_trafo:", y_unc_pred_trafo)

    # -----------------------------------
    # collect model variables that need to be saved
    # -----------------------------------
    model_vars_pred = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "model_pred"
    )
    model_vars_unc = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "model_unc"
    )

    return y_pred_trafo, y_unc_pred_trafo, model_vars_pred, model_vars_unc


def general_model_IC86_opt4(
    is_training,
    config,
    data_handler,
    data_transformer,
    shared_objects,
    *args,
    **kwargs
):
    """A general NN model for the IceCube IC86 configuration.

    Parameters
    ----------
    is_training : bool
        True if model is in training mode, false if in inference mode.
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
        The label prediction tensor y_pred.
    tf.Tensor
        The uncertainty estimate for each label.
    list of tf.Tensor
        The trainable parameters of the prediction network.
    list of tf.Tensor
        The trainable parameters of the uncertainty sub network.
    """
    keep_prob_list = shared_objects["keep_prob_list"]
    num_labels = data_handler.label_shape[-1]

    with tf.compat.v1.variable_scope("model_pred"):

        # apply DOM dropout, split and reshape DeepCore input
        X_IC78, X_DeepCore_upper, X_DeepCore_lower = preprocess_icecube_data(
            is_training,
            shared_objects,
            seed=config["tf_random_seed"],
        )

        # -----------------------------------
        # convolutional layers over DeepCore_1 : upper part
        # -----------------------------------
        conv2d_1_layers, kernels, biases = tfs.new_conv_nd_layers(
            X_DeepCore_upper,
            is_training=is_training,
            name="Upper DeepCore",
            method_list="convolution",
            keep_prob=keep_prob_list[1],
            seed=config["tf_random_seed"],
            **config["conv_upper_DeepCore_settings"]
        )

        # -----------------------------------
        # convolutional layers over DeepCore_2 : lower part
        # -----------------------------------
        conv2d_2_layers, kernels, biases = tfs.new_conv_nd_layers(
            X_DeepCore_lower,
            is_training=is_training,
            name="Lower DeepCore",
            method_list="convolution",
            keep_prob=keep_prob_list[1],
            seed=config["tf_random_seed"],
            **config["conv_lower_DeepCore_settings"]
        )

        # -----------------------------------
        # convolutional hex3d layers over X_IC78 data
        # -----------------------------------
        conv_hex3d_layers, kernels, biases = tfs.new_conv_nd_layers(
            X_IC78,
            name="IC78",
            is_training=is_training,
            method_list="hex_convolution",
            keep_prob=keep_prob_list[1],
            seed=config["tf_random_seed"],
            **config["conv_IC78_settings"]
        )

        # -----------------------------------
        # combine results of convolution and flatten
        # -----------------------------------
        # flatten layer
        layer_flat_IC78, num_features_IC78 = tfs.flatten_layer(
            conv_hex3d_layers[-1]
        )
        layer_flat_DeepCore_1, num_features_DeepCore_1 = tfs.flatten_layer(
            conv2d_1_layers[-1]
        )
        layer_flat_DeepCore_2, num_features_DeepCore_2 = tfs.flatten_layer(
            conv2d_2_layers[-1]
        )

        # combine layers
        layer_flat = tf.concat(
            [layer_flat_IC78, layer_flat_DeepCore_1, layer_flat_DeepCore_2],
            axis=1,
        )

        # dropout
        layer_flat = tf.nn.dropout(
            layer_flat,
            rate=1 - (keep_prob_list[2]),
            seed=config["tf_random_seed"],
        )

        # -----------------------------------
        # fully connected layers
        # -----------------------------------
        fc_settings = dict(config["fc_settings"])
        fc_settings["fc_sizes"][-1] = num_labels

        layers, weights, biases = tfs.new_fc_layers(
            input=layer_flat,
            keep_prob=keep_prob_list[3],
            is_training=is_training,
            seed=config["tf_random_seed"],
            **fc_settings
        )

        y_pred_trafo_orig = layers[-1]

        # -----------------------------------
        # Enforce Normalisation
        # -----------------------------------
        assert len(y_pred_trafo_orig.get_shape().as_list()) == 2

        # transform back
        y_pred = data_transformer.inverse_transform(
            y_pred_trafo_orig, data_type="label"
        )
        y_pred_list = tf.unstack(y_pred, axis=1)

        if "model_enforce_direction_norm" not in config:
            # backward compatibility
            enforce_direction_norm = True
        else:
            enforce_direction_norm = config["model_enforce_direction_norm"]

        if "model_dir_z_independent" not in config:
            # backward compatibility
            model_dir_z_independent = False
        else:
            model_dir_z_independent = config["model_dir_z_independent"]

        if "model_limit_dir_vec" not in config:
            # backward compatibility
            model_limit_dir_vec = None
        else:
            model_limit_dir_vec = config["model_limit_dir_vec"]

        if enforce_direction_norm or model_limit_dir_vec:

            index_dir_x = data_handler.get_label_index(
                config["label_dir_x_key"]
            )
            index_dir_y = data_handler.get_label_index(
                config["label_dir_y_key"]
            )
            index_dir_z = data_handler.get_label_index(
                config["label_dir_z_key"]
            )
            index_zenith = data_handler.get_label_index(
                config["label_zenith_key"]
            )
            index_azimuth = data_handler.get_label_index(
                config["label_azimuth_key"]
            )

            # limit direction vector components to certain value range
            if model_limit_dir_vec is not None:
                y_pred_list[index_dir_x] = (
                    tf.math.tanh(y_pred_list[index_dir_x])
                    * model_limit_dir_vec
                )
                y_pred_list[index_dir_y] = (
                    tf.math.tanh(y_pred_list[index_dir_y])
                    * model_limit_dir_vec
                )
                y_pred_list[index_dir_z] = (
                    tf.math.tanh(y_pred_list[index_dir_z])
                    * model_limit_dir_vec
                )

            trafo_indices = [
                index_dir_x,
                index_dir_y,
                index_dir_z,
                index_azimuth,
                index_zenith,
            ]

            if enforce_direction_norm:
                if model_dir_z_independent:
                    # since we are not normalizing the direction-z component
                    # we need to ensure that the allowed value range is
                    # within [-1, 1]. We'll leave a little bit of leeway
                    # to allow for [-1.1, 1.1] due to asymptotic behavior
                    # of tanh
                    assert np.abs(model_limit_dir_vec - 1) < 0.1
                    norm_xy = tf.math.sqrt(
                        (
                            y_pred_list[index_dir_x] ** 2
                            + y_pred_list[index_dir_y] ** 2
                        )
                        / (1 - tf.stop_gradient(y_pred_list[index_dir_z]) ** 2)
                    )
                    y_pred_list[index_dir_x] /= norm_xy
                    y_pred_list[index_dir_y] /= norm_xy
                else:
                    norm = tf.sqrt(
                        y_pred_list[index_dir_x] ** 2
                        + y_pred_list[index_dir_y] ** 2
                        + y_pred_list[index_dir_z] ** 2
                    )

                    y_pred_list[index_dir_x] /= norm
                    y_pred_list[index_dir_y] /= norm
                    y_pred_list[index_dir_z] /= norm

            # calculate zenith
            y_pred_list[index_zenith] = tf.acos(
                tf.clip_by_value(-y_pred_list[index_dir_z], -1, 1)
            )

            # calculate azimuth
            y_pred_list[index_azimuth] = (
                tf.atan2(-y_pred_list[index_dir_y], -y_pred_list[index_dir_x])
                + 2 * np.pi
            ) % (2 * np.pi)
        else:
            trafo_indices = []

        # -----------------------------------
        # limit PID variables to range 0 to 1
        # -----------------------------------

        # safety check
        for k in data_handler.label_names:
            if k[0:2] == "p_" and k not in config["label_pid_keys"]:
                raise ValueError("Did you forget about {!r}?".format(k))

        logit_tensors = {}
        for pid_key in config["label_pid_keys"]:
            if pid_key in data_handler.label_names:
                index_pid = data_handler.get_label_index(pid_key)
                trafo_indices.append(index_pid)
                logit_tensors[pid_key] = y_pred_list[index_pid]
                y_pred_list[index_pid] = tf.sigmoid(y_pred_list[index_pid])

        # save logit tensors. These will be needed if we want to use
        # cross entropy as a loss function. Tensorflow's cross entropy
        # functions use logits as input as opposed to sigmoid(logits) due
        # to numerical stability.
        shared_objects["logit_tensors"] = logit_tensors

        # put it back together
        y_pred = tf.stack(y_pred_list, axis=1)

        # transform
        y_pred_trafo = data_transformer.transform(y_pred, data_type="label")

        # Only do inv_trafo(trafo(y)) if necessary (numerical problems ...)
        y_pred_trafo_orig_list = tf.unstack(y_pred_trafo_orig, axis=1)
        y_pred_trafo_list = tf.unstack(y_pred_trafo, axis=1)
        y_pred_trafo_final_list = []
        for i in range(len(y_pred_trafo_orig_list)):
            if i in trafo_indices:
                y_pred_trafo_final_list.append(y_pred_trafo_list[i])
            else:
                y_pred_trafo_final_list.append(y_pred_trafo_orig_list[i])

        # # zero out labels with weights == 0 if they are nan
        # for i, non_zero in enumerate(shared_objects['non_zero_mask']):
        #     if not non_zero:
        #         y_pred_trafo_final_list[i] = tf.where(
        #                         tf.is_finite(y_pred_trafo_final_list[i]),
        #                         y_pred_trafo_final_list[i],
        #                         tf.zeros_like(y_pred_trafo_final_list[i]))

        # put it back together
        y_pred_trafo = tf.stack(y_pred_trafo_final_list, axis=1)

    with tf.compat.v1.variable_scope("model_unc"):

        # -----------------------------------
        # Uncertainty estimate
        # -----------------------------------
        fc_unc_settings = dict(config["fc_unc_settings"])
        fc_unc_settings["fc_sizes"][-1] = num_labels

        uncertainty_layers, weights, biases = tfs.new_fc_layers(
            input=tf.stop_gradient(layer_flat),
            is_training=is_training,
            keep_prob=keep_prob_list[3],
            seed=config["tf_random_seed"],
            **fc_unc_settings
        )
        y_unc_pred_trafo = uncertainty_layers[-1]
        y_unc_pred_trafo = tf.abs(y_unc_pred_trafo) + 1e-3

    # -----------------------------------
    # print architecture
    # -----------------------------------
    print("flat IC78:", layer_flat_IC78)
    print("layer_flat:", layer_flat)
    print("y_pred_trafo:", y_pred_trafo)
    print("y_unc_pred_trafo:", y_unc_pred_trafo)

    # -----------------------------------
    # collect model variables that need to be saved
    # -----------------------------------
    model_vars_pred = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "model_pred"
    )
    model_vars_unc = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, "model_unc"
    )

    return y_pred_trafo, y_unc_pred_trafo, model_vars_pred, model_vars_unc
