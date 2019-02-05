from __future__ import division, print_function
import tensorflow as tf

from tfscripts import layers as tfs

from dnn_reco.modules.models.utils.model_utils import preprocess_icecube_data

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
        A dictionary containg settings and objects that are shared and passed
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


def general_model_IC86_opt4(is_training, config, data_handler,
                            data_transformer, shared_objects, *args, **kwargs):
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
        A dictionary containg settings and objects that are shared and passed
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
    X_IC78 = shared_objects['x_ic78']
    X_DeepCore = shared_objects['x_deepcore']
    keep_prob_list = shared_objects['keep_prob_list']
    num_labels = data_handler.label_shape[-1]

    use_batch_norm = False
    use_residual = True

    # uncertainty layers
    fc_uncertainty_sizes = [300, num_labels]
    fc_uncertainty_use_dropout_list = [True, False]
    activation_list_uncertainty = ['elu','']
    fc_uncertainty_max_out_size_list = None
    fc_uncertainty_keep_prob = keep_prob_list[3]
    fc_uncertainty_use_batch_normalisation_list=use_batch_norm
    fc_uncertainty_use_residual_list=False

    with tf.variable_scope('model_pred'):

        # apply DOM dropout, split and reshape DeepCore input
        X_IC78, X_DeepCore_upper, X_DeepCore_lower = preprocess_icecube_data(
                                                is_training, shared_objects)

        # -----------------------------------
        # convolutional layers over DeepCore_1 : upper part
        # -----------------------------------
        conv2d_1_layers, kernels, biases = tfs.new_conv_nd_layers(
                        X_DeepCore_upper,
                        is_training=is_training,
                        name='Upper DeepCore',
                        method_list='convolution',
                        keep_prob=keep_prob_list[1],
                        **config['conv_upper_DeepCore_settings']
                        )

        # -----------------------------------
        # convolutional layers over DeepCore_2 : lower part
        # -----------------------------------
        conv2d_2_layers, kernels, biases = tfs.new_conv_nd_layers(
                        X_DeepCore_lower,
                        is_training=is_training,
                        name='Lower DeepCore',
                        method_list='convolution',
                        keep_prob=keep_prob_list[1],
                        **config['conv_lower_DeepCore_settings']
                        )

        # -----------------------------------
        # convolutional hex3d layers over X_IC78 data
        # -----------------------------------
        conv_hex3d_layers, kernels, biases = tfs.new_conv_nd_layers(
                                        X_IC78,
                                        name='IC78',
                                        is_training=is_training,
                                        method_list='hex_convolution',
                                        keep_prob=keep_prob_list[1],
                                        **config['conv_IC78_settings']
                                        )

        # -----------------------------------
        # combine results of convolution and flatten
        # -----------------------------------
        # flatten layer
        layer_flat_IC78, num_features_IC78 = tfs.flatten_layer(
                                                        conv_hex3d_layers[-1])
        layer_flat_DeepCore_1, num_features_DeepCore_1 = tfs.flatten_layer(
                                                        conv2d_1_layers[-1])
        layer_flat_DeepCore_2, num_features_DeepCore_2 = tfs.flatten_layer(
                                                        conv2d_2_layers[-1])

        # combine layers
        num_features = (num_features_DeepCore_1 + num_features_DeepCore_2
                        + num_features_IC78)
        layer_flat = tf.concat([layer_flat_IC78, layer_flat_DeepCore_1,
                                layer_flat_DeepCore_2], axis=1)

        # dropout
        layer_flat = tf.nn.dropout(layer_flat, keep_prob_list[2])

        # -----------------------------------
        # fully connected layers
        # -----------------------------------
        fc_settings = dict(config['fc_settings'])
        fc_settings['fc_sizes'][-1] = num_labels

        layers, weights, biases = tfs.new_fc_layers(
                                                input=layer_flat,
                                                keep_prob=keep_prob_list[3],
                                                is_training=is_training,
                                                **fc_settings)

        y_pred = layers[-1]

        # # -----------------------------------
        # # Enforce Normalisation
        # # -----------------------------------
        # Y_namesDict = settings['Y_namesDict']
        # if settings['normalizeData']:

        #     # load norm model
        #     normModel = np.load(settings['normModelFile'])
        #     normalizationConstant = settings['normalizationConstant']

        #     # Denormalize:
        #     y_pred = y_pred * (normModel[5] + normalizationConstant) + normModel[4]

        # y_pred_list = tf.unstack(y_pred, axis=1)

        # norm = tf.sqrt( y_pred_list[Y_namesDict['PrimaryDirectionX']]**2 + \
        #                 y_pred_list[Y_namesDict['PrimaryDirectionY']]**2 + \
        #                 y_pred_list[Y_namesDict['PrimaryDirectionZ']]**2)

        # y_pred_list[Y_namesDict['PrimaryDirectionX']] /= norm
        # y_pred_list[Y_namesDict['PrimaryDirectionY']] /= norm
        # y_pred_list[Y_namesDict['PrimaryDirectionZ']] /= norm

        # # calculate zenith
        # y_pred_list[Y_namesDict['PrimaryZenith']] = tf.acos( tf.clip_by_value(
        #                     -y_pred_list[Y_namesDict['PrimaryDirectionZ']],
        #                     -1,1))

        # # calculate azimuth
        # y_pred_list[Y_namesDict['PrimaryAzimuth']] = \
        #                 (tf.atan2( -y_pred_list[Y_namesDict['PrimaryDirectionY']],
        #                          -y_pred_list[Y_namesDict['PrimaryDirectionX']]) \
        #                 + 2 * np.pi) % (2 * np.pi)

        # # limit PID variables to range 0 to 1
        # pid_keys = [
        #     'p_cc_e', 'p_cc_mu', 'p_cc_tau', 'p_nc',
        #     'p_starting', 'p_starting_300m', 'p_starting_glashow',
        #     'p_starting_nc', 'p_starting_cc', 'p_starting_cc_e',
        #     'p_starting_cc_mu', 'p_starting_cc_tau',
        #     'p_starting_cc_tau_muon_decay',
        #     'p_starting_cc_tau_double_bang', 'p_entering',
        #     'p_entering_muon_single', 'p_entering_muon_bundle',
        #     'p_outside_cascade',
        # ]

        # # safety check
        # for k in Y_namesDict.keys():
        #     if k[0:2] == 'p_' and k not in pid_keys:
        #         raise ValueError('Did you forget about {!r}?'.format(k))

        # for pid_key in pid_keys:
        #     if pid_key in Y_namesDict:
        #         y_pred_list[Y_namesDict[pid_key]] = tf.sigmoid(
        #                                     y_pred_list[Y_namesDict[pid_key]])

        # y_pred = tf.stack(y_pred_list, axis=1)

        # if settings['normalizeData']:
        #     # Normalize:
        #     y_pred = ( y_pred - normModel[4] ) / (normModel[5] + normalizationConstant)

    with tf.variable_scope('model_unc'):

        # -----------------------------------
        # Uncertainty estimate
        # -----------------------------------
        fc_unc_settings = dict(config['fc_unc_settings'])
        fc_unc_settings['fc_sizes'][-1] = num_labels

        uncertainty_layers, weights, biases = tfs.new_fc_layers(
                                            input=tf.stop_gradient(layer_flat),
                                            is_training=is_training,
                                            keep_prob=keep_prob_list[3],
                                            **fc_unc_settings
                                            )
        y_uncertainty_pred = uncertainty_layers[-1]

    # -----------------------------------
    # print architecture
    # -----------------------------------
    print('flat IC78:', layer_flat_IC78)
    print('layer_flat:', layer_flat)
    print('y_pred:', y_pred)
    print('y_uncertainty_pred:', y_uncertainty_pred)

    # -----------------------------------
    # collect model variables that need to be saved
    # -----------------------------------
    model_vars_pred = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        'model_pred')
    model_vars_unc = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       'model_unc')

    return y_pred, y_uncertainty_pred, model_vars_pred, model_vars_unc
