from __future__ import division, print_function
import tensorflow as tf
import numpy as np

from dnn_reco import misc


class NNModel(object):
    """Base class for neural network architecture

    Attributes
    ----------
    config : dict
            Dictionary containing all settings as read in from config file.
    data_handler : :obj: of class DataHandler
            An instance of the DataHandler class. The object is used to obtain
            meta data.
    data_transformer : :obj: of class DataTransformer
            An instance of the DataTransformer class. The object is used to
            transform data.
    is_training : bool
            True if model is in training mode, false if in inference mode.
    saver : tensorflow.train.Saver
        A tensorflow saver used to save and load model weights.
    shared_objects : dict
        A dictionary containg settings and objects that are shared and passed
        on to sub modules.
    """

    def __init__(self, is_training, config, data_handler, data_transformer):
        """Initializes neural network base class.

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
        """
        self.is_training = is_training
        self.config = config
        self.data_handler = data_handler
        self.data_transformer = data_transformer

        self.shared_objects = {}

        # create tensorflow placeholders for input data
        self._setup_placeholders()

        # build NN architecture
        self._build_model()

    def _setup_placeholders(self):
        """Sets up placeholders for input data.
        """
        # define placeholders for keep probability for dropout
        if 'keep_probability_list' in self.config:
            keep_prob_list = [tf.placeholder(
                                        self.config['float_precision'],
                                        name='keep_prob_{:0.2f}'.format(i))
                              for i in self.config['keep_probability_list']]
            self.shared_objects['keep_prob_list'] = keep_prob_list

        # IC78: main IceCube array
        self.shared_objects['x_ic78'] = tf.placeholder(
                        self.config['float_precision'],
                        shape=[None, 10, 10, 60,  self.data_handler.num_bins],
                        name='x_ic78',
                        )
        self.shared_objects['x_ic78_trafo'] = self.data_transformer.transform(
                            self.shared_objects['x_ic78'], data_type='ic78')

        # DeepCore
        self.shared_objects['x_deepcore'] = tf.placeholder(
                        self.config['float_precision'],
                        shape=[None, 8, 60,  self.data_handler.num_bins],
                        name='x_deepcore',
                        )
        self.shared_objects['x_deepcore_trafo'] = \
            self.data_transformer.transform(self.shared_objects['x_deepcore'],
                                            data_type='deepcore')

        # labels
        self.shared_objects['y_true'] = tf.placeholder(
                        self.config['float_precision'],
                        shape=[None] + self.data_handler.label_shape,
                        name='y_true',
                        )
        self.shared_objects['y_true_trafo'] = self.data_transformer.transform(
                        self.shared_objects['y_true'], data_type='label')

        # misc data
        if self.data_handler.misc_shape is not None:
            self.shared_objects['x_misc'] = tf.placeholder(
                        self.config['float_precision'],
                        shape=[None] + self.data_handler.misc_shape,
                        name='x_misc',
                        )
            self.shared_objects['x_misc_trafo'] = \
                self.data_transformer.transform(self.shared_objects['x_misc'],
                                                data_type='misc')

    def _build_model(self):
        """Build neural network architecture.
        """
        class_string = 'dnn_reco.modules.models.{}.{}'.format(
                                self.config['model_file'],
                                self.config['model_name'],
                                )
        nn_model = misc.load_class(class_string)

        y_pred_trafo, y_unc_trafo, model_vars_pred, model_vars_unc = nn_model(
                                        is_training=self.is_training,
                                        config=self.config,
                                        data_handler=self.data_handler,
                                        data_transformer=self.data_transformer,
                                        shared_objects=self.shared_objects)

        # transform back
        y_pred = self.data_transformer.inverse_transform(y_pred_trafo,
                                                         data_type='label')
        y_unc = self.data_transformer.inverse_transform(y_unc_trafo,
                                                        data_type='label',
                                                        bias_correction=False)

        self.shared_objects['y_pred_trafo'] = y_pred_trafo
        self.shared_objects['y_unc_trafo'] = y_unc_trafo
        self.shared_objects['y_pred'] = y_pred
        self.shared_objects['y_unc'] = y_unc
        self.shared_objects['model_vars_pred'] = model_vars_pred
        self.shared_objects['model_vars_unc'] = model_vars_unc
        self.shared_objects['model_vars'] = model_vars_pred + model_vars_unc

        # count number of trainable parameters
        print('Number of free parameters in NN model: {}'.format(
                    self.count_parameters(self.shared_objects['model_vars'])))

        # create saver
        self.saver = tf.train.Saver(self.shared_objects['model_vars'])

    def count_parameters(self, var_list=None):
        """Count number of trainable parameters

        Parameters
        ----------
        var_list : list of tf.Tensors, optional
            A list of tensorflow tensors for which to calculate the nubmer of
            trainable parameters. If None, then all trainable parameters
            available will be counted.

        Returns
        -------
        int
            Number of trainable parameters
        """
        if var_list is None:
            var_list = tf.trainable_variables()
        return np.sum([np.prod(x.get_shape().as_list()) for x in var_list])
