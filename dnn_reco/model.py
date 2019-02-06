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
                                        self.config['tf_float_precision'],
                                        name='keep_prob_{:0.2f}'.format(i))
                              for i in self.config['keep_probability_list']]
            self.shared_objects['keep_prob_list'] = keep_prob_list

        # IC78: main IceCube array
        self.shared_objects['x_ic78'] = tf.placeholder(
                        self.config['tf_float_precision'],
                        shape=[None, 10, 10, 60,  self.data_handler.num_bins],
                        name='x_ic78',
                        )
        self.shared_objects['x_ic78_trafo'] = self.data_transformer.transform(
                            self.shared_objects['x_ic78'], data_type='ic78')

        # DeepCore
        self.shared_objects['x_deepcore'] = tf.placeholder(
                        self.config['tf_float_precision'],
                        shape=[None, 8, 60,  self.data_handler.num_bins],
                        name='x_deepcore',
                        )
        self.shared_objects['x_deepcore_trafo'] = \
            self.data_transformer.transform(self.shared_objects['x_deepcore'],
                                            data_type='deepcore')

        # labels
        self.shared_objects['y_true'] = tf.placeholder(
                        self.config['tf_float_precision'],
                        shape=[None] + self.data_handler.label_shape,
                        name='y_true',
                        )
        self.shared_objects['y_true_trafo'] = self.data_transformer.transform(
                        self.shared_objects['y_true'], data_type='label')

        # misc data
        if self.data_handler.misc_shape is not None:
            self.shared_objects['x_misc'] = tf.placeholder(
                        self.config['tf_float_precision'],
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

    def _create_label_weights(self):
        """Create label weights and update operation
        """
        weights = np.ones(self.data_handler.label_shape)

        if 'label_weight_dict' in self.config:
            for key in self.config['label_weight_dict'].keys():
                weights[self.data_handler.get_label_index(key)] = \
                    self.config['label_weight_dict'][key]

        if self.config['label_update_weights']:
            label_weights = tf.Variable(
                                    weights,
                                    name='label_weights',
                                    trainable=False,
                                    dtype=self.config['tf_float_precision'])

            new_label_weight_values = tf.placeholder(
                                        self.config['tf_float_precision'],
                                        shape=self.data_handler.label_shape,
                                        name='new_label_weight_values')

            label_weight_decay = 0.5
            self.shared_objects['assign_new_label_weights'] = \
                label_weights.assign(
                                label_weights * (1. - label_weight_decay) +
                                new_label_weight_values * label_weight_decay)

        else:
            label_weights = tf.constant(
                                    weights,
                                    shape=self.data_handler.label_shape,
                                    dtype=self.config['tf_float_precision'])
        self.shared_objects['label_weights'] = label_weights

        tf.summary.histogram('label_weights', label_weights)
        tf.summary.scalar('label_weights_benchmark',
                          tf.reduce_sum(label_weights))

        misc.print_warning('Total Benchmark should be: {:3.3f}'.format(
                                                                sum(weights)))

    def _get_optimizers_and_loss(self):
        optimizer_dict = dict(self.config['model_optimizer_dict'])

        # create each optimizer
        for name, opt_config in optimizer_dict:

            # sanity check: make sure loss file and name have same length
            if isinstance(opt_config['loss_file'], str):
                assert isinstance(opt_config['loss_name'], str)
                opt_config['loss_file'] = [opt_config['loss_file']]
                opt_config['loss_name'] = [opt_config['loss_name']]

            assert len(opt_config['loss_file']) == len(opt_config['loss_name'])

            # aggregate over all defined loss functions
            label_loss = None
            for file, name in zip(opt_config['loss_file'],
                                  opt_config['loss_name']):

                # get loss function
                class_string = 'dnn_reco.modules.loss.{}.{}'.format(file, name)
                loss_function = misc.load_class(class_string)

                # compute loss
                label_loss_i = loss_function(
                                        config=self.config,
                                        data_handler=self.data_handler,
                                        data_transformer=self.data_transformer,
                                        shared_objects=self.shared_objects)

                loss_shape = label_loss_i.get_shape().as_list()
                if loss_shape != self.data_handler.label_shape:
                    error_msg = 'Shape of label loss {!r} does not match {!r}'
                    raise ValueError(error_msg.format(
                                                loss_shape,
                                                self.data_handler.label_shape))

                if label_loss is None:
                    label_loss = label_loss_i
                else:
                    label_loss += label_loss_i

            # weight label_losses
            weighted_label_loss = label_losses*shared_objects['label_weights']
            weighted_loss_sum = tf.reduce_sum(weighted_label_loss)

            self.shared_objects['label_loss_dict'] = {
                'loss_label_' + name: weighted_label_loss,
                'loss_sum_' + name: weighted_loss_sum,
            }

            tf.summary.histogram('loss_label_' + name, weighted_label_loss)
            tf.summary.scalar('loss_sum_' + name, weighted_loss_sum)

            # get optimizer
            optimizer = getattr(tf.train,
                                opt_config['optimizer'])(
                                **opt_config['optimizer_settings']
                                )

            # get variable list
            if isinstance(opt_config['vars'], str):
                opt_config['vars'] = [opt_config['vars']]
            var_list =
            gvs = optimizer.compute_gradients(weighted_loss_sum,
                                              var_list=generator.model_vars)
            clip_gradients = False
            if clip_gradients:
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in gvs]
            else:
                capped_gvs = gvs
            self.generator_optimizer = optimizer.apply_gradients(capped_gvs)
            if self._config['generator_perform_training']:
                self.optimizers.append(self.generator_optimizer)

        print(optimizer_list)

    def compile(self):

        # create label_weights and assign op
        self._create_label_weights()

        self._get_optimizers_and_loss()

        # tukey scaling
        # non_zero_mask
        # med_abs_dev
        # update importance

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
