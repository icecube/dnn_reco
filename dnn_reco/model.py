from __future__ import division, print_function
import os
import tensorflow as tf
import numpy as np
import ruamel.yaml as yaml
import click
import timeit
import glob

from dnn_reco import misc
from dnn_reco.modules.loss.utils import loss_utils


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
        self._model_is_compiled = False
        self._step_offset = 0
        self.is_training = is_training
        self.config = config
        self.data_handler = data_handler
        self.data_transformer = data_transformer

        if self.is_training:
            # create necessary directories
            self._setup_directories()

            # create necessary variables to save training config files
            self._setup_training_config_saver()

        self.shared_objects = {}

        # create tensorflow placeholders for input data
        self._setup_placeholders()

        # initalize label weights and non zero mask
        self._intialize_label_weights()

        # build NN architecture
        self._build_model()

        # get or create new default session
        sess = tf.get_default_session()
        if sess is None:
            if 'tf_parallelism_threads' in self.config:
                n_cpus = self.config['tf_parallelism_threads']
                sess = tf.Session(config=tf.ConfigProto(
                            gpu_options=tf.GPUOptions(allow_growth=True),
                            device_count={'GPU': 1},
                            intra_op_parallelism_threads=n_cpus,
                            inter_op_parallelism_threads=n_cpus,
                          )).__enter__()
            else:
                sess = tf.Session(config=tf.ConfigProto(
                            gpu_options=tf.GPUOptions(allow_growth=True),
                            device_count={'GPU': 1},
                          )).__enter__()
        self.sess = sess
        tf.set_random_seed(self.config['tf_random_seed'])

    def _setup_directories(self):
        """Creates necessary directories
        """
        # Create directories
        directories = [self.config['model_checkpoint_path'],
                       self.config['log_path'],
                       ]
        for directory in directories:
            directory = os.path.dirname(directory)
            if not os.path.isdir(directory):
                os.makedirs(directory)
                misc.print_warning('Creating directory: {}'.format(directory))

    def _setup_training_config_saver(self):
        """Setup variables and check training step in order to save the
        training config during training.

        Previous training configs and training step files will only be deleted
        if the model is actually being overwritten, e.g. if model is saved
        in the model.fit method (further below).
        These will not yet be deleted here.
        """
        self._check_point_path = os.path.dirname(self.config[
                                                    'model_checkpoint_path'])
        self._training_steps_file = os.path.join(self._check_point_path,
                                                 'training_steps.yaml')

        # Load training iterations dict
        if os.path.isfile(self._training_steps_file):
            self._training_iterations_dict = yaml.safe_load(
                                            open(self._training_steps_file))
        else:
            misc.print_warning('Did not find {!r}. Creating new one'.format(
                self._training_steps_file))
            self._training_iterations_dict = {}

        # get the training step number
        if self.config['model_restore_model']:
            files = glob.glob(os.path.join(self._check_point_path,
                                           'config_training_*.yaml'))
            if files:
                max_file = os.path.basename(np.sort(files)[-1])
                self._training_step = int(max_file.replace(
                                        'config_training_',
                                        '').replace('.yaml', '')) + 1
            else:
                self._training_step = 0
        else:
            self._training_iterations_dict = {}
            self._training_step = 0

        self._training_config_file = os.path.join(
                    self._check_point_path,
                    'config_training_{:04d}.yaml'.format(self._training_step))

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

    def _intialize_label_weights(self):
        """Initialize label weights and non zero mask
        """
        label_weight_config = np.ones(self.data_handler.label_shape)
        label_weight_config *= self.config['label_weight_initialization']

        if 'label_weight_dict' in self.config:
            for key in self.config['label_weight_dict'].keys():
                label_weight_config[self.data_handler.get_label_index(key)] = \
                    self.config['label_weight_dict'][key]
        self.shared_objects['label_weight_config'] = label_weight_config
        self.shared_objects['non_zero_mask'] = label_weight_config > 0

    def _create_label_weights(self):
        """Create label weights and update operation
        """
        if self.config['label_update_weights']:
            label_weights = tf.Variable(
                                    self.shared_objects['label_weight_config'],
                                    name='label_weights',
                                    trainable=False,
                                    dtype=self.config['tf_float_precision'])

            self.shared_objects['new_label_weight_values'] = tf.placeholder(
                                        self.config['tf_float_precision'],
                                        shape=self.data_handler.label_shape,
                                        name='new_label_weight_values')

            label_weight_decay = 0.5
            self.shared_objects['assign_new_label_weights'] = \
                label_weights.assign(
                                label_weights * (1. - label_weight_decay) +
                                self.shared_objects['new_label_weight_values']
                                * label_weight_decay)

            # calculate MSE for each label
            y_diff_trafo = loss_utils.get_y_diff_trafo(
                                    config=self.config,
                                    data_handler=self.data_handler,
                                    data_transformer=self.data_transformer,
                                    shared_objects=self.shared_objects)

            self.shared_objects['mse_values_trafo'] = tf.reduce_mean(
                                                    tf.square(y_diff_trafo), 0)

        else:
            label_weights = tf.constant(
                                    self.shared_objects['label_weight_config'],
                                    shape=self.data_handler.label_shape,
                                    dtype=self.config['tf_float_precision'])

        self.shared_objects['label_weights'] = label_weights
        self.shared_objects['label_weights_benchmark'] = tf.reduce_sum(
                                                            label_weights)

        tf.summary.histogram('label_weights', label_weights)
        tf.summary.scalar('label_weights_benchmark',
                          self.shared_objects['label_weights_benchmark'])

        misc.print_warning('Total Benchmark should be: {:3.3f}'.format(
                            sum(self.shared_objects['label_weight_config'])))

    def _get_optimizers_and_loss(self):
        """Get optimizers and loss terms as defined in config.

        Raises
        ------
        ValueError
            Description
        """
        optimizer_dict = dict(self.config['model_optimizer_dict'])

        # create empy list to hold tensorflow optimizer operations
        optimizer_ops = []

        # create each optimizer
        for name, opt_config in optimizer_dict.items():

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

                # sanity check: make sure loss has expected shape
                loss_shape = label_loss_i.get_shape().as_list()
                if loss_shape != self.data_handler.label_shape:
                    error_msg = 'Shape of label loss {!r} does not match {!r}'
                    raise ValueError(error_msg.format(
                                                loss_shape,
                                                self.data_handler.label_shape))

                # accumulate loss terms
                if label_loss is None:
                    label_loss = label_loss_i
                else:
                    label_loss += label_loss_i

            # weight label_losses
            label_loss = tf.where(self.shared_objects['non_zero_mask'],
                                  label_loss, tf.zeros_like(label_loss))
            weighted_label_loss = label_loss * self.shared_objects[
                                                            'label_weights']
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

            var_list = []
            for var_name in opt_config['vars']:
                var_list.extend(self.shared_objects['model_vars_' + var_name])

            gvs = optimizer.compute_gradients(weighted_loss_sum,
                                              var_list=var_list)
            clip_gradients = False
            if clip_gradients:
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in gvs]
            else:
                capped_gvs = gvs
            optimizer_ops.append(optimizer.apply_gradients(capped_gvs))

        self.shared_objects['optimizer_ops'] = optimizer_ops

    def _merge_tensorboard_summaries(self):
        """Merges summary variables for TensorBoard visualization and logging.
        """
        self._merged_summary = tf.summary.merge_all()
        self._train_writer = tf.summary.FileWriter(
                           self.config['log_path'] + '_train')
        self._val_writer = tf.summary.FileWriter(
                           self.config['log_path'] + '_val')

    def _initialize_and_finalize_model(self):
        """Initalize and finalize model weights
        """

        # initialize variables
        self.sess.run(tf.global_variables_initializer())

        # finalize model: forbid any changes to computational graph
        tf.get_default_graph().finalize()

    def compile(self):

        if self.is_training:

            # create label_weights and assign op
            self._create_label_weights()

            self._get_optimizers_and_loss()

            self._merge_tensorboard_summaries()

        self._initialize_and_finalize_model()

        self._model_is_compiled = True

        # tukey scaling
        # med_abs_dev
        # update importance

    def restore(self):
        """Restore model weights from checkpoints
        """
        latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(
                                self.config['model_checkpoint_path']))
        if latest_checkpoint is None:
            misc.print_warning(
                    'Could not find previous checkpoint. Creating new one!')
        else:
            self._step_offset = int(latest_checkpoint.split('-')[-1])
            self.saver.restore(sess=self.sess, save_path=latest_checkpoint)

    def predict(self, x_ic78, x_deepcore, *args, **kwargs):
        """Reconstruct events.

        Parameters
        ----------
        x_ic78 : float, list or numpy.ndarray
            The input data for the main IceCube array.
        x_deepcore : float, list or numpy.ndarray
            The input data for the DeepCore array.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        feed_dict = {
            self.shared_objects['x_ic78']: x_ic78,
            self.shared_objects['x_deepcore']: x_deepcore,
        }

        # Fill in keep rates for dropout
        for keep_prob in self.shared_objects['keep_prob_list']:
            feed_dict[keep_prob] = 1.0

        y_pred, y_unc = self.sess.run([self.shared_objects['y_pred'],
                                       self.shared_objects['y_unc']],
                                      feed_dict=feed_dict)

        return y_pred, y_unc

    def _feed_placeholders(self, data_generator, is_validation):
        """Feed placeholder variables with a batch from the data_generator

        Parameters
        ----------
        data_generator : generator object
            A python generator object which generates batches of input data.
        is_validation : bool
            Description

        Returns
        -------
        dict
            The feed dictionary for the tf.session.run call.
        """
        x_ic78, x_deepcore, labels, misc_data = next(data_generator)

        feed_dict = {
            self.shared_objects['x_ic78']: x_ic78,
            self.shared_objects['x_deepcore']: x_deepcore,
            self.shared_objects['y_true']: labels,
        }
        if self.data_handler.misc_shape is not None:
            feed_dict[self.shared_objects['x_misc']] = misc_data

        # Fill in keep rates for dropout
        if is_validation:
            for keep_prob in self.shared_objects['keep_prob_list']:
                feed_dict[keep_prob] = 1.0
        else:
            for keep_prob, prob in zip(self.shared_objects['keep_prob_list'],
                                       self.config['keep_probability_list']):
                feed_dict[keep_prob] = prob

        return feed_dict

    def fit(self, num_training_iterations, train_data_generator,
            val_data_generator,
            evaluation_methods=None,
            *args, **kwargs):
        """Trains the NN model with the data provided by the data iterators.

        Parameters
        ----------
        num_training_iterations : int
            The number of training iterations to perform.
        train_data_generator : generator object
            A python generator object which generates batches of training data.
        val_data_generator : generator object
            A python generator object which generates batches of validation
            data.
        evaluation_methods : None, optional
            Description
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Raises
        ------
        ValueError
            Description
        """
        if not self._model_is_compiled:
            raise ValueError(
                        'Model must be compiled prior to call of fit method')

        # training operations to run
        train_ops = {'optimizer_{:03d}'.format(i): opt for i, opt in
                     enumerate(self.shared_objects['optimizer_ops'])}

        # add parameters and op if label weights are to be updated
        if self.config['label_update_weights']:

            label_weight_n = 0.
            label_weight_mean = np.zeros(self.data_handler.label_shape)
            label_weight_M2 = np.zeros(self.data_handler.label_shape)

            train_ops['mse_values_trafo'] = \
                self.shared_objects['mse_values_trafo']

        # ----------------
        # training loop
        # ----------------
        start_time = timeit.default_timer()
        for i in range(num_training_iterations):

            feed_dict = self._feed_placeholders(train_data_generator,
                                                is_validation=False)
            train_result = self.sess.run(train_ops,
                                         feed_dict=feed_dict)

            # --------------------------------------------
            # calculate online variabels for label weights
            # --------------------------------------------
            if self.config['label_update_weights']:
                mse_values_trafo = train_result['mse_values_trafo']
                mse_values_trafo[~self.shared_objects['non_zero_mask']] = 1.

                if np.isfinite(mse_values_trafo).all():
                    label_weight_n += 1
                    delta = mse_values_trafo - label_weight_mean
                    label_weight_mean += delta / label_weight_n
                    delta2 = mse_values_trafo - label_weight_mean
                    label_weight_M2 += delta * delta2
                else:
                    misc.print_warning('Found NaNs: {}'.format(
                                       mse_values_trafo))
                    for i, name in enumerate(self.data_handler.label_names):
                        print(name, mse_values_trafo[i])

                # every n steps: update label_weights
                if i % self.config['validation_frequency'] == 0:
                    new_weights = 1.0 / (np.sqrt(label_weight_mean) + 1e-3)
                    new_weights[new_weights < 1] = 1
                    new_weights *= self.shared_objects['label_weight_config']

                    # assign new label weight updates
                    feed_dict_assign = {
                        self.shared_objects['new_label_weight_values']:
                            new_weights}

                    self.sess.run(
                            self.shared_objects['assign_new_label_weights'],
                            feed_dict=feed_dict_assign)
                    updated_weights = self.sess.run(
                                        self.shared_objects['label_weights'])

                    # reset values
                    label_weight_n = 0.
                    label_weight_mean = np.zeros(self.data_handler.label_shape)
                    label_weight_M2 = np.zeros(self.data_handler.label_shape)

            # ----------------
            # validate performance
            # ----------------
            if i % self.config['validation_frequency'] == 0:
                eval_dict = {
                    'merged_summary': self._merged_summary,
                    'weights': self.shared_objects['label_weights_benchmark'],
                    'rmse_trafo': self.shared_objects['rmse_values_trafo'],
                    'y_pred': self.shared_objects['y_pred'],
                    'y_unc': self.shared_objects['y_unc'],
                }
                result_msg = ''
                for k, loss in self.shared_objects['label_loss_dict'].items():
                    if k[:9] == 'loss_sum_':
                        eval_dict[k] = loss
                        result_msg += k + ': {' + k + ':2.3f} '

                # -------------------------------------
                # Test performance on training data
                # -------------------------------------
                feed_dict_train = self._feed_placeholders(train_data_generator,
                                                          is_validation=True)
                results_train = self.sess.run(eval_dict,
                                              feed_dict=feed_dict_train)

                # -------------------------------------
                # Test performance on validation data
                # -------------------------------------
                feed_dict_val = self._feed_placeholders(val_data_generator,
                                                        is_validation=True)
                results_val = self.sess.run(eval_dict, feed_dict=feed_dict_val)

                self._train_writer.add_summary(
                                        results_train['merged_summary'], i)
                self._val_writer.add_summary(results_val['merged_summary'], i)
                msg = 'Step: {:08d}, Runtime: {:2.2f}s, Benchmark: {:3.3f}'
                print(msg.format(i, timeit.default_timer() - start_time,
                                 np.sum(updated_weights)))
                print('\t[Train]      '+result_msg.format(**results_train))
                print('\t[Validation] '+result_msg.format(**results_val))

                # print info for each label
                for name, index in self.data_handler.label_name_dict.items():
                    if updated_weights[index] > 0:
                        msg = '\tweight: {weight:2.3f},'
                        msg += ' train: {train:2.3f}, val: {val:2.3f} [{name}]'
                        print(msg.format(
                                    weight=updated_weights[index],
                                    train=results_train['rmse_trafo'][index],
                                    val=results_val['rmse_trafo'][index],
                                    name=name,
                                    ))

                # Call user defined evaluation method
                if self.config['evaluation_file'] is not None:
                    class_string = 'dnn_reco.modules.evaluation.{}.{}'.format(
                                self.config['evaluation_file'],
                                self.config['evaluation_name'],
                                )
                    eval_func = misc.load_class(class_string)
                    eval_func(feed_dict_train=feed_dict_train,
                              feed_dict_val=feed_dict_val,
                              results_train=results_train,
                              results_val=results_val,
                              config=self.config,
                              data_handler=self.data_handler,
                              data_transformer=self.data_transformer,
                              shared_objects=self.shared_objects)

            # ----------------
            # save models
            # ----------------
            if i % self.config['save_frequency'] == 0:
                self._save_training_config(i)
                if self.config['model_save_model']:
                    self.saver.save(
                            sess=self.sess,
                            global_step=self._step_offset + i,
                            save_path=self.config['model_checkpoint_path'])
            # ----------------

    def _save_training_config(self, iteration):
        """Save Training config and iterations to file.

        Parameters
        ----------
        iteration : int
            The current training iteration.

        Raises
        ------
        ValueError
            Description
        """
        if iteration == 0:
            if not self.config['model_restore_model']:
                # Delete old training config files and create a new and empty
                # training_steps.txt, since we are training a new model
                # from scratch and overwriting the old one

                # delete all previous training config files (if they exist)
                files = glob.glob(os.path.join(self._check_point_path,
                                               'config_training_*.yaml'))
                if files:
                    misc.print_warning("Please confirm the deletion of the "
                                       "previous trainin configs:")
                for file in files:
                    if click.confirm('Delete {!r}?'.format(file),
                                     default=True):
                        os.remove(file)
                    else:
                        raise ValueError(
                                    'Old training configs must be deleted!')

            # save training step config under appropriate name
            training_config = dict(self.config)
            del training_config['np_float_precision']
            del training_config['tf_float_precision']

            with open(self._training_config_file, 'w') as yaml_file:
                yaml.dump(training_config, yaml_file, default_flow_style=False)

        # update number of training iterations in training_steps.yaml
        self._training_iterations_dict[self._training_step] = iteration
        with open(self._training_steps_file, 'w') as yaml_file:
            yaml.dump(self._training_iterations_dict, yaml_file,
                      default_flow_style=False)

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
