import os
import tensorflow as tf
import numpy as np
import click
import timeit
import glob
import logging
from copy import deepcopy

from dnn_reco import misc
from dnn_reco.settings import yaml
from dnn_reco.modules.loss.utils import loss_utils


class BaseNNModel(tf.Module):
    """Base class for neural network architecture

    Derived classes must implement the __call__ method,
    create all necessary variables in the __init__ method,
    and set the following attributes in the __init__ method:
        - self.vars_pred: a list of all variables used for the prediction
        - self.vars_unc: a list of all variables used for the uncertainty

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
        A dictionary containing settings and objects that are shared and passed
        on to sub modules.
    """

    def __init__(
        self,
        is_training,
        config,
        data_handler,
        data_transformer,
        dtype: str = "float32",
        logger=None,
        name: str | None = None,
    ) -> None:
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
        dtype : str, optional
            The data type to use for the model, by default "float32".
        logger : logging.logger, optional
            A logging instance.
        name : str, optional
            The name of the model, by default "BaseNNModel".
        """
        if name is None:
            name = self.__class__.__name__
        tf.Module.__init__(self, name=name)

        self.dtype = dtype
        self._logger = logger or logging.getLogger(
            misc.get_full_class_string_of_object(self)
        )

        self._model_is_compiled = False
        self.is_training = is_training
        self.config = dict(deepcopy(config))
        self.data_handler = data_handler
        self.data_transformer = data_transformer
        self.vars_pred = None
        self.vars_unc = None

        self.shared_objects = {}

        if self.is_training:
            # create necessary directories
            self._setup_directories()

            # create necessary variables to save training config files
            self._setup_training_config_saver()

            # create summary writers
            self._train_writer = tf.summary.create_file_writer(
                os.path.join(self.config["log_path"], "train")
            )
            self._val_writer = tf.summary.create_file_writer(
                os.path.join(self.config["log_path"], "val")
            )

        # create label weights and non zero mask
        self._create_label_weights()

        # create variables necessary for tukey loss
        self._create_tukey_vars()

    def _setup_directories(self):
        """Creates necessary directories"""
        # Create directories
        directories = [
            self.config["model_checkpoint_path"],
            self.config["log_path"],
        ]
        for directory in directories:
            directory = os.path.dirname(directory)
            if not os.path.isdir(directory):
                os.makedirs(directory)
                misc.print_warning("Creating directory: {}".format(directory))

    def _setup_training_config_saver(self):
        """Setup variables and check training step in order to save the
        training config during training.

        Previous training configs and training step files will only be deleted
        if the model is actually being overwritten, e.g. if model is saved
        in the model.fit method (further below).
        These will not yet be deleted here.
        """
        self._check_point_path = os.path.dirname(
            self.config["model_checkpoint_path"]
        )
        self._training_steps_file = os.path.join(
            self._check_point_path, "training_steps.yaml"
        )

        # Load training iterations dict
        if os.path.isfile(self._training_steps_file):
            with open(self._training_steps_file, "r") as stream:
                self._training_iterations_dict = yaml.yaml_loader.load(stream)
        else:
            misc.print_warning(
                "Did not find {!r}. Creating new one".format(
                    self._training_steps_file
                )
            )
            self._training_iterations_dict = {}

        # get the training step number
        if self.config["model_restore_model"]:
            files = glob.glob(
                os.path.join(self._check_point_path, "config_training_*.yaml")
            )
            if files:
                max_file = os.path.basename(np.sort(files)[-1])
                self._training_step = (
                    int(
                        max_file.replace("config_training_", "").replace(
                            ".yaml", ""
                        )
                    )
                    + 1
                )
            else:
                self._training_step = 0
        else:
            self._training_iterations_dict = {}
            self._training_step = 0

        self._training_config_file = os.path.join(
            self._check_point_path,
            "config_training_{:04d}.yaml".format(self._training_step),
        )

    def _create_label_weights(self):
        """Create label weights and non zero mask"""
        label_weight_config = np.ones(self.data_handler.label_shape)
        label_weight_config *= self.config["label_weight_initialization"]

        if "label_weight_dict" in self.config:
            for key in self.config["label_weight_dict"].keys():
                label_weight_config[self.data_handler.get_label_index(key)] = (
                    self.config["label_weight_dict"][key]
                )
        self.shared_objects["label_weight_config"] = label_weight_config
        self.shared_objects["non_zero_mask"] = label_weight_config > 0

        if self.config["label_update_weights"]:
            label_weights = tf.Variable(
                self.shared_objects["label_weight_config"],
                name="label_weights",
                trainable=False,
                dtype=self.dtype,
            )
        else:
            label_weights = tf.constant(
                self.shared_objects["label_weight_config"],
                shape=self.data_handler.label_shape,
                dtype=self.dtype,
            )

        self.shared_objects["label_weights"] = label_weights

        if self.is_training:
            misc.print_warning(
                "Total Benchmark should be: {:3.3f}".format(
                    sum(self.shared_objects["label_weight_config"])
                )
            )

    def _update_tukey_vars(self, new_values, tukey_decay=0.001):
        """Update tukey variables"""
        self.shared_objects["median_abs_dev"].assign(
            self.shared_objects["median_abs_dev"] * (1.0 - tukey_decay)
            + new_values * tukey_decay
        )

    def _create_tukey_vars(self):
        """Create variables required for tukey loss"""
        if self.config["label_scale_tukey"]:
            median_abs_dev = tf.Variable(
                np.ones(shape=self.data_handler.label_shape) * 0.67449,
                name="median_abs_dev",
                trainable=False,
                dtype=self.dtype,
            )

        else:
            median_abs_dev = tf.constant(
                np.ones(shape=self.data_handler.label_shape) * 0.67449,
                shape=self.data_handler.label_shape,
                dtype=self.dtype,
            )

        self.shared_objects["median_abs_dev"] = median_abs_dev

    def _update_label_weights(
        self,
        new_values,
        label_weight_decay=0.5,
        summary_writer=None,
    ):
        """Update label weights"""
        label_weights = self.shared_objects["label_weights"]
        label_weights.assign(
            label_weights * (1.0 - label_weight_decay)
            + new_values * label_weight_decay
        )

        if summary_writer is not None:
            with summary_writer.as_default():
                tf.summary.histogram(
                    "label_weights", label_weights, step=self.step
                )
                tf.summary.scalar(
                    "label_weights_benchmark",
                    tf.reduce_sum(label_weights),
                    step=self.step,
                )

    def _get_event_weights(self, shared_objects):
        """Compute event weights"""
        event_weights = None

        if (
            "event_weight_class" in self.config
            and self.config["event_weight_class"] is not None
        ):

            # get event weight function
            event_weight_function = misc.load_class(
                self.config["event_weight_class"],
            )

            # compute loss
            event_weights = event_weight_function(
                config=self.config,
                data_handler=self.data_handler,
                data_transformer=self.data_transformer,
                shared_objects=shared_objects,
            )

            shape = event_weights.get_shape().as_list()
            assert (
                len(shape) == 2 and shape[1] == 1
            ), "Expected shape [-1, 1] but got {!r}".format(shape)
        return event_weights

    @tf.function
    def _compile_optimizers(self):
        """Compile the optimizer by running it with zero gradients."""
        for optimizer in self.optimizers.values():
            zero_grads = [tf.zeros_like(w) for w in self.trainable_variables]
            optimizer.apply_gradients(
                zip(zero_grads, self.trainable_variables)
            )

    def _create_optimizers(self):
        """Create optimizers"""
        optimizer_dict = dict(self.config["model_optimizer_dict"])

        # create empty list to hold tensorflow optimizer operations
        self.optimizers = {}

        # create each optimizer
        for name, opt_config in sorted(optimizer_dict.items()):

            optimizer_settings = dict(opt_config["optimizer_settings"])

            # create learning rate schedule if learning rate is a dict
            if "learning_rate" in optimizer_settings:
                if isinstance(optimizer_settings["learning_rate"], dict):

                    # assume that the learning rate dictionary defines a schedule
                    # In this case the dictionary must have the following keys:
                    #   full_class_string: str
                    #       The full class string of the scheduler class to use.
                    #   settings: dict
                    #       keyword arguments that are passed on to the scheduler
                    #       class.
                    lr_cfg = optimizer_settings.pop("learning_rate")
                    scheduler_class = misc.load_class(
                        lr_cfg["full_class_string"]
                    )
                    scheduler = scheduler_class(**lr_cfg["settings"])
                    optimizer_settings["learning_rate"] = scheduler

            self.optimizers[name] = getattr(
                tf.optimizers, opt_config["optimizer"]
            )(**optimizer_settings)

        # run optimizers with zero gradients to create optimizer variables
        self._compile_optimizers()

    @tf.function(reduce_retracing=True)
    def __call__(self, data_batch_dict, is_training=True, summary_writer=None):
        """Forward pass through the model.

        This method is to be implemented by derived class.

        Parameters
        ----------
        data_batch_dict : dict
            A dictionary containing the input data.
            This includes:
                x_ic78, x_deepcore, y_true, x_misc,
                and the transformed versions of these.
        is_training : bool, optional
            True if model is in training mode, false if in inference mode.
        summary_writer : tf.summary.SummaryWriter, optional
            A summary writer to write summaries to.

        Returns
        -------
        dict
            A dictionary containing the model predictions.
            This must at least contain the following:
                "y_pred_trafo": tf.Tensor
                    The (transformed) predicted values, i.e. in
                    normalized space.
                "y_unc_pred_trafo": tf.Tensor
                    The (transformed) predicted uncertainties, i.e. in
                    normalized space.
        """
        raise NotImplementedError

    @tf.function(reduce_retracing=True)
    def get_tensors(self, data_batch, is_training=True, summary_writer=None):
        """Get result tensors from the model.

        Performs a forward pass through the model and adds
        additional auxiliary tensors to the result dictionary.

        Parameters
        ----------
        data_batch : list
            A list containing the input data.
            This is typically: x_ic78, x_deepcore, labels, misc_data
        is_training : bool, optional
            True if model is in training mode, false if in inference mode.
        summary_writer : tf.summary.SummaryWriter, optional
            A summary writer to write summaries to.

        Returns
        -------
        dict
            A dictionary containing the model predictions and
            auxiliary tensors.
        """
        print("Tracing get_tensors")
        x_ic78, x_deepcore, labels, misc_data = data_batch

        x_ic78 = tf.convert_to_tensor(x_ic78, dtype=self.dtype)
        x_deepcore = tf.convert_to_tensor(x_deepcore, dtype=self.dtype)
        labels = tf.convert_to_tensor(labels, dtype=self.dtype)
        misc_data = tf.convert_to_tensor(misc_data, dtype=self)

        data_batch_dict = {
            "x_ic78": x_ic78,
            "x_deepcore": x_deepcore,
            "y_true": labels,
            "x_misc": misc_data,
            "x_ic78_trafo": self.data_transformer.transform(
                x_ic78, data_type="ic78"
            ),
            "x_deepcore_trafo": self.data_transformer.transform(
                x_deepcore, data_type="deepcore"
            ),
            "y_true_trafo": self.data_transformer.transform(
                labels, data_type="label"
            ),
            "x_misc_trafo": self.data_transformer.transform(
                misc_data, data_type="misc"
            ),
        }
        result_tensors = self(data_batch_dict, is_training=is_training)

        # sanity check to verify contents of the result_tensors
        must_keys = ["y_pred_trafo", "y_unc_pred_trafo"]
        for key in must_keys:
            if key not in result_tensors:
                raise ValueError(f"Key '{key}' not found in result_tensors.")

        avoid_keys = [
            "x_ic78",
            "x_deepcore",
            "y_true",
            "x_misc",
            "x_ic78_trafo",
            "x_deepcore_trafo",
            "y_true_trafo",
            "x_misc_trafo",
            "y_pred",
            "y_unc_pred",
            "event_weights",
            "y_diff_trafo",
            "mse_values_trafo",
            "rmse_values_trafo",
            "label_weight_config",
            "non_zero_mask",
            "label_weights",
            "median_abs_dev",
            "label_loss_dict",
            "mse_values",
            "rmse_values",
            "y_diff",
        ]
        for key in avoid_keys:
            if key in result_tensors:
                raise ValueError(
                    f"Key '{key}' is not allowed in result_tensors."
                )

        # add data_batch_dict to result_tensors
        result_tensors.update(data_batch_dict)

        # add event weights
        event_weights = self._get_event_weights(data_batch_dict)
        if event_weights is not None:
            result_tensors["event_weights"] = event_weights

        # compute auxiliary tensors
        result_tensors["y_diff_trafo"] = loss_utils.get_y_diff_trafo(
            config=self.config,
            data_handler=self.data_handler,
            data_transformer=self.data_transformer,
            shared_objects=result_tensors,
        )
        result_tensors["mse_values_trafo"] = tf.reduce_mean(
            result_tensors["y_diff_trafo"] ** 2,
            axis=0,
        )
        result_tensors["rmse_values_trafo"] = tf.sqrt(
            result_tensors["mse_values_trafo"]
        )

        # transform back
        result_tensors["y_pred"] = self.data_transformer.inverse_transform(
            result_tensors["y_pred_trafo"], data_type="label"
        )
        result_tensors["y_unc"] = self.data_transformer.inverse_transform(
            result_tensors["y_unc_pred_trafo"],
            data_type="label",
            bias_correction=False,
        )

        # calculate RMSE of untransformed values
        result_tensors["y_diff"] = (
            result_tensors["y_pred"] - result_tensors["y_true"]
        )
        result_tensors["mse_values"] = tf.reduce_mean(
            result_tensors["y_diff"] ** 2, axis=0
        )
        result_tensors["rmse_values"] = tf.sqrt(result_tensors["mse_values"])

        if summary_writer is not None:
            y_pred_list = tf.unstack(result_tensors["y_pred"], axis=1)
            for i, name in enumerate(self.data_handler.label_names):
                with summary_writer.as_default():
                    tf.summary.histogram(
                        "y_pred_" + name, y_pred_list[i], step=self.step
                    )

            # add the RMSE of each label as a tf.summary.scalar
            for i, name in enumerate(self.data_handler.label_names):
                tf.summary.scalar(
                    "RMSE_" + name,
                    result_tensors["rmse_values"][i],
                    step=self.step,
                )

            tf.summary.scalar(
                "Benchmark",
                tf.reduce_sum(
                    input_tensor=result_tensors["rmse_values_trafo"], axis=0
                ),
                step=self.step,
            )

        return result_tensors

    @tf.function(reduce_retracing=True)
    def get_loss(self, data_batch, is_training=True, summary_writer=None):
        """Get optimizers and loss terms as defined in config.

        Parameters
        ----------
        data_batch : list
            A list containing the input data, containing:
                x_ic78, x_deepcore, labels, misc_data
        is_training : bool, optional
            True if model is in training mode, false if in inference mode.
        summary_writer : tf.summary.SummaryWriter, optional
            A summary writer to write summaries to.

        Returns
        -------
        dict
            A dictionary containing the model predictions and
            auxiliary tensors.
        """
        print("Tracing get_loss")
        optimizer_dict = dict(self.config["model_optimizer_dict"])

        # run forward pass through the model
        result_tensors = self.get_tensors(
            data_batch=data_batch,
            is_training=is_training,
            summary_writer=summary_writer,
        )
        shared_objects = dict(self.shared_objects)
        shared_objects.update(result_tensors)
        shared_objects["label_loss_dict"] = {}

        # collect all defined loss functions
        for name, opt_config in sorted(optimizer_dict.items()):

            # sanity check: make sure loss file and name have same length
            if isinstance(opt_config["loss_class"], str):
                opt_config["loss_class"] = [opt_config["loss_class"]]

            # aggregate over all defined loss functions
            label_loss = None
            for class_string in opt_config["loss_class"]:

                # get loss function
                loss_function = misc.load_class(class_string)

                # compute loss
                label_loss_i = loss_function(
                    config=self.config,
                    data_handler=self.data_handler,
                    data_transformer=self.data_transformer,
                    shared_objects=shared_objects,
                )

                # sanity check: make sure loss has expected shape
                loss_shape = label_loss_i.get_shape().as_list()
                if loss_shape != self.data_handler.label_shape:
                    error_msg = "Shape of label loss {!r} does not match {!r}"
                    raise ValueError(
                        error_msg.format(
                            loss_shape, self.data_handler.label_shape
                        )
                    )

                # accumulate loss terms
                if label_loss is None:
                    label_loss = label_loss_i
                else:
                    label_loss += label_loss_i

            # weight label_losses
            # use nested where trick to avoid NaNs:
            # https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
            label_loss_safe = tf.where(
                shared_objects["non_zero_mask"],
                label_loss,
                tf.zeros_like(label_loss),
            )
            weighted_label_loss = tf.where(
                shared_objects["non_zero_mask"],
                label_loss_safe * shared_objects["label_weights"],
                tf.zeros_like(label_loss),
            )
            weighted_loss_sum = tf.reduce_sum(weighted_label_loss)

            # get variable list
            if isinstance(opt_config["vars"], str):
                opt_config["vars"] = [opt_config["vars"]]

            var_list = []
            for var_name in opt_config["vars"]:
                var_list.extend(getattr(self, "vars_" + var_name))

            # apply regularization
            if (
                opt_config["l1_regularization"] > 0.0
                or opt_config["l2_regularization"] > 0.0
            ):
                reg_loss = 0.0

                # apply regularization
                if opt_config["l1_regularization"] > 0.0:
                    reg_loss += tf.add_n(
                        [tf.reduce_sum(tf.abs(v)) for v in var_list]
                    )

                if opt_config["l2_regularization"] > 0.0:
                    reg_loss += tf.add_n(
                        [tf.reduce_sum(v**2) for v in var_list]
                    )

                total_loss = weighted_loss_sum + reg_loss

            else:
                total_loss = weighted_loss_sum

            # logging
            shared_objects["label_loss_dict"].update(
                {
                    "loss_label_" + name: weighted_label_loss,
                    "loss_sum_" + name: weighted_loss_sum,
                    "loss_sum_total_" + name: total_loss,
                }
            )
            if summary_writer is not None:
                with summary_writer.as_default():
                    tf.summary.histogram(
                        "loss_label_" + name,
                        weighted_label_loss,
                        step=self.step,
                    )
                    tf.summary.scalar(
                        "loss_sum_" + name, weighted_loss_sum, step=self.step
                    )
                    tf.summary.scalar(
                        "loss_sum_total_" + name, total_loss, step=self.step
                    )

        return shared_objects

    @tf.function(reduce_retracing=True)
    def perform_training_step(self, data_batch, summary_writer=None):
        """Perform a single training step.

        Parameters
        ----------
        data_batch : list
            A list containing the input data.
            This is typically: x_ic78, x_deepcore, labels, misc_data
        summary_writer : tf.summary.SummaryWriter, optional
            A summary writer to write summaries to.

        Returns
        -------
        dict
            A dictionary containing the model predictions and
            auxiliary tensors.
        """
        print("Tracing perform_training_step")
        with tf.GradientTape(persistent=True) as tape:
            shared_objects = self.get_loss(
                data_batch=data_batch,
                is_training=True,
                summary_writer=summary_writer,
            )

        optimizer_dict = dict(self.config["model_optimizer_dict"])
        for name, opt_config in sorted(optimizer_dict.items()):

            # get variable list
            if isinstance(opt_config["vars"], str):
                opt_config["vars"] = [opt_config["vars"]]

            var_list = []
            for var_name in opt_config["vars"]:
                var_list.extend(getattr(self, "vars_" + var_name))

            gradients = tape.gradient(
                shared_objects["label_loss_dict"]["loss_sum_total_" + name],
                var_list,
            )

            # remove nans in gradients and replace these with zeros
            if opt_config["remove_nan_gradients"]:
                gradients = [
                    tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
                    for grad in gradients
                ]

            if opt_config["clip_gradients_value"] is not None:
                gradients, _ = tf.clip_by_global_norm(
                    gradients, opt_config["clip_gradients_value"]
                )

            # Ensure finite values
            asserts = []
            for gradient in gradients:
                assert_finite = tf.Assert(
                    tf.math.is_finite(tf.reduce_mean(gradient)),
                    [
                        tf.reduce_min(gradient),
                        tf.reduce_mean(gradient),
                        tf.reduce_max(gradient),
                    ],
                )
                asserts.append(assert_finite)
            with tf.control_dependencies(asserts):
                self.optimizers[name].apply_gradients(zip(gradients, var_list))

        return shared_objects

    def compile(self):
        """Compile the model and create optimizers."""

        checkpoint_vars = {"model": self.variables}

        # create step counter for this object
        self.step = tf.Variable(
            0, trainable=False, dtype=tf.int64, name=self.name + "_step"
        )
        checkpoint_vars["step"] = self.step

        # check that all trainable variables are set
        trainable_vars = set([v.ref() for v in self.trainable_variables])
        vars_model = set([v.ref() for v in self.vars_pred]) | set(
            [v.ref() for v in self.vars_unc]
        )
        if trainable_vars != vars_model:
            raise ValueError(
                "Trainable variables do not match model variables."
            )

        if self.is_training:
            self._create_optimizers()
            for name, optimizer in self.optimizers.items():
                assert name not in checkpoint_vars, name
                checkpoint_vars[name] = optimizer

        self._model_is_compiled = True

        # create a tensorflow checkpoint object and keep track of variables
        self._checkpoint = tf.train.Checkpoint(**checkpoint_vars)
        self._checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint,
            self.config["model_checkpoint_path"],
            **self.config["model_checkpoint_manager_kwargs"],
        )

        num_vars, num_total_vars = self._count_number_of_variables()
        msg = f"\nNumber of Model Variables for {self.name}:\n"
        msg += f"\tFree: {num_vars}\n"
        msg += f"\tTotal: {num_total_vars}"
        self._logger.info(msg)

    def restore(self, is_training=True):
        """Restore model weights from checkpoints"""
        latest_checkpoint = self._checkpoint_manager.latest_checkpoint
        if latest_checkpoint is None:
            misc.print_warning(
                "Could not find previous checkpoint. Creating new one!"
            )
        else:
            self._logger.info(
                f"[Model] Loading checkpoint: {latest_checkpoint}"
            )
            status = self._checkpoint.restore(latest_checkpoint)
            if is_training:
                status.assert_consumed()
            else:
                status.expect_partial()

    def predict(
        self, x_ic78, x_deepcore, transformed=False, is_training=False
    ):
        """Reconstruct events.

        Parameters
        ----------
        x_ic78 : float, list or numpy.ndarray
            The input data for the main IceCube array.
        x_deepcore : float, list or numpy.ndarray
            The input data for the DeepCore array.
        transformed : bool, optional
            If true, the normalized and transformed values are returned.
        is_training : bool, optional
            True if model is in training mode, false if in inference mode.

        Returns
        -------
        np.ndarray, np.ndarray
            The prediction and estimated uncertainties
        """
        data_batch_dict = {
            "x_ic78": x_ic78,
            "x_deepcore": x_deepcore,
            "x_ic78_trafo": self.data_transformer.transform(
                x_ic78, data_type="ic78"
            ),
            "x_deepcore_trafo": self.data_transformer.transform(
                x_deepcore, data_type="deepcore"
            ),
        }
        result_tensors = self(data_batch_dict, is_training=is_training)

        if transformed:
            return_values = (
                result_tensors["y_pred_trafo"].numpy(),
                result_tensors["y_unc_pred_trafo"].numpy(),
            )
        else:
            # transform back
            y_pred = self.data_transformer.inverse_transform(
                result_tensors["y_pred_trafo"], data_type="label"
            ).numpy()
            y_unc = self.data_transformer.inverse_transform(
                result_tensors["y_unc_pred_trafo"],
                data_type="label",
                bias_correction=False,
            ).numpy()
            return_values = (y_pred, y_unc)

        return return_values

    def predict_batched(
        self, x_ic78, x_deepcore, max_size, transformed=False, *args, **kwargs
    ):
        """Reconstruct events in multiple batches.

        Parameters
        ----------
        x_ic78 : float, list or numpy.ndarray
            The input data for the main IceCube array.
        x_deepcore : float, list or numpy.ndarray
            The input data for the DeepCore array.
        transformed : bool, optional
            If true, the normalized and transformed values are returned.
        max_size : int, optional
            The maximum number of events to predict at once in a batch.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        np.ndarray, np.ndarray
            The prediction and estimated uncertainties
        """
        y_pred_list = []
        y_unc_list = []

        split_indices_list = np.array_split(
            np.arange(x_ic78.shape[0]), np.ceil(x_ic78.shape[0] / max_size)
        )

        for split_indices in split_indices_list:

            y_pred, y_unc = self.predict(
                x_ic78[split_indices],
                x_deepcore[split_indices],
                transformed=transformed,
                *args,
                **kwargs
            )
            y_pred_list.append(y_pred)
            y_unc_list.append(y_unc)

        y_pred = np.concatenate(y_pred_list, axis=0)
        y_unc = np.concatenate(y_unc_list, axis=0)

        return y_pred, y_unc

    def fit(
        self,
        num_training_iterations,
        train_data_generator,
        val_data_generator,
        evaluation_methods=None,
        *args,
        **kwargs
    ):
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
                "Model must be compiled prior to call of fit method"
            )

        # add parameters and op if label weights are to be updated
        if self.config["label_update_weights"]:

            label_weight_n = 0.0
            label_weight_mean = np.zeros(self.data_handler.label_shape)
            label_weight_M2 = np.zeros(self.data_handler.label_shape)

        # ----------------
        # training loop
        # ----------------
        start_time = timeit.default_timer()
        t_validation = start_time
        num_training_steps = 0
        for step_i in range(self.step.numpy(), num_training_iterations):

            # perform a training step
            train_result = self.perform_training_step(
                data_batch=next(train_data_generator),
            )

            # -------------------------------------
            # calculate variables for tukey scaling
            # -------------------------------------
            if self.config["label_scale_tukey"]:
                batch_median_abs_dev = np.median(
                    np.abs(train_result["y_diff_trafo"]), axis=0
                )

                # assign new label weight updates
                self._update_tukey_vars(batch_median_abs_dev)

            # --------------------------------------------
            # calculate online variables for label weights
            # --------------------------------------------
            if self.config["label_update_weights"]:
                mse_values_trafo = train_result["mse_values_trafo"].numpy()
                mse_values_trafo[~self.shared_objects["non_zero_mask"]] = 1.0

                if np.isfinite(mse_values_trafo).all():
                    label_weight_n += 1
                    delta = mse_values_trafo - label_weight_mean
                    label_weight_mean += delta / label_weight_n
                    delta2 = mse_values_trafo - label_weight_mean
                    label_weight_M2 += delta * delta2
                else:
                    misc.print_warning(
                        "Found NaNs: {}".format(mse_values_trafo)
                    )
                    for i, name in enumerate(self.data_handler.label_names):
                        print(name, mse_values_trafo[i])

                if not np.isfinite(label_weight_mean).all():
                    for i, name in enumerate(self.data_handler.label_names):
                        print("weight", name, label_weight_mean[i])

                if not np.isfinite(mse_values_trafo).all():
                    raise ValueError("FOUND NANS!")

                # every n steps: update label_weights
                if step_i % self.config["validation_frequency"] == 0:
                    new_weights = 1.0 / (
                        np.sqrt(np.abs(label_weight_mean) + 1e-6) + 1e-3
                    )
                    new_weights[new_weights < 1] = 1
                    new_weights *= self.shared_objects["label_weight_config"]

                    # assign new label weight updates
                    self._update_label_weights(
                        new_weights,
                        summary_writer=self._train_writer,
                    )

                    # reset values
                    label_weight_n = 0.0
                    label_weight_mean = np.zeros(self.data_handler.label_shape)
                    label_weight_M2 = np.zeros(self.data_handler.label_shape)

            # ----------------
            # validate performance
            # ----------------
            if step_i % self.config["validation_frequency"] == 0:
                delta_t = timeit.default_timer() - t_validation
                t_step = delta_t / self.config["validation_frequency"]
                t_total = timeit.default_timer() - start_time
                t_validation = timeit.default_timer()

                updated_weights = np.array(
                    self.shared_objects["label_weights"]
                )

                # ----------------
                # Test performance
                # ----------------
                #  x_ic78, x_deepcore, labels, misc_data
                batch_train = next(train_data_generator)
                results_train = self.get_loss(
                    data_batch=batch_train,
                    is_training=True,
                    summary_writer=self._train_writer,
                )

                #  x_ic78, x_deepcore, labels, misc_data
                batch_val = next(val_data_generator)
                results_val = self.get_loss(
                    data_batch=batch_val,
                    is_training=False,
                    summary_writer=self._val_writer,
                )

                # write to file
                self._train_writer.flush()
                self._val_writer.flush()

                # -----------------
                # Print out results
                # -----------------
                def get_result_msg(loss_dict):
                    result_msg = ""
                    for name, loss in sorted(loss_dict.items()):
                        if name[:9] == "loss_sum_":
                            result_msg += f"{name}: {loss.numpy():2.3f} "
                    return result_msg

                result_msg_train = get_result_msg(
                    results_train["label_loss_dict"]
                )
                result_msg_val = get_result_msg(results_val["label_loss_dict"])

                y_true_train = batch_train[2]
                y_true_val = batch_val[2]

                y_true_trafo_train = self.data_transformer.transform(
                    y_true_train, data_type="label"
                )
                y_true_trafo_val = self.data_transformer.transform(
                    y_true_val, data_type="label"
                )

                print(
                    f"Step: {step_i:08d}, Runtime: {t_total:.1f}s, Per-Step: {t_step:1.3f}s, "
                    f"Benchmark: {np.sum(updated_weights):3.3f}"
                )
                print("\t[Train]      " + result_msg_train)
                print("\t[Validation] " + result_msg_val)

                # print info for each label
                for name, index in sorted(
                    self.data_handler.label_name_dict.items()
                ):
                    if updated_weights[index] > 0:

                        unc_pull_train = np.std(
                            (
                                results_train["y_pred_trafo"].numpy()[:, index]
                                - y_true_trafo_train[:, index]
                            )
                            / results_train["y_unc_pred_trafo"].numpy()[
                                :, index
                            ],
                            ddof=1,
                        )
                        unc_pull_val = np.std(
                            (
                                results_val["y_pred_trafo"].numpy()[:, index]
                                - y_true_trafo_val[:, index]
                            )
                            / results_val["y_unc_pred_trafo"].numpy()[
                                :, index
                            ],
                            ddof=1,
                        )

                        msg = "\tweight: {weight:2.3f},"
                        msg += " train: {train:2.3f} [{unc_pull_train:1.2f}],"
                        msg += "val: {val:2.3f} [{unc_pull_val:2.2f}] [{name}"
                        msg += ", mean: {mean_train:2.3f} {mean_val:2.3f}]"
                        print(
                            msg.format(
                                weight=updated_weights[index],
                                train=results_train[
                                    "rmse_values_trafo"
                                ].numpy()[index],
                                val=results_val["rmse_values_trafo"].numpy()[
                                    index
                                ],
                                name=name,
                                mean_train=np.mean(y_true_train[:, index]),
                                mean_val=np.mean(y_true_val[:, index]),
                                unc_pull_train=unc_pull_train,
                                unc_pull_val=unc_pull_val,
                            )
                        )

                # Call user defined evaluation method
                if self.config["evaluation_class"] is not None:
                    eval_func = misc.load_class(
                        self.config["evaluation_class"]
                    )
                    eval_func(
                        batch_train=batch_train,
                        batch_val=batch_val,
                        results_train=results_train,
                        results_val=results_val,
                        config=self.config,
                        data_handler=self.data_handler,
                        data_transformer=self.data_transformer,
                        shared_objects=self.shared_objects,
                    )

            # ----------------
            # save models
            # ----------------
            if num_training_steps % self.config["save_frequency"] == 0:
                if self.config["model_save_model"] and num_training_steps > 0:
                    self._save_training_config(num_training_steps)
                    self._checkpoint_manager.save(self.step)

            # ----------------------
            # increment step counter
            # ----------------------
            self.step.assign_add(1)
            num_training_steps += 1

    def _save_training_config(self, iteration):
        """Save Training config and iterations to file.

        Parameters
        ----------
        iteration : int
            The current training iteration of this specific
            execution of the training loop. note

        Raises
        ------
        ValueError
            Description
        """
        if iteration <= self.config["save_frequency"]:
            if not self.config["model_restore_model"]:
                # Delete old training config files and create a new and empty
                # training_steps.txt, since we are training a new model
                # from scratch and overwriting the old one

                # delete all previous training config files (if they exist)
                files = glob.glob(
                    os.path.join(
                        self._check_point_path, "config_training_*.yaml"
                    )
                )
                if files:
                    misc.print_warning(
                        "Please confirm the deletion of the "
                        "previous training configs:"
                    )
                for file in files:
                    if click.confirm(
                        "Delete {!r}?".format(file), default=True
                    ):
                        os.remove(file)
                    else:
                        raise ValueError(
                            "Old training configs must be deleted!"
                        )

            # save training step config under appropriate name
            training_config = dict(self.config)
            del training_config["np_float_precision"]
            del training_config["tf_float_precision"]
            training_config = yaml.convert_nested_list_wrapper(training_config)

            with open(self._training_config_file, "w") as yaml_file:
                yaml.yaml_dumper.dump(training_config, yaml_file)

        # update number of training iterations in training_steps.yaml
        self._training_iterations_dict[self._training_step] = iteration
        with open(self._training_steps_file, "w") as yaml_file:
            yaml.yaml_dumper.dump(
                dict(self._training_iterations_dict), yaml_file
            )

    def _count_number_of_variables(self):
        """Counts number of model variables

        Returns
        -------
        int
            The number of trainable variables of the model.
        int
            The total number of variables of the model.
            This includes the non-trainable ones.
        """
        num_trainable = np.sum(
            [
                np.prod(x.get_shape().as_list())
                for x in self.trainable_variables
            ]
        )
        num_total = np.sum(
            [np.prod(x.get_shape().as_list()) for x in self.variables]
        )
        return num_trainable, num_total
