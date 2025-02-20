"""
All defined classes must be derived from BaseNNModel.
The NN architecture must be fully defined in the __init__ method
and more specifically, all variables have to exist at this point.
The following attributes must be defined in the __init__ method:
    self.vars_pred: list of tf.Variable
        The trainable parameters of the prediction network.
    self.vars_unc: list of tf.Variable
        The trainable parameters of the uncertainty sub network.

The forward pass must be defined in the __call__ method.
It is expected to have the following signature:

    Parameters
    ----------
    data_batch_dict : dict
        A dictionary containing the input data.
        This includes:
            x_ic78, x_deepcore, y_true, x_misc,
            and the transformed versions (+ _trafo) of these.
    is_training : bool, optional
        True if model is in training mode, false if in inference mode.
    summary_writer : tf.summary.FileWriter, optional
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

import numpy as np
import tensorflow as tf
from tfscripts import layers as tfs

from dnn_reco.model import BaseNNModel
from dnn_reco.modules.models.utils.model_utils import (
    PreprocessIceCubeDataLayer,
)


class GeneralIC86CNN(BaseNNModel):
    """A general CNN model for the IceCube IC86 configuration.

    Convolutional layers are applied to the DeepCore and IC78 data
    separately. The output of the convolutional layers is then
    combined and fed into fully connected layers.
    A separate sub network is used to predict the uncertainties.
    """

    def __init__(
        self,
        conv_upper_deepcore_settings,
        conv_lower_deepcore_settings,
        conv_ic78_settings,
        fc_settings,
        fc_unc_settings,
        is_training,
        config,
        data_handler,
        data_transformer,
        min_uncertainty_values=1e-6,
        add_prediction_to_unc_input=False,
        enforce_direction_norm=True,
        dir_z_independent=False,
        limit_dir_vec=None,
        keep_prob_dom=1.0,
        keep_prob_conv=1.0,
        keep_prob_flat=1.0,
        keep_prob_fc=1.0,
        fc_input_shape=None,
        fc_unc_input_shape=None,
        random_seed=None,
        verbose: bool = True,
        dtype: str = "float32",
        logger=None,
        name: str | None = None,
    ) -> None:
        """Initializes neural network base class.

        Parameters
        ----------
        conv_upper_deepcore_settings : dict
            Dictionary containing settings for the upper DeepCore
            convolutional layers.
        conv_lower_deepcore_settings : dict
            Dictionary containing settings for the lower DeepCore
            convolutional layers.
        conv_ic78_settings : dict
            Dictionary containing settings for the IC78 convolutional layers.
        fc_settings : dict
            Dictionary containing settings for the fully connected layers.
        fc_unc_settings : dict
            Dictionary containing settings for the fully connected layers
            of the uncertainty sub network.
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
        min_uncertainty_values : float | array_like, optional
            The minimum value that the uncertainty predictions can have,
            by default 1e-6. This can either be a scalar or an array_like
            with the same shape as the output of the uncertainty sub network.
        add_prediction_to_unc_input : bool, optional
            If true, add the prediction to the input of the uncertainty
            sub network, by default False.
        enforce_direction_norm : bool
            If true, enforce normalization of direction prediction.
        dir_z_independent : bool
            If true, enforce normalization of direction is only applied
            in the xy-plane.
        limit_dir_vec : float
            Limit the direction vector components to a certain value range
            via tanh.
        keep_prob_dom : float, optional
            The probability that an input DOM is kept during dropout,
            by default 1.0.
        keep_prob_conv : float, optional
            The probability that a convolutional layer output is kept
            during dropout, by default 1.0.
        keep_prob_flat : float, optional
            The keep probability for the flatten layer, by default 1.0.
        keep_prob_fc : float, optional
            The keep probability for the fully connected layers, by default 1.0.
        fc_input_shape : list, optional
            The shape of the input to the fully connected layers.
            If None, the shape is determined automatically, by default None.
        fc_unc_input_shape : list, optional
            The shape of the input to the fully connected layers of the
            uncertainty sub network. If None, the shape is determined
            automatically, by default None.
        random_seed : int, optional
            Random seed for weight initialization.
        verbose : bool, optional
            If true, print additional information, by default True.
        dtype : str, optional
            The data type to use for the model, by default "float32".
        logger : logging.logger, optional
            A logging instance.
        name : str, optional
            The name of the model, by default "BaseNNModel".
        """

        super().__init__(
            is_training=is_training,
            config=config,
            data_handler=data_handler,
            data_transformer=data_transformer,
            dtype=dtype,
            logger=logger,
            name=name,
        )

        self.verbose = verbose
        self.random_seed = random_seed
        self.min_uncertainty_values = min_uncertainty_values
        self.add_prediction_to_unc_input = add_prediction_to_unc_input
        self.enforce_direction_norm = enforce_direction_norm
        self.dir_z_independent = dir_z_independent
        self.limit_dir_vec = limit_dir_vec
        self.keep_prob_dom = keep_prob_dom
        self.keep_prob_conv = keep_prob_conv
        self.keep_prob_flat = keep_prob_flat
        self.keep_prob_fc = keep_prob_fc

        self.preprocess = PreprocessIceCubeDataLayer(
            keep_prob=keep_prob_dom,
            name=name + "__preprocess",
            seed=random_seed,
        )

        self.input_shape_upper = [-1, 1, 10, 8 * data_handler.num_bins]
        self.input_shape_lower = [-1, 1, 50, 8 * data_handler.num_bins]
        self.input_shape_ic78 = [-1, 10, 10, 60, data_handler.num_bins]

        self.cnn_upper_deepcore = tfs.ConvNdLayers(
            input_shape=self.input_shape_upper,
            float_precision=dtype,
            name=name + "__upper_deepcore",
            verbose=verbose,
            seed=random_seed,
            **conv_upper_deepcore_settings,
        )

        self.cnn_lower_deepcore = tfs.ConvNdLayers(
            input_shape=self.input_shape_lower,
            float_precision=dtype,
            name=name + "__lower_deepcore",
            verbose=verbose,
            seed=random_seed,
            **conv_lower_deepcore_settings,
        )

        self.cnn_ic78 = tfs.ConvNdLayers(
            input_shape=self.input_shape_ic78,
            float_precision=dtype,
            name=name + "__ic78",
            verbose=verbose,
            seed=random_seed,
            **conv_ic78_settings,
        )

        # infer the input shape for the fully connected layers
        if fc_input_shape is None or fc_unc_input_shape is None:
            fc_input_shape = self.infer_fc_input_shape()
            fc_unc_input_shape = self.infer_fc_unc_input_shape(fc_input_shape)

            if verbose:
                print("Inferred fc_input_shape:", fc_input_shape)
                print("Inferred fc_unc_input_shape:", fc_unc_input_shape)

        self.fc_input_shape = fc_input_shape
        self.fc_unc_input_shape = fc_input_shape

        # set correct output shape for fully connected layers
        fc_settings = dict(fc_settings)
        fc_settings["fc_sizes"][-1] = self.data_handler.label_shape[-1]
        fc_unc_settings = dict(fc_unc_settings)
        fc_unc_settings["fc_sizes"][-1] = self.data_handler.label_shape[-1]

        self.fc_layers = tfs.FCLayers(
            input_shape=fc_input_shape,
            float_precision=dtype,
            name=name + "__fc",
            verbose=verbose,
            seed=random_seed,
            **fc_settings,
        )

        self.fc_unc_layers = tfs.FCLayers(
            input_shape=fc_unc_input_shape,
            float_precision=dtype,
            name=name + "__fc_unc",
            verbose=verbose,
            seed=random_seed,
            **fc_unc_settings,
        )

        # collect all trainable variables
        self.vars_pred = (
            self.cnn_upper_deepcore.trainable_variables
            + self.cnn_lower_deepcore.trainable_variables
            + self.cnn_ic78.trainable_variables
            + self.fc_layers.trainable_variables
        )
        self.vars_unc = self.fc_unc_layers.trainable_variables

    def infer_fc_unc_input_shape(self, fc_input_shape):
        """Infer input shape for the uncertainty sub network.

        Parameters
        ----------
        fc_input_shape : list
            The shape of the input to the fully connected layers.

        Returns
        -------
        list
            The shape of the input to the fully connected
            layers of the uncertainty sub network.
        """
        fc_unc_input_shape = list(fc_input_shape)
        if self.add_prediction_to_unc_input:
            fc_unc_input_shape[1] += self.data_handler.label_shape[-1]
        return fc_unc_input_shape

    def infer_fc_input_shape(self):
        """Infer the input shape for the fully connected layers.

        Returns
        -------
        list
            The shape of the input to the fully connected layers.
        """
        data_batch_dict = {
            "x_ic78_trafo": tf.zeros(
                [1] + self.input_shape_ic78[1:], dtype=self.dtype
            ),
            "x_deepcore_trafo": tf.zeros(
                [1, 8, 60, self.data_handler.num_bins], dtype=self.dtype
            ),
        }
        layer_flat = self._apply_convolutions(
            data_batch_dict=data_batch_dict,
            is_training=False,
            summary_writer=None,
        )
        return layer_flat.shape.as_list()

    def _apply_convolutions(
        self,
        data_batch_dict,
        is_training=True,
        summary_writer=None,
    ):
        """Apply convolutional layers to input data.

        Parameters
        ----------
        data_batch_dict : dict
            A dictionary containing the input data.
            This includes:
                x_ic78, x_deepcore, y_true, x_misc,
                and the transformed versions of these.
        is_training : bool, optional
            True if model is in training mode, false if in inference mode.
        summary_writer : tf.summary.FileWriter, optional
            A summary writer to write summaries to.

        Returns
        -------
        tf.Tensor
            The flattened and combined output of the convolutional layers.
        """
        x_ic78_trafo = tf.convert_to_tensor(
            data_batch_dict["x_ic78_trafo"],
            dtype=self.dtype,
        )
        x_deepcore_trafo = tf.convert_to_tensor(
            data_batch_dict["x_deepcore_trafo"],
            dtype=self.dtype,
        )

        # apply DOM dropout, split and reshape DeepCore input
        x_ic78, x_deepcore_upper, x_deepcore_lower = self.preprocess(
            inputs=(x_ic78_trafo, x_deepcore_trafo),
            is_training=is_training,
        )

        # --------------------------
        # Apply convolutional layers
        # --------------------------
        conv2d_1_layers = self.cnn_upper_deepcore(
            x_deepcore_upper,
            is_training=is_training,
            keep_prob=self.keep_prob_conv,
        )
        conv2d_2_layers = self.cnn_lower_deepcore(
            x_deepcore_lower,
            is_training=is_training,
            keep_prob=self.keep_prob_conv,
        )
        conv_hex3d_layers = self.cnn_ic78(
            x_ic78,
            is_training=is_training,
            keep_prob=self.keep_prob_conv,
        )

        # ------------------------------------------
        # combine results of convolution and flatten
        # ------------------------------------------
        # flatten layer
        layer_flat_ic78, _ = tfs.flatten_layer(conv_hex3d_layers[-1])
        layer_flat_deepcore_1, _ = tfs.flatten_layer(conv2d_1_layers[-1])
        layer_flat_deepcore_2, _ = tfs.flatten_layer(conv2d_2_layers[-1])

        # combine layers
        layer_flat = tf.concat(
            [layer_flat_ic78, layer_flat_deepcore_1, layer_flat_deepcore_2],
            axis=1,
        )

        # dropout
        layer_flat = tf.nn.dropout(
            layer_flat,
            rate=1 - self.keep_prob_flat,
            seed=self.cnt(),
        )

        if self.verbose:
            print("flat IC78:", layer_flat_ic78)
            print("flat Upper DeepCore:", layer_flat_deepcore_1)
            print("flat Lower DeepCore:", layer_flat_deepcore_2)

        return layer_flat

    @tf.function
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
        summary_writer : tf.summary.FileWriter, optional
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
        result = {}

        # apply preprocessing, convolutions and flatten
        layer_flat = self._apply_convolutions(
            data_batch_dict=data_batch_dict,
            is_training=is_training,
            summary_writer=summary_writer,
        )

        # -----------------------------------
        # fully connected layers
        # -----------------------------------
        layers = self.fc_layers(
            layer_flat,
            is_training=is_training,
            keep_prob=self.keep_prob_fc,
        )
        y_pred_trafo_orig = layers[-1]

        # -----------------------------------
        # Enforce Normalisation
        # -----------------------------------
        assert len(y_pred_trafo_orig.get_shape().as_list()) == 2

        # transform back
        y_pred = self.data_transformer.inverse_transform(
            y_pred_trafo_orig, data_type="label"
        )
        y_pred_list = tf.unstack(y_pred, axis=1)

        if self.enforce_direction_norm or self.limit_dir_vec:

            index_dir_x = self.data_handler.get_label_index(
                self.config["label_dir_x_key"]
            )
            index_dir_y = self.data_handler.get_label_index(
                self.config["label_dir_y_key"]
            )
            index_dir_z = self.data_handler.get_label_index(
                self.config["label_dir_z_key"]
            )
            index_zenith = self.data_handler.get_label_index(
                self.config["label_zenith_key"]
            )
            index_azimuth = self.data_handler.get_label_index(
                self.config["label_azimuth_key"]
            )

            # limit direction vector components to certain value range
            if self.limit_dir_vec is not None:
                y_pred_list[index_dir_x] = (
                    tf.math.tanh(y_pred_list[index_dir_x]) * self.limit_dir_vec
                )
                y_pred_list[index_dir_y] = (
                    tf.math.tanh(y_pred_list[index_dir_y]) * self.limit_dir_vec
                )
                y_pred_list[index_dir_z] = (
                    tf.math.tanh(y_pred_list[index_dir_z]) * self.limit_dir_vec
                )

            trafo_indices = [
                index_dir_x,
                index_dir_y,
                index_dir_z,
                index_azimuth,
                index_zenith,
            ]

            if self.enforce_direction_norm:
                if self.dir_z_independent:
                    # since we are not normalizing the direction-z component
                    # we need to ensure that the allowed value range is
                    # within [-1, 1]. We'll leave a little bit of leeway
                    # to allow for [-1.1, 1.1] due to asymptotic behavior
                    # of tanh
                    assert np.abs(self.limit_dir_vec - 1) < 0.1
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
        for k in self.data_handler.label_names:
            if k[0:2] == "p_" and k not in self.config["label_pid_keys"]:
                raise ValueError("Did you forget about {!r}?".format(k))

        logit_tensors = {}
        for pid_key in self.config["label_pid_keys"]:
            if pid_key in self.data_handler.label_names:
                index_pid = self.data_handler.get_label_index(pid_key)
                trafo_indices.append(index_pid)
                logit_tensors[pid_key] = y_pred_list[index_pid]
                y_pred_list[index_pid] = tf.sigmoid(y_pred_list[index_pid])

        # save logit tensors. These will be needed if we want to use
        # cross entropy as a loss function. Tensorflow's cross entropy
        # functions use logits as input as opposed to sigmoid(logits) due
        # to numerical stability.
        result["logit_tensors"] = logit_tensors

        # put it back together
        y_pred = tf.stack(y_pred_list, axis=1)

        # transform
        y_pred_trafo = self.data_transformer.transform(
            y_pred, data_type="label"
        )

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

        # -----------------------------------
        # Uncertainty estimate
        # -----------------------------------
        if self.add_prediction_to_unc_input:
            unc_input = tf.concat(
                (tf.stop_gradient(y_pred_trafo), layer_flat),
                axis=-1,
            )
        else:
            unc_input = layer_flat

        uncertainty_layers = self.fc_unc_layers(
            unc_input,
            is_training=is_training,
            keep_prob=self.keep_prob_fc,
        )
        y_unc_pred_trafo = uncertainty_layers[-1]
        y_unc_pred_trafo = (
            tf.nn.elu(y_unc_pred_trafo) + 1 + self.min_uncertainty_values
        )

        # -----------------------------------
        # print architecture
        # -----------------------------------
        if self.verbose:
            print("layer_flat:", layer_flat)
            print("unc_input:", unc_input)
            print("y_pred_trafo:", y_pred_trafo)
            print("y_unc_pred_trafo:", y_unc_pred_trafo)

        # -----------------------------------
        # collect model variables that need to be saved
        # -----------------------------------
        result["y_pred_trafo"] = y_pred_trafo
        result["y_unc_pred_trafo"] = y_unc_pred_trafo

        return result
