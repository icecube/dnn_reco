"""
This file contains commonly used utility functions to build icecube nn models.
"""

import tensorflow as tf
from tfscripts.utils import SeedCounter


class PreprocessIceCubeDataLayer(tf.Module):
    """A custom layer that applies preprocessing to input data."""

    def __init__(self, keep_prob, name=None, seed=None):
        """Initializes the layer.

        Parameters
        ----------
        keep_prob : float
            The probability that an input DOM is kept during dropout.
        seed : int, optional
            The seed to use for random number generation, by default None.
        name : str, optional
            The name of the layer, by default None.
        """
        super(PreprocessIceCubeDataLayer, self).__init__(name=name)
        self.cnt = SeedCounter(seed)
        self.keep_prob = keep_prob

    def call(self, inputs, is_training, keep_prob=None, seed=None):
        """Applies preprocessing to input data.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor (IC78, DeepCore).
        is_training : bool
            True if model is in training mode, false if in inference mode.
        keep_prob : float, optional
            The probability that an input DOM is kept during dropout.
            If provided, this overrides the value set during initialization,
            by default None.
        seed : int, optional
            The seed to use for random number generation, by default None.

        Returns
        -------
        tf.Tensor
            Preprocessed input tensor.
        """
        X_IC78, X_DeepCore = inputs

        if keep_prob is None:
            keep_prob = self.keep_prob

        if seed is None:
            seed = self.cnt()

        # -----------------------------------
        # DropOut on whole DOMs
        # -----------------------------------
        if is_training:
            noise_shape_IC78 = (
                [tf.shape(X_IC78)[0]]
                + X_IC78.get_shape().as_list()[1:-1]
                + [1]
            )
            noise_shape_DeepCore = (
                [tf.shape(X_DeepCore)[0]]
                + X_DeepCore.get_shape().as_list()[1:-1]
                + [1]
            )

            X_IC78 = tf.nn.dropout(
                X_IC78,
                rate=1 - keep_prob,
                noise_shape=noise_shape_IC78,
                seed=seed,
            )

            X_DeepCore = tf.nn.dropout(
                X_DeepCore,
                rate=1 - keep_prob,
                noise_shape=noise_shape_DeepCore,
                seed=seed,
            )

        # -----------------------------------
        # Reshape DeepCore:
        # Move strings in channel dimension:
        # only convolve over DOMs dimension
        # -----------------------------------
        X_DeepCore_upper = tf.transpose(
            a=X_DeepCore[:, :, 0:10, :], perm=[0, 2, 3, 1]
        )
        X_DeepCore_lower = tf.transpose(
            a=X_DeepCore[:, :, 10:, :], perm=[0, 2, 3, 1]
        )

        input_ch_size = X_DeepCore.get_shape().as_list()[-1]
        X_DeepCore_upper = tf.reshape(
            X_DeepCore_upper, [-1, 1, 10, input_ch_size * 8]
        )
        X_DeepCore_lower = tf.reshape(
            X_DeepCore_lower, [-1, 1, 50, input_ch_size * 8]
        )

        return X_IC78, X_DeepCore_upper, X_DeepCore_lower
