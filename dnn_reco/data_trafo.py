from __future__ import division, print_function
import numpy as np
import pickle
import tensorflow as tf

from dnn_reco import detector


class DataTransformer:

    """Transforms data

    Attributes
    ----------
    trafo_model : dictionary
        A dictionary containing the transformation settings and parameters.
    """

    def __init__(self, data_handler,
                 treat_doms_equally=True, normalize_dom_data=True,
                 normalize_label_data=True, normalize_misc_data=True,
                 log_dom_bins=False, log_label_bins=False, log_misc_bins=False,
                 norm_constant=1e-6, float_precision='float64'):
        """Initializes a DataTransformer object and saves the trafo settings.

        Parameters
        ----------
        data_handler : :obj: of class DataHandler
            An instance of the DataHandler class. The object is used to obtain
            meta data.
        treat_doms_equally : bool
            All DOMs are treated equally, e.g. the mean and variance is
            calculated over all DOMs and not individually.
        normalize_dom_data : bool, optional
            If true, dom data will be normalized to have a mean of 0 and a
            variance of 1.
        normalize_label_data : bool, optional
            If true, labels will be normalized to have a mean of 0 and a
            variance of 1.
        normalize_misc_data : bool, optional
            If true, misc data will be normalized to have a mean of 0 and a
            variance of 1.
        log_dom_bins : bool, list of bool
            The natural logarithm is applied to the DOM bins prior
            to normalization.
            If a list is given, the length of the list must match the number of
            bins 'num_bins'. The logarithm is applied to bin i if the ith entry
            of the log_dom_bins list is True.
        log_label_bins : bool, list of bool, dict
            The natural logarithm is applied to the label bins prior
            to normalization.
            If a list is given, the length of the list must match the number of
            labels label_shape[-1]. The logarithm is applied to bin i if the
            ith entry of the log_label_bins list is True.
            If a dictionary is provided, a list of length label_shape[-1] will
            be initialized with False and only the values of the labels as
            specified by the keys in the dictionary will be  updated.
        log_misc_bins : bool, list of bool, dict
            The natural logarithm is applied to the misc data bins prior
            to normalization.
            If a list is given, the length of the list must match the number of
            misc variabels misc_shape[-1]. The logarithm is applied to bin i
            if the ith entry of the log_misc_bins list is True.
            If a dictionary is provided, a list of length label_shape[-1] will
            be initialized with False and only the values of the labels as
            specified by the keys in the dictionary will be  updated.
        norm_constant : float
            A small constant that is added to the denominator during
            normalization to ensure finite values.
        float_precision : str, optional
            Float precision to use for trafo methods.
            Examples: 'float32', 'float64'

        Raises
        ------
        ValueError
            Description
        """
        self._setup_complete = False
        self._np_float_dtype = getattr(np, float_precision)
        self._tf_float_dtype = getattr(tf, float_precision)

        # If log_bins is a bool, logarithm is to be applied to all bins.
        # In this case, create a list of bool for each data bin.
        if isinstance(log_dom_bins, bool):
            log_dom_bins = [log_dom_bins for i in range(data_handler.num_bins)]

        if isinstance(log_label_bins, bool):
            log_label_bins = [log_label_bins
                              for i in range(data_handler.label_shape[-1])]
        elif isinstance(log_label_bins, dict):
            log_dict = dict(log_label_bins)
            log_label_bins = np.zeros(data_handler.label_shape[-1], dtype=bool)
            for key, value in log_dict.items():
                log_label_bins[data_handler.get_label_index(key)] = bool(value)

        if isinstance(log_misc_bins, bool) and data_handler.misc_shape:
            log_misc_bins = [log_misc_bins
                             for i in range(data_handler.misc_shape[-1])]
        elif isinstance(log_misc_bins, dict) and data_handler.misc_shape:
            log_dict = dict(log_misc_bins)
            log_misc_bins = np.zeros(data_handler.misc_shape[-1], dtype=bool)
            for key, value in log_dict.items():
                log_misc_bins[data_handler.get_misc_index(key)] = bool(value)

        # Some sanity checks
        if len(log_dom_bins) != data_handler.num_bins:
            raise ValueError('{!r} != {!r}. Wrong log_bins: {!r}'.format(
                                                        len(log_dom_bins),
                                                        data_handler.num_bins,
                                                        log_dom_bins))
        if len(log_label_bins) != data_handler.label_shape[-1]:
            raise ValueError('{!r} != {!r}. Wrong log_bins: {!r}'.format(
                                                len(log_label_bins),
                                                data_handler.label_shape[-1],
                                                log_label_bins))
        if data_handler.misc_shape is not None:
            if len(log_misc_bins) != data_handler.label_shape[-1]:
                raise ValueError('{!r} != {!r}. Wrong log_bins: {!r}'.format(
                                                len(log_misc_bins),
                                                data_handler.misc_shape[-1],
                                                log_misc_bins))

        # create trafo_model_dict
        self.trafo_model = {
            'num_bins': data_handler.num_bins,
            'label_shape': data_handler.label_shape,
            'misc_shape': data_handler.misc_shape,
            'misc_names': data_handler.misc_names,
            'label_names': data_handler.label_names,
            'treat_doms_equally': treat_doms_equally,
            'normalize_dom_data': normalize_dom_data,
            'normalize_label_data': normalize_label_data,
            'normalize_misc_data': normalize_misc_data,
            'log_dom_bins': log_dom_bins,
            'log_label_bins': log_label_bins,
            'log_misc_bins': log_misc_bins,
            'norm_constant': norm_constant,
        }

        self._ic78_shape = [10, 10, 60, self.trafo_model['num_bins']]
        self._deepcore_shape = [8, 60, self.trafo_model['num_bins']]

    def _update_online_variance_vars(self, data_batch, n, mean, M2):
        """Update online variance variables.

        This can be used to iteratively calculate the mean and variance of
        a dataset.

        Parameters
        ----------
        data_batch : numpy ndarray
            A batch of data for which to update the variance variables of the
            dataset.
        n : int
            Counter for number of data elements.
        mean : numpy ndarray
            Mean of dataset.
        M2 : numpy ndarray
            Variance * size of dataset

        Returns
        -------
        int, np.ndarray, np.ndarray
            n, mean, M2
            Returns the updated online variance variables
        """
        for x in data_batch:
            n += 1
            delta = x - mean
            mean += delta/n
            delta2 = x - mean
            M2 += delta*delta2
        return n, mean, M2

    def _perform_update_step(self, log_bins, data_batch, n, mean, M2):
        """Update online variance variables.

        This can be used to iteratively calculate the mean and variance of
        a dataset.

        Parameters
        ----------
        log_bins : list of bool
            Defines whether the natural logarithm is appllied to bins along
            last axis. Must have same length as data_batch.shape[-1].
        data_batch : numpy ndarray
            A batch of data for which to update the variance variables of the
            dataset.
        n : int
            Counter for number of data elements.
        mean : numpy ndarray
            Mean of dataset.
        M2 : numpy ndarray
            Variance * size of dataset

        Returns
        -------
        int, np.ndarray, np.ndarray
            n, mean, M2
            Returns the updated online variance variables
        """
        data_batch = np.array(data_batch, dtype=self._np_float_dtype)

        # perform logarithm on bins
        for bin_i, log_bin in enumerate(log_bins):
            if log_bin:
                data_batch[..., bin_i] = np.log(1.0 + data_batch[..., bin_i])

        # calculate onlince variance and mean for DOM responses
        return self._update_online_variance_vars(data_batch=data_batch, n=n,
                                                 mean=mean, M2=M2)

    def create_trafo_model_iteratively(self, data_iterator, num_batches):
        """Iteratively create a transformation model.

        Parameters
        ----------
        data_iterator : generator object
            A python generator object which generates batches of
            dom_responses and cascade_parameters.
        num_batches : int
            How many batches to use to create the transformation model.
        """

        # create empty onlince variance variables
        ic78_n = 0.
        ic78_mean = np.zeros(self._ic78_shape)
        ic78_M2 = np.zeros(self._ic78_shape)

        deepcore_n = 0.
        deepcore_mean = np.zeros(self._deepcore_shape)
        deepcore_M2 = np.zeros(self._deepcore_shape)

        label_n = 0.
        label_mean = np.zeros(self.trafo_model['label_shape'])
        label_M2 = np.zeros(self.trafo_model['label_shape'])

        if self.trafo_model['misc_shape'] is not None:
            misc_n = 0.
            misc_mean = np.zeros(self.trafo_model['misc_shape'])
            misc_M2 = np.zeros(self.trafo_model['misc_shape'])

        for i in range(num_batches):

            if i % 100 == 0:
                print('At batch {} of {}'.format(i, num_batches))

            x_ic78, x_deepcore, label, misc_data = next(data_iterator)

            ic78_n, ic78_mean, ic78_M2 = self._perform_update_step(
                                    log_bins=self.trafo_model['log_dom_bins'],
                                    data_batch=x_ic78,
                                    n=ic78_n,
                                    mean=ic78_mean,
                                    M2=ic78_M2)

            deepcore_n, deepcore_mean, deepcore_M2 = self._perform_update_step(
                                    log_bins=self.trafo_model['log_dom_bins'],
                                    data_batch=x_deepcore,
                                    n=deepcore_n,
                                    mean=deepcore_mean,
                                    M2=deepcore_M2)

            label_n, label_mean, label_M2 = self._perform_update_step(
                                log_bins=self.trafo_model['log_label_bins'],
                                data_batch=label,
                                n=label_n,
                                mean=label_mean,
                                M2=label_M2)

            if self.trafo_model['misc_shape'] is not None:
                label_n, label_mean, label_M2 = self._perform_update_step(
                                log_bins=self.trafo_model['log_label_bins'],
                                data_batch=misc_data,
                                n=label_n,
                                mean=label_mean,
                                M2=label_M2)

        # Calculate standard deviation
        ic78_std = np.sqrt(ic78_M2 / ic78_n)
        deepcore_std = np.sqrt(deepcore_M2 / deepcore_n)
        label_std = np.sqrt(label_M2 / label_n)

        if self.trafo_model['misc_shape'] is not None:
            misc_std = np.sqrt(misc_M2 / misc_n)

        # combine DOM data over all DOMs if desired
        if self.trafo_model['treat_doms_equally']:

            # initalize with zeros
            self.trafo_model['ic78_mean'] = np.zeros(self._ic78_shape)
            self.trafo_model['ic78_std'] = np.zeros(self._ic78_shape)

            # now calculate normalization for real DOMs
            self.trafo_model['ic78_mean'][detector.ic78_real_DOMs_mask] = \
                np.mean(ic78_mean[detector.ic78_real_DOMs_mask], axis=0)
            self.trafo_model['ic78_std'][detector.ic78_real_DOMs_mask] = \
                np.mean(ic78_std[detector.ic78_real_DOMs_mask], axis=0)

            # DeepCore
            self.trafo_model['deepcore_mean'] = np.mean(deepcore_mean,
                                                        axis=(0, 1),
                                                        keepdims=True)
            self.trafo_model['deepcore_std'] = np.mean(deepcore_std,
                                                       axis=(0, 1),
                                                       keepdims=True)
        else:
            self.trafo_model['ic78_mean'] = ic78_mean
            self.trafo_model['ic78_std'] = ic78_std
            self.trafo_model['deepcore_mean'] = deepcore_mean
            self.trafo_model['deepcore_std'] = deepcore_std

        self.trafo_model['label_mean'] = label_mean
        self.trafo_model['label_std'] = label_std

        if self.trafo_model['misc_shape'] is not None:
            self.trafo_model['misc_mean'] = misc_mean
            self.trafo_model['misc_std'] = misc_std

        # set constant parameters to have a std dev of 1 instead of zero
        std_names = ['ic78_std', 'deepcore_std', 'label_std']
        if self.trafo_model['misc_shape'] is not None:
            std_names.append('misc_std')
        for key in std_names:
            mask = self.trafo_model[key] == 0
            self.trafo_model[key][mask] = 1.

        self._setup_complete = True

    def load_trafo_model(self, model_path):
        """Load a transformation model from file.

        Parameters
        ----------
        model_path : str
            Path to trafo model file.

        Raises
        ------
        ValueError
            If settings in loaded transformation model do not match specified
            settings.
            If not all specified settings are defined in the loaded
            transformation model.
        """
        # load trafo model from file
        with open(model_path, 'rb') as handle:
            trafo_model = pickle.load(handle)

        # make sure that settings match
        for key in self.trafo_model:
            if key not in trafo_model:
                raise ValueError('Key {!r} does not exist in {!r}'.format(
                    key, model_path))

            mismatch = self.trafo_model[key] != trafo_model[key]
            error_msg = 'Setting {!r} does not match!'.format(key)
            if isinstance(mismatch, bool):
                if mismatch:
                    raise ValueError(error_msg)
            elif mismatch.any():
                raise ValueError(error_msg)

        # update trafo model
        self.trafo_model = trafo_model

        self._setup_complete = True

    def save_trafo_model(self, model_path):
        """Saves transformation model to file.

        Parameters
        ----------
        model_path : str
            Path to trafo model file.
        """
        with open(model_path, 'wb') as handle:
            pickle.dump(self.trafo_model, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def _check_settings(self, data, data_type):
        """Check settings and return necessary parameters for trafo and inverse
        trafo method.

        Parameters
        ----------
        data :  numpy.ndarray or tf.Tensor
            The data that will be transformed.
        data_type : str
            Specifies what kind of data this is. This must be one of:
                'ic78', 'deepcore', 'label', 'misc'

        Returns
        -------
        type(data)
            The transformed data

        Raises
        ------
        ValueError
            If DataTransformer object has not created or loaded a trafo model.
            If provided data_type is unkown.
        """
        dtype = data.dtype
        data_type = data_type.lower()

        if not self._setup_complete:
            raise ValueError('DataTransformer needs to create or load a trafo'
                             'model prior to transform call.')

        if data_type not in ['ic78', 'deepcore', 'label', 'misc']:
            raise ValueError('data_type {!r} is unknown!'.format(data_type))

        # check if shape of data matches expected shape
        if data_type == 'ic78':
            shape = [10, 10, 60, self.trafo_model['num_bins']]
        elif data_type == 'deepcore':
            shape = [8, 60, self.trafo_model['num_bins']]
        else:
            shape = self.trafo_model['{}_shape'.format(data_type)]

        if list(data.shape[1:]) != shape:
            raise ValueError('Shape of data {!r} does'.format(data.shape[1:]) +
                             ' not match expected shape {!r}'.format(shape))

        if data_type in ['ic78', 'deepcore']:
            log_name = 'log_dom_bins'
            normalize_name = 'normalize_dom_data'

        else:
            log_name = 'log_{}_bins'.format(data_type)
            normalize_name = 'normalize_{}_data'.format(data_type)

        is_tf = tf.contrib.framework.is_tensor(data)

        if is_tf:
            if dtype != self._tf_float_dtype:
                data = tf.cast(data, dtype=self._tf_float_dtype)
        else:
            data = np.array(data, dtype=self._np_float_dtype)

        # choose numpy or tensorflow log function
        if is_tf:
            log_func = tf.log
            exp_func = tf.exp
        else:
            log_func = np.log
            exp_func = np.exp

        return data, log_name, normalize_name, log_func, exp_func, is_tf, dtype

    def transform(self, data, data_type, bias_correction=True):
        """Applies transformation to the specified data.

        Parameters
        ----------
        data : numpy.ndarray or tf.Tensor
            The data that will be transformed.
        data_type : str
            Specifies what kind of data this is. This must be one of:
                'ic78', 'deepcore', 'label', 'misc'
        bias_correction : bool, optional
            If true, the transformation will correct the bias, e.g. subtract
            of the data mean to make sure that the transformed data is centered
            around zero. Usually this behaviour is desired. However, when
            transforming uncertainties, this might not be useful.

        Returns
        -------
        type(data)
            The transformed data.

        No Longer Raises
        ----------------
        ValueError
            If DataTransformer object has not created or loaded a trafo model.
            If provided data_type is unkown.
        """
        data, log_name, normalize_name, log_func, exp_func, is_tf, dtype = \
            self._check_settings(data, data_type)

        if is_tf:
            data = tf.Print(data, [tf.reduce_mean(data)], 'before trafo')
        else:
            print(np.isfinite(data).all())

        # perform logarithm on bins
        if np.all(self.trafo_model[log_name]):
            # logarithm is applied to all bins: one operation
            data = log_func(1.0 + data)

        else:
            # logarithm is only applied to some bins
            if is_tf:
                data_list = tf.unstack(data, axis=-1)
                for bin_i, log_bin in enumerate(self.trafo_model[log_name]):
                    if log_bin:
                        data_list[bin_i] = log_func(1.0 + data_list[bin_i])
                data = tf.stack(data_list, axis=-1)
            else:
                for bin_i, log_bin in enumerate(self.trafo_model[log_name]):
                    if log_bin:
                        data[..., bin_i] = log_func(1.0 + data[..., bin_i])

        # normalize data
        if self.trafo_model[normalize_name]:
            if bias_correction:
                data -= self.trafo_model['{}_mean'.format(data_type.lower())]
            data /= (self.trafo_model['norm_constant'] +
                     self.trafo_model['{}_std'.format(data_type.lower())])

        # cast back to original dtype
        if is_tf:
            if dtype != self._tf_float_dtype:
                data = tf.cast(data, dtype=dtype)
        else:
            data = data.astype(dtype)

        if is_tf:
            data = tf.Print(data, [tf.reduce_mean(data)], 'after trafo')
        else:
            print(np.isfinite(data).all())

        return data

    def inverse_transform(self, data, data_type, bias_correction=True):
        """Applies inverse transformation to the specified data.

        Parameters
        ----------
        data : numpy.ndarray or tf.Tensor
            The data that will be transformed.
        data_type : str
            Specifies what kind of data this is. This must be one of:
                'ic78', 'deepcore', 'label', 'misc'
        bias_correction : bool, optional
            If true, the transformation will correct the bias, e.g. subtract
            of the data mean to make sure that the transformed data is centered
            around zero. Usually this behaviour is desired. However, when
            transforming uncertainties, this might not be useful.

        Returns
        -------
        type(data)
            Returns the inverse transformed DOM respones and
            cascade_parameters.

        No Longer Raises
        ----------------
        ValueError
            If DataTransformer object has not created or loaded a trafo model.
            If provided data_type is unkown.
        """
        data, log_name, normalize_name, log_func, exp_func, is_tf, dtype = \
            self._check_settings(data, data_type)

        if is_tf:
            data = tf.Print(data, [tf.reduce_mean(data)], 'before')
        else:
            print(np.isfinite(data).all())

        # de-normalize data
        if self.trafo_model[normalize_name]:
            data *= (self.trafo_model['norm_constant'] +
                     self.trafo_model['{}_std'.format(data_type.lower())])
            if bias_correction:
                data += self.trafo_model['{}_mean'.format(data_type.lower())]

        if is_tf:
            data = tf.Print(data, [tf.reduce_mean(data)], 'after de norm')
        else:
            print(np.isfinite(data).all())

        # undo logarithm on bins
        if np.all(self.trafo_model[log_name]):
            # logarithm is applied to all bins: one operation
            data = exp_func(data) - 1.0

        else:
            # logarithm is only applied to some bins
            if is_tf:
                data_list = tf.unstack(data, axis=-1)
                for bin_i, log_bin in enumerate(self.trafo_model[log_name]):
                    if log_bin:
                        data_list[bin_i] = exp_func(data_list[bin_i]) - 1.0
                data = tf.stack(data_list, axis=-1)
            else:
                for bin_i, log_bin in enumerate(self.trafo_model[log_name]):
                    if log_bin:
                        data[..., bin_i] = exp_func(data[..., bin_i]) - 1.0

        if is_tf:
            data = tf.Print(data, [tf.reduce_mean(data)], 'after log')
        else:
            print(np.isfinite(data).all())

        # cast back to original dtype
        if is_tf:
            if dtype != self._tf_float_dtype:
                data = tf.cast(data, dtype=dtype)
        else:
            data = data.astype(dtype)

        if is_tf:
            data = tf.Print(data, [tf.reduce_mean(data)], 'after')
        else:
            print(np.isfinite(data).all())

        return data
