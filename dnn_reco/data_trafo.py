from __future__ import division, print_function
import numpy as np
import pickle
import tensorflow as tf


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
                 norm_constant=1e-6):
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
            labels misc_shape[-1]. The logarithm is applied to bin i if the
            ith entry of the log_misc_bins list is True.
            If a dictionary is provided, a list of length label_shape[-1] will
            be initialized with False and only the values of the labels as
            specified by the keys in the dictionary will be  updated.
        norm_constant : float
            A small constant that is added to the denominator during
            normalization to ensure finite values.

        Raises
        ------
        ValueError
            Description
        """
        self._setup_complete = False

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

        self.IC79_shape = [10, 10, 60, self.trafo_model['num_bins']]
        self.DeepCore_shape = [8, 60, self.trafo_model['num_bins']]

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
        IC79_n = 0.
        IC79_mean = np.zeros(self.IC79_shape)
        IC79_M2 = np.zeros(self.IC79_shape)

        DeepCore_n = 0.
        DeepCore_mean = np.zeros(self.DeepCore_shape)
        DeepCore_M2 = np.zeros(self.DeepCore_shape)

        labels_n = 0.
        labels_mean = np.zeros(self.trafo_model['label_shape'])
        labels_M2 = np.zeros(self.trafo_model['label_shape'])

        if self.trafo_model['misc_shape'] is not None:
            misc_n = 0.
            misc_mean = np.zeros(self.trafo_model['misc_shape'])
            misc_M2 = np.zeros(self.trafo_model['misc_shape'])

        for i in range(num_batches):

            if i % 100 == 0:
                print('At batch {} of {}'.format(i, num_batches))

            X_IC79, X_DeepCore, labels, misc_data = next(data_iterator)

            IC79_n, IC79_mean, IC79_M2 = self._perform_update_step(
                                    log_bins=self.trafo_model['log_dom_bins'],
                                    data_batch=X_IC79,
                                    n=IC79_n,
                                    mean=IC79_mean,
                                    M2=IC79_M2)

            DeepCore_n, DeepCore_mean, DeepCore_M2 = self._perform_update_step(
                                    log_bins=self.trafo_model['log_dom_bins'],
                                    data_batch=X_DeepCore,
                                    n=DeepCore_n,
                                    mean=DeepCore_mean,
                                    M2=DeepCore_M2)

            labels_n, labels_mean, labels_M2 = self._perform_update_step(
                                log_bins=self.trafo_model['log_label_bins'],
                                data_batch=labels,
                                n=labels_n,
                                mean=labels_mean,
                                M2=labels_M2)

            if self.trafo_model['misc_shape'] is not None:
                labels_n, labels_mean, labels_M2 = self._perform_update_step(
                                log_bins=self.trafo_model['log_label_bins'],
                                data_batch=misc_data,
                                n=labels_n,
                                mean=labels_mean,
                                M2=labels_M2)

        # Calculate standard deviation
        IC79_std = np.sqrt(IC79_M2 / IC79_n)
        DeepCore_std = np.sqrt(DeepCore_M2 / DeepCore_n)
        labels_std = np.sqrt(labels_M2 / labels_n)

        if self.trafo_model['misc_shape'] is not None:
            misc_std = np.sqrt(misc_M2 / misc_n)

        # combine DOM data over all DOMs if desired
        if self.trafo_model['treat_doms_equally']:
            # ToDo: make sure this is actually doing the right thing!
            # ToDo: Handle empty DOMs differenty (perform masking)
            self.trafo_model['IC79_mean'] = np.mean(IC79_mean,
                                                    axis=(0, 1, 2),
                                                    keepdims=True)
            self.trafo_model['IC79_std'] = np.mean(IC79_std,
                                                   axis=(0, 1, 2),
                                                   keepdims=True)
            self.trafo_model['DeepCore_mean'] = np.mean(DeepCore_mean,
                                                        axis=(0, 1),
                                                        keepdims=True)
            self.trafo_model['DeepCore_std'] = np.mean(DeepCore_std,
                                                       axis=(0, 1),
                                                       keepdims=True)
        else:
            self.trafo_model['IC79_mean'] = IC79_mean
            self.trafo_model['IC79_std'] = IC79_std
            self.trafo_model['DeepCore_mean'] = DeepCore_mean
            self.trafo_model['DeepCore_std'] = DeepCore_std

        self.trafo_model['labels_mean'] = labels_mean
        self.trafo_model['labels_std'] = labels_std

        if self.trafo_model['misc_shape'] is not None:
            self.trafo_model['misc_mean'] = misc_mean
            self.trafo_model['misc_std'] = misc_std

        # set constant parameters to have a std dev of 1 instead of zero
        std_names = ['IC79_std', 'DeepCore_std', 'labels_std']
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

    def transform(self, dom_responses=None, cascade_parameters=None,
                  is_tf=True):
        """Applies transformation to the DOM responses and cascade parameters.

        Parameters
        ----------
        dom_responses : np.ndarray or tf.Tensor, None, optional
            A tensorflow tensor or numpy ndarray defining the input
            DOM response.
            shape: [batch_size, x_dim, y_dim, z_dim, num_bins]
            Optionally this can also be None.
            In this case no trafo is perfomed and a None is returned.
        cascade_parameters : np.ndarray or tf.Tensor, None, optional
            A tensorflow tensor or numpy ndarray defining the cascades.
            shape: [batch_size, num_cascade_params]
            Optionally this can also be None.
            In this case no trafo is perfomed and a None is returned.
        is_tf : bool, optional
            Specifies if the given data are tensorflow objects.
            If true, tensorflow operations will be used instead of numpy.

        Returns
        -------
        type(dom_responses), type(cascade_parameters)
            Returns the transformed DOM respones and cascade_parameters.

        Raises
        ------
        ValueError
            If DataTransformer object has not created or loaded a trafo model.
        """
        if not self._setup_complete:
            raise ValueError('DataTransformer needs to create or load a trafo'
                             'model prior to transform call.')
        if not is_tf:
            if dom_responses is not None:
                dom_responses = np.array(dom_responses)
            if cascade_parameters is not None:
                cascade_parameters = np.array(cascade_parameters)

        # choose numpy or tensorflow log function
        if is_tf:
            log_func = tf.log
        else:
            log_func = np.log

        # perform log on energy
        if self.trafo_model['log_energy']:
            if (cascade_parameters is not None and
                    self.trafo_model['num_cascade_params'] >= 6):

                if is_tf:
                    cascade_parameters_list = tf.unstack(cascade_parameters,
                                                         axis=-1)
                    cascade_parameters_list[5] = log_func(
                                             1.0 + cascade_parameters_list[5])
                    cascade_parameters = tf.stack(cascade_parameters_list,
                                                  axis=-1)
                else:
                    cascade_parameters[:, 5] = log_func(
                                                1.0 + cascade_parameters[:, 5])

        # perform logarithm on bins
        if dom_responses is not None:

            if np.all(self.trafo_model['log_bins']):
                # logarithm is applied to all bins: one operation
                dom_responses = log_func(1.0 + dom_responses)

            else:
                # logarithm is only applied to some bins
                if is_tf:
                    dom_responses_list = tf.unstack(dom_responses, axis=-1)
                    for bin_i, log_bin in enumerate(
                                                self.trafo_model['log_bins']):
                        if log_bin:
                            dom_responses_list[bin_i] = log_func(
                                            1.0 + dom_responses_list[bin_i])
                    dom_responses = tf.stack(dom_responses_list, axis=-1)
                else:
                    for bin_i, log_bin in enumerate(
                                                self.trafo_model['log_bins']):
                        if log_bin:
                            dom_responses[..., bin_i] = log_func(
                                            1.0 + dom_responses[..., bin_i])

        # normalize data
        if self.trafo_model['normalize']:
            if cascade_parameters is not None:
                cascade_parameters -= self.trafo_model[
                                                    'cascade_parameters_mean']
                cascade_parameters /= (self.trafo_model['norm_constant'] +
                                       self.trafo_model[
                                                    'cascade_parameters_std'])
            if dom_responses is not None:
                dom_responses -= self.trafo_model['dom_responses_mean']
                dom_responses /= (self.trafo_model['norm_constant'] +
                                  self.trafo_model['dom_responses_std'])

        return dom_responses, cascade_parameters

    def inverse_transform(self, dom_responses=None, cascade_parameters=None,
                          is_tf=True):
        """Applies ivnerse transformation to the DOM responses and
           cascade parameters.

        Parameters
        ----------
        dom_responses : np.ndarray or tf.Tensor, None, optional
            A tensorflow tensor or numpy ndarray defining the input
            DOM response.
            shape: [batch_size, x_dim, y_dim, z_dim, num_bins]
            Optionally this can also be None.
            In this case no trafo is perfomed and a None is returned.
        cascade_parameters : np.ndarray or tf.Tensor, None, optional
            A tensorflow tensor or numpy ndarray defining the cascades.
            shape: [batch_size, num_cascade_params]
            Optionally this can also be None.
            In this case no trafo is perfomed and a None is returned.
        is_tf : bool, optional
            Specifies if the given data are tensorflow objects.
            If true, tensorflow operations will be used instead of numpy.

        Returns
        -------
        type(dom_responses), type(cascade_parameters)
            Returns the inverse transformed DOM respones and
            cascade_parameters.

        Raises
        ------
        ValueError
            If DataTransformer object has not created or loaded a trafo model.
        """
        if not self._setup_complete:
            raise ValueError('DataTransformer needs to create or load a trafo'
                             'model prior to inverse transform call.')
        if not is_tf:
            if dom_responses is not None:
                dom_responses = np.array(dom_responses)
            if cascade_parameters is not None:
                cascade_parameters = np.array(cascade_parameters)

        if self.trafo_model['normalize']:
            if cascade_parameters is not None:
                cascade_parameters *= (self.trafo_model['norm_constant'] +
                                       self.trafo_model[
                                                    'cascade_parameters_std'])
                cascade_parameters += self.trafo_model[
                                                    'cascade_parameters_mean']
            if dom_responses is not None:
                dom_responses *= (self.trafo_model['norm_constant'] +
                                  self.trafo_model['dom_responses_std'])
                dom_responses += self.trafo_model['dom_responses_mean']

        # choose numpy or tensorflow exp function
        if is_tf:
            exp_func = tf.exp
        else:
            exp_func = np.exp

        # undo natural logarithm on cascade energy
        if self.trafo_model['log_energy']:
            if (cascade_parameters is not None and
                    self.trafo_model['num_cascade_params'] >= 6):

                if is_tf:
                    cascade_parameters_list = tf.unstack(cascade_parameters,
                                                         axis=-1)
                    cascade_parameters_list[5] = exp_func(
                                              cascade_parameters_list[5]) - 1.0
                    cascade_parameters = tf.stack(cascade_parameters_list,
                                                  axis=-1)
                else:
                    cascade_parameters[:, 5] = exp_func(
                                                cascade_parameters[:, 5]) - 1.0

        # undo logarithm on bins
        if dom_responses is not None:

            if np.all(self.trafo_model['log_bins']):
                # logarithm is applied to all bins: one operation
                dom_responses = exp_func(dom_responses) - 1.0

            else:
                # logarithm is only applied to some bins
                if is_tf:
                    dom_responses_list = tf.unstack(dom_responses, axis=-1)
                    for bin_i, log_bin in enumerate(
                                                self.trafo_model['log_bins']):
                        if log_bin:
                            dom_responses_list[bin_i] = exp_func(
                                            dom_responses_list[bin_i]) - 1.0
                    dom_responses = tf.stack(dom_responses_list, axis=-1)
                else:
                    for bin_i, log_bin in enumerate(
                                                self.trafo_model['log_bins']):
                        if log_bin:
                            dom_responses[..., bin_i] = exp_func(
                                            dom_responses[..., bin_i]) - 1.0

        return dom_responses, cascade_parameters
