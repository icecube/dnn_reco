from __future__ import division, print_function
import pandas as pd
import numpy as np
import multiprocessing
import glob
import resource
import time
import timeit
import os
import ruamel.yaml as yaml
from copy import deepcopy
import tensorflow as tf

from dnn_reco import misc
from dnn_reco import detector
from dnn_reco.data_trafo import DataTransformer
from dnn_reco.model import NNModel


class DataHandler(object):
    """Data handler for IceCube simulation data.

    The DataHandler class manages the loading of IceCube hdf5 files. It can
    read DOM data, labels and misc data from while, while also able to filter
    events.

    In addition, the DataHandler manages the meta data of the input data into
    the network and also provides the placeholder variables for the tensorflow
    graph.

    Attributes
    ----------
    label_names : list of str
        Names of labels.
    label_shape : list of int
        Shape of label/y tensor without batch dimension.
    misc_data_exists : bool
        If true, misc data exists and is != None.
    misc_names : list of str
        Names of misc names. If no misc data exists, this is an emtpy list.
    misc_shape : list of int
        Shape of misc data without batch dimension.
    num_bins : int
        Number of bins for each DOM input.
    num_labels : int
        Number of labels.
    num_misc : int
        Number of misc values that will be loaded per event.
    relative_time_keys : TYPE
        Description
    test_input_data : list of str
        List of files that is used to obtain meta data.

    """

    def __init__(self, config):
        """Initializes DataHandler object and reads in input data as dataframe.

        Parameters
        ----------
        config : dict
            Dictionary containing all settings as read in from config file.
            Must contain:
                'data_handler_num_bins': int
                    The number of bins for each DOM input.
        """

        # read input data
        self._config = dict(deepcopy(config))
        self.num_bins = config['data_handler_num_bins']

        # keep track of multiprocessing processes
        self._mp_processes = []

        self.is_setup = False

    def _setup_time_keys(self):
        """Add relative time keys
        """
        self.relative_time_keys = \
            self._config['data_handler_relative_time_keys']

        pattern = self._config['data_handler_relative_time_key_pattern']
        if pattern is not None:
            self.relative_time_keys.extend([n for n in self.label_names
                                            if pattern in n.lower()])
            self.relative_time_keys.extend([n for n in self.misc_names
                                            if pattern in n.lower()])

    def setup_with_test_data(self, test_input_data):
        """Setup the datahandler with a test input file.

        Parameters
        ----------
        test_input_data : str or list of str
            File name pattern or list of file patterns which define the paths
            to input data files. The first of the specified files will be
            read in to obtain meta data.
        """
        if isinstance(test_input_data, list):
            self.test_input_data = []
            for input_pattern in test_input_data[:3]:
                self.test_input_data.extend(glob.glob(input_pattern))
        else:
            self.test_input_data = glob.glob(test_input_data)

        # ToDo: option to pass in the meta data, such that the test file
        # does not need to be read
        self._get_label_meta_data()
        self._get_misc_meta_data()

        self._setup_time_keys()
        self.is_setup = True

    def setup_with_config(self, config_file):
        """Setup the datahandler with settings from a yaml configuration file.

        Parameters
        ----------
        config_file : str
            The path to the configuration file

        Raises
        ------
        NotImplementedError
            Description
        """
        with open(config_file, 'r') as stream:
            config_meta = yaml.safe_load(stream)

        self.label_names = config_meta['label_names']
        self.label_name_dict = config_meta['label_name_dict']
        self.label_shape = config_meta['label_shape']
        self.num_labels = config_meta['num_labels']
        self.misc_names = config_meta['misc_names']
        self.misc_name_dict = config_meta['misc_name_dict']
        self.misc_data_exists = config_meta['misc_data_exists']
        self.misc_shape = config_meta['misc_shape']
        self.num_misc = config_meta['num_misc']

        self._setup_time_keys()
        self.is_setup = True

    def _get_label_meta_data(self):
        """Loads labels from a sample file to obtain label meta data.
        """
        class_string = 'dnn_reco.modules.data.labels.{}.{}'.format(
                            self._config['data_handler_label_file'],
                            self._config['data_handler_label_name'],
                        )
        label_reader = misc.load_class(class_string)
        labels, label_names = label_reader(self.test_input_data[0],
                                           self._config)

        self.label_names = label_names
        self.label_name_dict = {n: i for i, n in enumerate(label_names)}
        self.label_shape = list(labels.shape[1:])
        self.num_labels = int(np.prod(self.label_shape))

    def _get_misc_meta_data(self):
        """Loads misc data from a sample file to obtain misc meta data.
        """
        class_string = 'dnn_reco.modules.data.misc.{}.{}'.format(
                            self._config['data_handler_misc_file'],
                            self._config['data_handler_misc_name'],
                        )
        misc_reader = misc.load_class(class_string)
        misc_data, misc_names = misc_reader(self.test_input_data[0],
                                            self._config)

        self.misc_names = misc_names
        self.misc_name_dict = {n: i for i, n in enumerate(misc_names)}
        if misc_data is None:
            self.misc_data_exists = False
            self.misc_shape = None
            self.num_misc = 0
        else:
            self.misc_data_exists = True
            self.misc_shape = list(misc_data.shape[1:])
            self.num_misc = int(np.prod(self.misc_shape))

    def _get_indices_from_string(self, string):
        """Get hexagonal indices assuming detector is centered around (0,0).

        Parameters
        ----------
        string : int
            IceCube string number of given DOM.

        Returns
        -------
        tuple: (int, int)
            Returns the hexagonal coordinate indices as a tuple: (x, y).
            Assumes the detector is centered around (0,0), which means
            negative indices are possible.
        """
        return detector.string_hex_coord_dict[string]

    def get_label_index(self, label_name):
        """Get index of a label.

        Parameters
        ----------
        label_name : str
            Name of label.

        Returns
        -------
        int
            Index.
        """
        return self.label_name_dict[label_name]

    def get_misc_index(self, misc_name):
        """Get index of a misc variable.

        Parameters
        ----------
        misc_name : str
            Name of misc variable.

        Returns
        -------
        int
            Index.
        """
        return self.misc_name_dict[misc_name]

    def read_icecube_data(self, input_data, nan_fill_value=None,
                          init_values=0., verbose=False):
        """Read IceCube hdf5 data files

        Parameters
        ----------
        input_data : str
            Path to input data file.
        nan_fill_value : float, optional
            Fill value for nan values in loaded data.
            Entries with nan values will be replaced by this value.
            If None, no replacement will be performed.
        init_values : float, optional
            The x_ic78 array will be initalized with these values via:
            np.zeros_like(x_ic78) * np.array(init_values)
        verbose : bool, optional
            Print out additional information on runtimes for loading and
            processing of files.

        Returns
        -------
        x_ic78 : numpy.ndarray
            DOM input data of main IceCube array.
            shape: [batch_size, 10, 10, 60, num_bins]
        x_deepcore : numpy.ndarray
            DOM input data of DeepCore array.
            shape: [batch_size, 8, 60, num_bins]
        labels : numpy.ndarray
            Labels.
            shape: [batch_size] + label_shape
        misc : numpy.ndarray
            Misc variables.
            shape: [batch_size] + misc_shape

        Raises
        ------
        ValueError
            Description
        """
        if not self.is_setup:
            raise ValueError('DataHandler needs to be set up first!')

        start_time = timeit.default_timer()

        try:
            with pd.HDFStore(input_data,  mode='r') as f:
                bin_values = f[self._config['data_handler_bin_values_name']]
                bin_indices = f[self._config['data_handler_bin_indices_name']]
                _time_range = f[self._config['data_handler_time_offset_name']]

        except Exception as e:
            print(e)
            print('Skipping file: {}'.format(input_data))
            return None

        time_range_start = _time_range['value']

        # create Dictionary with eventIDs
        size = len(_time_range['Event'])
        eventIDDict = {}
        for row in _time_range.iterrows():
            eventIDDict[(row[1][0], row[1][1], row[1][2], row[1][3])] = row[0]

        # Create arrays for input data
        x_ic78 = np.ones([size, 10, 10, 60, self.num_bins],
                         dtype=self._config['np_float_precision'],
                         ) * np.array(init_values)
        x_deepcore = np.ones([size, 8, 60, self.num_bins],
                             dtype=self._config['np_float_precision'],
                             ) * np.array(init_values)

        # ------------------
        # get DOM input data
        # ------------------
        for value_row, index_row in zip(bin_values.itertuples(),
                                        bin_indices.itertuples()):
            if value_row[1:5] != index_row[1:5]:
                raise ValueError(
                        'Event headers do not match! HDF5 version error?')
            string = index_row[6]
            dom = index_row[7] - 1
            index = eventIDDict[(index_row[1:5])]
            if string > 78:
                # deep core
                x_deepcore[index, string - 78 - 1, dom, index_row[10]] = \
                    value_row[10]
            else:
                # IC78
                a, b = self._get_indices_from_string(string)
                # Center of Detector is a,b = 0,0
                # a goes from -4 to 5
                # b goes from -5 to 4
                x_ic78[index, a+4, b+5, dom, index_row[10]] = value_row[10]

        # --------------
        # read in labels
        # --------------
        class_string = 'dnn_reco.modules.data.labels.{}.{}'.format(
                            self._config['data_handler_label_file'],
                            self._config['data_handler_label_name'],
                        )
        label_reader = misc.load_class(class_string)
        labels, _ = label_reader(input_data, self._config,
                                 label_names=self.label_names)
        assert list(labels.shape) == [size] + self.label_shape

        # perform label smoothing if provided in config
        if 'label_pid_smooth_labels' in self._config:
            smoothing = self._config['label_pid_smooth_labels']
            if smoothing is not None:
                for key, i in self.label_name_dict.items():
                    if key in self._config['label_pid_keys']:
                        assert ((labels[:, i] >= 0.).all()
                                and (labels[:, i] <= 1.).all()), \
                            'Values outside of [0, 1] for {!r}'.format(key)
                        labels[:, i] = \
                            labels[:, i] * (1 - smoothing) + smoothing / 2.

        # -------------------
        # read in misc values
        # -------------------
        class_string = 'dnn_reco.modules.data.misc.{}.{}'.format(
                            self._config['data_handler_misc_file'],
                            self._config['data_handler_misc_name'],
                        )
        misc_reader = misc.load_class(class_string)
        misc_data, _ = misc_reader(input_data, self._config,
                                   misc_names=self.misc_names)
        if self.misc_data_exists:
            assert list(misc_data.shape) == [size, self.num_misc]

        # -------------
        # filter events
        # -------------
        class_string = 'dnn_reco.modules.data.filter.{}.{}'.format(
                            self._config['data_handler_filter_file'],
                            self._config['data_handler_filter_name'],
                        )
        filter_func = misc.load_class(class_string)
        mask = filter_func(self, input_data, self._config, x_ic78, x_deepcore,
                           labels, misc_data, time_range_start)

        # mask out events not passing filter:
        x_ic78 = x_ic78[mask]
        x_deepcore = x_deepcore[mask]
        labels = labels[mask]
        if self.misc_data_exists:
            misc_data = misc_data[mask]
        time_range_start = time_range_start[mask]

        # ---------------
        # Fix time offset
        # ---------------
        if self.relative_time_keys:

            # fix misc relative time variables
            for i, name in enumerate(self.misc_names):
                if name in self.relative_time_keys:
                    misc_data[:, i] -= time_range_start

            # fix relative time labels
            for i, name in enumerate(self.label_names):
                if name in self.relative_time_keys:
                    labels[:, i] -= time_range_start

        # --------------------------
        # fill nan values if desired
        # --------------------------
        if nan_fill_value is None:
            mask = np.isfinite(np.sum(x_ic78, axis=(1, 2, 3, 4)))
            mask = np.logical_and(
                mask, np.isfinite(np.sum(x_deepcore, axis=(1, 2, 3))))
            mask = np.logical_and(
                mask, np.isfinite(np.sum(labels,
                                         axis=tuple(range(1, labels.ndim)))))
            if not mask.all():
                misc.print_warning('Found NaNs. ' +
                                   'Removing {} events from batch.'.format(
                                                len(mask) - np.sum(mask)))

                x_ic78 = x_ic78[mask]
                x_deepcore = x_deepcore[mask]
                labels = labels[mask]
                if self.misc_data_exists:
                    misc_data = misc_data[mask]
        else:

            # Raise Error if NaNs found in input data.
            # This should never be the case!
            mask = np.isfinite(np.sum(x_ic78, axis=(1, 2, 3, 4)))
            mask = np.logical_and(
                mask, np.isfinite(np.sum(x_deepcore, axis=(1, 2, 3))))
            if not mask.all():
                raise ValueError('Found NaN values in input data!')

            # Fixing NaNs in labels and misc data is ok, but warn about this
            mask = np.isfinite(np.sum(labels,
                                      axis=tuple(range(1, labels.ndim))))
            if self.misc_data_exists:
                mask = np.logical_and(mask, np.isfinite(
                    np.sum(misc_data, axis=tuple(range(1, misc_data.ndim)))))
            if not mask.all():
                misc.print_warning('Found NaNs in labels and/or misc data. ' +
                                   'Replacing NaNs in {} events'.format(
                                                len(mask) - np.sum(mask)))
            labels[~np.isfinite(labels)] = nan_fill_value
            if self.misc_data_exists:
                misc_data[~np.isfinite(misc_data)] = nan_fill_value
        # --------------------------

        if verbose:
            final_time = timeit.default_timer() - start_time
            print("=== Time needed to process Data: {:5.3f} seconds ==".format(
                                                                final_time))

        return x_ic78, x_deepcore, labels, misc_data

    def get_batch_generator(self,
                            input_data,
                            batch_size,
                            sample_randomly=True,
                            pick_random_files_forever=True,
                            file_capacity=1,
                            batch_capacity=5,
                            num_jobs=1,
                            num_add_files=0,
                            num_repetitions=1,
                            init_values=0.,
                            num_splits=None,
                            nan_fill_value=None,
                            verbose=False,
                            *args, **kwargs
                            ):
        """Get an IceCube data batch generator.

        This is a multiprocessing data iterator.
        There are 3 levels:

            1) A number of 'num_jobs' workers load files from the file list
               into memory and extract the DOM input data, labels, and misc
               data if defined.
               The events (input data, labels, misc data) of the loaded
               file is then queued onto a multiprocessing queue named
               'data_batch_queue'.

            2) Another worker aggregates the events of several files
               (number of files defined by 'num_add_files') together
               by dequeing elements from the 'data_batch_queue'.
               It then creates batches from these events
               (randomly if sample_randomly == True ).
               These batches are then put onto the 'final_batch_queue'.
               Elements in the 'final_batch_queue' now include 'batch_size'
               many events ( tuples of dom_responses, cascade_parameters).

            3) The third level consists of the actual generator object.
               It pops elements off of the 'final_batch_queue' and yields
               these as the desired batches of
               (input data, labels, misc data).

        Parameters
        ----------
        input_data : str or list of str
            File name pattern or list of file patterns which define the paths
            to input data files.
        batch_size : int
            Number of events per batch.
        sample_randomly : bool, optional
            If True, random files and events will be sampled.
            If False, file list and events will not be shuffled.
                Although the order will most likely stay the same, this
                can not be guaranteed, since batches are queued as soon as the
                workers finish loading and processing the files.
        pick_random_files_forever : bool, optional
            If True, random files are sampled from the file list in an infinite
                loop.
            If False, all files will only be loaded once. The 'num_repetitions'
                key defines how many times the events of a file will be used.
        file_capacity : int, optional
            Defines the maximum size of the queue which holds the loaded and
            processed events of a whole file.
        batch_capacity : int, optional
            Defines the maximum size of the batch queue which holds the batches
            of size 'batch_size'. This queue is what is used to obtain the
            final batches, which the generator yields.
        num_jobs : int, optional
            Number of jobs to run in parrallel to load and process input files.
        num_add_files : int, optional
            Defines how many files are additionaly loaded at once.
            Batches will be generated among events of these
            (1 + num_add_files) files
        num_repetitions : int, optional
            Number of times the events in a loaded file are to be used, before
            new files are loaded.
        init_values : float, optional
            The x_ic78 array will be initalized with these values via:
            np.zeros_like(x_ic78) * np.array(init_values)
        num_splits : int, optional
            If num_splits is given, the loaded file will be divided into
            num_splits chunks of about equal size. This can be useful when
            the input files contain a lot of events, since the multiprocessing
            queue can not handle elements of arbitrary size.
        nan_fill_value : float, optional
            Fill value for nan values in loaded data.
            Entries with nan values will be replaced by this value.
            If None, no replacement will be performed.
        verbose : bool, optional
            If True, verbose output with additional information on queues.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        generator
            A generator object which yields batches of:
            np.ndarry, np.ndarray
                dom_responses: [batch_size, x_dim, y_dim, z_dim, num_bins]
                cascade_parameters: [batch_size, num_cascade_parameters]

        Raises
        ------
        ValueError
            Description
        """
        if not self.is_setup:
            raise ValueError('DataHandler needs to be set up first!')

        if isinstance(input_data, list):
            file_list = []
            for input_pattern in input_data:
                file_list.extend(glob.glob(input_pattern))
        else:
            file_list = glob.glob(input_data)

        # define shared memory variables
        num_files_processed = multiprocessing.Value('i')
        processed_all_files = multiprocessing.Value('b')
        data_left_in_queue = multiprocessing.Value('b')

        # initialize shared variables
        num_files_processed.value = 0
        processed_all_files.value = False
        data_left_in_queue.value = True

        # create Locks
        file_counter_lock = multiprocessing.Lock()

        # create and randomly fill file_list queue
        file_list_queue = multiprocessing.Manager().Queue(maxsize=0)
        number_of_files = 0
        if sample_randomly:
            np.random.shuffle(file_list)

        if not pick_random_files_forever:
            # Only go through given file list once
            for file in file_list:
                number_of_files += 1
                file_list_queue.put(file)

        # create data_batch_queue
        data_batch_queue = multiprocessing.Manager().Queue(
                                                    maxsize=file_capacity)

        # create final_batch_queue
        final_batch_queue = multiprocessing.Manager().Queue(
                                                    maxsize=batch_capacity)

        def create_nn_biased_selection_func():
            """Helper Method to create NN model instance for  biased selection
            """
            local_random_state = np.random.RandomState()
            cfg = dict(deepcopy(self._config))
            cfg_sel = cfg['nn_biased_selection']

            default_settings = {
                'max_size': 32,
                'reload_frequency': 100,
                'biased_fraction': 0.1,
                'true_minus_pred_greater': {},
                'true_minus_pred_less': {},
                'true_minus_pred_trafo_greater': {},
                'true_minus_pred_trafo_less': {},
                'cut_abs_diff': {},
                'cut_abs_diff_trafo': {},
                'cut_unc_weighted_diff_trafo': {},
                'tf_parallelism_threads': 10,
                'GPU_device_count': 1,
            }
            for key, value in default_settings.items():
                if key not in cfg_sel:
                    cfg_sel[key] = value

            if cfg_sel['biased_fraction'] <= 0. or \
                    cfg_sel['biased_fraction'] > 1.:
                raise ValueError('Biased fraction {!r} not in (0, 1]'.format(
                    cfg_sel['biased_fraction']))

            # Create a new tf graph and session for this model instance
            g = tf.Graph()
            if 'tf_parallelism_threads' in cfg_sel:
                n_cpus = cfg_sel['tf_parallelism_threads']
                sess = tf.Session(graph=g, config=tf.ConfigProto(
                            gpu_options=tf.GPUOptions(allow_growth=True),
                            device_count={'GPU': cfg_sel['GPU_device_count']},
                            intra_op_parallelism_threads=n_cpus,
                            inter_op_parallelism_threads=n_cpus,
                          )).__enter__()
            else:
                sess = tf.Session(graph=g, config=tf.ConfigProto(
                            gpu_options=tf.GPUOptions(allow_growth=True),
                            device_count={'GPU': cfg_sel['GPU_device_count']},
                          )).__enter__()
            with g.as_default():
                # Create Data Handler object
                data_handler = DataHandler(cfg)
                data_handler.setup_with_test_data(cfg['training_data_file'])

                # create data transformer
                data_transformer = DataTransformer(
                    data_handler=data_handler,
                    treat_doms_equally=cfg['trafo_treat_doms_equally'],
                    normalize_dom_data=cfg['trafo_normalize_dom_data'],
                    normalize_label_data=cfg['trafo_normalize_label_data'],
                    normalize_misc_data=cfg['trafo_normalize_misc_data'],
                    log_dom_bins=cfg['trafo_log_dom_bins'],
                    log_label_bins=cfg['trafo_log_label_bins'],
                    log_misc_bins=cfg['trafo_log_misc_bins'],
                    norm_constant=cfg['trafo_norm_constant'])

                # load trafo model from file
                data_transformer.load_trafo_model(cfg['trafo_model_path'])

                # create NN model
                model = NNModel(is_training=False,
                                config=cfg,
                                data_handler=data_handler,
                                data_transformer=data_transformer,
                                sess=sess)

                # compile model: initalize and finalize graph
                model.compile()

                # restore model weights
                model.restore()

            # create nn biased selection masking function
            def nn_biased_selection_func(icecube_data):
                """Mask a batch of IceCube data to obtain a biased selection"""

                if icecube_data is None:
                    return None

                if len(icecube_data[0]) == 0:
                    return None

                # reload model weights if necessary
                nn_biased_selection_func.counter += 1
                if cfg_sel['reload_frequency'] is not None:
                    if nn_biased_selection_func.counter \
                            % cfg_sel['reload_frequency'] == 0:
                        model.restore()

                # apply model on loaded data
                y_pred, y_unc = model.predict_batched(
                                            icecube_data[0], icecube_data[1],
                                            max_size=cfg_sel['max_size'])
                y_true = icecube_data[2]

                # transform data
                y_true_trafo = data_transformer.transform(
                            y_true, data_type='label')
                y_pred_trafo = data_transformer.transform(
                            y_pred, data_type='label')
                y_unc_trafo = data_transformer.transform(
                            y_unc, data_type='label', bias_correction=False)

                diff = y_true - y_pred
                diff_trafo = y_true_trafo - y_pred_trafo
                abs_diff = np.abs(diff)
                abs_diff_trafo = np.abs(diff_trafo)

                # create mask
                mask = np.zeros(len(y_true))

                # select biased events based on difference greater than value
                for key, value in cfg_sel['true_minus_pred_greater'].items():
                    index = data_handler.get_label_index(key)
                    mask = np.logical_or(mask, diff[:, index] > value)

                # select biased events based on difference less than value
                for key, value in cfg_sel['true_minus_pred_less'].items():
                    index = data_handler.get_label_index(key)
                    mask = np.logical_or(mask, diff[:, index] < value)

                # select biased events based on transformed difference
                # greater than the specified value
                for key, value in \
                        cfg_sel['true_minus_pred_trafo_greater'].items():
                    index = data_handler.get_label_index(key)
                    mask = np.logical_or(mask, diff_trafo[:, index] > value)

                # select biased events based on transformed difference
                # greater than the specified value
                for key, value in \
                        cfg_sel['true_minus_pred_trafo_less'].items():
                    index = data_handler.get_label_index(key)
                    mask = np.logical_or(mask, diff_trafo[:, index] < value)

                # select biased events based on absolute difference
                for key, value in cfg_sel['cut_abs_diff'].items():
                    index = data_handler.get_label_index(key)
                    mask = np.logical_or(mask, abs_diff[:, index] >= value)

                # select biased events based on transformed absolute difference
                for key, value in cfg_sel['cut_abs_diff_trafo'].items():
                    index = data_handler.get_label_index(key)
                    mask = np.logical_or(mask,
                                         abs_diff_trafo[:, index] >= value)

                # select biased events based on uncertainty weighted difference
                for key, value in \
                        cfg_sel['cut_unc_weighted_diff_trafo'].items():
                    index = data_handler.get_label_index(key)
                    unc_diff_trafo = \
                        abs_diff_trafo[:, index] / y_unc_trafo[:, index]
                    mask = np.logical_or(mask, unc_diff_trafo >= value)

                # Check if any events are selected
                num_biased_events = np.sum(mask)

                # calculate how many unbiased events should be chosen
                num_to_choose = \
                    int(num_biased_events * (1./cfg_sel['biased_fraction']-1))

                # add random events to obtain approximate correct fraction
                indices_unbiased = np.arange(len(y_true))[~mask]

                # Choose at least 1 event from a file if imbalance is not too
                # big to make sure to still take events from all files.
                # Keep track of imbalance.
                num_chosen = num_to_choose - nn_biased_selection_func.balance
                if nn_biased_selection_func.balance > 100:
                    num_chosen = max(num_chosen, 0)
                else:
                    num_chosen = max(num_chosen, 1)
                num_chosen = int(min(len(indices_unbiased), num_chosen))
                nn_biased_selection_func.balance += num_chosen - num_to_choose

                if num_chosen > 0:
                    indices = local_random_state.choice(indices_unbiased,
                                                        size=num_chosen,
                                                        replace=False)
                    mask_random = np.zeros(len(y_true))
                    mask_random[indices] = True
                    mask = np.logical_or(mask, mask_random)

                # check if any events are being selected
                if np.sum(mask) == 0:
                    # print('None found from', len(y_true))
                    return None

                # apply mask
                icecube_data_masked = [np.array(icecube_data[0][mask]),
                                       np.array(icecube_data[1][mask]),
                                       np.array(icecube_data[2][mask])]
                if self.misc_data_exists:
                    icecube_data_masked.append(np.array(icecube_data[3][mask]))
                else:
                    icecube_data_masked.append(None)

                return icecube_data_masked

            nn_biased_selection_func.counter = 0
            nn_biased_selection_func.balance = 0
            return nn_biased_selection_func

        def file_loader():
            """Helper Method to load files.

            Loads a file from the file list, processes the data and creates
            the dom_responses and cascade_parameters of all events in the
            given file.
            It then puts these on the 'data_batch_queue' multiprocessing queue.
            """
            local_random_state = np.random.RandomState()

            # ----------------------------------------------
            # Create NN model instance for  biased selection
            # ----------------------------------------------
            if 'nn_biased_selection' in self._config and \
                    self._config['nn_biased_selection'] is not None:

                cfg_sel = self._config['nn_biased_selection']
                if cfg_sel['apply_biased_selection'] and \
                        cfg_sel['biased_fraction'] > 0:
                    nn_biased_selection_func = \
                        create_nn_biased_selection_func()
                    nn_biased_selection = True
                else:
                    nn_biased_selection = False

            else:
                nn_biased_selection = False
            # ----------------------------------------------

            while not processed_all_files.value:

                # get file
                if pick_random_files_forever:
                    file = local_random_state.choice(file_list)
                else:
                    with file_counter_lock:
                        if not file_list_queue.empty():
                            file = file_list_queue.get()
                        else:
                            file = None

                if file:
                    if os.path.exists(file):

                        if verbose:
                            usage = resource.getrusage(resource.RUSAGE_SELF)
                            msg = "{} {:02.1f} GB. file_list_queue:" \
                                  " {}. data_batch_queue: {}"
                            print(msg.format(file,
                                             usage.ru_maxrss / 1024.0 / 1024.0,
                                             file_list_queue.qsize(),
                                             data_batch_queue.qsize()))
                        icecube_data = self.read_icecube_data(
                                    input_data=file,
                                    init_values=init_values,
                                    nan_fill_value=nan_fill_value)

                        # biased selection
                        if nn_biased_selection:
                            icecube_data = \
                                nn_biased_selection_func(icecube_data)

                        if icecube_data is not None:

                            if num_splits is None:

                                # put batch in queue
                                data_batch_queue.put(icecube_data)
                            else:

                                # split data into several smaller chunks
                                # (Multiprocessing queue can only handle
                                #  a certain size)
                                split_indices_list = np.array_split(
                                        np.arange(icecube_data[0].shape[0]),
                                        num_splits)

                                for split_indices in split_indices_list:

                                    batch = [icecube_data[0][split_indices],
                                             icecube_data[1][split_indices],
                                             icecube_data[2][split_indices],
                                             ]
                                    if self.misc_data_exists:
                                        batch.append(
                                                icecube_data[3][split_indices])
                                    else:
                                        batch.append(None)

                                    # put batch in queue
                                    data_batch_queue.put(batch)
                    else:
                        misc.print_warning(
                            'WARNING: File {} does not exist.\033[0m'.format(
                                                                        file))

                    if not pick_random_files_forever:
                        with file_counter_lock:
                            num_files_processed.value += 1
                            if num_files_processed.value == number_of_files:
                                processed_all_files.value = True

        def data_queue_iterator(sample_randomly):
            """Helper Method to create batches.

            Takes (1 + num_add_files) many elements off of the
            'data_batch_queue' (if available). This corresponds to taking
            events of (1 + num_add_files) many files.
            Batches are then generated from these events.

            Parameters
            ----------
            sample_randomly : bool
                If True, a random event order will be sampled.
                If False, events will not be shuffled.
            """
            if not pick_random_files_forever:
                with file_counter_lock:
                    if processed_all_files.value and data_batch_queue.empty():
                        data_left_in_queue.value = False

            # reset event batch
            size = 0
            ic78_batch = []
            deepcore_batch = []
            labels_batch = []
            if self.misc_data_exists:
                misc_batch = []

            while data_left_in_queue.value:
                # create lists and concatenate at end
                # (faster than concatenating in each step)
                data_batch = data_batch_queue.get()
                current_queue_size = len(data_batch[0])
                x_ic78_list = [data_batch[0]]
                x_deepcore_list = [data_batch[1]]
                label_list = [data_batch[2]]

                if self.misc_data_exists:
                    misc_list = [data_batch[3]]

                while current_queue_size < num_repetitions * batch_size and \
                        data_left_in_queue.value:

                    # avoid dead lock and delay for a bit
                    time.sleep(0.1)

                    for i in range(num_add_files):
                        if (data_batch_queue.qsize() > 1 or
                                not data_batch_queue.empty()):
                            data_batch = data_batch_queue.get()
                            current_queue_size += len(data_batch[0])
                            x_ic78_list.append(data_batch[0])
                            x_deepcore_list.append(data_batch[1])
                            label_list.append(data_batch[2])

                            if self.misc_data_exists:
                                misc_list.append(data_batch[3])

                # concatenate into one numpy array:
                x_ic78 = np.concatenate(x_ic78_list, axis=0)
                x_deepcore = np.concatenate(x_deepcore_list, axis=0)
                labels = np.concatenate(label_list, axis=0)
                if self.misc_data_exists:
                    misc_data = np.concatenate(misc_list, axis=0)

                queue_size = x_ic78.shape[0]
                if verbose:
                    print('queue_size', queue_size, x_ic78.shape)

                # num_repetitions:
                #   potentially dangerous for batch_size approx file_size
                for epoch in range(num_repetitions):
                    if not sample_randomly:
                        shuffled_indices = range(queue_size)
                    else:
                        shuffled_indices = np.random.permutation(queue_size)

                    # ---------
                    # Version A
                    # ---------
                    # # pick at most as many events as needed to complete batch
                    # queue_index = 0
                    # while queue_index < queue_size:
                    #     # number of events needed to fill batch
                    #     num_needed = batch_size - size

                    #     # number of events available in this epoch
                    #     num_available = queue_size - queue_index
                    #     num_events_to_add = min(num_needed, num_available)

                    #     # add events to batch lists
                    #     new_queue_index = queue_index + num_events_to_add
                    #     indices = shuffled_indices[queue_index:new_queue_index]
                    #     if isinstance(indices, int):
                    #         indices = [indices]
                    #     ic78_batch.extend(x_ic78[indices])
                    #     deepcore_batch.extend(x_deepcore[indices])
                    #     labels_batch.extend(labels[indices])
                    #     if self.misc_data_exists:
                    #         misc_batch.extend(misc_data[indices])

                    #     queue_index = new_queue_index
                    #     size += num_events_to_add

                    # ---------
                    # Version B
                    # ---------
                    for index in shuffled_indices:

                        # add event to batch lists
                        ic78_batch.append(x_ic78[index])
                        deepcore_batch.append(x_deepcore[index])
                        labels_batch.append(labels[index])
                        if self.misc_data_exists:
                            misc_batch.append(misc_data[index])

                        size += 1
                    # ---------
                        if size == batch_size:
                            batch = [np.array(ic78_batch),
                                     np.array(deepcore_batch),
                                     np.array(labels_batch)]
                            if self.misc_data_exists:
                                batch.append(np.array(misc_batch))
                            else:
                                batch.append(None)

                            if verbose:
                                usage = resource.getrusage(
                                            resource.RUSAGE_SELF).ru_maxrss
                                msg = "{:02.1f} GB. file_list_queue: {}." \
                                      " data_batch_queue: {}. " \
                                      "final_batch_queue: {}"
                                print(msg.format(usage / 1024.0 / 1024.0,
                                      file_list_queue.qsize(),
                                      data_batch_queue.qsize(),
                                      final_batch_queue.qsize()))
                            final_batch_queue.put(batch)

                            # reset event batch
                            size = 0
                            ic78_batch = []
                            deepcore_batch = []
                            labels_batch = []
                            if self.misc_data_exists:
                                misc_batch = []

                if not pick_random_files_forever:
                    with file_counter_lock:
                        if (processed_all_files.value and
                                data_batch_queue.empty()):
                            data_left_in_queue.value = False

            # collect leftovers and put them in an (incomplete) batch
            if ic78_batch:
                batch = [np.array(ic78_batch),
                         np.array(deepcore_batch),
                         np.array(labels_batch)]
                if self.misc_data_exists:
                    batch.append(np.array(misc_batch))
                else:
                    batch.append(None)
                final_batch_queue.put(batch)

        def batch_iterator():
            """Create batch generator

            Yields
            ------
            np.ndarry, np.ndarray
                Returns a batch of DOM_responses and cascade_parameters.
                dom_responses: [batch_size, x_dim, y_dim, z_dim, num_bins]
                cascade_parameters: [batch_size, num_cascade_parameters]
            """
            while data_left_in_queue.value or not final_batch_queue.empty():
                batch = final_batch_queue.get()
                yield batch

        # create processes
        for i in range(num_jobs):
            process = multiprocessing.Process(target=file_loader, args=())
            process.daemon = True
            process.start()
            self._mp_processes.append(process)

        process = multiprocessing.Process(target=data_queue_iterator,
                                          args=(sample_randomly,))
        process.daemon = True
        process.start()
        self._mp_processes.append(process)

        return batch_iterator()

    def kill(self):
        """Kill Multiprocessing queues and workers
        """
        for process in self._mp_processes:
            process.terminate()

        time.sleep(1.)
        for process in self._mp_processes:
            process.join(timeout=1.0)
