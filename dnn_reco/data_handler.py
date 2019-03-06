from __future__ import division, print_function
import pandas as pd
import numpy as np
import multiprocessing
import glob
import resource
import timeit
import os
import ruamel.yaml as yaml

from dnn_reco import misc
from dnn_reco import detector


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
        self._config = dict(config)
        self.num_bins = config['data_handler_num_bins']

        self.is_setup = False

    def _setup_time_keys(self):
        """Add relative time keys
        """
        self.relative_time_keys = \
            self.config['data_handler_relative_time_keys']

        pattern = self.config['data_handler_relative_time_key_pattern']
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
            for input_pattern in test_input_data:
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
        with open(data_settings, 'r') as stream:
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
        self.num_labels = np.prod(self.label_shape)

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
            self.num_misc = np.prod(self.misc_shape)

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
        nan_fill_value : int, optional
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
                raise ValueError('Values and Indices not in same order!')
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
        mask = filter_func(input_data, self._config, x_ic78, x_deepcore,
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
            x_ic78[~np.isfinite(x_ic78)] = nan_fill_value
            x_deepcore[~np.isfinite(x_deepcore)] = nan_fill_value
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
                            verbose=False,
                            *args, **kwargs
                            ):
        """Get an IceCube data batch generator.

        This is a multiprocessing data iterator.
        There are 3 levels:

            1) A number of 'num_jobs' workers load files from the file list
               into memory and extract dom_responses and cascade_parameters.
               The events (dom_responses, cascade_parameters) of the loaded
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
               (dom_responses, cascade_parameters).

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
        file_list_queue = multiprocessing.Queue(maxsize=0)
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

        def file_loader():
            """Helper Method to load files.

            Loads a file from the file list, processes the data and creates
            the dom_responses and cascade_parameters of all events in the
            given file.
            It then puts these on the 'data_batch_queue' multiprocessing queue.
            """
            local_random_state = np.random.RandomState()
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
                                    init_values=init_values)

                        if icecube_data is not None:

                            if num_splits is None:

                                # put batch in queue
                                data_batch_queue.put(icecube_data)
                            else:

                                # split data into several smaller chunks
                                # (Multiprocessing queue can only handle
                                #  a certain size)
                                split_indices_list = np.array_split(
                                        np.arange(x_ic78.shape[0]), num_splits)

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
                x_ic78_list = [data_batch[0]]
                x_deepcore_list = [data_batch[1]]
                label_list = [data_batch[2]]

                if self.misc_data_exists:
                    misc_list = [data_batch[3]]

                for i in range(num_add_files):
                    if (data_batch_queue.qsize() > 1 or
                            not data_batch_queue.empty()):
                        data_batch = data_batch_queue.get()
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

                    # todo: optimize this, get rid of loop
                    for index in shuffled_indices:

                        # add event to batch lists
                        ic78_batch.append(x_ic78[index])
                        deepcore_batch.append(x_deepcore[index])
                        labels_batch.append(labels[index])
                        if self.misc_data_exists:
                            misc_batch.append(misc_data[index])

                        size += 1
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

        process = multiprocessing.Process(target=data_queue_iterator,
                                          args=(sample_randomly,))
        process.daemon = True
        process.start()

        return batch_iterator()
