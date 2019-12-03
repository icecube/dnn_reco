#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import logging
import glob
import numpy as np
import tensorflow as tf
import timeit
from collections import deque

from icecube import icetray, dataclasses
import ruamel.yaml as yaml

from ic3_data.container import DNNDataContainer
from ic3_data.data import DNNContainerHandler

from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer
from dnn_reco.model import NNModel


class DeepLearningReco(icetray.I3ConditionalModule):

    """Module to apply dnn reco.

    Attributes
    ----------
    batch_size : int, optional
        The number of events to accumulate and pass through the network in
        parallel. A higher batch size than 1 can usually improve recontruction
        runtime, but will also increase the memory footprint.
    config : dict
        Dictionary with configuration settings
    data_handler : dnn_reco.data_handler.DataHanlder
        A data handler object. Handles nn model input meta data and provides
        tensorflow placeholders.
    data_transformer : dnn_reco.data_trafo.DataTransformer
        The data transformer.
    model : dnn_reco.model.NNModel
        The neural network model
    """

    def __init__(self, context):
        """Initialize DeepLearningReco Module
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('ModelPath', 'Path to DNN model', None)
        self.AddParameter('DNNDataContainer',
                          'Data container that will be used to feed model',
                          None)
        self.AddParameter('IgnoreMisconfiguredSettingsList',
                          "The model automatically checks whether the "
                          "configured settings for the 'DNNDataContainer' "
                          "match those settings that were exported in this "
                          "model. If a mismatch is found, an error will be "
                          "raised. This helps to ensure the correct use of "
                          "the trained models. Sometimes it is necessary to "
                          "use the model with slightly different settings. In "
                          "this case a list of setting names can be passed "
                          "for which the mismatches will be ignored. Doing so "
                          "will relax the raised error to a warning that is "
                          "issued. This should be used with caution.",
                          None)
        self.AddParameter('OutputBaseName',
                          'Output key under which the result will be written',
                          'DeepLearningReco')
        self.AddParameter("MeasureTime",
                          "If True, time for preprocessing and prediction will"
                          " be measured and printed", False)
        self.AddParameter("ParallelismThreads",
                          "Tensorflow config option for 'intra_op_parallelism_"
                          "threads' and 'inter_op_parallelism_threads'"
                          "[# CPUs]", None)

    def Configure(self):
        """Configure DeepLearningReco module.

        Read in configuration and build nn model.

        Raises
        ------
        ValueError
            If settings do not match the expected settings by the nn model.
        """
        self._model_path = self.GetParameter('ModelPath')
        self._container = self.GetParameter('DNNDataContainer')
        self._output_key = self.GetParameter("OutputBaseName")
        self._measure_time = self.GetParameter("MeasureTime")
        self._parallelism_threads = self.GetParameter("ParallelismThreads")
        self._ingore_list = \
            self.GetParameter('IgnoreMisconfiguredSettingsList')
        if self._ingore_list is None:
            self._ingore_list = []

        # read in and combine config files and set up
        training_files = glob.glob(os.path.join(self._model_path,
                                                'config_training_*.yaml'))
        last_training_file = np.sort(training_files)[-1]
        setup_manager = SetupManager([last_training_file])
        self.config = setup_manager.get_config()

        # ToDo: Adjust necessary values in config
        self.config['model_checkpoint_path'] = os.path.join(self._model_path,
                                                            'model')
        self.config['model_is_training'] = False
        self.config['trafo_model_path'] = os.path.join(self._model_path,
                                                       'trafo_model.npy')
        if self._parallelism_threads is not None:
            self.config['tf_parallelism_threads'] = self._parallelism_threads

        # ----------------------------------------------------------------
        # Check if settings of data container match settings in model path
        # ----------------------------------------------------------------
        cfg_file = os.path.join(self._model_path, 'config_data_settings.yaml')
        with open(cfg_file, 'r') as stream:
            data_config = yaml.safe_load(stream)

        # Backwards compatibility for older exported models which did not
        # include this setting. In this case the separated format, e.g.
        # icecube array + deepcore array is used as opposed to the string-dom
        # format: [batch, 86, 60, num_bins]
        if 'is_str_dom_format' not in data_config:
            data_config['is_str_dom_format'] = False

        for k in self._container.config:

            # backwards compatibility for older exported models which did not
            # export these settings
            if k not in data_config and k in ['pulse_key', 'dom_exclusions',
                                              'partial_exclusion',
                                              'cascade_key',
                                              'allowed_pulse_keys',
                                              'allowed_cascade_keys']:
                msg = 'Warning: not checking if parameter {!r} is correctly '
                msg += 'configured for model {!r} because the setting '
                msg += 'was not exported.'
                logging.warning(msg.format(k, self._model_path))
                continue

            # check for allowed pulse keys
            if (k == 'pulse_key' and 'allowed_pulse_keys' in data_config and
                    data_config['allowed_pulse_keys'] is not None and
                    self._container.config[k]
                    in data_config['allowed_pulse_keys']):

                # this is an allowed pulse, so everything is ok
                continue

            # check for allowed cascade keys
            if (k == 'cascade_key' and 'allowed_cascade_keys' in data_config
                    and data_config['allowed_cascade_keys'] is not None and
                    self._container.config[k]
                    in data_config['allowed_cascade_keys']):

                # this is an allowed cascade key, so everything is ok
                continue

            if not self._container.config[k] == data_config[k]:
                if k in self._ingore_list:
                    msg = 'Warning: parameter {!r} is set to {!r} which '
                    msg += 'differs from the model [{!r}] default value {!r}. '
                    msg += 'This mismatch will be ingored since the parameter '
                    msg += 'is in the IgnoreMisconfiguredSettingsList. '
                    msg += 'Make sure this is what you intend to do!'
                    logging.warning(msg.format(k, self._container.config[k],
                                               self._model_path,
                                               data_config[k]))
                else:
                    msg = 'Fatal: parameter {!r} is set to {!r} which '
                    msg += 'differs from the model [{!r}] default value {!r}.'
                    msg += 'If you are sure you want to use this model '
                    msg += 'with these settings, then you can add the '
                    msg += 'parameter to the IgnoreMisconfiguredSettingsList.'
                    raise ValueError(msg.format(k, self._container.config[k],
                                                self._model_path,
                                                data_config[k]))
        # ----------------------------------------------------------------

        # create variables and frame buffer for batching
        self._frame_buffer = deque()
        self._pframe_counter = 0
        self._batch_event_index = 0

        # Create a new tensorflow graph and session for this instance of
        # dnn reco
        g = tf.Graph()
        if 'tf_parallelism_threads' in self.config:
            n_cpus = self.config['tf_parallelism_threads']
            sess = tf.Session(graph=g, config=tf.ConfigProto(
                        gpu_options=tf.GPUOptions(allow_growth=True),
                        device_count={'GPU': 1},
                        intra_op_parallelism_threads=n_cpus,
                        inter_op_parallelism_threads=n_cpus,
                      )).__enter__()
        else:
            sess = tf.Session(graph=g, config=tf.ConfigProto(
                        gpu_options=tf.GPUOptions(allow_growth=True),
                        device_count={'GPU': 1},
                      )).__enter__()
        with g.as_default():
            # Create Data Handler object
            self.data_handler = DataHandler(self.config)
            self.data_handler.setup_with_config(
                    os.path.join(self._model_path, 'config_meta_data.yaml'))

            # Get time vars that need to be corrected by global time offset
            self._mask_time = []
            for i, name in enumerate(self.data_handler.label_names):
                if name in self.data_handler.relative_time_keys:
                    self._mask_time.append(True)
                else:
                    self._mask_time.append(False)
            self._mask_time = np.expand_dims(np.array(self._mask_time), axis=0)

            # create data transformer
            self.data_transformer = DataTransformer(
                data_handler=self.data_handler,
                treat_doms_equally=self.config['trafo_treat_doms_equally'],
                normalize_dom_data=self.config['trafo_normalize_dom_data'],
                normalize_label_data=self.config['trafo_normalize_label_data'],
                normalize_misc_data=self.config['trafo_normalize_misc_data'],
                log_dom_bins=self.config['trafo_log_dom_bins'],
                log_label_bins=self.config['trafo_log_label_bins'],
                log_misc_bins=self.config['trafo_log_misc_bins'],
                norm_constant=self.config['trafo_norm_constant'])

            # load trafo model from file
            self.data_transformer.load_trafo_model(
                                            self.config['trafo_model_path'])

            # create NN model
            self.model = NNModel(is_training=False,
                                 config=self.config,
                                 data_handler=self.data_handler,
                                 data_transformer=self.data_transformer,
                                 sess=sess)

            # compile model: initalize and finalize graph
            self.model.compile()

            # restore model weights
            self.model.restore()

            # Get trained labels, e.g. labels with weights greater than zero
            self._mask_labels = \
                self.model.shared_objects['label_weight_config'] > 0
            self._non_zero_labels = [n for n, b in
                                     zip(self.data_handler.label_names,
                                         self._mask_labels) if b]
            self._non_zero_log_bins = \
                [l for l, b in
                 zip(self.data_transformer.trafo_model['log_label_bins'],
                     self._mask_labels) if b]

    def Process(self):
        """Process incoming frames.

        Pop frames and put them in the frame buffer.
        When a physics frame is popped, accumulate the input data to form
        a batch of events. Once a full batch of physics events is accumulated,
        perform the prediction and push the buffered frames.
        The Physics method can then write the results to the physics frame
        by using the results:
            self.y_pred_batch, self.y_unc_batch
            self._runtime_prediction, self._runtime_preprocess_batch
        and the current event index self._batch_event_index
        """
        frame = self.PopFrame()

        # put frame on buffer
        self._frame_buffer.append(frame)

        # check if the current frame is a physics frame
        if frame.Stop == icetray.I3Frame.Physics:

            self._pframe_counter += 1

            # check if we have a full batch of events
            if self._pframe_counter == self._container.batch_size:

                # we have now accumulated a full batch of events so
                # that we can perform the prediction
                self._process_frame_buffer()

    def Finish(self):
        """Run prediciton on last incomplete batch of events.

        If there are still frames left in the frame buffer there is an
        incomplete batch of events, that still needs to be passed through.
        This method will run the prediction on the incomplete batch and then
        write the results to the physics frame. All frames in the frame buffer
        will be pushed.
        """
        if self._frame_buffer:

            # there is an incomplete batch of events that we need to complete
            self._process_frame_buffer()

    def _process_frame_buffer(self):
        """Performs prediction for accumulated batch.
        Then writes results to physics frames in frame buffer and eventually
        pushes all of the frames in the order they came in.
        """
        self._perform_prediction(size=self._pframe_counter)

        # reset counters and indices
        self._batch_event_index = 0
        self._pframe_counter = 0

        # push frames
        while self._frame_buffer:
            fr = self._frame_buffer.popleft()

            if fr.Stop == icetray.I3Frame.Physics:

                # write results at current batch index to frame
                self._write_to_frame(fr, self._batch_event_index)

                # increase the batch event index
                self._batch_event_index += 1

            self.PushFrame(fr)

    def _perform_prediction(self, size):
        """Perform the prediction for a batch of events.

        Parameters
        ----------
        size : int
            The size of the current batch.
        """
        if size > 0:
            if self._measure_time:
                start_time = timeit.default_timer()

            self.y_pred_batch, self.y_unc_batch = self.model.predict(
                                x_ic78=self._container.x_ic78[:size],
                                x_deepcore=self._container.x_deepcore[:size])

            # Fix time offset
            if self.data_handler.relative_time_keys:
                mask = np.broadcast_to(self._mask_time,
                                       self.y_pred_batch.shape)
                self.y_pred_batch[mask] += \
                    self._container.global_time_offset_batch[:size]

            if self._measure_time:
                self._runtime_prediction = \
                    (timeit.default_timer() - start_time) / size
        else:
            self.y_pred_batch = None
            self.y_unc_batch = None
            self._runtime_prediction = None

    def _write_to_frame(self, frame, batch_event_index):
        """Writes the prediction results of the given batch event index to
        the frame.

        Parameters
        ----------
        frame : I3Frame
            The physics frame to which the results should be written to.
        batch_event_index : int
            The batch event index. This defines which event in the batch is to
            be written to the frame.
        """
        if self._measure_time:
            start_time = timeit.default_timer()

        # Write prediction and uncertainty estimate to frame
        results = {}
        for name, pred, unc, log_label in zip(
                self._non_zero_labels,
                self.y_pred_batch[batch_event_index][self._mask_labels],
                self.y_unc_batch[batch_event_index][self._mask_labels],
                self._non_zero_log_bins):

            # save prediction
            results[name] = float(pred)

            # save uncertainty estimate
            if log_label:
                results[name + '_log_uncertainty'] = float(unc)
            else:
                results[name + '_uncertainty'] = float(unc)

        # Create combined I3Particle
        if 'label_particle_keys' in self.config:
            particle_keys = self.config['label_particle_keys']

            particle = dataclasses.I3Particle()
            if 'energy' in particle_keys:
                if particle_keys['energy'] in self._non_zero_labels:
                    particle.energy = results[particle_keys['energy']]
            if 'time' in particle_keys:
                if particle_keys['time'] in self._non_zero_labels:
                    particle.time = results[particle_keys['time']]
            if 'length' in particle_keys:
                if particle_keys['length'] in self._non_zero_labels:
                    particle.length = results[particle_keys['length']]
            if 'dir_x' in particle_keys:
                if particle_keys['dir_x'] in self._non_zero_labels:
                    particle.dir = dataclasses.I3Direction(
                                            results[particle_keys['dir_x']],
                                            results[particle_keys['dir_y']],
                                            results[particle_keys['dir_z']])
            elif 'azimuth' in particle_keys:
                if particle_keys['azimuth'] in self._non_zero_labels:
                    particle.dir = dataclasses.I3Direction(
                                            results[particle_keys['azimuth']],
                                            results[particle_keys['zenith']])

            if 'pos_x' in particle_keys:
                if particle_keys['pos_x'] in self._non_zero_labels:
                    particle.pos = dataclasses.I3Position(
                                            results[particle_keys['pos_x']],
                                            results[particle_keys['pos_y']],
                                            results[particle_keys['pos_z']])

            frame[self._output_key + '_I3Particle'] = particle

        # write time measurement to frame
        if self._measure_time:
            results['runtime_prediction'] = self._runtime_prediction
            results['runtime_write'] = \
                timeit.default_timer() - start_time
            results['runtime_preprocess'] = \
                self._container.runtime_batch[batch_event_index]

        # write to frame
        frame[self._output_key] = dataclasses.I3MapStringDouble(results)
