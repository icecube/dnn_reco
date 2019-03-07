#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import glob
import numpy as np

from icecube import icetray, dataclasses
import ruamel.yaml as yaml

from ic3_data.container import DNNDataContainer
from ic3_data.data import DNNContainerHandler

from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer
from dnn_reco.model import NNModel


class DeepLearningReco(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter('ModelPath', 'Path to DNN model', None)
        self.AddParameter('DNNDataContainer',
                          'Data container that will be used to feed model',
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
        self._model_path = self.GetParameter('ModelPath')
        self._container = self.GetParameter('DNNDataContainer')
        self._output_key = self.GetParameter("OutputBaseName")
        self._measure_time = self.GetParameter("MeasureTime")
        self._parallelism_threads = self.GetParameter("ParallelismThreads")

        # read in and combine config files and set up
        training_files = glob.glob(os.path.join(self._model_path,
                                                'config_training_*.yaml'))
        last_training_file = np.sort(training_files)[-1]
        setup_manager = SetupManager([last_training_file])
        self.config = setup_manager.get_config()

        # ToDo: Adjust necessary values in config
        self.config['model_checkpoint_path'] = self._model_path
        self.config['model_is_training'] = False
        self.config['trafo_model_path'] = os.path.join(self._model_path,
                                                       'trafo_model.npy')
        if self._parallelism_threads is not None:
            self.config['tf_parallelism_threads'] = self._parallelism_threads

        # Check if settings of data container match settings in model path
        cfg_file = os.path.join(self._model_path, 'config_data_settings.yaml')
        with open(cfg_file, 'r') as stream:
            data_config = yaml.safe_load(stream)
        for k in self._container.config:
            if not self._container.config[k] == data_config[k]:
                raise ValueError('Settings do not match: {!r} != {!r}'.format(
                        self._container.config[k], data_config[k]))

        # Create Data Handler object
        self.data_handler = DataHandler(self.config)
        self.data_handler.setup_with_config(
                    os.path.join(self._model_path, 'config_meta_data.yaml'))

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
        self.data_transformer.load_trafo_model(self.config['trafo_model_path'])

        # create NN model
        self.model = NNModel(is_training=False,
                             config=self.config,
                             data_handler=self.data_handler,
                             data_transformer=self.data_transformer)

        # compile model: initalize and finalize graph
        self.model.compile()

        # restore model weights
        self.model.restore()

    def Physics(self, frame):
        raise NotImplementedError()
