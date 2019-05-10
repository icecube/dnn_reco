#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os

from icecube import icetray

from ic3_data.container import DNNDataContainer
from ic3_data.data import DNNContainerHandler
from dnn_reco.ic3.modules import DeepLearningReco


@icetray.traysegment
def ApplyDNNRecos(
        tray, name,
        pulse_key,
        model_names,
        output_keys=None,
        models_dir='/data/user/mhuennefeld/DNN_reco/models/exported_models',
        cascade_key='MCCascade',
        measure_time=True,
        batch_size=1,
        num_cpus=1,
        ):
    """Apply DNN reco

    Parameters
    ----------
    tray : icecube.icetray
        Description
    name : str
        Name of module
    pulse_key : str
        Name of pulses to use.
    model_names : str or list of str
        A list of strings or a single string that define the models to apply.
        If a list of model names is given, the reco will be applied with each
        model.
    output_keys : None, optional
        A list of output keys for the reco results.
        If None, the output will be saved as dnn_reco_{ModelName}.
    models_dir : str, optional
        The main model directory. The final model directory will be:
            os.path.join(models_dir, ModelName)
    cascade_key : str, optional
        The particle to use if the relative time method is 'vertex' or
        'first_light_at_dom'.
    measure_time : bool, optional
        If True, the run-time will be measured.
    batch_size : int, optional
        The number of events to accumulate and pass through the network in
        parallel. A higher batch size than 1 can usually improve recontruction
        runtime, but will also increase the memory footprint.
    num_cpus : int, optional
        Number of CPU cores to use if CPUs are used instead of a GPU.
    """
    if isinstance(model_names, str):
        model_names = [model_names]

    if output_keys is None:
        output_keys = ['DeepLearningReco_{}'.format(m) for m in model_names]

    # create DNN data container object
    container = DNNDataContainer()

    # configure container
    container.load_configuration(os.path.join(models_dir, model_names[0]))

    # set up container
    container.set_up()

    tray.AddModule(DNNContainerHandler, 'DNNContainerHandler',
                   DNNDataContainer=container,
                   PulseKey=pulse_key,
                   CascadeKey=cascade_key,
                   If=lambda f: pulse_key in f)

    for model_name, output_key in zip(model_names, output_keys):
        tray.AddModule(DeepLearningReco, 'DeepLearningReco_' + model_name,
                       ModelPath=os.path.join(models_dir, model_name),
                       DNNDataContainer=container,
                       OutputBaseName=output_key,
                       MeasureTime=measure_time,
                       BatchSize=batch_size,
                       ParallelismThreads=num_cpus,
                       )
