#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os

from icecube import icetray

from ic3_data.container import DNNDataContainer
from ic3_data.data import DNNContainerHandler
from dnn_reco.ic3.module import DeepLearningReco


@icetray.traysegment
def ApplyDNNRecos(
        tray, name,
        pulse_key,
        model_names,
        output_keys=None,
        models_dir='/data/user/mhuennefeld/DNN_reco/models/exported_models',
        cascade_key='MCCascade',
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
    OutputKeys : None, optional
        A list of output keys for the reco results.
        If None, the output will be saved as dnn_reco_{ModelName}.
    models_dir : str, optional
        The main model directory. The final model directory will be:
            os.path.join(models_dir, ModelName)
    CascadeKey : str, optional
        The particle to use if the relative time method is 'vertex' or
        'first_light_at_dom'.
    """
    if isinstance(model_names, str):
        model_names = [model_names]

    if output_keys is None:
        output_keys = []

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
                       MeasureTime=True,
                       )
