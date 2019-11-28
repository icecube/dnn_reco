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
        model_names,
        pulse_key=None,
        dom_exclusions=None,
        partial_exclusion=None,
        output_keys=None,
        models_dir='/data/user/mhuennefeld/DNN_reco/models/exported_models',
        cascade_key='MCCascade',
        check_settings=True,
        measure_time=True,
        batch_size=1,
        num_cpus=1,
        verbose=True,
        ):
    """Apply DNN reco

    Parameters
    ----------
    tray : icecube.icetray
        Description
    name : str
        Name of module
    model_names : str or list of str
        A list of strings or a single string that define the models to apply.
        If a list of model names is given, the reco will be applied with each
        model.
    pulse_key : str
        Name of pulses to use.
    dom_exclusions : list of str, optional
        List of frame keys that define DOMs or TimeWindows that should be
        excluded. Typical values for this are:
        ['BrightDOMs','SaturationWindows','BadDomsList','CalibrationErrata']
    partial_exclusion : bool, optional
        If True, partially exclude DOMS, e.g. only omit pulses from excluded
        TimeWindows defined in 'dom_exclusions'.
        If False, all pulses from a DOM will be excluded if the omkey exists
        in the dom_exclusions.
    output_keys : None, optional
        A list of output keys for the reco results.
        If None, the output will be saved as dnn_reco_{ModelName}.
    models_dir : str, optional
        The main model directory. The final model directory will be:
            os.path.join(models_dir, ModelName)
    cascade_key : str, optional
        The particle to use if the relative time method is 'vertex' or
        'first_light_at_dom'.
    check_settings : bool, optional
        If True, the set values will be checked against configured or
        loaded settings if they were set. It these settings do not match
        an error will be raised. This ensures the correct use of the
        trained models.
        Sometimes it is necessary to use the model with slightly different
        settings. In this case 'check_settings' can be set to False.
        However, when doing so it can't be guaranteed that the model is
        used in a correct way.
        TLTR: use 'False' with caution.
    measure_time : bool, optional
        If True, the run-time will be measured.
    batch_size : int, optional
        The number of events to accumulate and pass through the network in
        parallel. A higher batch size than 1 can usually improve recontruction
        runtime, but will also increase the memory footprint.
    num_cpus : int, optional
        Number of CPU cores to use if CPUs are used instead of a GPU.
    verbose : bool, optional
        If True, output pulse masking information.
    """
    if isinstance(model_names, str):
        model_names = [model_names]

    if output_keys is None:
        output_keys = ['DeepLearningReco_{}'.format(m) for m in model_names]

    # create DNN data container object
    container = DNNDataContainer(batch_size=batch_size)

    # configure container
    container.load_configuration(os.path.join(models_dir, model_names[0]))

    # set up container and define pulse settings and DOM exclusions
    container.set_up(pulse_key=pulse_key,
                     dom_exclusions=dom_exclusions,
                     partial_exclusion=partial_exclusion,
                     cascade_key=cascade_key,
                     check_settings=check_settings)

    tray.AddModule(DNNContainerHandler, 'DNNContainerHandler_' + name,
                   DNNDataContainer=container, Verbose=verbose)

    for model_name, output_key in zip(model_names, output_keys):
        tray.AddModule(DeepLearningReco, 'DeepLearningReco_'+model_name+name,
                       ModelPath=os.path.join(models_dir, model_name),
                       DNNDataContainer=container,
                       OutputBaseName=output_key,
                       MeasureTime=measure_time,
                       ParallelismThreads=num_cpus,
                       )
