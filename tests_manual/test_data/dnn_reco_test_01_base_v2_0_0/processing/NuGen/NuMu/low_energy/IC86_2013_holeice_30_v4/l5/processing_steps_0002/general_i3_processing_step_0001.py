#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/icetray-start
#METAPROJECT icetray/v1.12.0
# Note: this file must unfortunately be python 2.7 compatible!
from __future__ import print_function, division
import os
import sys

if "ENV_SITE_PACKAGES" in os.environ:
    sys.path.insert(1, os.environ["ENV_SITE_PACKAGES"])

if "PYTHON_PACKAGE_IMPORTS" in os.environ:
    import importlib

    for package in os.environ["PYTHON_PACKAGE_IMPORTS"].split(","):
        importlib.import_module(package)

import timeit
import click

from I3Tray import I3Tray
from icecube import icetray, hdfwriter

from ic3_labels.weights.segments import AddWeightMetaData, UpdateMergedWeights


from ic3_processing.utils.exp_data import livetime
from ic3_processing.utils import setup
from ic3_processing.modules.utils import tray_timer


@click.command()
@click.argument("cfg", type=click.Path(exists=True))
@click.argument("run_number", type=int)
@click.option("--scratch/--no-scratch", default=True)
def main(cfg, run_number, scratch):
    # start timer
    start_time = timeit.default_timer()

    # --------------------------------
    # load configuration and setup job
    # --------------------------------
    cfg, context = setup.setup_job_and_config(cfg, run_number, scratch)
    # --------------------------------

    tray = I3Tray()

    tray.context["ic3_processing"] = context
    tray.context["ic3_processing"]["HDF_keys"] = []

    tray.Add("I3Reader", FilenameList=context["infiles"])

    # -----------------------------------------
    # Write livetime data for experimental data
    # -----------------------------------------
    livetime.write_exp_livetime_data(
        tray,
        name="write_exp_livetime_data",
        cfg=cfg,
    )

    # ----------------------------------------------
    # Add tray segments defined in configuration cfg
    # ----------------------------------------------
    for i, settings in enumerate(cfg["tray_segments"]):
        # get module/segment class
        if "." not in settings["ModuleClass"]:
            module_class = settings["ModuleClass"]
        else:
            module_class = setup.load_class(settings["ModuleClass"])

        # sanity check to make sure the user didn't have a typo
        allowed_keys = ["ModuleClass", "ModuleKwargs", "ModuleTimer"]
        for key in settings.keys():
            if key not in allowed_keys:
                msg = "The key '{}'' is not in the allowed keys: {}!"
                raise KeyError(msg.format(key, allowed_keys))

        # dynamically replace values of the form
        #       context-->a.b.c
        # with tray.context[a][b][c]
        search_key = "context-->"
        kwargs = {}
        if "ModuleKwargs" in settings:
            for key, value in settings["ModuleKwargs"].items():
                # dynamically replace key
                if isinstance(value, str):
                    if search_key in value:
                        context_keys = value.replace(search_key, "").split(".")
                        value = tray.context
                        for context_key in context_keys:
                            value = value[context_key]

                    elif value == "<config>":
                        value = cfg

                    else:
                        # expand with parameters in config
                        value = value.format(**cfg)

                kwargs[key] = value

        if "OutputKey" in kwargs:
            tray.context["ic3_processing"]["HDF_keys"].append(
                kwargs["OutputKey"]
            )

        # get name of module under which it will be added to the tray
        name = settings["ModuleClass"].split(".")[-1] + "_{:05d}".format(i)

        # start timer if specified
        if "ModuleTimer" in settings and settings["ModuleTimer"]:
            tray.Add(tray_timer.TimerStart, TimerName=name + "_timer")
            tray.context["ic3_processing"]["HDF_keys"].append("DurationQ")
            tray.context["ic3_processing"]["HDF_keys"].append("DurationP")

        # add module
        tray.Add(module_class, name, **kwargs)

        # stop timer if specified
        if "ModuleTimer" in settings and settings["ModuleTimer"]:
            tray.Add(tray_timer.TimerStop, TimerName=name + "_timer")

    # -----------------------------------------------------------
    # keep track of merged files and update weights if they exist
    # -----------------------------------------------------------
    if cfg["data_type"] != "exp" and not cfg["skip_meta_data_tracking"]:
        # Official simulation sets save the n_event_per_run information
        # in keys such as I3MCWeightDict or CorsikaWeightMap. However,
        # there are some older custom datasets that lack this information.
        if "dataset_n_events_per_run" in cfg:
            n_evets_per_run = cfg["dataset_n_events_per_run"]
        else:
            n_evets_per_run = -1

        if not context["weights_meta_info_exists"]:
            # check if "AddWeightMetaData" was already added to the I3Tray
            if "AddWeightMetaData" not in tray.TrayInfo().modules_in_order:
                tray.AddModule(
                    AddWeightMetaData,
                    "AddWeightMetaData",
                    NFiles=tray.context["ic3_processing"]["total_n_files"],
                    NEventsPerRun=n_evets_per_run,
                )

        tray.AddModule(
            UpdateMergedWeights,
            "UpdateMergedWeights",
            TotalNFiles=tray.context["ic3_processing"]["total_n_files"],
        )

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    if cfg["write_i3"]:
        if "i3_streams" in cfg["write_i3_kwargs"]:
            i3_streams = [
                icetray.I3Frame.Stream(s)
                for s in cfg["write_i3_kwargs"].pop("i3_streams")
            ]
        else:
            i3_streams = [
                icetray.I3Frame.DAQ,
                icetray.I3Frame.Physics,
                icetray.I3Frame.TrayInfo,
                icetray.I3Frame.Simulation,
                icetray.I3Frame.Stream("S"),
                icetray.I3Frame.Stream("M"),
                icetray.I3Frame.Stream("m"),
                icetray.I3Frame.Stream("W"),
                icetray.I3Frame.Stream("X"),
            ]
        print("Only writing the following streams:", i3_streams)

        tray.AddModule(
            "I3Writer",
            "EventWriter",
            filename="{}.{}".format(context["outfile"], cfg["i3_ending"]),
            Streams=i3_streams,
            **cfg["write_i3_kwargs"]
        )

    if cfg["write_hdf5"]:
        keys = cfg["write_hdf5_kwargs"].pop("Keys")
        keys += tray.context["ic3_processing"]["HDF_keys"]
        tray.AddSegment(
            hdfwriter.I3HDFWriter,
            "hdf",
            Output="{}.hdf5".format(context["outfile"]),
            Keys=[k for k in set(keys)],
            **cfg["write_hdf5_kwargs"]
        )
    # --------------------------------------------------

    tray.AddModule("TrashCan", "the can")
    tray.Execute()
    tray.Finish()

    print("\nUsage:")
    print("------")
    msg = "    systime: {:7.2f} | usertime {:7.2f} | ncall: {:5d} | {}"
    for entry in tray.Usage():
        key = entry.key()
        usage = entry.data()
        print(msg.format(usage.systime, usage.usertime, usage.ncall, key))

    end_time = timeit.default_timer()
    print("\nDuration: {:5.3f}s".format(end_time - start_time))


if __name__ == "__main__":
    main()
