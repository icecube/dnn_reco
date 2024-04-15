.. IceCube DNN reconstruction

.. _apply_model:

Apply Model
***********

The |dnn_reco| software package provides a method to export your trained
models which can be applied to i3 files via the provided I3TraySegment.
To export our trained model we must provide the training configuration file
as well as the configuration file that was used to create the training data.
Consult ``--help`` for additional options. We will export our model by running:

.. code-block:: bash

    python export_model.py $CONFIG_DIR/getting_started.yaml -s $DNN_HOME/training_data/processing/datasets/11883/clsim-base-4.0.5.0.99_eff/output/summaryV2_clipped/create_training_data_01.yaml_0000 -o $DNN_HOME/exported_models/getting_started_model

This should complete with the message:

.. code-block:: php

    ====================================
    = Successfully exported model to:  =
    ====================================
    $DNN_HOME/exported_models/getting_started_model

To apply our new model to i3 files we can use the provided I3TraySegment
``dnn_reco.ic3.segments.ApplyDNNRecos`` in a simple script such as the
following:

.. code-block:: python

    import os
    import click

    from I3Tray import I3Tray
    from icecube import icetray, dataio, hdfwriter
    from icecube.weighting import get_weighted_primary

    from ic3_labels.labels.modules import MCLabelsCascades
    from dnn_reco.ic3.segments import ApplyDNNRecos


    @click.command()
    @click.argument("input_file_pattern", type=click.Path(exists=True), required=True, nargs=-1)
    @click.option(
        "-o", "--outfile", default="dnn_output", help="Name of output file without file ending."
    )
    @click.option(
        "-m",
        "--model_names",
        default="getting_started_model",
        help="Parent directory of exported models.",
    )
    @click.option(
        "-d",
        "--models_dir",
        default="{DNN_HOME}/exported_models",
        help="Parent directory of exported models.",
    )
    @click.option(
        "-g",
        "--gcd_file",
        default="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2012.56063_V1.i3.gz",
        help="GCD File to use.",
    )
    @click.option(
        "-j", "--num_cpus", default=8, help="Number of CPUs to use if run on CPU instead of GPU"
    )
    @click.option("--i3/--no-i3", default=True)
    @click.option("--hdf5/--no-hdf5", default=True)
    def main(input_file_pattern, outfile, model_names, models_dir, gcd_file, num_cpus, i3, hdf5):

        # create output directory if necessary
        base_path = os.path.dirname(outfile)
        if not os.path.isdir(base_path):
            print("\nCreating directory: {}\n".format(base_path))
            os.makedirs(base_path)

        # expand models_dir with environment variable
        models_dir = models_dir.format(DNN_HOME=os.environ["DNN_HOME"])

        HDF_keys = [
            "LabelsDeepLearning",
            "MCPrimary",
            "OnlineL2_PoleL2MPEFit_MuEx",
            "OnlineL2_PoleL2MPEFit_TruncatedEnergy_AllBINS_Muon",
        ]

        tray = I3Tray()

        # read in files
        file_name_list = [str(gcd_file)]
        file_name_list.extend(list(input_file_pattern))
        tray.AddModule("I3Reader", "reader", Filenamelist=file_name_list)

        # Add labels
        tray.AddModule(get_weighted_primary, "getWeightedPrimary", If=lambda f: not f.Has("MCPrimary"))
        tray.AddModule(
            MCLabelsCascades,
            "MCLabelsCascades",
            PulseMapString="InIceDSTPulses",
            PrimaryKey="MCPrimary",
            ExtendBoundary=0.0,
            OutputKey="LabelsDeepLearning",
        )

        # collect model and output names
        if isinstance(model_names, (str, unicode)):
            model_names = [str(model_names)]
        output_names = ["DeepLearningReco_{}".format(m) for m in model_names]

        # Make sure DNN reco will be written to hdf5 file
        for outbox in output_names:
            if outbox not in HDF_keys:
                HDF_keys.append(outbox)
                HDF_keys.append(outbox + "_I3Particle")

        # Apply DNN Reco
        tray.AddSegment(
            ApplyDNNRecos,
            "ApplyDNNRecos",
            pulse_key="InIceDSTPulses",
            model_names=model_names,
            output_keys=output_names,
            models_dir=models_dir,
            num_cpus=num_cpus,
        )

        # Write output
        if i3:
            tray.AddModule("I3Writer", "EventWriter", filename="{}.i3.bz2".format(outfile))

        if hdf5:
            tray.AddSegment(
                hdfwriter.I3HDFWriter,
                "hdf",
                Output="{}.hdf5".format(outfile),
                CompressionLevel=9,
                Keys=HDF_keys,
                SubEventStreams=["InIceSplit"],
            )
        tray.AddModule("TrashCan", "YesWeCan")
        tray.Execute()


    if __name__ == "__main__":
        main()


This script loads the specified i3 files, adds the labels, applies our
model, and saves the output to i3/ hdf5 files as specified.
Create a file ``apply_dnn_reco.py`` in the ``$DNN_HOME`` directory
with the above content with your editor of choice.

.. code-block:: bash

    # Create a file apply_dnn_reco in the $DNN_HOME directory and save
    # the above example script to that file
    vim $DNN_HOME/apply_dnn_reco.py

We can then apply our model to some of the i3 files of NuGen dataset 11883
which we have not used in our training set with the following:

.. code-block:: bash

    python $DNN_HOME/apply_dnn_reco.py /data/sim/IceCube/2012/filtered/level2/neutrino-generator/11883/01000-01999/clsim-base-4.0.5.0.99_eff/Level2_IC86.2012_nugen_numu.011883.001011.clsim-base-4.0.5.0.99_eff.i3.bz2 -o $DNN_HOME/output/dnn_reco_output

This will create an hdf5 and an i3 file with the specified file names:
``$DNN_HOME/output/dnn_reco_output.hdf5`` and
``$DNN_HOME/output/dnn_reco_output.i3.bz2``.

.. note::
    Running the |dnn_reco| on a CPU is much slower than running it on a GPU.
    If performance is an issue, then you should consider using a GPU.


..
    As we previously did for the creation of the training data, we will use
    the processing framework from link to svn sandbox.

    Modify the configuration file (link) to use the correct model
    add: model_dir, model_names
    and set GPU to 0.=?

    Then we create the job files

    and run them
    (no need to run dagman for just one file, we can simply execute the )
