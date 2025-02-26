.. IceCube DNN reconstruction

.. _create_training_data:

.. note::
    If you already have training data files available, then you can skip to the :ref:`next section<Train Model>`. You may also use the training files
    located here: ``/data/user/mhuennefeld/DNN_reco/tutorials/training_data``.

Create Training Data
********************

To create the training data we will use the tray
segment ``ic3_data.segments.CreateDNNData`` from the |ic3_data| project
and for the labels we will use ``ic3_labels.labels.modules.MCLabelsCascades``
from the |ic3_labels| repository.
You are free to include these modules in your processing set up of choice.
Here we will use
`these processing scripts <https://github.com/mhuen/ic3-processing>`_.

The tray segment ``ic3_data.segments.CreateDNNData`` can write out different
types of input data.
Options include (more available):

* ``charge_bins``: Histogram charge in time bins as given by 'TimeBins'
* ``charge_weighted_time_quantiles``: Calculate charge weighted time quantiles for the quantiles specified in 'TimeQuantiles'
* ``autoencoder``: Encodes pulse data with an autoencoder (deprecated)
* ``pulse_summmary_clipped``: Calculates 9 summary values from pulses (used in `DNN-reco paper <https://arxiv.org//abs/2101.11589>`_, see paper for more information)
* ``reduced_summary_statistics_data``: Total charge, time of first pulse, std. deviation of pulse times. These are computed entirely in c++. If you need a NN that executes in < 1ms/event, then this is a good choice.

If your application needs different input data, you can easily add a function
in ``ic3_data.data_formats``.
The 'DataFormat' key of the tray segment ``ic3_data.segments.CreateDNNData``
defines which function in ``ic3_data.data_formats`` will be used
to create the input data.
The only requirement is that the input data must be a vector of length n for
each DOM.

.. note::
    Recommended data input formats are ``pulse_summmary_clipped`` or
    ``reduced_summary_statistics_data``. Support for ``ml_suite``-based
    input data will be added in the future.

We will use ``pulse_summmary_clipped`` in this tutorial.
The input data to our network will therefore consist of summary values
calculated from the pulses of each DOM.
The computed summary values for each DOM are:


    1. Total DOM charge
    2. Charge within 500ns of first pulse.
    3. Charge within 100ns of first pulse.
    4. Relative time of first pulse. (relative to total time offset)
    5. Charge weighted quantile with q = 0.2
    6. Charge weighted quantile with q = 0.5 (median)
    7. Relative time of last pulse. (relative to total time offset)
    8. Charge weighted mean pulse arrival time
    9. Charge weighted std of pulse arrival time

The settings we will use to achieve this are:

.. code-block:: python

    from ic3_data.segments import CreateDNNData

    tray.AddSegment(
        CreateDNNData,
        "CreateDNNData",
        NumDataBins=9,
        RelativeTimeMethod="time_range",
        DataFormat="pulse_summmary_clipped",
        PulseKey="InIceDSTPulses",
        DOMExclusions=["SaturationWindows", "BadDomsList", "CalibrationErrata"],
        PartialExclusion=True,
    )

To be able to train our neural network, we must also define labels.
In this tutorial we will use:

.. code-block:: python

    from ic3_labels.labels.modules import MCLabelsCascades

    tray.AddModule(
        MCLabelsCascades,
        "MCLabelsCascades",
        PulseMapString="InIceDSTPulses",
        PrimaryKey="MCPrimary",
        ExtendBoundary=0.0,
        OutputKey="LabelsDeepLearning",
    )

This will create a number of different labels and
write them to an I3MapStringDouble.
Amongst others, these include:

* ``EnergyVisible``: visible energy in Detector
* ``num_muons_at_entry``: number of muons that enter the convex hull around the detector
* ``VertexX/Y/Z``: For cascades: x/y/z-coordinate of vertex; For muons: x/y/z-coordinate of entry point/starting point
* ``PrimaryZenith``: zenith angle of primary particle as defined by 'PrimaryKey'
* ``PrimaryAzimuth``: azimuth angle of primary particle as defined by 'PrimaryKey'
* ``PrimaryDirectionX/Y/Z``: direction vector components of primary particle as defined by 'PrimaryKey'
* ``PrimaryEnergy``: energy of primary particle as defined by 'PrimaryKey'
* ``Length``: For cascades: length of cascade; For muons: length of muon
* ``LengthInDetector``: For cascades: length of cascade within convex hull; For muons: length of muon within convex hull
* ``p_starting_300m``: 1 if neutrino event with vertex within 300m of convex hull around detector, else 0
* ``p_starting``: 1 if starting event, else 0
* ``p_outside_cascade``: 1 if neutrino event with vertex outside of convex hull, else 0
* ``p_entering``: 1, if entering muon, else 0

Now we are ready to save the training data to hdf5 files:

.. code-block:: python

    from icecube import hdfwriter

    tray.AddSegment(
        hdfwriter.I3HDFWriter,
        "hdf",
        Output="name_of_output_file.hdf5",
        CompressionLevel=9,
        Keys=[
            "dnn_data_bin_values",
            "dnn_data_bin_indices",
            "dnn_data_global_time_offset",
            "LabelsDeepLearning",
        ],
        SubEventStreams=["InIceSplit"],
    )

We can now put these modules together in a script and process 1000 files for each of the NuMu datasets 22644, 226465, 22646.
To facilitate this process, we will use the mentioned `processing scripts <https://github.com/mhuen/ic3-processing>`_.
This repository allows to easily create processing scripts via a yaml configuration
file. The configuration file we will use is provided within the |dnn_reco| repository.

If we have installed ``ic3-processing`` (see :ref:`Installation and Requirements`), we can create the job files via:

.. code-block:: bash

    # create job files (--help for more options)
    ic3_create_job_files $DNN_HOME/repositories/dnn_reco/configs/tutorial/create_training_data.yaml -d $DNN_HOME/training_data/

This will write the executable job files and the configuration file that was used
to the directory ``$DNN_HOME/training_data/processing``.
The output files will be written to ``$DNN_HOME/training_data/datasets``.
You may also write DAGMan files if you pass the option ``--dagman``.
Make sure to write the DAGMan files to condor scratch.
If you created DAGMan files, you then start the DAGMan by executing the ``start_dagman.sh`` script.
Alternatively, you can process the job files locally with the command ``ic3_process_local``.
Check ``--help`` for options.
To process a single file, you can also directly execute the shell script in a fresh shell:

.. code-block:: bash

    # Open a new terminal with a fresh shell without loading an icecube
    # environment. Redefine our $DNN_HOME variable.
    export DNN_HOME=/data/user/${USER}/DNN_tutorial

    # process file number 1 of dataset 22644 (part of our training set)
    $DNN_HOME/training_data/processing/NuGen/22644/level2_dev/jobs/0000000-0000999/job_Level2_NuMu_NuGenCCNC.022644.000001.sh

    # process file number 0 of dataset 22644 (part of our validation set)
    $DNN_HOME/training_data/processing/NuGen/22644/level2_dev/jobs/0000000-0000999/job_Level2_NuMu_NuGenCCNC.022644.000000.sh

.. note::
    Make sure to open a fresh shell without loading an icecube environment to execute the job shell scripts. The shell scripts are set up such that they will load an icecube environment. Hence, if you already have
    one loaded in current shell, it will cause problems.

To test the rest of the tutorial, it is enough to process one file
from the training and validation set (run numbers ending with 0).
However, the network will overfit on the training data due to the low number of available training events.
