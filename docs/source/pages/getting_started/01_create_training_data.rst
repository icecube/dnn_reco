.. IceCube DNN reconstruction

.. _create_training_data:

Create Training Data
********************

To create the training data we will use the tray
segment ``ic3_data.segements.CreateDNNData`` from the |ic3_data| project
and for the labels we will use ``ic3_labels.labels.modules.MCLabelsCascades``
from the |ic3_labels| repository.
You are free to include these modules in your processing set up of choice.
Here we will use
`these processing scripts <https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/mhuennefeld/processing_scripts>`_.

.. note::
    :ref:`If you already have training data files available, then you can skip
    to the next section<Train Model>`

The tray segment ``ic3_data.segements.CreateDNNData`` can write out different
types of input data.
Options include:

* ``charge_bins``: Histogram charge in time bins as given by 'TimeBins'
* ``charge_weighted_time_quantiles``: Calculate charge weighted time quantiles for the quantiles specified in 'TimeQuantiles'
* ``autoencoder``: Encodes pulse data with an autoencoder
* ``pulse_summmary_clipped``: Calculate summary values from pulses

If your application needs different input data, you can easily add a function
in ``ic3_data.data_formats``.
The 'DataFormat' key of the tray segment ``ic3_data.segements.CreateDNNData``
defines which function in ``ic3_data.data_formats`` will be used
to create the input data.
The only requirement is that the input data must be a vector of length n for
each DOM.

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

    tray.AddSegment(CreateDNNData, 'CreateDNNData',
                    NumDataBins=9,
                    RelativeTimeMethod='time_range',
                    DataFormat='pulse_summmary_clipped',
                    PulseKey='InIceDSTPulses')

To be able to train our neural network, we must also define labels.
In this tutorial we will use:

.. code-block:: python

    from ic3_labels.labels.modules import MCLabelsCascades

    tray.AddModule(MCLabelsCascades, 'MCLabelsCascades',
                   PulseMapString='InIceDSTPulses',
                   PrimaryKey='MCPrimary',
                   ExtendBoundary=0.,
                   OutputKey='LabelsDeepLearning')

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

    tray.AddSegment(hdfwriter.I3HDFWriter, 'hdf',
                    Output='name_of_output_file.hdf5',
                    CompressionLevel=9,
                    Keys=['dnn_data_bin_values',
                          'dnn_data_bin_indices',
                          'dnn_data_global_time_offset',
                          'LabelsDeepLearning'],
                    SubEventStreams=['InIceSplit'])

We can now put these modules together in a script and process the dataset 11883.
To facilitate this process, we will use the mentioned `processing scripts <https://code.icecube.wisc.edu/projects/icecube/browser/IceCube/sandbox/mhuennefeld/processing_scripts>`_, in which this is already done.
First we must fetch the processing scripts:

.. code-block:: bash

    svn co http://code.icecube.wisc.edu/svn/sandbox/mhuennefeld/processing_scripts/trunk/processing/ $DNN_HOME/processing


Within the svn repository, there is a already a configuration file available
that we will use to create the training data.

..
    There is already a template configuration file available.
    We will copy this file to another location and make our edits.

    .. code-block:: bash

        mkdir --parents $DNN_HOME/configs/processing/
        cp $DNN_HOME/processing/configs/tutorial_dnn_reco/getting_started/create_training_data_01.yaml $DNN_HOME/configs/processing/

Create the job files via:

.. code-block:: bash

    cd $DNN_HOME/processing

    # create job files (--help for more options)
    python create_job_files.py configs/tutorial_dnn_reco/getting_started/create_training_data_01.yaml -d $DNN_HOME/training_data/

This will write the exectuable job files and the configuration file that was used
to the directory ``$DNN_HOME/training_data/processing``.
The output files will be written to ``$DNN_HOME/training_data/datasets``.
You may also write DAGMan files if you pass the option ``--dagman``.
Make sure to write the DAGMan files to condor scratch.
If you created DAGMan files, you then start the DAGMan by executing the ``start_dagman.sh`` script.
Alternatively, you can process the job files locally with the script ``process_local.py``.
Check ``--help`` for options.
To process a single file, you can also directly execute the shell script in a fresh shell:

.. code-block:: bash

    # Open a new terminal with a fresh shell without loading an icecube
    # environment. Redefine our $DNN_HOME variable.
    export DNN_HOME=/data/user/${USER}/DNN_tutorial

    # process file number 0 (part of our training set)
    $DNN_HOME/training_data/processing/datasets/11883/clsim-base-4.0.5.0.99_eff/output/summaryV2_clipped/jobs/00000-00999/job_11883_clsim-base-4.0.5.0.99_effDOMPulseData_00000000.sh

    # process file number 1000 (part of our validation set)
    $DNN_HOME/training_data/processing/datasets/11883/clsim-base-4.0.5.0.99_eff/output/summaryV2_clipped/jobs/01000-01999/job_11883_clsim-base-4.0.5.0.99_effDOMPulseData_00001000.sh

.. note::
    Make sure to open a fresh shell without loading an icecube environment to execute the job shell scripts. The shell scripts are set up such that they will load an icecube environment. Hence, if you already have
    one loaded in current shell, it will cause problems.

To test the rest of the tutorial, it is enough to process one file
from the training and validation set.
However, the network will overfit on the training data which then only consists
of about 700 events.

