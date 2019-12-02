.. IceCube DNN reconstruction

.. _general_remarks:

General Remarks
***************

|dnn_reco| is intended to be a highly modularized software framework for neural
network applications in IceCube. As such, there are many modules that you can
easily modify or switch out, depending on your needs.
A central configuration file defines which modules and which settings
will be used.
This configuration file is necessary for each of the individual steps.
There are many options that can be configured.
Here we will only briefly touch a few basic concepts.
An extensive documentation of the configuration file options is provided in
:ref:`Configuration Options<Configuration Options>`.

In this tutorial we will use the directory and virtual environment that we
created during the installation steps (see :ref:`Installation and Requirements`).

.. code-block:: bash

    # Create environment variable if it doesn't already exist
    export DNN_HOME=/data/user/${USER}/DNN_tutorial

    # load icecube environment if not already loaded
    eval $(/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh)
    source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/icerec/V05-02-04/env-shell.sh

    # Activate virtual environment if not already activated
    source ${DNN_HOME}/dnn_reco_env/bin/activate

.. note::
    It is important to first load the icecube environment and then
    activate the virtual environment. If done the other way around, the
    environment variables defined by the virtual environment will be
    overwritten by the icecube env-shell.sh. As a result, a different
    python version will be used than the one from the virtual environment.