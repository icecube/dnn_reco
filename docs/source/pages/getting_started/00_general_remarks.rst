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

    # Activate virtual environment if not already activated
    source ${DNN_HOME}/dnn_reco_env/bin/activate

