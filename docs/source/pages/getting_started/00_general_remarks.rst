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

    # Redefine the environment variable
    export DNN_HOME=/data/user/${USER}/DNN_tutorial

    # load icecube environment
    eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)
    /cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/env-shell.sh

Now that the icecube environment is loaded, we can activate our virtual environment.

.. code-block:: bash

    # activate python virtual environment
    source ${DNN_HOME}/py3-v4.3.0_tensorflow2.14/bin/activate

    # set CUDA environment variables
    export CUDA_HOME=/data/user/mhuennefeld/software/cuda/cuda-11.8
    export PATH=$PATH:${CUDA_HOME}/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64


.. note::
    It is important to first load the icecube environment and then
    activate the virtual environment. If done the other way around, the
    environment variables defined by the virtual environment will be
    overwritten by the icecube env-shell.sh. As a result, a different
    python version will be used than the one from the virtual environment.
