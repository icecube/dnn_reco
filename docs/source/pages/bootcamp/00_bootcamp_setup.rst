.. IceCube DNN reconstruction

.. _bootcamp_setup:

Installation and Setup
**********************

We will use https://hub.gzk.io/ for this tutorial.
After you have logged in, we will open a terminal to
setup the environment and to install |dnn_reco|.

Setup Environment
=================

To facilitate the installation process we will define and create necessary
environment variables here.
Unfortunately we can not use the local anaconda and icecube installation
since the hdf5 version is slightly different than what we have available on
NPX to create the training files.
We will therefore clear environment variables and instead use a cvmfs
python with the following slightly dirty hack:

.. code-block:: bash

    # -----------
    # clear shell
    # -----------
    unset I3_SHELL
    unset I3_SRC
    unset I3_TESTDATA
    unset I3_DATA
    unset I3_BUILD
    unset I3_PLATFORM

    PATH=$(echo "$PATH" | sed -e 's|/opt/icetray/combo/build/bin:||')
    LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's|/opt/icetray/combo/build/lib/tools:||')
    LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's|/opt/icetray/combo/build/lib:||')
    PYTHONPATH=$(echo "$PYTHONPATH" | sed -e 's|/opt/icetray/combo/build/lib:||')
    # -----------

    # Use python and icecube metaproject from cvmfs
    eval $(/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh)
    source /shared/dnn_reco_tutorial/software/parasitic_icerec/build/env-shell.sh

    # use CUDA 10.0 (latest supported version by pip install tensorflow-gpu)
    export CUDA_HOME=/shared/dnn_reco_tutorial/cuda-10.0;
    export PATH=${CUDA_HOME}/bin:$PATH
    export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

    # Create working directory
    # We will create a virtual environment and clone the repositories here
    export DNN_HOME=${HOME}/DNN_tutorial

    # create directories
    mkdir --parents $DNN_HOME

It is recommended to set up a python virtual environment for the installation
of DNN reco and its dependencies.
In the following we will create a virtual environment with virtualenv.

.. code-block:: bash

    # Create virtualenv
    cd $DNN_HOME
    virtualenv dnn_reco_env

    # activate virtual environment
    source ${DNN_HOME}/dnn_reco_env/bin/activate

..    # make sure h5py is not newly installed in virtual env (pip uninstall h5py)



Install DNN Reco
================

We are now ready to install the necessary prerequisites and |dnn_reco|.

.. code-block:: bash

    # install tensorflow
    pip install tensorflow-gpu

    mkdir ${DNN_HOME}/repositories
    cd  ${DNN_HOME}/repositories

    # clone repositories
    git clone https://github.com/mhuen/TFScripts.git
    git clone https://github.com/mhuen/ic3-data.git
    git clone https://github.com/mhuen/ic3-labels.git
    git clone https://github.com/mhuen/dnn_reco.git

    # make sure that your virtualenv is activated
    # you can check this by exectuting
    which pip
    # It should point to:
    echo ${DNN_HOME}/dnn_reco_env/bin/pip

    # install packages
    pip install -e  ${DNN_HOME}/repositories/TFScripts
    pip install -e  ${DNN_HOME}/repositories/ic3-data
    pip install -e  ${DNN_HOME}/repositories/ic3-labels
    pip install -e  ${DNN_HOME}/repositories/dnn_reco

    # uninstall h5py (problems with hdf5 version...)
    pip uninstall h5py

Verify Installation
===================

Try to create a tensorflow session and to import |dnn_reco|.

.. code-block:: bash

    # the following should successfully create a tensorflow session
    python -c 'import tensorflow as tf; print(tf.__version__); tf.Session()'

    # try to import dnn_reco (This should run without giving any output)
    python -c 'import dnn_reco; import tfscripts; import ic3_labels; import ic3_data'