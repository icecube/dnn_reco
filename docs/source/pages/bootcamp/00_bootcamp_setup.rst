.. IceCube DNN reconstruction

.. _bootcamp_setup:

Installation and Setup
**********************

We will use https://hub.gzk.io/ for this tutorial.
After you have logged in, we will open a terminal to install |dnn_reco| and to
setup the environment.

Setup Environment
=================

To facilitate the installation process we will define and create necessary
environment variables here.

.. code-block:: bash

    # Create working directory
    # We will create a virtual environment and clone the repositories here
    export DNN_HOME=${HOME}/DNN_tutorial

    # create directories
    mkdir --parents $DNN_HOME


It is recommended to set up a python virtual environment for the installation
of DNN reco and its dependencies.
We will skip that for this tutorial though.



Install DNN Reco
================

We are now ready to install the necessary prerequisites and |dnn_reco|.

.. code-block:: bash

    mkdir ${DNN_HOME}/repositories
    cd  ${DNN_HOME}/repositories

    # clone repositories
    git clone https://github.com/mhuen/TFScripts.git
    git clone https://github.com/mhuen/ic3-data.git
    git clone https://github.com/mhuen/ic3-labels.git
    git clone https://github.com/mhuen/dnn_reco.git

    # install packages
    pip install -e  ${DNN_HOME}/repositories/TFScripts
    pip install -e  ${DNN_HOME}/repositories/ic3-data
    pip install -e  ${DNN_HOME}/repositories/ic3-labels
    pip install -e  ${DNN_HOME}/repositories/dnn_reco

Verify Installation
===================

Try to create a tensorflow session and to import |dnn_reco|.

.. code-block:: bash

    # the following should successfully create a tensorflow session
    python -c 'import tensorflow as tf; print(tf.__version__); tf.Session()'

    # try to import dnn_reco (This should run without giving any output)
    python -c 'import dnn_reco; import tfscripts; import ic3_labels; import ic3_data'