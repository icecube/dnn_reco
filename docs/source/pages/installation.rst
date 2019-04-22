.. IceCube DNN reconstruction

Installation and Requirements
*****************************

The following packages need to be installed to be able to use DNN reco:

* `Tensorflow <https://www.tensorflow.org/>`_
* https://github.com/mhuen/TFScripts

Additionally, to create the training data we need the following:


* https://github.com/mhuen/ic3-data
* https://github.com/mhuen/ic3-labels

Define Environment Variables for Installation
=============================================

To facilitate the installation process we will define and create necessary
environment variables here.

.. code-block:: bash

    # Create working directory
    # We will create a virtual environment and clone the repositories here
    export DNN_HOME=/data/user/${USER}/DNN_tutorial

    # create directories
    mkdir --parents $DNN_HOME

Set up a Python Virtual Environment
===================================

It is recommended to set up a python virtual environment for the installation
of DNN reco and its dependencies.
In the following we will create a virtual environment with virtualenv.

.. code-block:: bash

    #If you have not yet installed virtualenv, you can do so via
    pip install virtualenv

    # Create virtualenv
    cd $DNN_HOME
    virtualenv --no-site-packages dnn_reco_env

    # activate virtual environment
    source ${DNN_HOME}/dnn_reco_env/bin/activate


Install Prerequisites and DNN reco
==================================

We are now ready to install the necessary prerequisites and |dnn_reco|.

Install Tensorflow
------------------

Follow the instructions on `<https://www.tensorflow.org/install>`_ to install
tensorflow. Version 1.13 is currently recommended. Version 2.0 is not yet
supported by |dnn_reco|.

On NPX/Cobalts you can install tensorflow 1.8 with the prebuilt pip wheels:

.. code-block:: bash

    # CPU version (make sure your virutal environement is activated)
    pip install /data/user/mhuennefeld/software/tensorflow/cpu/py2-v3.0.1/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl

    # GPU version (make sure your virutal environement is activated)
    pip install /data/user/mhuennefeld/software/tensorflow/gpu/py2-v3.0.1/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl

To run the GPU version with the prebuilt wheel you will need to have a GPU and
CUDA 8.0 available. You can use:

.. code-block:: bash

    export CUDA_HOME=/data/user/mhuennefeld/software/condor_cuda3/cuda-8.0;
    export PATH=$PATH:/data/user/mhuennefeld/software/condor_cuda3/cuda-8.0/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/user/mhuennefeld/software/condor_cuda3/cuda-8.0/lib64


Install DNN Reco
----------------

.. code-block:: bash

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

Verify Installation
-------------------

Try to create a tensorflow session and to import |dnn_reco|.

.. code-block:: bash

    # the following should successfully create a tensorflow session
    python -c 'import tensorflow as tf; print(tf.__version__); tf.Session()'

    # try to import dnn_reco (This should run without giving any output)
    python -c 'import dnn_reco; import tfscripts; import ic3_labels; import ic3_data'