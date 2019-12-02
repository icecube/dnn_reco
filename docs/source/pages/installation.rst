.. IceCube DNN reconstruction

.. _installation_and_requirements:

Installation and Requirements
*****************************

The following packages need to be installed to be able to use DNN reco:

* `Tensorflow <https://www.tensorflow.org/>`_
* https://github.com/mhuen/TFScripts
* https://github.com/mhuen/ic3-data

Additionally, to create the training data we need the following:

* https://github.com/mhuen/ic3-labels

This guide will explain how to create a virtual environment with the necessary
software packages. An alternative is to use singularity containers which have
tensorflow installed. Documentation on how to use the singularity containers will follow in the future.

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

    # load icecube environment
    eval $(/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh)
    source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/icerec/V05-02-04/env-shell.sh

.. source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/simulation/V06-01-01/env-shell.sh

Set up a Python Virtual Environment
===================================

It is recommended to set up a python virtual environment for the installation
of DNN reco and its dependencies.
In the following we will create a virtual environment with virtualenv.

.. code-block:: bash

    #If you have not yet installed virtualenv, you can do so via
    pip install virtualenv


    # We do not want to use any external python packages:
    # Unsetting PYTHONPATH is needed during pip installing packages to our
    # environment. If it is not unset, then we will find packages from cvmfs
    # and pip itself will not work.
    # When later using this virtualenv we don't need to unset PYTHONPATH, so that
    # we can find packages in cvfms that do not exist in our environment
    unset PYTHONPATH

    # Create virtualenv
    cd $DNN_HOME
    virtualenv --no-site-package dnn_reco_env

    # activate virtual environment
    source ${DNN_HOME}/dnn_reco_env/bin/activate

    # install specific h5py version that matches hdf5 libs in cvmfs py2-v3.0.1
    pip install h5py==2.7.1


Modify Virtual Environment Activation Script
============================================

Loading a cvmfs python version as well as activating an IceCube build will
modify the ``$PYTHONPATH`` variable. The paths to the cmvfs python and icecube
libs will be prepended. This means that python will first look into those
directories to find packages before it looks inside the virtual environment.
We want to avoid this and enforce that packages from our virtual environment
are used first. To do this we need to prepend the path to our virtual
environment in the ``$PYTHONPATH``. Since we don't want to do this manually
everytime we load the environment, we can add this to the ``activate`` shell
script that starts the virtual environment.
We will need to add the following to the ``deactivate ()`` function

.. code-block:: bash

    if ! [ -z "${_OLD_VIRTUAL_PYTHONHOME+_}" ] ; then
        PYTHONHOME="$_OLD_VIRTUAL_PYTHONHOME"
        export PYTHONHOME
        unset _OLD_VIRTUAL_PYTHONHOME
    fi

and this in the main body:

.. code-block:: bash

    # unset PYTHONHOME if set
    if ! [ -z "${PYTHONHOME+_}" ] ; then
        _OLD_VIRTUAL_PYTHONHOME="$PYTHONHOME"
        unset PYTHONHOME
    fi

We can change the activation scripts manually or by executing the following commands:

.. code-block:: bash

    # update active script from cpu environment
    perl -i -0pe 's/_OLD_VIRTUAL_PATH\="\$PATH"\nPATH\="\$VIRTUAL_ENV\/bin:\$PATH"\nexport PATH/_OLD_VIRTUAL_PATH\="\$PATH"\nPATH\="\$VIRTUAL_ENV\/bin:\$PATH"\nexport PATH\n\n# prepend virtual env path to PYTHONPATH if set\nif ! \[ -z "\$\{PYTHONPATH+_\}" \] ; then\n    _OLD_VIRTUAL_PYTHONPATH\="\$PYTHONPATH"\n    export PYTHONPATH\=\$VIRTUAL_ENV\/lib\/python2.7\/site-packages:\$PYTHONPATH\nfi/' ${DNN_HOME}/dnn_reco_env/bin/activate
    perl -i -0pe 's/        export PYTHONHOME\n        unset _OLD_VIRTUAL_PYTHONHOME\n    fi/        export PYTHONHOME\n        unset _OLD_VIRTUAL_PYTHONHOME\n    fi\n\n    if ! \[ -z "\$\{_OLD_VIRTUAL_PYTHONPATH+_\}" \] ; then\n        PYTHONPATH\="\$_OLD_VIRTUAL_PYTHONPATH"\n        export PYTHONPATH\n        unset _OLD_VIRTUAL_PYTHONPATH\n    fi/' ${DNN_HOME}/dnn_reco_env/bin/activate


Install Prerequisites and DNN reco
==================================

We are now ready to install the necessary prerequisites and |dnn_reco|.

.. _install_tensorflow:

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
    export PATH=$PATH:${CUDA_HOME}/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64

which you can also add to the virtual environment activate script.

.. _install_dnn_reco:

Install DNN Reco
----------------

We are now ready to install |dnn_reco|.
To do so we must clone the repositories and then install them via pip.
The flag ``-e`` or ``--editable`` enables us to edit the source files and use
these changes without having to reinstall the package after each change.

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

    # Make sure h5py version is still 2.7.1 due to hdf5 lib version
    # problems with cmvfs. If tables is installed, uninstall it, since
    # cmvfs provides a matching tables version, which will become
    # availabe if we don't unset PYTHONPATH
    # You can check the h5py version via:
    python -c 'import h5py; print h5py.__version__'

.. _verify_installation:

Verify Installation
-------------------

We are now done and can use our new environment.
Log in to a fresh shell and load the environment via:

.. code-block:: bash

    # Redefine the environment variable
    export DNN_HOME=/data/user/${USER}/DNN_tutorial

    # load icecube environment
    eval $(/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh)
    source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/icerec/V05-02-04/env-shell.sh

    # activate python virtual environment
    source ${DNN_HOME}/dnn_reco_env/bin/activate

To verify if our environment was installed correctly, we can
try to create a tensorflow session and to import |dnn_reco|.

.. code-block:: bash

    # the following should successfully create a tensorflow session
    python -c 'import tensorflow as tf; print(tf.__version__); tf.Session()'

    # try to import dnn_reco (This should run without giving any output)
    python -c 'import dnn_reco; import tfscripts; import ic3_labels; import ic3_data'