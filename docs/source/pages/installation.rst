.. IceCube DNN reconstruction

.. _installation_and_requirements:

Installation and Requirements
*****************************

The following packages need to be installed to be able to use DNN reco:

* `Tensorflow <https://www.tensorflow.org/>`_
* https://github.com/icecube/TFScripts
* https://github.com/icecube/ic3-data

Additionally, to create the training data we need the following:

* https://github.com/icecube/ic3-labels
* https://github.com/mhuen/ic3-processing

This guide will explain how to create a virtual environment with the necessary
software packages. An alternative is to use singularity containers which have
tensorflow installed. Documentation on how to use the singularity containers will follow in the future.
Here we will create a virtual environment based on cvmfs python py3-v4.3.0
and icetray v1.12.0.

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

    # load cvmfs python environment
    eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/setup.sh)

.. source /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/simulation/V06-01-01/env-shell.sh

Set up a Python Virtual Environment
===================================

It is recommended to set up a python virtual environment for the installation
of DNN reco and its dependencies.
In the following we will create a virtual environment with virtualenv.

.. code-block:: bash

    # If virtualenv is not included in the chosen cvmfs python installation,
    # it can be installed via: `pip install --user virtualenv`

    # Create virtualenv
    cd $DNN_HOME
    python -m virtualenv py3-v4.3.0_tensorflow2.14

    # activate virtual environment
    source ${DNN_HOME}/py3-v4.3.0_tensorflow2.14/bin/activate


Modify Virtual Environment Activation Script
============================================

Loading a cvmfs python version as well as activating an IceCube build will
modify the ``$PYTHONPATH`` variable. The paths to the cmvfs python and icecube
libs will be prepended. This means that python will first look into those
directories to find packages before it looks inside the virtual environment.
We want to avoid this and enforce that packages from our virtual environment
are used first. To do this we need to prepend the path to our virtual
environment in the ``$PYTHONPATH``. Since we don't want to do this manually
every time we load the environment, we can add this to the ``activate`` shell
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

    # change activation script such that it prepends the path
    # to the virtual environment to the PYTHONPATH environment variable
    perl -i -0pe 's/_OLD_VIRTUAL_PATH\="\$PATH"\nPATH\="\$VIRTUAL_ENV\/bin:\$PATH"\nexport PATH/_OLD_VIRTUAL_PATH\="\$PATH"\nPATH\="\$VIRTUAL_ENV\/bin:\$PATH"\nexport PATH\n\n# prepend virtual env path to PYTHONPATH if set\nif ! \[ -z "\$\{PYTHONPATH+_\}" \] ; then\n    _OLD_VIRTUAL_PYTHONPATH\="\$PYTHONPATH"\n    export PYTHONPATH\=\$VIRTUAL_ENV\/lib\/python3.7\/site-packages:\$PYTHONPATH\nfi/' ${DNN_HOME}/py3-v4.3.0_tensorflow2.14/bin/activate
    perl -i -0pe 's/        export PYTHONHOME\n        unset _OLD_VIRTUAL_PYTHONHOME\n    fi/        export PYTHONHOME\n        unset _OLD_VIRTUAL_PYTHONHOME\n    fi\n\n    if ! \[ -z "\$\{_OLD_VIRTUAL_PYTHONPATH+_\}" \] ; then\n        PYTHONPATH\="\$_OLD_VIRTUAL_PYTHONPATH"\n        export PYTHONPATH\n        unset _OLD_VIRTUAL_PYTHONPATH\n    fi/' ${DNN_HOME}/py3-v4.3.0_tensorflow2.14/bin/activate


Note that the following commands via pip are meant to be executed with
the virtual environment activated.
If unsure, whether the correct env is activated and/or whether the correct
pip is being used, you can execute the following.

.. code-block:: bash

    # make sure that your virtualenv is activated
    # you can check this by executing
    which pip
    # It should point to:
    echo ${DNN_HOME}/py3-v4.3.0_tensorflow2.14/bin/pip


Install Prerequisites and DNN reco
==================================

We are now ready to install the necessary prerequisites and |dnn_reco|.

.. _install_tensorflow:

Install Tensorflow
------------------

Tensorflow may be installed via pip.
However, the prebuilt wheels are build against a specific version of
CUDA and cuDNN. This is irrelevant if tensorflow is meant to be run on
the CPU, but if run on the GPU, the correct CUDA and cuDNN versions must
be available.
The table located `here <https://www.tensorflow.org/install/source#gpu>`_
may be used to find which versions are necessary for which version of
tensorflow.
As of writing this, the CUDA versions 10.0 and 10.2 are available on NPX,
but without the necessary cuDNN version.
Therefore, we'll use a local installation of CUDA 10.1 with cuDNN 7.6
for tensorflow version 2.3.

.. code-block:: bash

    # install tensorflow 2.14 (CUDA 11 and cuDNN 8.7)
    pip install tensorflow==2.14 tensorflow_probability==0.22.1

To run the GPU version with the prebuilt wheel you will need to have a GPU and
CUDA 10.1 + cuDNN 7.6 available. You can use:

.. code-block:: bash

    export CUDA_HOME=/data/user/mhuennefeld/software/cuda/cuda-11.8
    export PATH=$PATH:${CUDA_HOME}/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64

which you can also add to the virtual environment activate script.

.. _install_dnn_reco:

Install Additional Packages
---------------------------

We'll install the other required packages now. Note that ``ic3-data``
needs to be compiled as it uses c++ in the backend.
For the compilation to succeed, the icecube headers need to be found.
The package searches in the ``$I3_SRC`` and ``$I3_BUILD`` directories to
find these.
Some virtual environments in cvmfs do not properly set these variables,
so we'll skip the icecube environment activation all-together and
simply manually set them prior to the installation of ``ic3-data``.

.. code-block:: bash

    # this will technically also be installed in ic3-data installation,
    # but the resulting warning/error might be confusing, so we'll just
    # install it first
    pip install pybind11

    # set I3_BUILD and I3_SRC to correct directories
    # these are needed for ic3-data to find the icecube headers
    export I3_BUILD=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/metaprojects/icetray/v1.12.0/
    export I3_SRC=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/metaprojects/icetray/v1.12.0/

    # install required repositories
    # specific versions may be checked out by adding tag, e.g. "@v2.1.0"
    # Note: if there are issues with installing directly from the
    # repositories, you can clone the repositories first and then install
    # them via pip install /path/to/repo [see instructions below for dnn_reco]
    pip install git+ssh://git@github.com/icecube/TFScripts
    pip install git+ssh://git@github.com/icecube/ic3-labels
    pip install git+ssh://git@github.com/icecube/ic3-data

The prebuilt binaries for python package h5py are
built against a specific hdf version, which usually differs
from what we have in cvmfs. Therefore we need to compile
it from source.

.. code-block:: bash

    # typically
    # if there is a HDF5 version mismatch we must install h5py from source
    # Use: 'h5cc -showconfig' to obtain hdf5 configuration and library version
    # use: 'ls -lah $(which h5cc)' to obtain path to hdf5 directory
    pip uninstall h5py
    HDF5_VERSION=1.14.0 HDF5_DIR=/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_7_x86_64/spack/opt/spack/linux-centos7-x86_64_v2/gcc-13.1.0/hdf5-1.14.0-4p2djysy6f7vful3egmycsguijjddkah pip install --no-binary=h5py h5py==3.11.0


Install DNN Reco
----------------

We are now ready to install |dnn_reco|.
To do so we must clone the repositories and then install them via pip.
The flag ``-e`` or ``--editable`` enables us to edit the source files and use
these changes without having to reinstall the package after each change.

.. code-block:: bash

    mkdir ${DNN_HOME}/repositories
    cd  ${DNN_HOME}/repositories

    # clone repository (or clone via ssh)
    git clone https://github.com/mhuen/dnn_reco.git

    # install package
    pip install -e  ${DNN_HOME}/repositories/dnn_reco

Install Processing Scripts (ic3-processing)
-------------------------------------------

We will use this package to create the training data.

.. code-block:: bash

    cd  ${DNN_HOME}/repositories

    # clone repository (or clone via ssh)
    git clone https://github.com/mhuen/ic3-processing.git

    # install package
    pip install -e  ${DNN_HOME}/repositories/ic3-processing


.. _verify_installation:

Verify Installation
-------------------

We are now done and can use our new environment.
Log in to a fresh shell and load the environment via:

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

To verify if our environment was installed correctly, we can
try to create a tensorflow session and to import |dnn_reco|.

.. code-block:: bash

    # the following commands should run without any errors
    python -c 'import tensorflow as tf; print(tf.__version__)'
    python -c 'import dnn_reco; import tfscripts; import ic3_labels; import ic3_data'


.. note::
    The prebuilt tensorflow binary is built to use avx2 and ssse3 instructions among others.
    These are not available on cobalts 1 through 4.
    Attempting to import tensorflow will lead to an "illegal instructions"
    error. Therefore, if running on the cobalts, simply choose one of the
    newer machines: cobalt >=5.
    On NPX, if running CPU jobs, you can request nodes with avx2 and ssse3
    support by adding: ``requirements = (TARGET.has_avx2) && (TARGET.has_ssse3)``. This is only necessary for CPU jobs. For GPU jobs,
    these requirements should not be set.
