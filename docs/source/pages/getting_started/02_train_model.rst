.. IceCube DNN reconstruction

.. _train_model:

Train Model
***********

Now that we have created our training data, we can move on to training the
neural network.
We will have to perform two steps: create a data transformation model and then
train the neural network.
The necessary scripts for these steps are located in the main directory of the
|dnn_reco| software package.
As previously mentioned, we need to define settings in our central
configuration file.
We will copy and edit a template configuration file for the purpose of this
tutorial.

.. code-block:: bash

    # Define the directory, where we will store the training configuration file
    export CONFIG_DIR=$DNN_HOME/configs/training

    # create the configuration directory
    mkdir --parents $CONFIG_DIR

    # copy config template over to our newly created directory
    cp $DNN_HOME/repositories/dnn_reco/configs/tutorial/getting_started.yaml $CONFIG_DIR/

We now need to edit the keys:
``training_data_file``, ``trafo_data_file``, ``validation_data_file``,
``test_data_file``,
so that they point to the paths of our training data.
To train our model we are going to use the first 1000 hdf5 files.
The transformation model will be built by using the same files.
Our validation and test data are files 1000 to 1009.
We can make these changes by hand or by executing the following command which
will replace the string '{insert_DNN_HOME}' with our environment variable
$DNN_HOME:

.. code-block:: bash

    sed -i -e 's,{insert_DNN_HOME},'"$DNN_HOME"',g' $CONFIG_DIR/getting_started.yaml

Keep in mind that you have to point to a different path, if you are using
the data in ``/data/user/mhuennefeld/DNN_reco/tutorials/training_data``, or
if your data is located elsewhere.


Create Data Transformation Model
================================

The training files that we will use in this tutorial were created with the
``pulse_summmary_clipped`` input format
(see :ref:`Create Training Data` for more info),
which means that we reduced the pulses of each DOM to the following
summary values:

    1. Total DOM charge
    2. Charge within 500ns of first pulse.
    3. Charge within 100ns of first pulse.
    4. Relative time of first pulse. (relative to total time offset)
    5. Charge weighted quantile with q = 0.2
    6. Charge weighted quantile with q = 0.5 (median)
    7. Relative time of last pulse. (relative to total time offset)
    8. Charge weighted mean pulse arrival time
    9. Charge weighted std of pulse arrival time

The input tensor which is fed into our network therefore has the shape
(-1, 10, 10, 60, 9) for the main IceCube array and (-1, 8, 60, 9) for the
DeepCore strings.

It is helpful to transform the input data as well as the labels.
A common transformation is to normalize the data to have a mean of zero and
a standard deviation of 1. Additionally, the logarithm should be applied to
features and labels that span over several decades.

The software framework includes a data transformer class that takes care
of all of these transformations.
All that is necessary is to define the settings of the transformer class
in the configuration file.
We are going to highlight a few options in the following:

``trafo_data_file``:
    Defines the files that will be used to compute the mean
    and standard deviation. Usually we will keep this the same as the files
    used for training the neural network (``training_data_file``).

``trafo_num_jobs``:
    This defines the number of CPU workers that will be used
    in parallel to load the data

``trafo_num_batches``:
    The number of batches of size ``batch_size`` to iterate over.
    We should make sure, that we compute the mean and standard deviation
    over enough events.

``trafo_model_path``:
    Path to which the transformation model will be saved.

``trafo_normalize_dom_data``/ ``trafo_normalize_label_data``/ ``trafo_normalize_misc_data``:
    If true, the input data per DOM, labels, and miscellanous data will be
    normalized to have a mean of zero and a standard deviation of one.

``trafo_log_dom_bins``:
    Defines whether or not the logarithm should be applied to the input
    data of each DOM.
    This can either be a bool in which case the logarithm will be applied
    to the whole input vector if set to True, or you can define a bool
    for each input feature.
    The provided configuration file applies the logarithm to the first three
    input features.
    You are free to change this as you wish.

``trafo_log_label_bins``:
    Defines whether or not to apply the logarithm to the labels.
    This can be a bool, a list of bool, or a dictionary in which you can
    define this for a specific label.
    The default value will be False, if a dictionary is passed, e.g. the
    logarithm will not be applied to any labels
    that are not contained in the dictionary.

Once we are certain that we filled in the correct values, we can create
the data transformation model.
This step needs to process data as defined in the ``trafo_data_file`` key,
because the mean and standard deviation depend on the data.

.. code-block:: bash

    # cd into the dnn_reco directory
    cd $DNN_HOME/repositories/dnn_reco/dnn_reco

    # create the transformation Model
    python create_trafo_model.py $CONFIG_DIR/getting_started.yaml

.. note::

    If you only created one training file you will not have enough training
    data to generate 100 batches of 32 events. As a result, the above will
    fail with a ``StopIteration`` error. You will either have to process a
    few more training data files, or lower the number of batches that you
    would like to use to create the transformation model. You can do this
    by setting the ``trafo_num_batches`` key in
    ``$CONFIG_DIR/getting_started.yaml``
    to a lower value such as 20.

Upon succesful completion this should print:

.. code-block:: php

    =======================================
    = Successfully saved trafo model to:  =
    =======================================
    '../data/trafo_models/dnn_reco_11883_tutorial_trafo_model.npy'




Train Neural Network Model
==========================

The network architecture that will be used in this tutorial is the
``general_model_IC86`` architecture which is defined in the module
``dnn_reco.modules.models.general_IC86_models``.
This is a deep convolutional neural network with 8 convolutional layers for
the upper and 14 convolutional layers for the lower DeepCore part.
20 convolutional layers are performed over the main IceCube array.
The three output tensors of each of these convolutional blocks are then
concatenated and fed into a fully connected sub network of 3 layers.
Additionally, we define a second fully connected sub network of 3 layers, that
is used to predict the uncertainties on each of the reconstructed quantities.
You may change the architecture by modifying the settings below
::

    #----------------------
    # NN Model Architecture
    #----------------------

in the configuration file.
You can also define your own neural network architecture, by changing the keys
``model_file`` and ``model_name`` to point to the correct file and function.

During training, we can provide weights to each of the labels.
That way we can force the training to focus on the labels that we care about.
In this tutorial we will focus on reconstructing the visible energy in the
detector (``EnergyVisible``), while also providing a smaller weight to the primary energy of the neutrino (``PrimaryEnergy``).
For throughgoing muons, ``EnergyVisible`` is the energy of the muon as it enters the
detector.
For starting muons, this is the sum of the deposited energy by the cascade
plus the energy of the outgoing muon.
There are several ways how we can define the weights for all labels.
The key ``label_weight_initialization``
defines the default weight for the labels.
We can specify the weight of certain variables with the ``label_weight_dict``
key.

.. note::
    If certain variables are included in the logarithm/exponential transformation of the data transformer, but not trained, e.g. weights set to zero, then it can happen that the values for these drift out of bound leading to NaNs. If this happens, you can also set the weights of the affected variables to very small positive weights such as 0.00001

Other important settings for the training procedure are the ``batch_size``
and the choice of loss functions and minimizers which are defined
in the ``model_optimizer_dict``.
For now we will use a simple Mean Squared Error (MSE) for the prediction and
uncertainty estimate.
The structure of the setting ``model_optimizer_dict`` is a bit complicated,
but it is very powerful.
We can define as many optimizers with as many loss funtions as we like.
A few basic loss functions are already implemented in
``dnn_reco.modules.loss``.
You are free to add your custom loss functions by adding a file/function in
the ``dnn_reco.modules.loss`` module and by then adjusting the ``loss_file``
and ``loss_name`` keys.

It generally helps to start off with something robust such as MSE and a
learning rate of 0.001.
After this training step has converged
(see :ref:`Monitor Progress<bootcamp_monitor>`),
we can reduce the learning rate and/or change the loss function to something
more robust towards outliers such as tukey loss
(https://arxiv.org/abs/1505.06606).
To start training we run:

.. code-block:: bash

    # If on a system with multiple GPUs, we can define the GPU device that we
    # want to use by setting the CUDA_VISIBLE_DEVICES to the the device number
    # In this case, we will run on GPU 0.
    CUDA_VISIBLE_DEVICES=0 python train_model.py $CONFIG_DIR/getting_started.yaml

.. note::
    Running this on one of the cobalts should work,
    but will be extremely slow.
    Training on a GPU is highly recommended.
    NPX isn't suited well for training, since the job ideally needs
    1 GPU in addition to ~10 CPUs. However, this will be hard to obtain
    on NPX. Reducing the number of requested CPUs may help.
    In this case, the number of worker jobs for the data input pipeline should be reduced by setting the ``num_jobs`` key in the configuration.
    If possible, it is recommended to run this on other resources,
    if available.

This will run for ``num_training_iterations`` many iterations or
until we kill the process via ``ctrl + c``.
The current model is saved every ``save_frequency`` (default value: 500)
iterations, so you may abort and restart at any time.

Every call of ``train_model.py`` will keep track of the number of
training iterations as well as the configuration options used.
This means that you do not have to keep track yourself.
Moreover, the currently installed python packages and
the git revision is logged.
This information will be exported together with the model, to ensure
reproducability.
The keys ``model_checkpoint_path`` and ``log_path`` define where the model
checkpoints and the tensorboard log files will be saved to.
The ``model_checkpoint_path`` also defines the path from which the weights of
the neural network will be recovered from in a subsequent call to ``train_model.py``
if ``model_restore_model`` is set to True.
If you wish to start from scratch, you can set ``model_restore_model``
to False or manually delete the checkpoint and log directory of your model.
In order not to get models mixed up, you should make sure that each of your
trained models has a unique name as defined in the key ``unique_name``.
The easiest way to achieve this is to have a separate configuration file for
each of your models.

.. note::
    Many more configuration options are available which are documented in
    :ref:`Configuration Options`.
    The software framework is meant to provide high flexibility.
    Therefore you can easily swap out modules and create custom ones.
    We have briefly touched the option to create your own neural network
    architecture here as well as the option to add custom loss functions.
    More information on the exchangable modules is provided in
    :ref:`Code Documentation`.

