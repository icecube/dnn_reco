.. IceCube DNN reconstruction

.. _bootcamp_train:
Train Model
***********

Now that we have installed |dnn_reco|, we can move on to training the
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
To train our model we are going to use the first 100 hdf5 files.
The transformation model will be built by using the same files.
Our validation and test data are files 100 to 109.
We can make these changes by hand or by executing the following command which
will replace the string '{insert_DNN_HOME}' with our environment variable
$DNN_HOME:

.. code-block:: bash

    sed -i -e 's,{insert_DNN_HOME},'"$DNN_HOME"',g' $CONFIG_DIR/getting_started.yaml


Create Data Transformation Model
================================

It is helpful to transform the input data as well as the labels.
A common transformation is to normalize the data to have a mean of zero and
a standard deviation of 1. Additionally, the logarithm should be applied to
features and labels that span over several decades.

The software framework includes a data tranformer class that takes care
of all of these transformations.
All that is necessary is to define the settings of the transformer class
in the configuration file.

We are going to highlight the following settings.

trafo_normalize_dom_data
trafo_normalize_label_data
trafo_log_dom_bins
trafo_log_label_bins

Once we are certain that we filled in the correct values, we can create
the data transformation model.
This step needs to process data as defined in the `trafo_data_file` key,
because the mean and standard deviation depend on the data.

.. code-block:: bash

    # cd into the dnn_reco directory
    cd $DNN_HOME/repositories/dnn_reco/dnn_reco

    # create the transformation Model
    python create_trafo_model.py $CONFIG_DIR/getting_started.yaml

Upon succesful completion this should print:

.. code-block:: php

    =======================================
    = Successfully saved trafo model to:  =
    =======================================
    '../data/trafo_models/dnn_reco_11883_tutorial_trafo_model.npy'




Train Neural Network Model
==========================

define label weights: EnergyVisible, all other zero

batch_size

Training loss and optimizer is defined in the `model_optimizer_dict` dictionary.
For now we will use a simple Mean Squared Error (MSE) for the prediction and
uncertainty estimate.

!Add formular for MSE of pred and sigma here!

It generally helps to start off with something robust such as MSE and a
learning rate of 0.001.
After this training step has converged (see monitor training progress),
we can reduce the learning rate and/or change the loss function to something
more robust towards outliers such as tukey loss (link to paper).
There are more loss functions defined in modules/loss.
You can also add a custom loss function by adding a file and a loss
function with the specified signature.
Afterwards you must adjust the keys `loss_file` and `loss_name` such that these
hold the file and function name of your newly created loss function.

To start training we run:

.. code-block:: bash

    # If on a system with multiple GPUs, we can define the GPU device that we
    # want to use by setting the CUDA_VISIBLE_DEVICES to the the device number
    # In this case, we will run on GPU 0.
    CUDA_VISIBLE_DEVICES=0 python train_model.py $CONFIG_DIR/getting_started.yaml

This will run indefinetely until we kill the process via `ctrl + c`.
The current model is saved every `save_frequency` (default value: 500) times.

Every call to train_model.py will keep track of the number of trainng iterations
as well as the configuration options used,
This means that you do not have to keep track yourself.
Additionally, the currently installed python packages and
the git revision is logged.



