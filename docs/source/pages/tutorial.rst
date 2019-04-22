.. IceCube DNN reconstruction

Getting Started
***************

In this tutorial we will learn how to:

* :ref:`Create training data from IceCube i3 files<Create Training Data>`
* :ref:`Train our neural network model<Train Model>`
* :ref:`Monitor the training progress<Monitor Progress>`
* :ref:`Export and apply our trained model to IceCube i3 files<Apply Model>`

We will use NuMu NuGen files (first 100 files of dataset 11883) to train a
deep convolutional neural network that will predict the energy of the muon as
it enters the convex hull around the IceCube detector.

figure of IceCube detecotr and muon entering it?
figure of Energy Resolution?

:ref:`Let's get started!<General Remarks>`


General Remarks
===============

|dnn_reco| is intended to be a highly modularized software framework for neural
network applications in IceCube. As such, there are many modules that you can
easily modify or switch out, depending on your needs.
A central configuration file defines which modules and which settings to use.
This configuration file is necessary for each of the individual steps.
There are many options that can be configured.
Here we will only briefly touch a few basic concepts.
An extensive documentation of the configuration file options is provided in
:ref:`Configuration Options<Configuration Options>`.

Create Training Data
====================

.. note::
    :ref:`If you already have training data files available, then you can skip
    to the next section<Train Model>`


Train Model
===========

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

    # Define the directory, where we will store the configuration file
    export CONFIG_DIR=$DNN_HOME/configs

    # create the configuration directory
    mkdir --parents $CONFIG_DIR

    # copy config template over to our newly created directory
    cp $DNN_HOME/repositories/dnn_reco/configs/tutorial/getting_started.yaml $CONFIG_DIR/

We now need to edit the keys:
`training_data_file`, `trafo_data_file`, 'validation_data_file', `test_data_file`,
so that they point to the paths of our training data.
To train our model we are going to use the first 100 hdf5 files.
The transformation model will be built by using the same files.
Our validation and test data are files 100 to 109.
We can make these changes by hand or by executing the following commands:

.. code-block:: bash

    provide sed command


Create Data Transformation Model
--------------------------------

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

.. code-block:: bash

    output of create trafo model py



Train Neural Network Model
--------------------------

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






Monitor Progress
================

We can verify the GPU utilization by the training procedure with
nvidia-smi

To keep track, we can do something like:

.. code-block:: bash

    watch -n 3 nvidia-smi

All labels as well as the losses are logged with tensorboard.
If you would like to add more variables to log,
just add these with the standard functions tf.log.xxxx. in your custom modules.
Variables that need to be logged are collected via tf.get_log_vars....

We can then use tensorboard to render these logs.

.. conde-block:: bash

    # If we run tensorboard remotely we must provide a port and make sure
    # to forward this port in the ssh connection
    tensorboard --logdir= --port 7475

If the port forwading is correctly set up, you can now point your browser to
(address).

More info on tensorboard is provided here (link to tensorboard).

figure of Tensorboard training curve


Apply Model
===========

The |dnn_reco| software package provides a method to export your trained
models which can be applied to i3 files via the provided I3TraySegment.

To export our trained model run:

.. code-block:: bash

    python export_model.py $CONFIG_DIR/getting_started.yaml -s $DNN_HOME/data/path/to/yaml -o $DNN_HOME/exported_models/getting_started_model

This should complete with the message:

.. code-block:: bash

    print output

To apply our new model to i3 files we can use the I3TraySegment
dnn_reco.ic3.segments.xx

As we previously did for the creation of the training data, we will use
the processing framework from link to svn sandbox.

Modify the configuration file (link) to use the correct model
add: model_dir, model_names
and set GPU to 0.=?

Then we create the job files

and run them
(no need to run dagman for just one file, we can simply execute the )