.. IceCube DNN reconstruction

Monitor Progress
****************

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

.. code-block:: bash

    # If we run tensorboard remotely we must provide a port and make sure
    # to forward this port in the ssh connection
    tensorboard --logdir= --port 7475

If the port forwading is correctly set up, you can now point your browser to
(address).

More info on tensorboard is provided here (link to tensorboard).

figure of Tensorboard training curve
