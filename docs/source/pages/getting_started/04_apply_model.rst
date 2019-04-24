.. IceCube DNN reconstruction

Apply Model
***********

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