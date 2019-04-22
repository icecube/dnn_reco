.. IceCube DNN reconstruction

Code Documentation
******************

General information about modularity, scripts/steps to run

Steering by one central config file


Scripts/Steps
====================

Create Data Transformation Model
--------------------------------

create_trafo_model.py

Train Model
-----------

train_model.py

exports logs: use tensorboard to view
(can also define any tensors to log in modules, these will be collected)

Export Model
------------

export_model.py

needs config to yaml file that was used for data creation.
The settings that are being used from that config are: ...



Modules
=======


Training Data Filter
--------------------

dnn_reco.data.filter


Load Training Labels
--------------------
dnn_reco.data.labels


Load Miscellaneous Data
-----------------------
dnn_reco.data.misc


Evaluate Training Progress
--------------------------
dnn_reco.evaluate

Loss Functions
--------------
dnn_reco.loss

Neural Network Models
---------------------
dnn_reco.models



Classes
=======

Additional documentation of classes here or better in the API section?

DataHandler
DataTrafo
NNModel
SetupManager

references to API and source code