.. IceCube DNN reconstruction

.. _code_scripts:

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
