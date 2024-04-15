.. IceCube DNN reconstruction

.. note::
    This tutorial was specifically created for the DNN Bootcamp at the 2019 Madison meeting. The virtual machine is no longer available.
    You can still run this tutorial, but will have to create new training
    data files and setup the environment elsewhere. Please refer to
    :ref:`Getting Started tutorial<getting_started>` and
    :ref:`Installation and Requirements<installation_and_requirements>`.

IceCube DNN Bootcamp
********************

This tutorial is a simplified version of the :ref:`Getting Started tutorial<getting_started>`
where we will focus on:


* :ref:`Training a neural network model<bootcamp_train>`
* :ref:`Monitoring the training progress<bootcamp_monitor>`
* :ref:`Exporting and applying the trained model to IceCube i3 files<bootcamp_apply>`
* :ref:`Visualizing the results<bootcamp_visualize>`

We will use NuMu NuGen files (first 1010 files of dataset 11883) to train a
deep convolutional neural network that will predict the energy of the muon as
it enters the convex hull around the IceCube detector.

:ref:`Let's get started!<bootcamp_setup>`

.. image:: figures/scatter_PrimaryMuonEnergyEntry_compare_BINS.png
    :width: 49 %
.. image:: figures/scatter_PrimaryMuonEnergyEntry_compare_DOMS.png
    :width: 49 %

.. image:: figures/scatter_PrimaryMuonEnergyEntry_compare_MuEx.png
    :width: 49 %
.. image:: figures/scatter_PrimaryMuonEnergyEntry_compare_DNN.png
    :width: 49 %

.. toctree::
   :maxdepth: 2
   :hidden:

   bootcamp/00_bootcamp_setup
   bootcamp/01_bootcamp_train
   bootcamp/02_bootcamp_monitor
   bootcamp/03_bootcamp_apply
   bootcamp/04_bootcamp_visualize
