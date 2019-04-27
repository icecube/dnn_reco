.. IceCube DNN reconstruction

IceCube DNN Bootcamp
********************

This tutorial is a simplifed version of the :ref:`Getting Started tutorial<getting_started>`
where we will focus on:


* :ref:`Training a neural network model<bootcamp_train>`
* :ref:`Monitoring the training progress<bootcamp_monitor>`
* :ref:`Exporting and applying the trained model to IceCube i3 files<bootcamp_apply>`
* :ref:`Visualizing the results<bootcamp_visualize>`

We will use NuMu NuGen files (first 1010 files of dataset 11883) to train a
deep convolutional neural network that will predict the energy of the muon as
it enters the convex hull around the IceCube detector.

figure of IceCube detecotr and muon entering it?
figure of Energy Resolution?

:ref:`Let's get started!<bootcamp_setup>`

.. toctree::
   :maxdepth: 2
   :hidden:

   bootcamp/00_bootcamp_setup
   bootcamp/01_bootcamp_train
   bootcamp/02_bootcamp_monitor
   bootcamp/03_bootcamp_apply
   bootcamp/04_bootcamp_visualize