.. IceCube DNN reconstruction

.. _models:

Apply Exported Models
*********************

Applying an exported model can be achieved with the
``ApplyDNNRecos`` I3TraySegment.
As an example, the models ``example_model1``
and ``example_model2`` can be applied via:

.. code-block:: bash

    from dnn_reco.ic3.segments import ApplyDNNRecos

    tray.AddSegment(
        ApplyDNNRecos,
        'ApplyDNNRecos',
        pulse_key='InIceDSTPulses',
        dom_exclusions=['SaturationWindows', 'BadDomsList','CalibrationErrata'],
        partial_exclusion=True,
        model_names=['example_model1', 'example_model2'],
    )

Models which use the same input settings may be grouped in a single tray
segment via the ``model_names`` parameter which accepts a list of model names.
These models will then share the same input pipeline. As a result, the
preprocessing only needs to be performed once.
On a GPU, this is the most time consuming step.
If run on a CPU, the number of CPUs to run the model on may be passed
via ``num_cpus``.
Especially if on a GPU, it is advisable to run the |dnn_reco| on batches of
events at a time. This can be controlled via ``batch_size`` which defines the
number of events to reconstruct simulateneously.
The best settings depend on the hardware setup.
A good staring point could be 32 or 64.

.. The models described in the following are located in
.. ``/data/user/mhuennefeld/DNN_reco/models/exported_models/``.
.. In the future these might also be made available in the user_cvmfs space.
.. There are also a number of models used for the ``DNNCascade`` selection.
.. These are described `here <https://wiki.icecube.wisc.edu/index.php/Cascade_Neutrino_Source_Dataset/Machine_Learning_Models#DNN_reco_Models>`_
.. and available at ``/data/ana/PointSource/DNNCascade/utils/exported_models/<version>/dnn_reco/``


.. **List of trained models:**

.. * :ref:`mese_v2__all_gl_both2: MESC Cascades (SpiceLea 30cm Holeice)<models_mese_v2__all_gl_both2>`
.. * :ref:`dnn_reco_paper_hese__m4_before_GL_unc_sys: HESE Cascades (Spice3.2 + Spice3.2 systematics)<models_dnn_reco_paper_hese__m4_before_GL_unc_sys>`
.. * :ref:`dnn_reco_paper_hese__m5_after_GL_unc_sys: HESE Cascades (Spice3.2 + SpiceLea & Spice3.2 systematics): <models_dnn_reco_paper_hese__m5_after_GL_unc_sys>`



.. .. _models_mese_v2__all_gl_both2:

.. mese_v2__all_gl_both2
.. ---------------------

.. This model is used for the |dnn_reco| paper.
.. It is a model focused on the cascade directional reconstruction for MESC.

.. ``IceModel``:
..     Baseline is SpiceLea 30cm Holeice. Also trained on SpiceLea systematics in earlier training steps. The model is fine-tuned to the baseline for the prediction as well as uncertainty estimates. This means that coverage should hold on the baseline dataset, but will under-cover for systematic
..     sets.

.. ``Pulses``:
..     InIceDSTPulses (or equivalent)

.. ``DOM Exclusions``:
..     ['BrightDOMs','SaturationWindows', 'BadDomsList','CalibrationErrata']

.. ``Partial Exclusion``:
..     True

.. ``Training Data``:
..     First half of each dataset in ``/data/ana/Cscd/StartingEvents/NuGen/*/*/IC86_2013*``.





.. .. _models_dnn_reco_paper_hese__m4_before_GL_unc_sys:

.. dnn_reco_paper_hese__m4_before_GL_unc_sys
.. -----------------------------------------

.. This is a model focused on the cascade directional reconstruction for HESE.

.. ``IceModel``:
..     The model is trained on Spice3.2 with all of the available systematic
..     datasets.

.. ``Pulses``:
..     InIceDSTPulses (or equivalent)

.. ``DOM Exclusions``:
..     ['BrightDOMs','SaturationWindows', 'BadDomsList','CalibrationErrata']

.. ``Partial Exclusion``:
..     True

.. ``Training Data``:
..     First half of each dataset in ``/data/ana/Cscd/StartingEvents/NuGen/*/*/IC86_flasher*``.



.. .. _models_dnn_reco_paper_hese__m5_after_GL_unc_sys:

.. dnn_reco_paper_hese__m5_after_GL_unc_sys
.. ----------------------------------------

.. This is a model focused on the cascade directional reconstruction for HESE.
.. It uses ``dnn_reco_paper_hese__m4_before_GL_unc_sys`` and adds some additional
.. training steps broaden uncertainty estimates.


.. ``IceModel``:
..     The model is trained on Spice3.2 with all of the available systematic datasets for the prediction. Further training steps for the uncertainty estimate were performed on Spice3.2 + SpiceLea systematics. The uncertainty estimates are therefore broadened to include additional systemtatic uncertainties.

.. ``Pulses``:
..     InIceDSTPulses (or equivalent)

.. ``DOM Exclusions``:
..     ['BrightDOMs','SaturationWindows', 'BadDomsList','CalibrationErrata']

.. ``Partial Exclusion``:
..     True

.. ``Training Data``:
..     First half of each dataset in ``/data/ana/Cscd/StartingEvents/NuGen/*/*/IC86_flasher*``.
