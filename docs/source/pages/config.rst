.. IceCube DNN reconstruction

Configuration Options
*********************

The |dnn_reco|-framework uses a central configuration file to define
the settings for the various steps.
Configuration options used by the default models
are described in the following.
Configuration files may contain more options and keys than specified here.
Any custom model may use and define options.

General Settings
================

``unique_name``:
    This is the name of the neural network model. If you are using the default
    checkpoint and log paths, this name is used to correctly identify which
    model you want to load/train/save/export.

``training_data_file``:
    A list of glob file patterns that defines which files will be used for the
    training set.

``trafo_data_file``:
    A list of glob file patterns that defines which files will be used to
    create the data transformation model. Usually it's fine to use the same
    files as used for training.

``validation_data_file``:
    A list of glob file patterns that defines which files will be used for the
    validation set.

``test_data_file``:
    If the data handler object is not instanciated with a config, it will load
    one of these files to obtain meta data, such as the labels and misc data
    names.

``tf_random_seed``:
    Random seed for tensorflow weight initialization.
    Note: due to the parallel reading and processing of the training data,
    this will not guarantee the exact same results for multiple trained models
    with the same configuration files.

``float_precision``:
    Float precision to be used.

``num_jobs``:
    Number of workers that are used in parallel to load files and populate the
    'data_batch_queue'.
    See :ref:`Monitor Progress<monitor_progress>`
    for some additional information on the data input pipeline.

``file_capacity``:
    The capacity of the 'data_batch_queue'. The workers can only enqueue
    data to the 'data_batch_queue' if the capacity has not been reached.
    If the 'data_batch_queue' has reached its capacity, the workers will halt
    until elements get dequeued.
    See :ref:`Monitor Progress<monitor_progress>`
    for some additional information on the data input pipeline.

``batch_capacity``:
    The capacity of the 'final_batch_queue' which holds the batches of size
    ``batch_size``. The batches dequeued from this queue are fed directly
    into the neural netwok input tensors.
    See :ref:`Monitor Progress<monitor_progress>`
    for some additional information on the data input pipeline.

``num_add_files``:
    This defines how many files should be aggregated before randomly sampling
    batches from the loaded events.
    Ideally, one would want to load as many files as possible to make sure that
    the batches get mixed together well with events from different training
    files. However, the training datasets are typically so large, that we
    are only able to load a small subset of events at a time.
    See :ref:`Monitor Progress<monitor_progress>`
    for some additional information on the data input pipeline.

``num_repetitions``:
    This defines how many epocs to run over the loaded data.
    If ``num_add_files`` is high enough, e.g. if we load enough events to
    sample from, we can reuse these events before loading the next files.
    This is much more efficient that only using the loaded events once.
    However, care must be taken since a too high number of repetitions in
    connection with a small number of loaded events can result in overfitting
    to those events.
    See :ref:`Monitor Progress<monitor_progress>`
    for some additional information on the data input pipeline.

``batch_size``:
    The number of events in a batch.

``log_path``:
    The directory to which the log files will be written to.
    You can use other keys as variables in the string:
    ``log_path = log_path.format(**config)`` will be applied.


Data Handler
============

``data_handler_bin_values_name``:
    The name of the key which holds the data bin values.
    Default: `dnn_data_bin_values`

``data_handler_bin_indices_name``:
    The name of the key which holds the data bin indices.
    Default: `dnn_data_bin_indices`

``data_handler_time_offset_name``:
    The name of the key which holds the global time offset.
    Time variables are usually computed relative to this time offset.
    Default: `dnn_data_global_time_offset`

``data_handler_num_bins``:
    Size of per DOM input feature vector.

``data_handler_label_file``:
    How and which labels will be loaded from the hdf5 files is defined by
    a function in the ``dnn_reco.modules.data.labels`` directory.
    Which function to use is defined by the ``data_handler_label_file``
    and ``data_handler_label_name`` keys.
    This key specifies the file to use in that directory.
    Default: `default_labels`

``data_handler_label_name``:
    How and which labels will be loaded from the hdf5 files is defined by
    a function in the ``dnn_reco.modules.data.labels`` directory.
    Which function to use is defined by the ``data_handler_label_file``
    and ``data_handler_label_name`` keys.
    This key specifies the function name.
    Default: `simple_label_loader`

``data_handler_misc_file``:
    How and which misc data will be loaded from the hdf5 files is defined by
    a function in the ``dnn_reco.modules.data.misc`` directory.
    Which function to use is defined by the ``data_handler_misc_file``
    and ``data_handler_misc_name`` keys.
    This key specifies the file to use in that directory.
    Default: `default_misc`

``data_handler_misc_name``:
    How and which misc data will be loaded from the hdf5 files is defined by
    a function in the ``dnn_reco.modules.data.misc`` directory.
    Which function to use is defined by the ``data_handler_misc_file``
    and ``data_handler_misc_name`` keys.
    This key specifies the function name.
    Default: `general_misc_loader`

``data_handler_filter_file``:
    Sometime it can be helpful to choose a subselection of events to train on.
    The ``data_handler_filter_file`` and ``data_handler_filter_name`` keys
    define the function to use to filter the events.
    This key specifies the file to use in the
    ``dnn_reco.modules.data.filter`` directory.
    Default: `default_filter`

``data_handler_filter_name``:
    Sometime it can be helpful to choose a subselection of events to train on.
    The ``data_handler_filter_file`` and ``data_handler_filter_name`` keys
    define the function to use to filter the events.
    This key specifies the function name.
    Default: `general_filter`

``data_handler_label_key``:
    This is a key used by the default label loader.
    It specifies the name of the key in the hdf5 file that holds the labels.

``data_handler_relative_time_keys``:
    The time input DOM data is usually calculated relative to the time defined
    in `data_handler_time_offset_name`.
    If labels contain global times, it is recommendet to transform these to
    relative times.
    The labels provided here as a list will be transformed to relative time.

``data_handler_relative_time_key_pattern``:
    The time input DOM data is usually calculated relative to the time defined
    in `data_handler_time_offset_name`.
    If labels contain global times, it is recommendet to transform these to
    relative times.
    You can provide a pattern here.
    Labels will be transformed to relative times if
    ``'data_handler_relative_time_key_pattern' is in label_name.lower()``
    is true.

Misc Settings
=============

``misc_load_dict``:
    This is a key of the general_misc_loader misc data loader.
    The general_misc_loader will load the keys defined in the
    ``misc_load_dict``
    from the training files. The pattern is: 'hdf key': 'column'
    These values will then be added to the misc values under the name:
    'hdf key'_'column'


Filter Settings
===============

The general_filter will filter events according to the key value pairs
defined in the dicts ``filter_equal``, ``filter_greater_than``,
``filter_less_than``.
The keys used defined in the dicts must exist in the loaded misc data names.

``filter_equal``:
    For events to pass filter, the following must be True: misc[key] == value

``filter_greater_than``:
    For events to pass filter, the following must be True: misc[key] > value

``filter_less_than``:
    For events to pass filter, the following must be True: misc[key] < value

Label Settings
==============

``label_weight_initialization``:
    A weighting can be applied to the labels to focus on certain labels
    in the training process.
    The loss is computed as a vector where each entry corresponds to the loss
    for that given label.
    This vector is then multiplied by the weights for each label.
    The ``label_weight_initialization``-key defines the default weight for
    the labels.

``label_weight_dict``:
    A weighting can be applied to the labels to focus on certain labels
    in the training process.
    The loss is computed as a vector where each entry corresponds to the loss
    for that given label.
    This vector is then multiplied by the weights for each label.
    The default weight for all labels is set to the specified value in
    the ``label_weight_initialization``-key.
    The ``label_weight_dict`` is a dictionary where you can define the weights
    for certain labels.
    The syntax is: {label_name: label_weight}.

``label_particle_keys``:
    This defines which labels will be used to populate the I3Particle when
    the model is being applied to new events.
    Optional keys include:
    `energy`, `time`, `length`, `dir_x`, `dir_y`, `dir_z`,
    `pos_x`, `pos_y`, `pos_z`.
    If the keys are not defined in the ``label_particle_keys`` dictionary,
    the I3Particle will be populated with NaNs instead.
    The I3Particle will be written to: {output_name}_I3Particle.
    Additionally, a key with all labels with weights greater than zero will
    be saved to: {output_name}.


``label_update_weights``:
    If set to True,
    this will update the label weights during the training process
    to ensure that labels are learnt according to their difficulty.
    The weights of each label will be scaled by the inverse RMSE
    of that label, which is calculated with a moving average
    over the past training iterations.

``label_scale_tukey``:
    If set to True, the median absolute residuals that is used for
    the tukey loss will be updated via a moving average over the last
    training iterations.
    This key is only relevant, if the chosen loss function is tukey.

``label_zenith_key``:
    Specifies the name of the zenith direction label if it exists.

``label_azimuth_key``:
    Specifies the name of the azimuth direction label if it exists.

``label_dir_x_key``:
    Specifies the name of the direction vector x-component label if it exists.

``label_dir_y_key``:
    Specifies the name of the direction vector y-component label if it exists.

``label_dir_z_key``:
    Specifies the name of the direction vector z-component label if it exists.

``label_add_dir_vec``:
    This key is used inside the ``simple_label_loader`` function.
    If True, direction vector components will be calculated on the fly
    from the given azimuth and zenith labels as specified in the
    ``label_azimuth_key`` and ``label_zenith_key``, respectively.
    These will be added as labels under the names:
    `direction_x`, `direction_y`, `direction_z`.

``label_position_at_rel_time``:
    This key is used inside the ``simple_label_loader`` function.
    If True, the position at a certain time (relative to time offset of event)
    based on the vertex and particle direction will be calculated on the fly
    and added as labels under the names:
    `rel_pos_x`, `rel_pos_y`, `rel_pos_z`.
    The position is given as: vertex + dir * delta_t * c.

``label_pid_keys``:
    This key is used by the default network architectures.
    It defines a list of binary classification labels.
    The labels specified in this list will be forced to the value range (0, 1).


General Training Settings
=========================

``num_training_iterations``:
    Number of training iterations to run.

``validation_frequency``:
    Defines after how many training iterations to run evaluation on
    validation set.

``save_frequency``:
    Defines the frequency at which the model should be saved.
    The frequency is given in number of training iterations.

``keep_probability_list``:
    Keep rates for the dropout layers, if they are used within the specified
    neural network architecture.
    You may specify an arbitrary long list here.

``evaluation_file``:
    A custom evaluation method can be defined.
    This key defines which file to use in the ``dnn_reco.modules.evaluation``
    directory.
    Default: 'default_evaluation'

``evaluation_name``:
    A custom evaluation method can be defined.
    This key defines the name of the evaluation method to be run.
    Default: 'eval_direction'


Trafo Settings
==============

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

``trafo_log_label_bins``/ ``trafo_log_misc_bins``:
    Defines whether or not to apply the logarithm to the labels/ misc data.
    This can be a bool, a list of bool, or a dictionary in which you can
    define this for a specific label / misc data.
    The default value will be False, if a dictionary is passed, e.g. the
    logarithm will not be applied to any labels / misc data
    that are not contained in the dictionary.

``trafo_treat_doms_equally``:
    If true, all DOMs will be treated equally, e.g. the mean and std deviation
    of the input data will be computed the same over all DOMs.

``trafo_norm_constant``:
    A small constant to stabilize the normalization.

NN Model Training
=================

``model_checkpoint_path``:
    The path to the checkpoint directory.
    This is the directory to which the model will be saved and also where
    it will be loaded from.
    You can use other keys as variables in the string:
    ``model_checkpoint_path = model_checkpoint_path.format(**config)``
    will be applied.

``model_restore_model``:
    If set to True, the model will be loaded if a previous model was saved
    to the directory specified by the ``model_checkpoint_path``-key.
    If set to False, the model will be re-iniatlized, e.g. training will
    begin from scratch.

``model_save_model``:
    If set to True, the model will be saved after every ``save_frequency``
    training steps.

``model_optimizer_dict``:
    This dictionary defines the different loss functions and optimizer settings
    that will be applied during training.
    Define a dictionary of dictionaries of optimizers here.
    Each optimizer has to define the following fields:

    ``optimizer``:
        name of tf.train.Optimizer, e.g. 'AdamOptimizer'

    ``optimizer_settings``:
        a dictionary of settings for the optimizer

    ``vars``:
        str or list of str specifying the variables the optimizer is
        adujusting. E.g. ['unc', 'pred'] to optimize weights of the
        main prediction network and the uncertainty subnetwork.
    ``loss_file``:
        str or list of str, defines file of loss function
    ``loss_name``:
        str or list of str, defines name of loss function
        If loss_file and loss_name are lists, they must have the same
        length. In this case, a sum of each loss will be performed
    ``l1_regularization``:
        Regularization strength (lambda) for L1-Regularization
    ``l2_regularization``:
        Regularization strength (lambda) for L2-Regularization

    This structure might seem a bit confusing, but it enables the use of
    different tensorflow optimizer operations, which can each apply to
    different weights of the network and wrt different loss functions.


NN Model Architecture
=====================

``model_file``:
    The network architecture that will be used is defined by the
    ``model_file`` and ``model_name`` keys.
    The ``model_file`` defines the file that will be used in the
    ``dnn_reco.modules.models`` dicetory.
    Default: `general_IC86_models`

``model_name``:
    The network architecture that will be used is defined by the
    ``model_file`` and ``model_name`` keys.
    The ``model_name`` defines the function that will be used to
    create and buld the model.
    Default: `general_model_IC86`

``model_is_training``:
    A bool indicating whether the network is in training mode.
    This is needed for certain layers such as batch normalisation.
    Default: `True`

``conv_upper_DeepCore_settings``:
    This key is used by the default architectures and defines the
    convolutional layers over the upper DeepCore array.

``conv_lower_DeepCore_settings``:
    This key is used by the default architectures and defines the
    convolutional layers over the lower DeepCore array.

``conv_IC78_settings``:
    This key is used by the default architectures and defines the
    convolutional layers over the main IceCube array.

``fc_settings``:
    This key is used by the default architectures and defines the
    fully connected layers after the results of the different
    conovlutional branches are flattened and combined.

``fc_unc_settings``:
    This key is used by the default architectures and defines the
    fully connected layers used for the uncertainty estimate.
    The input of these layers are the combined and flattened results
    of the different conovlutional branches.

....
