.. IceCube DNN reconstruction

Configuration Options
*********************

General information about modularity, scripts/steps to run

Steering by one central config file

List of all config options (ideally in a way so that they are searchable!)
with references to corresponding code/module

config files may contain more options and keys than specified here.

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
    f

``label_weight_dict``:
    f

``label_particle_keys``:
    f

``label_update_weights``:
    f

``label_scale_tukey``:
    f

``label_zenith_key``:
    f

``label_azimuth_key``:
    f

``label_dir_x_key``:
    f

``label_dir_y_key``:
    f

``label_dir_z_key``:
    f

``label_add_dir_vec``:
    f

``label_position_at_rel_time``:
    f

``label_pid_keys``:
    f


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
    f

``model_restore_model``:
    f

``model_save_model``:
    f

``model_optimizer_dict``:
    f


NN Model Architecture
=====================

``model_file``:
    f

``model_name``:
    f

``model_is_training``:
    f

``conv_upper_DeepCore_settings``:
    f

``conv_lower_DeepCore_settings``:
    f

``conv_IC78_settings``:
    f

``fc_settings``:
    f

``fc_unc_settings``:
    f

....
