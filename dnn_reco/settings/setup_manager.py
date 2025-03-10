import os
import numpy as np
import tensorflow as tf

import dnn_reco
from dnn_reco import misc
from dnn_reco.settings.yaml import yaml_loader
from dnn_reco.settings import version_control

# suppress natural naming warnings
import warnings
from tables import NaturalNameWarning

warnings.filterwarnings("ignore", category=NaturalNameWarning)


class SetupManager:
    """Setup Manager for DNN reco project

    Handles loading and merging of yaml config files, sets up directories.

    Config Keys
    -----------

    Automatically created settings

        config_name : str
            An automatically created config name.
            Base names of yaml config files are concatenated with '__'.
            This can be used to create subdirectories for logging and
            checkpoints.

    General settings

        test_data_file : str
            Path to test data file.
        training_data_file : str
            Path to training data file.
        validation_data_file : str
            Path to validation data file.
        num_jobs : int
            Relevant for IceCubeDataHandler
            Defines number of workers (background processes) which load data.
        file_capacity : int
            Relevant for IceCubeDataHandler
            Defines size of queue which stores events of loaded files.
        batch_capacity
            Relevant for IceCubeDataHandler
            Defines size of final batch queue. This queue is directly used
            by the data iteratotor to yield batches of data.
        num_add_files
            Relevant for IceCubeDataHandler
            Defines number of files from which the batches are generated.
        num_repetitions
            Relevant for IceCubeDataHandler
            Defines how many times events of the loaded files are used before
            new files are loaded.
        DOM_init_values: float or array-like
            The x_ic78 and deepcore array will be initialized with these
            values via:
            np.zeros_like(x_ic78) * np.array(init_values)
        batch_size : int
            The batch size which will be used by the data generators.
        log_path : str
            Path to the directory in which the logs are to be stored.

    General Training settings

        num_training_iterations : int
            Number of training iterations to perform.
        validation_frequency : int
            Defines the interval at which validation results should be
            calculated and logged.
        save_frequency : int
            Defines the interval at which the model parameters will be stored
            to file.

    Trafo settings

        trafo_model_path : str
            Path to trafo model file.
        trafo_normalize_dom_data : bool
            If true, the DOM input data will be normalized to have a
            mean of 0 and a variance of 1.
        trafo_normalize_label_data : bool
            If true, the label data will be normalized to have a
            mean of 0 and a variance of 1.
        trafo_normalize_misc_data : bool
            If true, the misc data will be normalized to have a
            mean of 0 and a variance of 1.
        trafo_log_dom_bins : bool, list of bool
            The natural logarithm is to the DOM input data.
            If a list is given, the length of the list must match the number of
            bins 'num_bins'. The logarithm is applied to bin i if the ith entry
            of the log_bins list is True.
        trafo_log_label_bins : dict[key: bool]
            The natural logarithm is applied to the specified labels
        trafo_log_misc_bins : bool, list of bool
            The natural logarithm is applied to the misc data.
            If a list is given, the length of the list must match the number of
            misc keys. The logarithm is applied to key i if the ith entry
            of the log_bins list is True.
        trafo_treat_doms_equally : bool
            All DOMs are treated equally, e.g. the mean and variance is
            calculated over all DOMs and not individually.
        trafo_norm_constant : float
            A small constant that is added to the denominator during
            normalization to ensure finite values.

    NN Model Architecture

        model_class : str
            Name of class that is used to define the model.
        model_kwargs : dict
            A dictionary of arguments that are passed on to the model class.

    NN Model Training

        model_checkpoint_path : str
            Path to directory in which the model checkpoints are
            stored.
        model_restore_model : bool
            If true, model parameters are restored from file.
            If false, model parameters are randomly initialized.
        model_save_model : bool
            If true, the model parameters are saved to file.
                Note: This will overwrite possible existing files.
        model_optimizer_dict : dict
            Defines the loss functions and optimizers that are used for
            training.

    Attributes
    ----------
    config : dictionary
        Dictionary with defined settings.
    shared_objects : dictionary
        Dictionary with additional objects that are available in all modules.
        Keys:
            'data_transformer' : DataTransformer object used to transform data.
            'keep_prob_list' : Tensorflow placeholders for keep probabilities
                                for dropout layers
    """

    # define default config
    _default_config = {
        "float_precision": "float32",
        "DOM_init_values": 0.0,
        "trafo_norm_constant": 1e-6,
        "data_handler_nan_fill_value": None,
        "data_handler_max_events_per_file": None,
        "data_handler_max_file_chunk_size": None,
    }

    def __init__(self, config_files, num_threads=None):
        """Initializes the DNN reco Setup Manager

        Loads and merges yaml config files, sets up necessary directories.

        Parameters
        ----------
        config_files : list of strings
            List of yaml config files.
        program_options : str
            A string defining the program options.
        num_threads : int, optional
            Number of threads to use for tensorflow operations.
            If not given, the number of threads is not limited.
        """
        self._config_files = config_files

        # gather objects that will be passed and shared to all modules
        # (generator, discriminator and loss modules)
        self.shared_objects = {}

        # load and combine configs
        self._setup_config()

        # set up tensorflow
        # limit GPU usage
        gpu_devices = tf.config.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        # limit number of CPU threads
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)

    def _setup_config(self):
        """Loads and merges config

        Raises
        ------
        ValueError
            If no config files are given.
            If a setting is defined in multiplie config files.
        """
        if len(self._config_files) == 0:
            raise ValueError("You must specify at least one config file!")

        # ----------------------------------
        # load config
        # ----------------------------------
        new_config = {}
        config_name = None
        for config_file in self._config_files:

            # append yaml file to config_name
            file_base_name = os.path.basename(config_file).replace(".yaml", "")
            if config_name is None:
                config_name = file_base_name
            else:
                config_name += "__" + file_base_name

            with open(config_file, "r") as stream:
                config_update = yaml_loader.load(stream)

            duplicates = set(new_config.keys()).intersection(
                set(config_update.keys())
            )

            # make sure no options are defined multiple times
            if duplicates:
                raise ValueError(
                    "Keys are defined multiple times {!r}".format(duplicates)
                )
            # update config
            new_config.update(config_update)
        config = dict(self._default_config)
        config.update(new_config)

        # define numpy and tensorflow float precision
        config["tf_float_precision"] = getattr(tf, config["float_precision"])
        config["np_float_precision"] = getattr(np, config["float_precision"])

        # check for version mismatch of dnn-reco (only major version)
        if "dnn_reco_version" in config:
            restore_version = config["dnn_reco_version"]
            restore_major = int(restore_version.split(".")[0])
            if restore_version != dnn_reco.__version__:
                misc.print_warning(
                    f"Resoring model with version {restore_version}. "
                    f"Version of dnn-reco is {dnn_reco.__version__}."
                )
            if dnn_reco.__version_major__ != restore_major:
                raise ValueError(
                    "Mismatch of major version number of dnn-reco. "
                    f"The model was created with version {restore_version}, "
                    f"but version is {dnn_reco.__version__} "
                    "is currently being used."
                )
        else:
            config["dnn_reco_version"] = dnn_reco.__version__

        # get git repo information
        config["git_short_sha"] = str(version_control.short_sha)
        config["git_sha"] = str(version_control.sha)
        config["git_origin"] = str(version_control.origin)
        config["git_uncommited_changes"] = version_control.uncommitted_changes
        config["pip_installed_packages"] = version_control.installed_packages

        # ----------------------------------
        # expand all strings with variables
        # ----------------------------------
        config["config_name"] = str(config_name)
        for key in config:
            if isinstance(config[key], str):
                config[key] = config[key].format(**config)

        # Todo: save the merged config together with logs and checkpoints
        #       and make sure this defines a unique name.

        self.config = config

    def get_config(self):
        """Returns config

        Returns
        -------
        dictionary
            Dictionary with defined settings.
        """
        return dict(self.config)
