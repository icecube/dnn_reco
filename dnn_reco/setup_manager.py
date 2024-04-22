from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
import ruamel.yaml as yaml

from dnn_reco import version_control

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
        dom_response_shape : list of int
            The shape of the DOM response tensor excluding the batch dimension.
            E.g.: [x_dim, y_dim, z_dim, num_bins]
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
        keep_global_count : bool
            If true, a global count of training iterations is performed.
            The amount of previous training iterations is inferred
            from the latest checkpoint file.
            If false, counting of training iterations will start at zero.
        keep_probability_list : list of float
            A list of keep probabilities for dropout layers.
            A tensorflow placeholder is created for each float given in this
            list. These placeholders can then be used in the generator or
            discriminator networks by defining the index of the correct keep
            probability.
            During training these placeholders will be set to the values as
            defined in 'keep_probability_list'. During testing and validation
            these will be set to 1.0, e.g. no dropout is applied.
        hypothesis_smearing : None or list of float
            Smearing to be used for cascade hypothesis during training.
            'hypothesis_smearing' is a list of floats with the length equal to
            the number of cascade parameters.
            The ith value in 'smearing'  resembles the std deviation of the
            gaussian that will be added as a smearing.
            Cascade Parameters: x, y, z, zenith, azimuth, energy, t
            If smearing is None, no smearing will be applied.

    Trafo settings

        trafo_model_path : str
            Path to trafo model file.
        trafo_data_file : str
            Path to trafo data file.
        trafo_load_model : bool
            If true, the transformation model will be loaded from file.
            If false, a new transorfmation model will be created from the data
                      specified by the 'trafo_data' key.
        trafo_save_model : bool
            If true, the transformation model will be saved to the file
                     specified by the 'trafo_model_path' key.
                     Note: This will overwrite the file!
        trafo_normalize : bool
            If true, data will be normalized to have a mean of 0 and a variance
            of 1.
        trafo_log_bins : bool, list of bool
            The natural logarithm is applied to the hits prior
            to normalization.
            If a list is given, the length of the list must match the number of
            bins 'num_bins'. The logarithm is applied to bin i if the ith entry
            of the log_bins list is True.
        trafo_log_energy : bool, list of bool
            The natural logarithm is applied to the cascade energy.
        trafo_treat_doms_equally : bool
            All DOMs are treated equally, e.g. the mean and variance is
            calculated over all DOMs and not individually.
        trafo_norm_constant : float
            A small constant that is added to the denominator during
            normalization to ensure finite values.

    NN Model Architecture

        generator_model_file : str
            Name of python file in dnn_reco/modules/model/ directory in
            which the generator is defined.
        generator_model_name : str
            Name of function in the 'generator_model_file' that is used to
            define the generator.

    NN Model Training

        model_checkpoint_path : str
            Path to directory in which the generator checkpoints are
            stored.
        generator_restore_model : bool
            If true, generator parameters are restored from file.
            If false, generator parameters are randomly initialized.
        generator_save_model : bool
            If true, the generator parameters are saved to file.
                Note: This will overwrite possible existing files.
        generator_perform_training : bool
            If true, the generator will be trained in a perform_training call
                     if it has trainable parameters.
        generator_loss_file : str
            Name of python file in dnn_reco/modules/loss/ directory in
            which the loss function for the generator optimizer is defined.
        generator_loss_name : str
            Name of function in the 'generator_loss_file' that is used to
            define the loss function for the generator optimizer.

        generator_optimizer_name : str
            Name of the tensorflow optimizer to use for the generator training.

        generator_optimizer_settings : dict
            Settings for the chosen generator optimizer.

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
        "data_handler_num_splits": None,
    }

    def __init__(self, config_files):
        """Initializes the DNN reco Setup Manager

        Loads and merges yaml config files, sets up necessary directories.

        Parameters
        ----------
        config_files : list of strings
            List of yaml config files.
        program_options : str
            A string defining the program options.
        """
        self._config_files = config_files

        # gather objects that will be passed and shared to all modules
        # (generator, discriminator and loss modules)
        self.shared_objects = {}

        # load and combine configs
        self._setup_config()

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
                config_update = yaml.YAML(typ="safe", pure=True).load(stream)

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
        import tfscripts as tfs

        tfs.FLOAT_PRECISION = config["tf_float_precision"]

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
