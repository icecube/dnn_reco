#!/usr/bin/env python
import os
import shutil
import glob
import click
import logging

from dnn_reco import misc
from dnn_reco.settings.yaml import yaml_loader, yaml_dumper
from dnn_reco.settings.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer


@click.command()
@click.argument("config_files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--output_folder",
    "-o",
    default=None,
    type=click.Path(),
    help="folder to which the model will be exported",
)
@click.option(
    "--data_settings",
    "-s",
    default=None,
    type=click.Path(exists=True),
    help="Config file used to create training data",
)
@click.option(
    "--logs/--no-logs", default=True, help="Export tensorflow log files."
)
@click.option(
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
)
def main(config_files, output_folder, data_settings, logs, log_level):
    """Script to export dnn reco model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    output_folder : str
        Path to model output directory to which the exported model will be
        written to.
    data_settings : str
        Path to config file that was used to create the training data,
        or alternatively a config file that contains the data settings.
    logs : bool
        Export tensorflow log files.
    log_level : str
        Log level for logging.
    """
    # set up logging
    logging.basicConfig(level=log_level)

    # Check paths and define output names
    if not os.path.isdir(output_folder):
        print("Creating directory: {!r}".format(output_folder))
        os.makedirs(output_folder)
    else:
        if len(os.listdir(output_folder)) > 0:
            if click.confirm(
                "Directory already exists and contains files! "
                "Delete {!r}?".format(output_folder),
                default=False,
            ):
                shutil.rmtree(output_folder)
                os.makedirs(output_folder)
            else:
                raise ValueError("Aborting!")

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # Create Data Handler object
    data_handler = DataHandler(config)
    data_handler.setup_with_test_data(config["training_data_file"])

    # create data transformer
    data_transformer = DataTransformer(
        data_handler=data_handler,
        treat_doms_equally=config["trafo_treat_doms_equally"],
        normalize_dom_data=config["trafo_normalize_dom_data"],
        normalize_label_data=config["trafo_normalize_label_data"],
        normalize_misc_data=config["trafo_normalize_misc_data"],
        log_dom_bins=config["trafo_log_dom_bins"],
        log_label_bins=config["trafo_log_label_bins"],
        log_misc_bins=config["trafo_log_misc_bins"],
        norm_constant=config["trafo_norm_constant"],
    )

    # load trafo model from file
    data_transformer.load_trafo_model(config["trafo_model_path"])

    # create NN model
    ModelClass = misc.load_class(config["model_class"])
    model = ModelClass(
        config=config,
        data_handler=data_handler,
        data_transformer=data_transformer,
        **config["model_kwargs"]
    )

    # compile model: define loss function and optimizer
    model.compile()

    # -------------------------
    # Export latest checkpoints
    # -------------------------
    latest_checkpoint = model._checkpoint_manager.latest_checkpoint
    checkpoint_dir = os.path.dirname(latest_checkpoint)
    if latest_checkpoint is None:
        raise ValueError("Could not find a checkpoint. Aborting export!")
    else:
        for ending in [".index", ".data-00000-of-00001"]:
            shutil.copy2(src=latest_checkpoint + ending, dst=output_folder)
        shutil.copy2(
            src=os.path.join(checkpoint_dir, "checkpoint"), dst=output_folder
        )

    # -----------------------------
    # read and export data settings
    # -----------------------------
    export_data_settings(
        data_settings=data_settings,
        output_folder=output_folder,
        config=config,
    )

    # -----------------------------
    # Export trafo model and config
    # -----------------------------
    base_name = os.path.basename(config["trafo_model_path"])
    if "." in base_name:
        file_ending = base_name.split(".")[-1]
        base_name = base_name.replace("." + file_ending, "")

    shutil.copy2(
        src=config["trafo_model_path"],
        dst=os.path.join(output_folder, "trafo_model.npy"),
    )
    shutil.copy2(
        src=os.path.join(
            os.path.dirname(config["trafo_model_path"]),
            "config_trafo__{}.yaml".format(base_name),
        ),
        dst=os.path.join(output_folder, "config_trafo.yaml"),
    )

    # ----------------------------
    # Export training config files
    # ----------------------------
    checkpoint_directory = os.path.dirname(config["model_checkpoint_path"])
    training_files = glob.glob(
        os.path.join(checkpoint_directory, "config_training_*.yaml")
    )
    for training_file in training_files:
        shutil.copy2(
            src=training_file,
            dst=os.path.join(output_folder, os.path.basename(training_file)),
        )
    shutil.copy2(
        src=os.path.join(checkpoint_directory, "training_steps.yaml"),
        dst=os.path.join(output_folder, "training_steps.yaml"),
    )

    # ----------------------
    # Export model meta data
    # ----------------------
    # Export all the information that the datahandler and data trafo collect
    # via the test file
    # ToDo: implement DataHandler.setup_with_config(config_meta_data.yaml)
    #       (instead of DataHandler.setup_with_data_container)

    meta_data = {
        "label_names": data_handler.label_names,
        "label_name_dict": data_handler.label_name_dict,
        "label_shape": data_handler.label_shape,
        "num_labels": data_handler.num_labels,
        "misc_names": data_handler.misc_names,
        "misc_name_dict": data_handler.misc_name_dict,
        "misc_data_exists": data_handler.misc_data_exists,
        "misc_shape": data_handler.misc_shape,
        "num_misc": data_handler.num_misc,
    }
    with open(os.path.join(output_folder, "config_meta_data.yaml"), "w") as f:
        yaml_dumper.dump(meta_data, f)

    # ------------------------------------
    # Export package versions and git hash
    # ------------------------------------
    version_control = {
        "dnn_reco_version": config["dnn_reco_version"],
        "git_short_sha": config["git_short_sha"],
        "git_sha": config["git_sha"],
        "git_origin": config["git_origin"],
        "git_uncommited_changes": config["git_uncommited_changes"],
        "pip_installed_packages": config["pip_installed_packages"],
    }
    with open(os.path.join(output_folder, "version_control.yaml"), "w") as f:
        yaml_dumper.dump(version_control, f)

    # -------------------------------
    # Export tensorflow training logs
    # -------------------------------
    if logs:
        log_directory = os.path.dirname(config["log_path"])
        shutil.copytree(
            src=log_directory, dst=os.path.join(output_folder, "logs")
        )

    print("\n====================================")
    print("= Successfully exported model to:  =")
    print("====================================")
    print("{!r}\n".format(output_folder))


# ---------------------
# Collect Data Settings
# ---------------------


def export_data_settings(data_settings, output_folder, config):
    """Read and export data settings.

    Parameters
    ----------
    data_settings : str
        Path to config file that was used to create the training data.
    output_folder : str
        Path to model output directory to which the exported model will be
        written to.
    config : dict
        Configuration of the NN model.
    """
    with open(data_settings, "r") as stream:
        data_config = yaml_loader.load(stream)

    for k in [
        "pulse_time_quantiles",
        "pulse_time_binning",
        "autoencoder_settings",
        "autoencoder_encoder_name",
    ]:
        if k not in data_config or data_config[k] is None:
            data_config[k] = None
    for k in ["pulse_time_quantiles", "pulse_time_binning"]:
        if data_config[k] is not None:
            data_config[k] = list(data_config[k])

    try:
        data_settings = ic3_processing_scripts(data_config, config)
    except Exception as e:
        print(e)
        print("Falling back to SVN processing scripts")
        try:
            data_settings = svn_processing_scripts(data_config)
        except Exception as e:
            print(e)
            print("Could not extract data settings. Aborting export!")
            raise e

    if "is_str_dom_format" not in data_settings:
        data_settings["is_str_dom_format"] = False

    print("\n=========================")
    print("= Found Data Settings:  =")
    print("=========================")
    for key, value in data_settings.items():
        print("{}: {}".format(key, value))
    misc.print_warning(
        "Please check if the extracted data settings are correct!"
    )

    with open(
        os.path.join(output_folder, "config_data_settings.yaml"), "w"
    ) as f:
        yaml_dumper.dump(data_settings, f)


# ----------------------------------------
# Extract settings from processing scripts
# ----------------------------------------


def ic3_processing_scripts(data_config, config):
    """Extract settings from IC3 processing scripts.

    Note that there is some ambiguity if multiple instances of
    CreateDNNData are used in the processing script. It will
    be attempted to find the correct one based on the specified
    output keys.
    """
    data_settings = {}

    # find CreateDNNData instance based on output keys
    dnn_data_configs = []
    for step_config in data_config["processing_steps"]:
        for segment_cfg in step_config["tray_segments"]:
            if segment_cfg["ModuleClass"] == "ic3_data.segments.CreateDNNData":

                # check correct output names
                if "OutputKey" in segment_cfg["ModuleKwargs"]:
                    base = segment_cfg["ModuleKwargs"]["OutputKey"]
                else:
                    base = "dnn_data"

                if (
                    config["data_handler_bin_values_name"]
                    == base + "_bin_values"
                    and config["data_handler_bin_indices_name"]
                    == base + "_bin_indices"
                    and config["data_handler_time_offset_name"]
                    == base + "_global_time_offset"
                ):
                    dnn_data_configs.append(segment_cfg["ModuleKwargs"])

    if len(dnn_data_configs) != 1:
        raise ValueError(
            "Expected to find exactly one CreateDNNData instance with "
            f"matching output keys, but found {dnn_data_configs}"
        )

    cfg = dnn_data_configs[0]

    # now extract settings
    data_settings["num_bins"] = cfg["NumDataBins"]
    data_settings["relative_time_method"] = cfg["RelativeTimeMethod"]
    data_settings["data_format"] = cfg["DataFormat"]
    data_settings["pulse_key"] = cfg.get("PulseKey", None)
    data_settings["dom_exclusions"] = cfg.get("DOMExclusions", None)
    data_settings["partial_exclusion"] = cfg.get("PartialExclusion", None)
    data_settings["time_bins"] = cfg.get("TimeBins", None)
    data_settings["time_quantiles"] = cfg.get("TimeQuantiles", None)
    data_settings["autoencoder_settings"] = cfg.get(
        "AutoencoderSettings", None
    )
    data_settings["autoencoder_name"] = cfg.get("AutoencoderEncoderName", None)
    data_settings["cascade_key"] = cfg.get("CascadeKey", "MCCascade")
    data_settings["output_key"] = cfg.get("OutputKey", "dnn_data")

    # collect mutable objects
    allowed_pulse_keys = []
    allowed_cascade_keys = []
    if data_settings["pulse_key"] is not None:
        allowed_pulse_keys.append(data_settings["pulse_key"])
    if data_settings["cascade_key"] is not None:
        allowed_cascade_keys.append(data_settings["cascade_key"])

    if len(allowed_pulse_keys) > 0:
        data_settings["allowed_pulse_keys"] = allowed_pulse_keys
    if len(allowed_cascade_keys) > 0:
        data_settings["allowed_cascade_keys"] = allowed_cascade_keys

    return data_settings


def svn_processing_scripts(data_config):
    """Extract settings from old SVN processing scripts."""
    data_settings = {}
    data_settings["num_bins"] = data_config["num_data_bins"]
    data_settings["relative_time_method"] = data_config["relative_time_method"]
    data_settings["data_format"] = data_config["pulse_data_format"]
    data_settings["time_bins"] = data_config["pulse_time_binning"]
    data_settings["time_quantiles"] = data_config["pulse_time_quantiles"]
    data_settings["autoencoder_settings"] = data_config["autoencoder_settings"]
    data_settings["autoencoder_name"] = data_config["autoencoder_encoder_name"]

    # ---------------------------------
    # Find and save optional parameters
    # ---------------------------------
    if "DNN_excluded_doms" in data_config:
        data_settings["dom_exclusions"] = data_config["DNN_excluded_doms"]
    if "DNN_partial_exclusion" in data_config:
        data_settings["partial_exclusion"] = data_config[
            "DNN_partial_exclusion"
        ]
    if "DNN_pulse_key" in data_config:
        data_settings["pulse_key"] = data_config["DNN_pulse_key"]
    elif "pulse_map_string" in data_config:
        data_settings["pulse_key"] = data_config["pulse_map_string"]

    if "DNN_cascade_key" in data_config:
        data_settings["cascade_key"] = data_config["DNN_cascade_key"]

    allowed_pulse_keys = []
    allowed_cascade_keys = []
    if "pulse_key" in data_settings:
        allowed_pulse_keys.append(data_settings["pulse_key"])
    if "cascade_key" in data_settings:
        allowed_cascade_keys.append(data_settings["cascade_key"])

    if "datasets" in data_config:
        for dataset in data_config["datasets"].values():
            if "DNN_pulse_key" in dataset:
                allowed_pulse_keys.append(dataset["DNN_pulse_key"])
            elif "pulse_map_string" in dataset:
                allowed_pulse_keys.append(dataset["pulse_map_string"])

            if "DNN_cascade_key" in dataset:
                allowed_cascade_keys.append(dataset["DNN_cascade_key"])

    if len(allowed_pulse_keys) > 0:
        data_settings["allowed_pulse_keys"] = allowed_pulse_keys
    if len(allowed_cascade_keys) > 0:
        data_settings["allowed_cascade_keys"] = allowed_cascade_keys

    return data_settings


if __name__ == "__main__":
    main()
