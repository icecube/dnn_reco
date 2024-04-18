#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import click
import ruamel.yaml as yaml

from dnn_reco import misc
from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer


@click.command()
@click.argument("config_files", type=click.Path(exists=True), nargs=-1)
def main(config_files):
    """Script to generate trafo model.

    Creates the desired trafo model as defined in the yaml configuration files
    and saves the trafo model to disc.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    """

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # Create Data Handler object
    data_handler = DataHandler(config)
    data_handler.setup_with_test_data(config["trafo_data_file"])

    settings_trafo = {
        "input_data": config["trafo_data_file"],
        "batch_size": config["batch_size"],
        "sample_randomly": True,
        "pick_random_files_forever": False,
        "file_capacity": 1,
        "batch_capacity": 2,
        "num_jobs": config["trafo_num_jobs"],
        "num_add_files": 1,
        "num_repetitions": 1,
        "num_splits": config["data_handler_num_splits"],
        "init_values": config["DOM_init_values"],
        "nan_fill_value": config["data_handler_nan_fill_value"],
    }
    trafo_data_generator = data_handler.get_batch_generator(**settings_trafo)

    # create TrafoModel
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

    data_transformer.create_trafo_model_iteratively(
        data_iterator=trafo_data_generator,
        num_batches=config["trafo_num_batches"],
    )

    # save trafo model to file
    base_name = os.path.basename(config["trafo_model_path"])
    if "." in base_name:
        file_ending = base_name.split(".")[-1]
        base_name = base_name.replace("." + file_ending, "")

    directory = os.path.dirname(config["trafo_model_path"])
    if not os.path.isdir(directory):
        os.makedirs(directory)
        misc.print_warning("Creating directory: {}".format(directory))

    trafo_config_file = os.path.join(
        directory, "config_trafo__{}.yaml".format(base_name)
    )

    with open(trafo_config_file, "w") as yaml_file:
        yaml.YAML(typ="unsafe", pure=True).dump(
            config,
            yaml_file,
            default_flow_style=False,
        )
    data_transformer.save_trafo_model(config["trafo_model_path"])

    # kill multiprocessing queues and workers
    data_handler.kill()

    print("\n=======================================")
    print("= Successfully saved trafo model to:  =")
    print("=======================================")
    print("{!r}\n".format(config["trafo_model_path"]))


if __name__ == "__main__":
    main()
