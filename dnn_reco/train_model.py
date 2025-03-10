#!/usr/bin/env python
import click
import logging

from dnn_reco import misc
from dnn_reco.settings.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer


@click.command()
@click.argument("config_files", type=click.Path(exists=True), nargs=-1)
@click.option(
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
)
@click.option(
    "--num_threads",
    type=int,
    default=None,
)
def main(config_files, log_level, num_threads):
    """Script to train the NN model.

    Creates data handler, data transformer, and build NN model as specified
    in the config files. Compiles and trains the NN model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    log_level : str
        Log level for logging.
    num_threads : int
        Number of threads to use for tensorflow.
    """
    # set up logging
    logging.basicConfig(level=log_level)

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files, num_threads=num_threads)
    config = setup_manager.get_config()

    if not config["model_kwargs"]["is_training"]:
        raise ValueError("Model must be in training mode!")

    # Create Data Handler object
    data_handler = DataHandler(config)
    data_handler.setup_with_test_data(config["training_data_file"])

    # Create Data iterators for training and validation data
    train_data_generator = data_handler.get_batch_generator(
        input_data=config["training_data_file"],
        batch_size=config["batch_size"],
        sample_randomly=True,
        pick_random_files_forever=True,
        file_capacity=config["file_capacity"],
        batch_capacity=config["batch_capacity"],
        num_jobs=config["num_jobs"],
        num_add_files=config["num_add_files"],
        num_repetitions=config["num_repetitions"],
        max_events_per_file=config["data_handler_max_events_per_file"],
        max_file_chunk_size=config["data_handler_max_file_chunk_size"],
        init_values=config["DOM_init_values"],
        nan_fill_value=config["data_handler_nan_fill_value"],
    )

    val_data_generator = data_handler.get_batch_generator(
        input_data=config["validation_data_file"],
        batch_size=config["batch_size"],
        sample_randomly=True,
        pick_random_files_forever=True,
        file_capacity=1,
        batch_capacity=5,
        num_jobs=1,
        num_add_files=1,
        num_repetitions=1,
        max_events_per_file=config["data_handler_max_events_per_file"],
        max_file_chunk_size=config["data_handler_max_file_chunk_size"],
        init_values=config["DOM_init_values"],
        nan_fill_value=config["data_handler_nan_fill_value"],
    )

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

    # restore model weights
    if config["model_restore_model"]:
        model.restore()

    # train model
    model.fit(
        num_training_iterations=config["num_training_iterations"],
        train_data_generator=train_data_generator,
        val_data_generator=val_data_generator,
        evaluation_methods=None,
    )


if __name__ == "__main__":
    main()
