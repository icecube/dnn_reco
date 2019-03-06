#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import click

from dnn_reco import misc
from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer
from dnn_reco.model import NNModel


@click.command()
@click.argument('config_files', click.Path(exists=True), nargs=-1)
def main(config_files):
    """Script to train the NN model.

    Creates data handler, data transformer, and build NN model as specified
    in the config files. Compiles and trains the NN model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    """

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    if not config['model_is_training']:
        raise ValueError('Model must be in training mode!')

    # Create Data Handler object
    data_handler = DataHandler(config)
    data_handler.setup_with_test_data(config['training_data_file'])

    # Create Data iterators for training and validation data
    train_data_generator = data_handler.get_batch_generator(
                                    input_data=config['training_data_file'],
                                    batch_size=config['batch_size'],
                                    sample_randomly=True,
                                    pick_random_files_forever=True,
                                    file_capacity=config['file_capacity'],
                                    batch_capacity=config['batch_capacity'],
                                    num_jobs=config['num_jobs'],
                                    num_add_files=config['num_add_files'],
                                    num_repetitions=config['num_repetitions'],
                                    init_values=config['DOM_init_values'],
                                    )

    val_data_generator = data_handler.get_batch_generator(
                                    input_data=config['validation_data_file'],
                                    batch_size=config['batch_size'],
                                    sample_randomly=True,
                                    pick_random_files_forever=True,
                                    file_capacity=1,
                                    batch_capacity=5,
                                    num_jobs=1,
                                    num_add_files=0,
                                    num_repetitions=1,
                                    init_values=config['DOM_init_values'],
                                    )

    # create data transformer
    data_transformer = DataTransformer(
                    data_handler=data_handler,
                    treat_doms_equally=config['trafo_treat_doms_equally'],
                    normalize_dom_data=config['trafo_normalize_dom_data'],
                    normalize_label_data=config['trafo_normalize_label_data'],
                    normalize_misc_data=config['trafo_normalize_misc_data'],
                    log_dom_bins=config['trafo_log_dom_bins'],
                    log_label_bins=config['trafo_log_label_bins'],
                    log_misc_bins=config['trafo_log_misc_bins'],
                    norm_constant=config['trafo_norm_constant'])

    # load trafo model from file
    data_transformer.load_trafo_model(config['trafo_model_path'])

    # create NN model
    model = NNModel(is_training=True,
                    config=config,
                    data_handler=data_handler,
                    data_transformer=data_transformer)

    # compile model: define loss function and optimizer
    model.compile()

    # restore model weights
    if config['model_restore_model']:
        model.restore()

    # train model
    model.fit(num_training_iterations=config['num_training_iterations'],
              train_data_generator=train_data_generator,
              val_data_generator=val_data_generator,
              evaluation_methods=None,)


if __name__ == '__main__':
    main()
