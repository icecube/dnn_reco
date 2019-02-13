from __future__ import division, print_function
import click
import numpy as np
import tensorflow as tf

from dnn_reco import misc
from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer


@click.command()
@click.argument('config_files', click.Path(exists=True), nargs=-1)
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
    data_handler = DataHandler(test_input_data=config['training_data_file'],
                               config=config)

    settings_trafo = {
                'input_data': config['training_data_file'],
                'batch_size': config['batch_size'],
                'sample_randomly': True,
                'pick_random_files_forever': False,
                'file_capacity': 1,
                'batch_capacity': 2,
                'num_jobs': config['num_jobs'],
                'num_add_files': 0,
                'num_repetitions': 1,
                'init_values': config['DOM_init_values'],
            }
    trafo_data_generator = data_handler.get_batch_generator(**settings_trafo)

    # create TrafoModel
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

    # Perform some basic tests
    for i in range(3):

        # get a new batch
        batch = next(trafo_data_generator)

        for test_data, data_type in zip(batch,
                                        ['ic78', 'deepcore',
                                         'label', 'misc']):
            if test_data is None:
                continue

            test_data_trafo = data_transformer.transform(test_data, data_type)
            test_data_inv_trafo = data_transformer.inverse_transform(
                                                    test_data_trafo, data_type)
            deviations = np.reshape(np.abs(test_data_inv_trafo - test_data),
                                    [-1])

            print('Checking numpy trafor for {}:'.format(data_type))
            print('\tChecking Mean:')
            print('\tOrig: {:2.5f}, Trafo: {:2.5f}, Inv-Trafo: {:2.5f}'.format(
                    np.mean(test_data),
                    np.mean(test_data_trafo),
                    np.mean(test_data_inv_trafo)))
            print('\tChecking Standard Deviation:')
            print('\tOrig: {:2.5f}, Trafo: {:2.5f}, Inv-Trafo: {:2.5f}'.format(
                    np.std(test_data, ddof=1),
                    np.std(test_data_trafo, ddof=1),
                    np.std(test_data_inv_trafo, ddof=1)))
            print('\tOriginal is same as inv-trafo(trafo(original)): ',
                  np.allclose(test_data, test_data_inv_trafo),
                  (test_data_inv_trafo == test_data).all())
            print('\tResiduals: min {:2.8f}, max {:2.8f}, mean {:2.8f}'.format(
                    np.min(deviations),
                    np.max(deviations),
                    np.mean(deviations)))

            test_data = tf.constant(test_data)
            test_data_trafo = data_transformer.transform(test_data, data_type)
            test_data_inv_trafo = data_transformer.inverse_transform(
                                                    test_data_trafo, data_type)

            sess = tf.Session()
            test_data, test_data_trafo, test_data_inv_trafo = sess.run(
                            [test_data, test_data_trafo, test_data_inv_trafo])
            deviations = np.reshape(np.abs(test_data_inv_trafo - test_data),
                                    [-1])

            print('Checking tensorflow trafor for {}:'.format(data_type))
            print('\tChecking Mean:')
            print('\tOrig: {:2.5f}, Trafo: {:2.5f}, Inv-Trafo: {:2.5f}'.format(
                    np.mean(test_data),
                    np.mean(test_data_trafo),
                    np.mean(test_data_inv_trafo)))
            print('\tChecking Standard Deviation:')
            print('\tOrig: {:2.5f}, Trafo: {:2.5f}, Inv-Trafo: {:2.5f}'.format(
                    np.std(test_data, ddof=1),
                    np.std(test_data_trafo, ddof=1),
                    np.std(test_data_inv_trafo, ddof=1)))
            print('\tOriginal is same as inv-trafo(trafo(original)): ',
                  np.allclose(test_data, test_data_inv_trafo),
                  (test_data_inv_trafo == test_data).all())
            print('\tResiduals: min {:2.8f}, max {:2.8f}, mean {:2.8f}'.format(
                np.min(deviations), np.max(deviations), np.mean(deviations)))


if __name__ == '__main__':
    main()
