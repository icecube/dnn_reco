from __future__ import division, print_function
import os
import shutil
import glob
import click
import ruamel.yaml as yaml
import tensorflow as tf

from dnn_reco import misc
from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer
from dnn_reco.model import NNModel


@click.command()
@click.argument('config_files', click.Path(exists=True), nargs=-1)
@click.option('--output_folder', '-o', default=None,
              help='folder to which the model will be exported')
@click.option('--data_settings', '-s',  default=None,
              help='Config file used to create training data')
@click.option('--logs/--no-logs', default=True,
              help='Export tensorflow log files.')
def main(config_files, output_folder, data_settings, logs):
    """Script to export dnn reco model.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    """

    # Check paths and define output names
    if not os.path.isdir(output_folder):
        print('Creating directory: {!r}'.format(output_folder))
        os.makedirs(output_folder)
    else:
        if len(os.listdir(output_folder)) > 0:
            raise ValueError('Directory already exists and contains files!')

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    # Create Data Handler object
    data_handler = DataHandler(config)
    data_handler.setup_with_test_data(config['training_data_file'])

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

    # -------------------------
    # Export latest checkpoints
    # -------------------------
    latest_checkpoint = tf.train.latest_checkpoint(os.path.dirname(
                                config['model_checkpoint_path']))
    if latest_checkpoint is None:
        raise ValueError('Could not find a checkpoint. Aborting export!')
    else:
        print(latest_checkpoint)
        raise NotImplementedError('Need to copy the checkpoints over')

    # -----------------------------
    # read and export data settings
    # -----------------------------
    export_data_settings(data_settings=data_settings,
                         output_folder=output_folder)

    # -----------------------------
    # Export trafo model and config
    # -----------------------------
    base_name = os.path.basename(config['trafo_model_path'])
    if '.' in base_name:
        file_ending = base_name.split('.')[-1]
        base_name = base_name.replace('.' + file_ending, '')

    shutil.copy2(src=config['trafo_model_path'],
                 dst=os.path.join(output_folder, 'trafo_model.npy'))
    shutil.copy2(src=os.path.join(os.path.dirname(config['trafo_model_path']),
                                  'config_trafo__{}.yaml'.format(base_name)),
                 dst=os.path.join(output_folder, 'config_trafo.yaml'))

    # ----------------------------
    # Export training config files
    # ----------------------------
    training_files = glob.glob(os.path.join(self._check_point_path,
                                            'config_training_*.yaml'))
    for training_file in training_files:
        shutil.copy2(src=training_file,
                     dst=os.path.join(output_folder,
                                      os.path.basename(training_file)))
    shutil.copy2(src=os.path.join(config['model_checkpoint_path'],
                                  'training_steps.yaml'),
                 dst=os.path.join(output_folder, 'training_steps.yaml'))

    # ----------------------
    # Export model meta data
    # ----------------------
    # Export all the information that the datahandler and data trafo collect
    # via the test file
    # ToDo: implement DataHandler.setup_with_config(config_meta_data.yaml)
    #       (instead of DataHandler.setup_with_data_container)

    # ------------------------------------
    # Export package versions and git hash
    # ------------------------------------

    # -------------------------------
    # Export tensorflow training logs
    # -------------------------------
    if logs:
        raise NotImplementedError()

    # Todo:

    # # restore model weights
    # if config['model_restore_model']:
    #     model.restore()

    # if i % self.config['save_frequency'] == 0:
    #         self._save_training_config(i)
    #         if self.config['model_save_model']:
    #             self.saver.save(
    #                     sess=self.sess,
    #                     global_step=self._step_offset + i,
    #                     save_path=self.config['model_checkpoint_path'])

    # ToDo: save model correctly in output path
    # data_transformer.save_trafo_model(config['trafo_model_path'])


def export_data_settings(data_settings, output_folder):
    """Read and export data settings.

    Parameters
    ----------
    data_settings : str
        Path to config file that was used to create the training data.
    output_folder : str
        Path to model output directory to which the exported model will be
        written to.
    """
    with open(data_settings, 'r') as stream:
        data_config = yaml.safe_load(stream)

    data_settings = {}
    if 'pulse_time_quantiles' not in data_config:
        data_config['pulse_time_quantiles'] = None
    if 'pulse_time_binning' not in data_config:
        data_config['pulse_time_binning'] = None
    if 'autoencoder_settings' not in data_config:
        data_config['autoencoder_settings'] = None
    if 'autoencoder_encoder_name' not in data_config:
        data_config['autoencoder_encoder_name'] = None

    data_settings['num_bins'] = data_config['num_data_bins']
    data_settings['relative_time_method'] = data_config['relative_time_method']
    data_settings['data_format'] = data_config['pulse_data_format']
    data_settings['time_bins'] = data_config['pulse_time_binning'],
    data_settings['time_quantiles'] = data_config['pulse_time_quantiles']
    data_settings['autoencoder_settings'] = data_config['autoencoder_settings']
    data_settings['autoencoder_name'] = data_config['autoencoder_encoder_name']

    with open(os.path.join(output_folder, 'config_data_settings'), 'w') as f:
        data_config = yaml.dump(f)


if __name__ == '__main__':
    main()
