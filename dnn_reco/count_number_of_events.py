#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import click
import ruamel.yaml as yaml
from tqdm import tqdm
import glob
import pandas as pd
from multiprocessing import Pool

from dnn_reco import misc
from dnn_reco.setup_manager import SetupManager
from dnn_reco.data_handler import DataHandler
from dnn_reco.data_trafo import DataTransformer


def count_num_events(data):
    input_data, config = data
    try:
        with pd.HDFStore(input_data,  mode='r') as f:
            time_offset = f[config['data_handler_time_offset_name']]['value']
        return len(time_offset)
    except KeyError:
        return 0


@click.command()
@click.argument('config_files', type=click.Path(exists=True), nargs=-1)
@click.option('--n_jobs', '-j',
              default=1, help='Number of jobs to run in parallel.')
def main(config_files, n_jobs):
    """Script to count number of files.

    Parameters
    ----------
    config_files : list of strings
        List of yaml config files.
    n_jobs : int
        Number of jobs to run in parallel.
    """

    # read in and combine config files and set up
    setup_manager = SetupManager(config_files)
    config = setup_manager.get_config()

    names = ['test_data_file', 'validation_data_file',
             'training_data_file', 'trafo_data_file']
    num_events_list = []
    for name in names:
        # get files
        print('Creating file list for {!r}'.format(name))
        input_data = config[name]
        if isinstance(input_data, list):
            input_data = set(input_data)
            file_list = []
            for input_pattern in input_data:
                file_list.extend(glob.glob(input_pattern))
        else:
            file_list = glob.glob(input_data)
        file_list = set(file_list)

        print('Starting counting')
        pool = Pool(processes=n_jobs)
        num_files = len(file_list)
        num_events = 0
        with tqdm(total=num_files) as pbar:
            for i, n in tqdm(enumerate(pool.imap_unordered(
                            count_num_events, [(f, config) for f in file_list]))):
                pbar.update()
                num_events += n

        num_events_list.append(num_events)
        print('Found {!r} events for {!r}\n'.format(num_events, name))

    print('\n===============================')
    print('= Completed Counting Events:  =')
    print('===============================')
    for num_events, name in zip(num_events_list, names):
        print('Found {!r} events for {!r}'.format(num_events, name))


if __name__ == '__main__':
    main()
