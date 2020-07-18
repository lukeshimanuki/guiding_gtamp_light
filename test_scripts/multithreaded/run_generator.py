import os
import multiprocessing
import argparse
import socket
import numpy as np

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs
from test_scripts.run_generator import get_logfile_name, parse_arguments
from test_scripts.run_generator import convert_seed_epoch_idxs_to_seed_and_epoch
import time

def worker_p(config):
    if 'do_uploading' in config:
        print "Running upload generator results"
        cmd = 'python upload_generator_results.py'
        print cmd
        os.system(cmd)
    else:
        command = 'python ./test_scripts/run_generator.py'

        for key, value in zip(config.keys(), config.values()):
            option = ' -' + str(key) + ' ' + str(value)
            command += option
        print command
    #time.sleep(10)
    #os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def get_all_configs(target_pidx_idxs, setup):
    if setup.use_learning:
        if 'pick' in setup.learned_sampler_atype:
            _, _, total_epochs = convert_seed_epoch_idxs_to_seed_and_epoch('pick', '', setup)
        else:
            region = setup.learned_sampler_atype.split('_')[1]
            _, _, total_epochs = convert_seed_epoch_idxs_to_seed_and_epoch('place', region + '_region', setup)
        total_epochs = range(len(total_epochs))
    else:
        total_epochs = [0]
        
    configs = []
    print "Total number of epochs", len(total_epochs)
    for epoch in total_epochs:
        for idx in target_pidx_idxs:
            config = {}
            for k, v in setup._get_kwargs():
                if type(v) is bool and v is True:
                    config[k] = ''
                elif type(v) is not bool:
                    config[k] = v
            config['sampler_epoch_idx'] = epoch
            config['target_pidx_idx'] = idx
            configs.append(config)
    return configs


def main():
    target_pidxs = [40200,40201,40202,40204,40205,40206,40207,40208,40209]
    target_pidx_idxs = range(len(target_pidxs))
    setup = parse_arguments()

    #configs = get_all_configs(target_pidx_idxs, setup)
    configs = [{'do_uploading'}]+[{'a':'b'}]*10
    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)
    pool.close()
    pool.join()
    #print results


if __name__ == '__main__':
    main()
