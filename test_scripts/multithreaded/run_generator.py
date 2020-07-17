import os
import multiprocessing
import argparse
import socket
import numpy as np

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs
from test_scripts.run_generator import get_logfile_name, parse_arguments
from test_scripts.run_greedy import get_seed_and_epochs


def worker_p(config):
    command = 'python ./test_scripts/run_generator.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    target_pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                    40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]

    target_pidx_idxs = range(len(target_pidxs))

    setup = parse_arguments()
    sampler_seeds = range(3)
    configs = []

    action_types = ['place_loading']
    for action_type in action_types:
        for sampler_seed in sampler_seeds:
            setup.sampler_seed = sampler_seed
            if 'pick' in action_type:
                _, _, total_epochs = get_seed_and_epochs('pick', '', setup)
            else:
                region = action_type.split('_')[1]
                _, _, total_epochs = get_seed_and_epochs('place', region+'_region', setup)
            total_epochs = range(len(total_epochs))
            for epoch in total_epochs:
                for idx in target_pidx_idxs:
                    config = {}
                    for k, v in setup._get_kwargs():
                        if type(v) is bool and v is True:
                            config[k] = ''
                        elif type(v) is not bool:
                            config[k] = v
                    config['sampler_epoch'] = epoch
                    config['sampler_seed'] = sampler_seed
                    config['target_pidx_idx'] = idx
                    config['learned_sampler_atype'] = action_type
                    configs.append(config)
    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
