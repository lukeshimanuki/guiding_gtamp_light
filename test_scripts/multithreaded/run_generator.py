import os
import multiprocessing
import argparse
import socket
import numpy as np

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs
from test_scripts.run_generator import get_logfile_name, parse_arguments


def worker_p(config):
    command = 'python ./test_scripts/run_generator.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    print command
    #os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    target_pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                    40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]

    target_pidx_idxs = range(len(target_pidxs))

    setup = parse_arguments()
    if setup.use_learning:
        sampler_seeds = range(3)
        planning_seeds = range(1,4)
    else:
        planning_seeds = range(4)
        sampler_seeds = [0]
    target_file = open(get_logfile_name(setup).name,'r')
    existing_results = target_file.read().splitlines()
    pidx_seed_already_exist = []
    for l in existing_results:
        pidx = int(l.split(',')[0])
        seed = int(l.split(',')[1])
        pidx_seed_already_exist.append((pidx,seed))
    pidx_seed_already_exist = []
    print np.all([(idx,seed) in pidx_seed_already_exist for seed in sampler_seeds for idx in target_pidx_idxs])
    configs = []
    for planning_seed in planning_seeds:
        for seed in sampler_seeds:
            for idx in target_pidx_idxs:
                if (idx,seed) in pidx_seed_already_exist:
                    continue
                config = {}
                for k,v in setup._get_kwargs():
                    if type(v) is bool and v is True:
                        config[k] = ''
                    elif type(v) is not bool:
                        config[k] = v
                config['target_pidx_idx'] = idx
                config['sampler_seed'] = seed
                config['seed'] = planning_seed
                if setup.use_learning:
                    config['use_learning'] = ''

                configs.append(config)

    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
