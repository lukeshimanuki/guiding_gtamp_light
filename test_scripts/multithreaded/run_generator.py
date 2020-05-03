import os
import multiprocessing
import argparse

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs


def worker_p(config):
    command = 'python ./test_scripts/run_generator.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    #command += ' -use_learning'
    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    raw_dir = './planning_experience/for_testing_generators/'
    all_plan_exp_files = os.listdir(raw_dir)

    pidxs = [int(f.split('_')[1]) for f in all_plan_exp_files]
    seeds = range(1, 5)

    configs = []
    sampling_strategy = 'unif'
    n_mp_limit = 5
    for seed in seeds:
        for pidx in pidxs:
            config = {
                'pidx': pidx,
                'seed': seed,
                'sampling_strategy': sampling_strategy,
                'n_mp_limit': n_mp_limit
            }
            configs.append(config)

    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
