import os
import multiprocessing
import argparse

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs
from test_scripts.run_generator import get_logfile_name, parse_arguments


def worker_p(config):
    command = 'python ./test_scripts/run_generator.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    #command += ' -use_learning'
    print command
    #os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    raw_dir = './planning_experience/for_testing_generators/'
    all_plan_exp_files = os.listdir(raw_dir)

    pidxs = [int(f.split('_')[1]) for f in all_plan_exp_files]
    seeds = range(0, 5)

    setup = parse_arguments()
    target_file = open(get_logfile_name(setup).name,'r')
    existing_results = target_file.read().splitlines()
    pidx_seed_already_exist = []
    for l in existing_results:
        pidx = int(l.split(',')[0])
        seed = int(l.split(',')[1])
        pidx_seed_already_exist.append((pidx,seed))

    configs = []
    for seed in seeds:
        for pidx in pidxs:
            if (pidx,seed) in pidx_seed_already_exist:
                continue
            config = {
                'pidx': pidx,
                'seed': seed,
                'sampling_strategy': setup.sampling_strategy,
                'n_mp_limit': setup.n_mp_limit
            }
            configs.append(config)

    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
