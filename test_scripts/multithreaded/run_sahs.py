import os
import multiprocessing

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
from test_scripts.run_greedy import parse_arguments


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py'

    for key, value in zip(config.keys(), config.values()):
        if 'sampling_strategy' in key:
            continue
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    print command
    #os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    setup = parse_arguments()

    pidxs = [60089, 60061, 60094, 60075, 60074, 60050, 60096, 60057, 60008, 60088, 60026, 60003, 60010, 60067, 60091,
             60031, 60006, 60024, 60030, 60062, 60099, 60018, 60011, 60029, 60098, 60083, 60079, 60016, 60045, 60038,
             60046, 60032, 60058, 60097, 60039]
    pidx_and_seeds = [(pidx,seed) for pidx in pidxs for seed in range(5)]
    import pdb;pdb.set_trace()
    configs = []
    print "total runs", len(pidxs)*len(range(5))
    for pidx_seed in pidx_and_seeds:
        config = {}
        for k, v in setup._get_kwargs():
            if type(v) is bool and v is True:
                config[k] = ''
            elif type(v) is not bool:
                config[k] = v
        config['pidx'] = pidx_seed[0]
        config['planner_seed'] = pidx_seed[1]
        config['timelimit'] = 9999
        configs.append(config)

    n_workers = 1 #multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
