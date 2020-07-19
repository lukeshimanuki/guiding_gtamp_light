import os
import multiprocessing
import numpy as np
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
from test_scripts.run_greedy import parse_arguments


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    setup = parse_arguments()
    setup.use_region_agnostic = True
    setup.absq_seed = 2
    setup.place_goal_region_epoch = 'best'
    setup.place_obj_region_epoch = 'best'
    setup.pick_epoch = 'best'
    setup.timelimit = np.inf
    setup.num_node_limit = 100

    pidxs = [40200, 40201, 40202, 40204, 40205, 40206, 40207, 40208, 40209]

    pidx_and_seeds = [(pidx, seed) for pidx in pidxs for seed in range(4)]
    configs = []
    print "total runs", len(pidxs) * len(range(4))
    for pidx_seed in pidx_and_seeds:
        config = {}
        for k, v in setup._get_kwargs():
            if type(v) is bool and v is True:
                config[k] = ''
            elif type(v) is not bool:
                config[k] = v
        config['pidx'] = pidx_seed[0]
        config['planner_seed'] = pidx_seed[1]
        configs.append(config)

    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
