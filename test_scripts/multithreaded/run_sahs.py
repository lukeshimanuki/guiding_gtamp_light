import os
import multiprocessing

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py'

    for key, value in zip(config.keys(), config.values()):
        if 'sampling_strategy' in key:
            continue
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    command += ' -timelimit 9999'
    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='Greedy Planner parameters')
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])
    params = parser.parse_args()
    pidx_begin = params.pidxs[0]
    pidx_end = params.pidxs[1]
    configs = []
    for pidx in range(pidx_begin, pidx_end):
        for planner_seed in range(5):
            config = {
                'pidx': pidx,
                'planner_seed': planner_seed
            }
            configs.append(config)

    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
