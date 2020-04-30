import os
import multiprocessing
import argparse

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
from threaded_test_utils import get_sahs_configs


def worker_p(config):
    command = 'python ./test_scripts/run_mcts.py'

    command += ' -use_shaped_reward'
    command += ' -use_learned_q'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    command += ' -pw'
    command += ' -use_ucb'
    command += ' -widening_parameter 0.2'
    command += ' -sampling_strategy voo'
    command += ' -switch_frequency 1000'

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    parser = argparse.ArgumentParser(description='Greedy Planner parameters')
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parameters = parser.parse_args()
    pidx_begin = parameters.pidxs[0]
    pidx_end = parameters.pidxs[1]
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
