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

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    raw_dir = './planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/' \
              'qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_' \
              'use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    all_plan_exp_files = os.listdir(raw_dir)

    pidxs = [int(f.split('_')[1]) for f in all_plan_exp_files]
    seeds = range(5)
    pidxs = pidxs[0:50]

    configs = []
    for seed in seeds:
        for pidx in pidxs:
            config = {
                'pidx': pidx,
                'seed': seed,
            }
            configs.append(config)

    n_workers = 20 #multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
