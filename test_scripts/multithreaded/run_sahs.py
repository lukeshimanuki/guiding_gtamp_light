import os
import multiprocessing

from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
from test_scripts.run_greedy import parse_arguments


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py -num_episode 4000 -use_learning -use_region_agnostic  -domain two_arm_mover -n_mp_limit 5 -num_node_limit 3000  -n_iter_limit 2000 -num_train 5000 ' \
              '-pidx {} -planner_seed {} -train_type {} -sampler_seed {} ' \
              '-n_objs_pack {} -timelimit {} -absq_seed {}'. \
        format(config['pidx'], config['planner_seed'], config['train_type'], config['sampler_seed'],
               config['n_objs_pack'], config['timelimit'], config['absq_seed'])

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def main():
    setup = parse_arguments()
    pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
             40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]

    pidx_and_seeds = [(pidx, seed) for pidx in pidxs for seed in range(5)]
    configs = []
    print "total runs", len(pidxs) * len(range(5))
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
