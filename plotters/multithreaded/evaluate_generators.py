import argparse
import os
import re
import numpy as np
import multiprocessing

from multiprocessing.pool import ThreadPool
from generators.learning.learning_algorithms.WGANGP import WGANgp


def worker_p(config):
    command = 'python ./plotters/evaluate_generators.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def get_max_iteration(weight_dir):
    all_weight_files = os.listdir(weight_dir)
    max_iter = -np.inf
    for wfile in all_weight_files:
        if '.pt' not in wfile:
            continue
        iteration = int(re.findall(r'[0-9]+', wfile)[0])

        if iteration > max_iter:
            max_iter = iteration
    return max_iter


def main():
    parser = argparse.ArgumentParser('config')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-architecture', type=str, default='fc')

    parameters = parser.parse_args()
    model = WGANgp(parameters.atype, parameters.region, parameters.architecture)

    max_iter = get_max_iteration(model.weight_dir)
    max_iter = min(250000, max_iter)
    iterations = range(100, max_iter, 100)
    configs = []
    for iteration in iterations:
        config = {
            'iteration': iteration,
            'atype': parameters.atype,
            'region': parameters.region,
            'architecture': parameters.architecture
        }

        configs.append(config)

    n_workers = 1 if parameters.architecture != 'fc' else multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
