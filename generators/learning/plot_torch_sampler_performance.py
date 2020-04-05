from generators.learning.learning_algorithms.WGANGP import WGANgp
from matplotlib import pyplot as plt
from train_torch_sampler import get_data_generator

import numpy as np
import pickle
import os
import argparse
import re
import copy


def plot(x_data, y_data, title, file_dir):
    plt.close('all')
    plt.figure()
    plt.plot(x_data, y_data)
    plt.title(title)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_dir + '{}.png'.format(title))


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


def plot_results(iterations, results, fdir):
    results = copy.deepcopy(np.array(results))
    iterations = copy.deepcopy(np.array(iterations)[:len(results)])
    in_bound_idxs = results[:, 2] != np.inf
    results = results[in_bound_idxs, :]
    if len(results) == 0:
        return
    iterations = iterations[in_bound_idxs]

    plot(iterations, results[:, 0], 'Min MSEs', fdir)
    plot(iterations, results[:, 1], 'KDE scores', fdir)
    plot(iterations, results[:, 2], 'Entropies', fdir)


def main():
    parser = argparse.ArgumentParser('config')
    parser.add_argument('-atype', type=str, default='pick')
    parser.add_argument('-region', type=str, default='loading_region')
    config = parser.parse_args()
    model = WGANgp(config.atype, config.region)

    max_iter = get_max_iteration(model.weight_dir)
    trainloader, trainset, testset = get_data_generator(config.atype, config.region)

    iterations = range(100, max_iter, 100)

    fdir = model.weight_dir + '/result_summary/'
    if not os.path.isdir(fdir):
        os.makedirs(fdir)

    fname = 'results.pkl'
    if os.path.isfile(fname):
        results = pickle.load(open(fname, 'r'))
    else:
        results = []
        for iteration in iterations:
            result = model.evaluate_generator(testset, iteration=iteration)
            results.append(result)

            if iteration % 100 == 0:
                plot_results(iterations, results, fdir)
        pickle.dump(results, open(fname, 'wb'))

    print "Min MSE", iterations[np.argsort(results[:, 0])][0:50]
    print "KDE scores", iterations[np.argsort(results[:, 1])][::-1][0:50]
    print "Entropies", iterations[np.argsort(results[:, 2])][::-1][0:50]


if __name__ == '__main__':
    main()
