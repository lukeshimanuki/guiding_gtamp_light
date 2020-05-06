from generators.learning.learning_algorithms.WGANGP import WGANgp
from matplotlib import pyplot as plt
from generators.learning.train_torch_sampler import get_data_generator

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
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='home_region')
    config = parser.parse_args()
    model = WGANgp(config.atype, config.region)

    max_iter = get_max_iteration(model.weight_dir)
    max_iter = min(250000, max_iter)
    iterations = range(100, max_iter, 100)

    fdir = model.weight_dir + '/result_summary/'
    if not os.path.isdir(fdir):
        os.makedirs(fdir)

    fname = 'results.pkl'
    summary_file_exists = os.path.isfile(fdir + fname)
    summary_file_exists = False
    if summary_file_exists:
        results = pickle.load(open(fdir + fname, 'r'))
    else:
        trainloader, trainset, testset = get_data_generator(config.atype, config.region)
        results = []
        for iteration in iterations:
            result = model.evaluate_generator(testset, iteration=iteration)
            results.append(result)

            if iteration % 100 == 0:
                plot_results(iterations, results, fdir)
                pickle.dump(results, open(fdir + fname, 'wb'))

    results = np.array(results)
    iterations = np.array(iterations)

    # I need to select non-inf entropy
    non_inf_idxs = np.where(results[:, 2] != np.inf)[0]
    iterations = iterations[non_inf_idxs]
    results = results[non_inf_idxs, :]
    max_kde_idx = np.argsort(results[:, 1])[::-1][0:100]
    print "Max KDE epoch", iterations[max_kde_idx]
    print "Max KDE", results[max_kde_idx, 1]
    print "Max KDE entropy", results[max_kde_idx, 2]
    print "Max KDE min MSE", results[max_kde_idx, 0]
    iters = iterations[max_kde_idx]
    entropies = results[max_kde_idx, 2]
    for i, e in zip(iters, entropies):
        print i, e


if __name__ == '__main__':
    main()
