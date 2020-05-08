from matplotlib import pyplot as plt
from generators.learning.learning_algorithms.WGANGP import WGANgp

import copy
import os
import numpy as np
import argparse
import pickle


def plot(x_data, y_data, title, file_dir):
    plt.close('all')
    plt.figure()
    plt.plot(x_data, y_data)
    plt.title(title)
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    plt.savefig(file_dir + '{}.png'.format(title))


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


def print_results(results, iterations):
    results = np.array(results)
    iterations = np.array(iterations)

    print "Raw numbers"
    for i, kde, mse, entropy in zip(iterations, results[:, 1], results[:, 0], results[:, 2]):
        print i, kde, mse, entropy

    max_kde_idx = np.argsort(results[:, 1])[::-1][0:100]
    print "Max KDE epoch", iterations[max_kde_idx][0]
    print "Max KDE", results[max_kde_idx, 1][0]
    print "Max KDE entropy", results[max_kde_idx, 2][0]
    print "Max KDE min MSE", results[max_kde_idx, 0][0]


def main():
    parser = argparse.ArgumentParser('config')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-iteration', type=int, default=0)
    parser.add_argument('-architecture', type=str, default='fc')
    config = parser.parse_args()

    model = WGANgp(config.atype, config.region, config.architecture)
    fdir = model.weight_dir + '/result_summary/'

    result_files = os.listdir(fdir)
    iters = [int(f.split('_')[-1].split('.')[0]) for f in result_files]
    result_files_sorted = result_files[np.argsort(iters)]

    results = [pickle.load(open(fdir + result_file, 'r')) for result_file in result_files_sorted]
    plot_results(iters, results, fdir)

    print_results(results, iters)
