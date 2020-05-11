import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
# from generators.learning.learning_algorithms.WGANGP import WGANgp

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
    print "Saving figure...", file_dir + '{}.png'.format(title)
    plt.savefig(file_dir + '{}.png'.format(title))


def plot_results(iterations, results, result_dir):
    results = copy.deepcopy(np.array(results))
    iterations = copy.deepcopy(np.array(iterations)[:len(results)])
    in_bound_idxs = results[:, 2] != np.inf
    results = results[in_bound_idxs, :]
    if len(results) == 0:
        return
    iterations = iterations[in_bound_idxs]

    plot(iterations, results[:, 0], 'Min MSEs', result_dir)
    plot(iterations, results[:, 1], 'kernel_density_estimates', result_dir)
    plot(iterations, results[:, 2], 'Entropies', result_dir)


def print_results(results, iterations, plot_dir, result_dir):
    results = np.array(results)
    iterations = np.array(iterations)

    print "Raw numbers"
    for i, kde, mse, entropy in zip(iterations, results[:, 1], results[:, 0], results[:, 2]):
        print i, kde, mse, entropy

    max_kde_idx = np.argsort(results[:, 1])[::-1]
    to_print = "Max KDE epoch {} \nMax KDE {} \nMax KDE entropy {} \nMax KDE min MSE {}".format(
        iterations[max_kde_idx][0], results[max_kde_idx, 1][0], results[max_kde_idx, 2][0], results[max_kde_idx, 0][0])
    print to_print

    best_iter = iterations[max_kde_idx][0]
    weight_dir = result_dir[:-15]
    for fin in os.listdir(weight_dir):
        if 'gen' not in fin:
            continue
        iteration = int(fin.split('_')[-1].split('.')[0])
        if iteration == best_iter:
            break
    print weight_dir + fin
    fin = open(plot_dir+'/results.txt', 'wb')
    fin.write(to_print)

def main():
    parser = argparse.ArgumentParser('config')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-iteration', type=int, default=0)
    parser.add_argument('-architecture', type=str, default='fc')
    parser.add_argument('-old', action='store_true', default=False)  # used for threaded runs
    config = parser.parse_args()
    if config.old:
        result_dir = 'plotters/generator_plots/before_adding_vmanip/{}/{}/wgangp/result_summary'.format(
            config.atype, config.region,
            config.architecture)
        results = pickle.load(open(result_dir + '/results.pkl', 'r'))
        print_results(results, range(len(results)), result_dir)
        return

    if config.atype == 'pick':
        result_dir = './generators/learning/learned_weights/{}/wgangp/{}/result_summary/'.format(config.atype,
                                                                                                    config.architecture)
    else:
        result_dir = './generators/learning/learned_weights/{}/{}/wgangp/{}/result_summary/'.format(config.atype,
                                                                                                    config.region,
                                                                                                    config.architecture)
    result_files = os.listdir(result_dir + '/')
    iters = [int(f.split('_')[-1].split('.')[0]) for f in result_files]
    result_files_sorted = np.array(result_files)[np.argsort(iters)]
    iters = np.sort(iters)
    result_files_sorted.tolist()
    results = [pickle.load(open(result_dir + result_file, 'r')) for result_file in result_files_sorted]

    plot_dir = './plotters/generator_plots/{}/{}/wgangp/{}/'.format(config.atype, config.region,
                                                                    config.architecture)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    plot_results(iters, results, plot_dir)
    print_results(results, iters, plot_dir, result_dir)


if __name__ == '__main__':
    main()
