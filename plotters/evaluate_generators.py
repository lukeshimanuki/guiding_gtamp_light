from generators.learning.learning_algorithms.WGANGP import WGANgp
from generators.learning.train_torch_sampler import get_data_generator

import pickle
import os
import argparse


def main():
    parser = argparse.ArgumentParser('config')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-iteration', type=int, default=0)
    parser.add_argument('-architecture', type=str, default='fc')
    config = parser.parse_args()

    model = WGANgp(config.atype, config.region, config.architecture)

    fdir = model.weight_dir + '/result_summary/'
    if not os.path.isdir(fdir):
        os.makedirs(fdir)

    fname = 'results_iter_%d.pkl' % config.iteration
    summary_file_exists = os.path.isfile(fdir + fname)
    if summary_file_exists:
        print "*******Already done*******"
        return
    else:
        trainloader, trainset, testset = get_data_generator(config.atype, config.region)
        result = model.evaluate_generator(testset, iteration=config.iteration)
        pickle.dump(result, open(fdir + fname, 'wb'))
    import pdb;
    pdb.set_trace()

    """
    results = np.array(results)
    iterations = np.array(iterations)
    import pdb;pdb.set_trace()

    # I need to select non-inf entropy
    non_inf_idxs = np.where(results[:, 2] != np.inf)[0]
    #iterations = iterations[non_inf_idxs]
    #results = results[non_inf_idxs, :]
    max_kde_idx = np.argsort(results[:, 1])[::-1][0:100]

    iters = iterations[max_kde_idx]
    entropies = results[max_kde_idx, 2]
    for i, e in zip(iters, entropies):
        print i, e
    for i, kde, mse in zip(iterations, results[:, 1], results[:, 0]):
        print i, kde, mse
    import pdb;pdb.set_trace()

    print "Max KDE epoch", iterations[max_kde_idx][0]
    print "Max KDE", results[max_kde_idx, 1][0]
    print "Max KDE entropy", results[max_kde_idx, 2][0]
    print "Max KDE min MSE", results[max_kde_idx, 0][0]
    """


if __name__ == '__main__':
    main()
