from generators.learning.PlaceWGANGP import PlaceWGANgp
from matplotlib import pyplot as plt
from train_torch_sampler import get_data_generator

import numpy as np
import pickle
import os


def plot(x_data, y_data, title):
    plt.figure()
    plt.plot(x_data, y_data)
    plt.title(title)
    plt.savefig('./generators/learning/{}.png'.format(title))


def main():
    action_type = 'place'
    region = 'loading_region'
    model = PlaceWGANgp(region)

    trainloader, trainset, testset = get_data_generator(action_type, region)
    iterations = range(100, 94000, 100)

    # todo
    #   write a visualization script for pytorch outputs
    #   save the results and load it if necessary
    if os.path.isfile('./generators/learning/results.pkl'):
        results = pickle.load(open('./generators/learning/results.pkl', 'r'))
    else:
        results = np.array([model.evaluate_generator(testset, iteration=iteration) for iteration in iterations])
        pickle.dump(results, open('./generators/learning/results.pkl', 'wb'))

    iterations = np.array(iterations)
    in_bound_idxs = results[:, 2] != np.inf
    results = results[in_bound_idxs, :]
    iterations = iterations[in_bound_idxs]
    print "Min MSE", iterations[np.argsort(results[:, 0])][0:50]
    print "KDE scores", iterations[np.argsort(results[:, 1])][::-1][0:50]
    print "Entropies", iterations[np.argsort(results[:, 2])][::-1][0:50]

    plot(iterations, results[:, 0], 'Min MSEs')
    plot(iterations, results[:, 1], 'KDE scores')
    plot(iterations, results[:, 2], 'Entropies')


if __name__ == '__main__':
    main()
