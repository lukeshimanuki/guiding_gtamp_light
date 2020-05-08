from generators.learning.learning_algorithms.WGANGP import WGANgp
from generators.learning.train_torch_sampler import get_data_generator
from generators.learning.datasets.GeneratorDataset import StandardDataset

import pickle
import os
import argparse


def get_data_generator(action_type, region):
    dataset = StandardDataset(action_type, region, True, is_testing=True)
    batch_size = 32
    return dataset

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
        testset = get_data_generator(config.atype, config.region)
        result = model.evaluate_generator(testset, iteration=config.iteration)
        pickle.dump(result, open(fdir + fname, 'wb'))


if __name__ == '__main__':
    main()
