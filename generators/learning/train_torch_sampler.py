import socket
import torch
import argparse
import os
import pickle

from datasets.GeneratorDataset import StandardDataset
from generators.learning.learning_algorithms.WGANGP import WGANgp

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp_light/learned_weights/'


def save_weights(net, epoch, action_type, seed, region):
    net_name = net.__class__._get_name(net)
    PATH = './learning/torch_weights/atype_%s_%s_region_%s_seed_%d_epoch_%d.pt' % (
    action_type, net_name, region, seed, epoch)
    torch.save(net.state_dict(), PATH)


def get_data_generator(action_type, region, seed):
    dataset = StandardDataset(action_type, region, True, seed)
    n_train = int(len(dataset) * 0.9)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20,
                                              pin_memory=True)
    test_idx_dir = './generators/datasets/testset_cache_file_idxs/'
    if not os.path.isdir(test_idx_dir):
        os.makedirs(test_idx_dir) 
    test_idx_file = 'seed_{}_atype_{}_region_{}.pkl'.format(seed, action_type, region)
    pickle.dump(testset.indices, open(test_idx_dir+test_idx_file,'wb'))
    return trainloader, trainset, testset


def main():
    parser = argparse.ArgumentParser("Sampler training")
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-architecture', type=str, default='fc')
    config = parser.parse_args()

    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)

    model = WGANgp(config.atype, config.region, config.architecture)

    trainloader, trainset, testset = get_data_generator(config.atype, config.region, config.seed)
    n_train = len(trainset)
    model.train(trainloader, testset, n_train)


if __name__ == '__main__':
    main()
