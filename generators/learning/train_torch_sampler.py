import torch
import argparse

from datasets.GeneratorDataset import StandardDataset, ImportanceEstimatorDataset
from generators.learning.learning_algorithms.WGANGP import WGANgp
from learning_algorithms.ImportanceWeightEstimation import ImportanceWeightEstimation
import numpy as np

ROOTDIR = './'


def get_data_generator(config):
    if config.train_type == 'w':
        dataset = ImportanceEstimatorDataset(config, True, is_testing=False)
    else:
        dataset = StandardDataset(config, True, is_testing=False)
    n_train = int(len(dataset) * 0.9)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20,
                                              pin_memory=True)
    n_test = len(testset.indices)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=n_test, shuffle=True, num_workers=20,
                                             pin_memory=True)
    print "number of training data", n_train

    return trainloader, testloader, trainset, testset


def get_w_data(config):
    batch_size = 32
    dataset = ImportanceEstimatorDataset(config, True, is_testing=False)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=20,
                                              pin_memory=True)
    return trainloader, dataset


def get_wgandi_data(config, w_model):
    dataset = ImportanceEstimatorDataset(config, True, is_testing=False)
    actions = torch.from_numpy(dataset.actions).float()
    konf_obsts = torch.from_numpy(dataset.konf_obsts).float()
    poses = torch.from_numpy(dataset.poses).float()

    # getting neutral dataset
    neu_idxs = dataset.labels != 1
    neu_actions = actions[neu_idxs]
    neu_konf_obsts = konf_obsts[neu_idxs]
    neu_poses = poses[neu_idxs]

    # computing importance values of neutral actions
    w_values = w_model.predict(neu_actions, neu_konf_obsts, neu_poses).detach()
    w_values[w_values < 0] = 0
    prob_of_data = (w_values / torch.sum(w_values)).cpu().numpy()

    # sampling neutral data according to their w values
    n_neu_data = len(neu_actions)
    data_idxs = range(n_neu_data)
    chosen = np.random.choice(data_idxs, n_neu_data, p=prob_of_data.squeeze())
    chosen_neu_actions = neu_actions[chosen]
    chosen_neu_konf_obsts = neu_konf_obsts[chosen]
    chosen_neu_poses = neu_poses[chosen]

    # concatenating the sampled neutral data with positive data
    pos_idxs = dataset.labels == 1
    pos_actions = actions[pos_idxs]
    pos_konf_obsts = konf_obsts[pos_idxs]
    pos_poses = poses[pos_idxs]
    dataset.actions = torch.cat([pos_actions, chosen_neu_actions])
    dataset.poses = torch.cat([pos_poses, chosen_neu_poses])
    dataset.konf_obsts = torch.cat([pos_konf_obsts, chosen_neu_konf_obsts])

    n_train = int(len(dataset) * 0.9)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20,
                                              pin_memory=True)
    n_test = len(testset.indices)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=n_test, shuffle=True, num_workers=20,
                                             pin_memory=True)
    print "number of training data", n_train
    return trainloader, testloader, trainset, testset


def main():
    parser = argparse.ArgumentParser("Sampler training")
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-atype', type=str, default='pick')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-architecture', type=str, default='fc')
    parser.add_argument('-num_episode', type=int, default=1000)
    parser.add_argument('-train_type', type=str, default='wgandi')
    config = parser.parse_args()

    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    if config.train_type == 'w':
        model = ImportanceWeightEstimation(config)
        trainloader, trainset = get_w_data(config)
        testloader = None
    elif config.train_type == 'wgandi':
        w_model = ImportanceWeightEstimation(config)
        w_model.load_weights()
        model = WGANgp(config)
        trainloader, testloader, trainset, testset = get_wgandi_data(config, w_model)
    elif config.train_type == 'wgangp':
        model = WGANgp(config)
        trainloader, testloader, trainset, testset = get_data_generator(config)
    else:
        raise NotImplementedError

    n_train = len(trainset)
    model.train(trainloader, testloader, n_train)


if __name__ == '__main__':
    main()
