import torch
import argparse

from datasets.GeneratorDataset import StandardDataset, ImportanceEstimatorDataset, GivenDataset
from generators.learning.learning_algorithms.WGANGP import WGANgp
from generators.learning.learning_algorithms.ActorCritic import ActorCritic
from learning_algorithms.ImportanceWeightEstimation import ImportanceWeightEstimation
import numpy as np
import os

ROOTDIR = './'


def get_data_generator(config):
    if config.train_type == 'w':
        dataset = ImportanceEstimatorDataset(config, True, is_testing=False)
    else:
        dataset = StandardDataset(config, True, is_testing=False)
    n_train = int(len(dataset) * 0.7)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    batch_size = 32

    batch_size = min(32, int(0.1*len(trainset)))
    #num_workers = 1 if batch_size < 10 else 10
    num_workers = 10
    print "Batch size {} num workers {}".format(batch_size, num_workers)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    n_test = len(testset.indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=n_test, shuffle=True, num_workers=num_workers,
                                             pin_memory=False)
    print "number of training data", n_train
    print "number of test data", n_test
    return trainloader, testloader, trainset, testset


def get_w_data(config):
    dataset = ImportanceEstimatorDataset(config, True, is_testing=False)
    batch_size = min(32, int(0.1*len(dataset)))
    num_workers = 1 if batch_size < 32 else 10
    print 'Batch size', batch_size
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)
    return trainloader, dataset


def get_wgandi_data(config, w_model):
    dataset = ImportanceEstimatorDataset(config, True, is_testing=False)
    actions = torch.from_numpy(dataset.actions).float()
    konf_obsts = torch.from_numpy(dataset.konf_obsts).float()
    poses = torch.from_numpy(dataset.poses).float()
    labels = torch.from_numpy(dataset.labels).float()

    # split the positive data into train and test sets
    pos_set = GivenDataset(actions[labels==1], konf_obsts[labels==1], poses[labels==1])
    n_train = int(len(pos_set) * 0.7)
    trainset, testset = torch.utils.data.random_split(pos_set, [n_train, len(pos_set) - n_train])
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=1,
                                             pin_memory=True)

    # get only the positive dataset, and merge with neutral dataset
    pos_actions = trainset.dataset[trainset.indices]
    neu_idxs = labels!=1
    tr_actions = torch.cat([actions[neu_idxs], trainset.dataset[trainset.indices]['actions']])
    tr_konf_obsts = torch.cat([konf_obsts[neu_idxs], trainset.dataset[trainset.indices]['konf_obsts']])
    tr_poses = torch.cat([poses[neu_idxs], trainset.dataset[trainset.indices]['poses']])
    tr_labels = torch.cat([torch.zeros(poses[neu_idxs].shape[0],1), torch.ones(len(trainset.indices),1)])

    dataset.actions = tr_actions.numpy()
    dataset.konf_obsts = tr_konf_obsts.numpy()
    dataset.poses = tr_poses.numpy()
    dataset.labels = tr_labels.numpy()
    actions = torch.from_numpy(dataset.actions).float()
    konf_obsts = torch.from_numpy(dataset.konf_obsts).float()
    poses = torch.from_numpy(dataset.poses).float()

    # Compute data of positive and neutral data
    w_values = w_model.predict(actions, konf_obsts, poses).detach()
    w_values[w_values < 0] = 0
    prob_of_data = (w_values / torch.sum(w_values)).cpu().numpy()

    # sampling data according to their w values
    n_data = len(actions)
    data_idxs = range(n_data)
    chosen = np.random.choice(data_idxs, n_data, p=prob_of_data.squeeze())
    chosen_actions = actions[chosen]
    chosen_konf_obsts = konf_obsts[chosen]
    chosen_poses = poses[chosen]
    pos_idxs = np.array(data_idxs)[dataset.labels.squeeze()==1]
    neu_idxs = np.array(data_idxs)[dataset.labels.squeeze()!=1]
    print 'n pos chosen', len([c for c in chosen if c in pos_idxs])
    print 'n neu chosen', len([c for c in chosen if c in neu_idxs])
    print 'n unique pos', len(np.unique([c for c in chosen if c in pos_idxs]))
    print 'n unique neg', len(np.unique([c for c in chosen if c in neu_idxs]))

    pos_w_values = w_model.predict(actions, konf_obsts, poses)[pos_idxs].detach().cpu().numpy()
    neu_w_values = w_model.predict(actions, konf_obsts, poses)[neu_idxs].detach().cpu().numpy()
    print 'num pos data', len(pos_idxs)
    print 'num pos data with neg w vals', len(pos_w_values[pos_w_values<=0])

    excluded_pos_idxs = [i for i in pos_idxs if i not in chosen] 
    print 'n excluded pos data', len(excluded_pos_idxs)
    to_add = chosen_actions

    dataset.actions = torch.cat([chosen_actions, actions[excluded_pos_idxs]])
    dataset.poses = torch.cat([chosen_poses, poses[excluded_pos_idxs]])
    dataset.konf_obsts = torch.cat([chosen_konf_obsts, konf_obsts[excluded_pos_idxs]])
    dataset.labels = torch.ones(dataset.poses.shape[0],1)
    trainset = dataset
    batch_size = min(32, int(0.1*len(dataset)))
    num_workers = 1 if batch_size < 32 else 10
    num_workers = 1
    print "Batch size {} num workers {}".format(batch_size, num_workers)
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)
    print "number of training data", len(dataset)
    return trainloader, testloader, trainset, testset


def main():
    parser = argparse.ArgumentParser("Sampler training")
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-atype', type=str, default='pick')
    parser.add_argument('-region', type=str, default='home_region')
    parser.add_argument('-state_mode', type=str, default='absolute')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-architecture', type=str, default='fc')
    parser.add_argument('-num_episode', type=int, default=1000)
    parser.add_argument('-train_type', type=str, default='wgangp')
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-patience', type=int, default=20)
    parser.add_argument('-wclip', type=int, default=10)
    config = parser.parse_args()

    achieved_counter = 0
    for num_episode in [config.num_episode]:
        config.num_episode = num_episode
        for seed in range(4):
            print "****NUM EPISODE {} SEED {}*****".format(num_episode, seed)
            config.seed = seed
            torch.cuda.manual_seed_all(config.seed)
            torch.manual_seed(config.seed)
            if config.train_type == 'wgandi':
                w_model = ImportanceWeightEstimation(config)
                if len(os.listdir(w_model.weight_dir)) == 1 or config.f:
                    trainloader, trainset = get_w_data(config)
                    w_model.train(trainloader, None, len(trainset))
                w_model.load_weights()
                print "Finished training w_model! Training WGANDI..."
                model = WGANgp(config)
                trainloader, testloader, trainset, testset = get_wgandi_data(config, w_model)
            elif config.train_type == 'wgangp':
                model = WGANgp(config)
                trainloader, testloader, trainset, testset = get_data_generator(config)
            elif config.train_type == 'actorcritic':
                model = ActorCritic(config)
                trainloader, testloader, trainset, testset = get_data_generator(config)
            else:
                raise NotImplementedError

            n_train = len(trainset)

            achieved_target_kde_and_entropy = model.train(trainloader, testloader, n_train)
            if achieved_target_kde_and_entropy:
                achieved_counter += 1

            if achieved_counter >= 4:
                break


if __name__ == '__main__':
    main()
