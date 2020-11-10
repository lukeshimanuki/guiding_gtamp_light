#!/bin/env python2

import argparse
import random
import numpy as np
import sys
import os
import pickle
import torch
import glob
import torch.optim as optim

from datasets.dataset import PoseBasedDataset
from learn.pose_based_models.fc import FullyConnected
from data_traj import get_actions
import csv


def create_model(config):
    m = FullyConnected(config)
    # if os.path.isfile(m.weight_file_name) and not config.donttrain and not config.f:
    #    print "Quitting because we've already trained with the given configuration"
    #    sys.exit(-1)
    return m


def create_dataset(inputs, targets, num_training):
    train_inputs = np.array(inputs[:num_training])
    train_targets = np.array(targets[:num_training])
    train_dataset = PoseBasedDataset(train_inputs, train_targets)

    test_inputs = np.array(inputs[num_training:])
    test_targets = np.array(targets[num_training:])
    test_dataset = PoseBasedDataset(test_inputs, test_targets)

    return train_dataset, test_dataset


def load_data(dirname, num_data, desired_operator_type='two_arm_pick'):
    cachefile = "{}{}-pose_based_num_data_{}_retired.pkl".format(dirname, desired_operator_type, num_data)
    # cachefile = './planning_experience/two_arm_pick_two_arm_place_before_submission.pkl'
    if os.path.isfile(cachefile):
        print "Loading the cached file:", cachefile
        return pickle.load(open(cachefile, 'rb'))
    print "Caching file..."
    file_list = glob.glob("{}/pap_traj_*.pkl".format(dirname))

    pose_based_states = []
    actions = []
    obj_list = [
        u'rectangular_packing_box4', u'rectangular_packing_box2', u'rectangular_packing_box3',
        u'rectangular_packing_box1', u'square_packing_box2', u'square_packing_box3', u'square_packing_box1',
        u'square_packing_box4'
    ]
    n_traj = 0
    for traj_data in file_list:
        data = pickle.load(open(traj_data, 'r'))
        assert len(data.states) == len(data.actions)

        for state, op_instance in zip(data.states, data.actions):
            # state processing
            pose_based_state = []
            for obj_name in obj_list:
                pose = state.nodes[obj_name][3:6]
                pose_based_state += pose
            robot_pose = state.robot_pose.squeeze()
            pose_based_state += robot_pose.tolist()
            pose_based_states.append(pose_based_state)

            # action processing
            entity_names = list(state.nodes.keys())[::-1]
            action = get_actions(op_instance, entity_names)
            actions.append(action)
        n_traj += 1
        n_data = len(np.vstack(pose_based_states))

        print "n episodes included {}/{} n_data {}".format(n_traj, len(file_list), n_data)
        if n_data >= num_data:
            break

    data = (pose_based_states, actions)
    pickle.dump(data, open(cachefile, 'wb'))
    return data


def train(config, train_dataloader, test_dataloader):
    model = create_model(config)

    optimizerD = optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

    dataset = test_dataloader.dataset[:]
    te_inputs = torch.from_numpy(dataset['inputs']).float()
    te_targets = torch.from_numpy(dataset['targets']).float()

    def data_generator():
        while True:
            for d in train_dataloader:
                yield d

    data_gen = data_generator()

    n_iters_per_epoch = 5000 / train_dataloader.batch_size
    n_epochs = 1000
    n_iters = n_iters_per_epoch * n_epochs
    k = 1
    best_performance = -np.inf
    patience = 0
    for i in range(n_iters):
        _data = data_gen.next()
        inputs = _data['inputs'].float()
        targets = _data['targets'].float()

        model.zero_grad()

        # how can I test if this is doing it correctly?

        # compute the loss
        pred = model(inputs)
        target_action_value = torch.sum(torch.sum(pred * targets, axis=-1), axis=-1)
        alternate_action_values = pred * (1 - targets) + -2e32 * targets
        max_of_alternate_action_values = torch.max(torch.max(alternate_action_values, axis=-1)[0], axis=-1)[0]
        action_value_delta = target_action_value - max_of_alternate_action_values
        action_ranking_cost = 1 - action_value_delta
        hinge_loss = torch.mean(torch.max(torch.zeros(action_ranking_cost.size()), action_ranking_cost))

        hinge_loss.backward()
        optimizerD.step()

        # compute the ranking loss - what is the top k rank?
        if i % n_iters_per_epoch == 0:
            te_pred = model(te_inputs)
            target_val = torch.sum(torch.sum(te_targets * te_pred, axis=-1), axis=-1)
            te_pred = te_pred.reshape((len(te_pred), 22))
            top_k_values = torch.sort(te_pred, -1)[0][:, -k:]
            target_values_repeated = target_val.reshape((len(target_val), 1)).repeat((1, k))

            target_value_in_top_k = torch.sum(target_values_repeated == top_k_values, axis=-1)
            percentage_in_top_k = torch.mean(target_value_in_top_k.float())
            print 'top ' + str(k) + ' loss', percentage_in_top_k

            if percentage_in_top_k > best_performance:
                torch.save(model.state_dict(), model.weight_file_name)
                best_performance = percentage_in_top_k
                patience = 0
            else:
                patience += 1

            if patience >= 50:
                print "Patience reached. Best performance is",best_performance
                break


def write_test_results_in_csv(top0, top1, top2, seed, num_training, loss_fcn):
    with open('./learn/top_k_results/' + loss_fcn + '_seed_' + str(seed) + '.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([num_training, top0, top1, top2])


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_hidden', type=int, default=32)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-num_test', type=int, default=1000)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-val_portion', type=float, default=0.1)
    parser.add_argument('-top_k', type=int, default=1)
    parser.add_argument('-donttrain', action='store_true', default=False)
    parser.add_argument('-same_vertex_model', action='store_true', default=False)
    parser.add_argument('-diff_weight_msg_passing', action='store_true', default=False)
    parser.add_argument('-operator', type=str, default='two_arm_pick_two_arm_place')
    parser.add_argument('-num_fc_layers', type=int, default=2)
    parser.add_argument('-no_goal_nodes', action='store_true', default=False)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-n_msg_passing', type=int, default=1)
    parser.add_argument('-weight_initializer', type=str, default='glorot_uniform')
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-mse_weight', type=float, default=0.0)
    parser.add_argument('-statetype', type=str, default='shortest')
    configs = parser.parse_args()
    return configs


if __name__ == '__main__':
    configs = parse_args()
    np.random.seed(configs.seed)
    random.seed(configs.seed)

    data_path = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/rsc/trajectory_data/shortest/'
    pose_based_states, actions = load_data(data_path, configs.num_train + configs.num_test)
    assert configs.num_train > 0
    train_dataset, test_dataset = create_dataset(pose_based_states, actions, configs.num_train)

    batch_size = 32
    num_workers = 10
    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, pin_memory=False)
    test_dataloder = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers, pin_memory=False)

    donttrain = configs.donttrain
    train(configs, train_dataloder, test_dataloder)
