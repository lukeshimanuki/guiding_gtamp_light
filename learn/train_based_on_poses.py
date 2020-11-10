#!/bin/env python2

import argparse
import random
import numpy as np
import sys
import os
import pickle
import glob

from learn.pose_based_models.fc import FullyConnected
from data_traj import get_actions
import csv


def top_k_accuracy(q_model, nodes, edges, actions, k):
    print "Testing on %d number of data" % len(nodes)
    """
    q_target_action = q_model.predict_with_raw_input_format(nodes, edges, actions)
    n_data = len(nodes)
    q_all_actions = q_model.alt_msg_layer.predict([nodes, edges, actions])
    accuracy = []
    top_zero_accuracy = []
    top_one_accuracy = []
    top_two_accuracy = []
    for i in range(n_data):
        n_actions_bigger_than_target = np.sum(q_target_action[i] < q_all_actions[i])
        accuracy.append(n_actions_bigger_than_target <= k)
        top_zero_accuracy.append(n_actions_bigger_than_target == 0)
        top_one_accuracy.append(n_actions_bigger_than_target <= 1)
        top_two_accuracy.append(n_actions_bigger_than_target <= 2)

    return np.mean(accuracy), np.mean(top_zero_accuracy), np.mean(top_one_accuracy), np.mean(top_two_accuracy)
    """
    raise NotImplementedError


def create_model(config):
    if 'two_arm' in config.domain:
        m = FullyConnected(num_actions=8)
    else:
        raise NotImplementedError
    if os.path.isfile(m.weight_file_name) and not config.donttrain and not config.f:
        print "Quitting because we've already trained with the given configuration"
        sys.exit(-1)
    return m


def create_train_data(inputs, targets, num_training):
    training_inputs = inputs[:num_training]
    training_targets = targets[:num_training]
    return training_inputs, training_targets


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


def train(config):
    data_path = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/rsc/trajectory_data/shortest/'
    pose_based_states, actions = load_data(data_path, config.num_train+config.num_test)

    num_training = config.num_train
    assert num_training > 0
    config.num_train = num_training
    model = create_model(config)
    training_inputs, training_targets = create_train_data(pose_based_states, actions, num_training)
    model.load_weights()

    """
    num_test = len(nodes) - num_training
    config.num_test = num_test
    _, post_top_zero_acc, post_top_one_acc, post_top_two_acc = top_k_accuracy(model, t_inputs, t_targets, config.top_k)

    # write_test_results_in_csv(post_top_zero_acc, post_top_one_acc, post_top_two_acc, seed, num_training, config.loss)
    print "Post-training top-0 accuracy %.2f" % post_top_zero_acc
    print "Post-training top-1 accuracy %.2f" % post_top_one_acc
    print "Post-training top-2 accuracy %.2f" % post_top_two_acc
    """


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

    donttrain = configs.donttrain
    train(configs)
