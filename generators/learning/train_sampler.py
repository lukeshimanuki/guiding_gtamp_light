import argparse
import os
import pickle
import numpy as np
import random
import socket

from generators.learning.utils.model_creation_utils import create_policy


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-algo', type=str, default='mse')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-tau', type=float, default=0.0)
    parser.add_argument('-dtype', type=str, default='n_objs_pack_4')
    args = parser.parse_args()
    return args


configs = parse_args()
np.random.seed(configs.seed)
random.seed(configs.seed)
os.environ['PYTHONHASHSEED'] = str(configs.seed)

import tensorflow as tf

tf.set_random_seed(configs.seed)

from PlacePolicyMSEFeedForward import PlacePolicyMSEFeedForward
from PlacePolicyMSEFeedForwardWithoutKeyConfig import PlacePolicyMSEFeedForwardWithoutKeyConfig
from PlacePolicyMSESelfAttention import PlacePolicyMSESelfAttention
from PlacePolicyMSESelfAttentionDenseEvalNet import PlacePolicyMSESelfAttentionDenseEvalNet
from PlacePolicyMSESelfAttentionDenseGenNetDenseEvalNet import PlacePolicyMSESelfAttentionDenseGenNetDenseEvalNet
from utils.data_processing_utils import get_processed_poses_from_state, get_processed_poses_from_action, \
    state_data_mode, action_data_mode, make_konfs_relative_to_pose
from utils import data_processing_utils
from gtamp_utils import utils


def load_data(traj_dir):
    traj_files = os.listdir(traj_dir)
    cache_file_name = 'no_collision_at_target_obj_poses_cache_state_data_mode_%s_action_data_mode_%s_loading_region_only.pkl' % (
        state_data_mode, action_data_mode)
    # cache_file_name = 'cache_state_data_mode_%s_action_data_mode_%s.pkl' % (state_data_mode, action_data_mode)
    if os.path.isfile(traj_dir + cache_file_name):
        print "Loading the cache file", traj_dir + cache_file_name
        return pickle.load(open(traj_dir + cache_file_name, 'r'))

    print 'caching file...'
    all_states = []
    all_actions = []
    all_sum_rewards = []
    all_poses = []
    all_rel_konfs = []
    all_konf_relevance = []

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)

    for traj_file in traj_files:
        if 'pidx' not in traj_file:
            continue
        if 'no_collision_at' not in traj_file:
            continue
        traj = pickle.load(open(traj_dir + traj_file, 'r'))
        if len(traj.states) == 0:
            continue

        states = []
        poses = []
        actions = []
        traj_rel_konfs = []
        konf_relevance = []
        for s, a in zip(traj.states, traj.actions):
            state_vec = s.collision_vector
            n_key_configs = state_vec.shape[1]

            is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([s.obj in s.goal_entities]))
            is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
            is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([s.region in s.goal_entities]))
            is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))

            filter_movements_to_goal_region = s.region in s.goal_entities
            if filter_movements_to_goal_region:
                continue
            state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)
            states.append(state_vec)
            poses.append(get_processed_poses_from_state(s, a))
            actions.append(get_processed_poses_from_action(s, a))
            rel_konfs = make_konfs_relative_to_pose(s.abs_obj_pose, key_configs)

            traj_rel_konfs.append(np.array(rel_konfs).reshape((1, 615, 4, 1)))

            place_motion = a['place_motion']
            binary_collision_vector = s.collision_vector.squeeze()[:, 0]
            place_relevance = data_processing_utils.get_relevance_info(key_configs, binary_collision_vector,
                                                                       place_motion)
            konf_relevance.append(place_relevance)

            pick_motion = a['pick_motion']
            # pick_relevance = data_processing_utils.get_relevance_info(key_configs, binary_collision_vector, pick_motion)

        states = np.array(states)
        poses = np.array(poses)
        actions = np.array(actions)
        traj_rel_konfs = np.array(traj_rel_konfs)

        rewards = traj.rewards
        sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])
        if len(states) == 0:
            continue
        all_poses.append(poses)
        all_states.append(states)
        all_actions.append(actions)
        all_sum_rewards.append(sum_rewards)
        all_rel_konfs.append(traj_rel_konfs)
        all_konf_relevance.append(konf_relevance)

        n_data = len(np.vstack(all_rel_konfs))
        assert len(np.vstack(all_states)) == n_data
        print n_data
        if n_data > 10:
            break
    all_rel_konfs = np.vstack(all_rel_konfs).squeeze(axis=1)
    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions).squeeze()
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    all_poses = np.vstack(all_poses).squeeze()
    all_konf_relevance = np.vstack(all_konf_relevance).squeeze()
    pickle.dump((all_states, all_konf_relevance, all_poses, all_rel_konfs, all_actions, all_sum_rewards),
                open(traj_dir + cache_file_name, 'wb'))
    return all_states, all_konf_relevance, all_poses, all_rel_konfs, all_actions, all_sum_rewards[:, None]


def get_data(datatype):
    if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/planning_experience/processed/'

    if datatype == 'n_objs_pack_4':
        data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_4/sahs/sampler_trajectory_data/'
    else:
        data_dir = '/planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/sampler_trajectory_data/'

    print "Loading data from", data_dir
    states, konf_relelvance, poses, rel_konfs, actions, sum_rewards = load_data(root_dir + data_dir)
    is_goal_flag = states[:, :, 2:, :]
    states = states[:, :, :2, :]  # collision vector

    n_data = 5000
    states = states[:5000, :]
    poses = poses[:n_data, :]
    actions = actions[:5000, :]
    sum_rewards = sum_rewards[:5000]
    is_goal_flags = is_goal_flag[:5000, :]
    konf_relelvance = konf_relelvance[:5000, :]

    print "Number of data", len(states)
    return states, konf_relelvance, poses, rel_konfs, is_goal_flags, actions, sum_rewards


def train_mse_ff(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/rel_konf_place_mse/' % (
        config.dtype, state_data_mode, action_data_mode)
    policy = PlacePolicyMSEFeedForwardWithoutKeyConfig(dim_action=dim_action, dim_collision=dim_state,
                                                       save_folder=savedir, tau=config.tau, config=config)
    policy.policy_model.summary()
    states, konf_relelvance, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)
    actions = actions[:, 4:]
    poses = poses[:, :4]
    goal_flags = goal_flags[:, 0, :].squeeze()
    policy.train_policy(states, konf_relelvance, poses, rel_konfs, goal_flags, actions, sum_rewards)


def train_mse_ff_keyconfigs(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/rel_konf_place_mse/' \
              % (config.dtype, state_data_mode, action_data_mode)
    policy = PlacePolicyMSEFeedForward(dim_action=dim_action, dim_collision=dim_state,
                                       save_folder=savedir, tau=config.tau, config=config)
    policy.policy_model.summary()
    states, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)
    actions = actions[:, 4:]
    poses = poses[:, 4:8]
    goal_flags = goal_flags[:, 0, :].squeeze()
    policy.train_policy(states, poses, rel_konfs, goal_flags, actions, sum_rewards)


def train_mse_selfattention_conv_evalnet(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/rel_konf_place_mse/' \
              % (config.dtype, state_data_mode, action_data_mode)
    policy = PlacePolicyMSESelfAttention(dim_action=dim_action, dim_collision=dim_state,
                                         save_folder=savedir, tau=config.tau, config=config)
    policy.policy_model.summary()
    states, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)
    actions = actions[:, 4:]
    poses = poses[:, 4:8]

    policy.train_policy(states, poses, rel_konfs, goal_flags, actions, sum_rewards)


def train_mse_selfattention_dense_evalnet(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/' \
              'selfattention_dense_evalnet/' \
              % (config.dtype, state_data_mode, action_data_mode)
    policy = PlacePolicyMSESelfAttentionDenseEvalNet(dim_action=dim_action, dim_collision=dim_state,
                                                     save_folder=savedir, tau=config.tau, config=config)
    policy.policy_model.summary()
    states, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)
    actions = actions[:, 4:]
    poses = poses[:, 0:20]
    policy.train_policy(states, poses, rel_konfs, goal_flags, actions, sum_rewards)


def train_mse_selfattention_dense_gennet_dense_evalnet(config):
    n_key_configs = 615
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/' \
              'selfattention_dense_gennet_dense_evalnet/' \
              % (config.dtype, state_data_mode, action_data_mode)
    policy = PlacePolicyMSESelfAttentionDenseGenNetDenseEvalNet(dim_action=dim_action, dim_collision=dim_state,
                                                                save_folder=savedir, tau=config.tau, config=config)
    policy.policy_model.summary()
    states, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)

    actions = actions[:, 4:]
    poses = poses[:, 0:20]  # use the object pose to inform the collision net

    policy.train_policy(states, poses, rel_konfs, goal_flags, actions, sum_rewards)


def train(config):
    policy = create_policy(config)
    policy.policy_model.summary()
    states, konf_relevance, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)
    actions = np.array([utils.encode_pose_with_sin_and_cos_angle(a[3:]) for a in actions])
    poses = poses[:, 0:4]

    """
    #policy.load_weights()
    q0qg_vals_b4_sm = policy.q0_qg_b4_sm.predict([goal_flags, rel_konfs, states, poses]).squeeze()[0]
    q0qg_vals_after_sm = policy.q0_qg_after_sm.predict([goal_flags, rel_konfs, states, poses]).squeeze()[0]
    concat = policy.concat.predict([goal_flags, rel_konfs, states, poses]).squeeze()[0]
    evalnet_b4 = policy.evalnet_b4_sm.predict([goal_flags, rel_konfs, states, poses]).squeeze()[0]
    evalnet = policy.evalnet.predict([goal_flags, rel_konfs, states, poses]).squeeze()[0]

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    key_configs = key_configs.reshape((1, 615, 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    """

    def noise(z_size):
        return np.random.normal(size=z_size).astype('float32')

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])

    # to delete: 399, 274, 295, 297, 332, 352, 409, 410, 411, 412, 461, 488,
    xmin = -0.7; xmax = 4.3
    ymin = -8.55; ymax = -4.85
    indices_to_delete = np.hstack([np.where(key_configs[:, 1] > ymax)[0], np.where(key_configs[:, 1] < ymin)[0],
                                   np.where(key_configs[:, 0] > xmax)[0], np.where(key_configs[:, 0] < xmin)[0]])
    key_configs = np.delete(key_configs, indices_to_delete, axis=0)
    states = np.delete(states, indices_to_delete, axis=1)
    n_key_configs = len(key_configs)
    key_configs = key_configs.reshape((1, n_key_configs, 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    goal_flags=np.delete(goal_flags, indices_to_delete, axis=1)

    """
    noises = noise((len(goal_flags), 4))
    inp = [goal_flags[0:2, :], key_configs[0:2, :], states[0:2, :], poses[0:2, :], noises[0:2, :]]
    scores = policy.score_function.predict(inp)
    relval = policy.relevance_net.predict(inp)
    collision_diff = policy.collision_diff.predict(inp)
    losses = policy.loss_model.predict(inp)
    relevance = policy.relevance_loss_model.predict(inp).squeeze()
    true_konf_rel = konf_relevance[0:2,:]

    loss = []
    for p, t in zip(relevance, true_konf_rel):
        loss.append([trel * np.log(pred_rel) + (1-trel)*np.log(1-pred_rel) for trel, pred_rel in zip(p, t)])
    import pdb;pdb.set_trace()
    rel_loss_eval = policy.relevance_loss_model.evaluate(inp, konf_relevance[0:2, :])
    """


    # actions = policy.policy_model.predict(inp)
    # evaluated_loss = policy.loss_model.evaluate(inp, [actions[0:10]])
    # collision_loss = policy.collision_loss_model.predict(inp)
    policy.train_policy(states, konf_relevance, poses, key_configs, goal_flags, actions, sum_rewards)


def main():
    if configs.algo == 'ff':
        train_mse_ff(configs)
    elif configs.algo == 'ff_collision':
        train_mse_ff_keyconfigs(configs)
    elif configs.algo == 'sa_conv_evalnet':
        train_mse_selfattention_conv_evalnet(configs)
    elif configs.algo == 'sa_dense_gennet_dense_evalnet':
        train_mse_selfattention_dense_gennet_dense_evalnet(configs)
    else:
        train(configs)


if __name__ == '__main__':
    main()
