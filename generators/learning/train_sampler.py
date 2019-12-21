import argparse
import os
import pickle
import numpy as np
import random
import socket

from generators.learning.utils.model_creation_utils import create_policy
from test_scripts.visualize_learned_sampler import create_environment


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-algo', type=str, default='mse')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-tau', type=float, default=0.0)
    parser.add_argument('-dtype', type=str, default='n_objs_pack_4')
    parser.add_argument('-atype', type=str, default='pick')
    args = parser.parse_args()
    return args


configs = parse_args()
np.random.seed(configs.seed)
random.seed(configs.seed)
os.environ['PYTHONHASHSEED'] = str(configs.seed)

import tensorflow as tf

tf.set_random_seed(configs.seed)

from utils.data_processing_utils import get_processed_poses_from_state, get_processed_poses_from_action, \
    state_data_mode, action_data_mode, make_konfs_relative_to_pose
from utils import data_processing_utils
from gtamp_utils import utils


def load_data(traj_dir):
    traj_files = os.listdir(traj_dir)
    cache_file_name = 'no_collision_at_target_obj_state_data_mode_%s_action_data_mode_%s_loading_region_only.pkl' % (
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
    all_paths = []

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
        place_paths = []
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
            place_paths.append(place_motion)
            binary_collision_vector = s.collision_vector.squeeze()[:, 0]
            place_relevance = data_processing_utils.get_relevance_info(key_configs, binary_collision_vector,
                                                                       place_motion)

            """
            problem_env, openrave_env = create_environment(0)
            init = [utils.decode_pose_with_sin_and_cos_angle(poses[0][-4:])]
            utils.set_robot_config(init)
            utils.visualize_path(key_configs[place_relevance, :])
            """

            konf_relevance.append(place_relevance)

            pick_motion = a['pick_motion']
            # pick_relevance = data_processing_utils.get_relevance_info(key_configs, binary_collision_vector, pick_motion)

        states = np.array(states)
        poses = np.array(poses)
        actions = np.array(actions)
        traj_rel_konfs = np.array(traj_rel_konfs)

        rewards = traj.rewards
        sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])
        print sum_rewards, traj_file
        if len(states) == 0:
            continue
        all_poses.append(poses)
        all_states.append(states)
        all_actions.append(actions)
        all_sum_rewards.append(sum_rewards)
        all_rel_konfs.append(traj_rel_konfs)
        all_konf_relevance.append(konf_relevance)
        all_paths.append(place_paths)

        n_data = len(np.vstack(all_rel_konfs))
        assert len(np.vstack(all_states)) == n_data
        if n_data > 5000:
            break
    all_rel_konfs = np.vstack(all_rel_konfs).squeeze(axis=1)
    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions).squeeze()
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    all_poses = np.vstack(all_poses).squeeze()
    all_konf_relevance = np.vstack(all_konf_relevance).squeeze()
    pickle.dump((all_states, all_konf_relevance, all_poses, all_rel_konfs, all_actions, all_sum_rewards, all_paths),
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
    states, konf_relelvance, poses, rel_konfs, actions, sum_rewards, paths = load_data(root_dir + data_dir)
    is_goal_flag = states[:, :, 2:, :]
    states = states[:, :, :2, :]  # collision vector

    n_data = 5000
    states = states[:5000, :]
    poses = poses[:n_data, :]
    actions = actions[:5000, :]
    sum_rewards = sum_rewards[:5000]
    is_goal_flags = is_goal_flag[:5000, :]
    konf_relelvance = konf_relelvance[:5000, :]

    return states, konf_relelvance, poses, rel_konfs, is_goal_flags, actions, sum_rewards


def train(config):
    policy = create_policy(config)
    policy.policy_model.summary()
    states, konf_relevance, poses, rel_konfs, goal_flags, actions, sum_rewards = get_data(config.dtype)
    if config.atype == 'pick':
        actions = np.array([utils.encode_pose_with_sin_and_cos_angle(a[0:3]) for a in actions])
    elif config.atype == 'place':
        actions = np.array([utils.encode_pose_with_sin_and_cos_angle(a[3:]) for a in actions])
    else:
        raise NotImplementedError
    if config.atype == 'pick':
        poses = poses[:, :-4]  # It is currently: [obj_pose, goal_obj_poses, robot_pick_pose].
    elif config.atype == 'place':
        # todo use obj_pose, goal_obj_poses, q_pick
        poses = poses[:, :-4]  # It is currently: [obj_pose, goal_obj_poses, robot_pick_pose].
    else:
        raise NotImplementedError

    ####
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])

    # to delete: 399, 274, 295, 297, 332, 352, 409, 410, 411, 412, 461, 488,
    xmin = -0.7
    xmax = 4.3
    ymin = -8.55
    ymax = -4.85
    indices_to_delete = np.hstack([np.where(key_configs[:, 1] > ymax)[0], np.where(key_configs[:, 1] < ymin)[0],
                                   np.where(key_configs[:, 0] > xmax)[0], np.where(key_configs[:, 0] < xmin)[0]])
    key_configs = np.delete(key_configs, indices_to_delete, axis=0)
    states = np.delete(states, indices_to_delete, axis=1)
    konf_relevance = np.delete(konf_relevance, indices_to_delete, axis=1)

    n_key_configs = len(key_configs)
    key_configs = key_configs.reshape((1, n_key_configs, 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)
    ###

    print "Number of data", len(states)
    policy.train_policy(states, konf_relevance, poses, key_configs, goal_flags, actions, sum_rewards)


def main():
    train(configs)


if __name__ == '__main__':
    main()
