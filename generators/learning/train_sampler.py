import os
import pickle
import numpy as np
import random
import socket
import argparse

from generators.learning.utils.model_creation_utils import create_policy
from generators.learning.utils.data_processing_utils import filter_configs_that_are_too_close
from generators.learning.utils import sampler_utils
from test_scripts.visualize_learned_sampler import create_environment

from trajectory_representation.operator import Operator


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
    parser.add_argument('-region', type=str, default='loading_region')
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


def load_data(traj_dir, action_type, desired_region):
    traj_files = os.listdir(traj_dir)
    # cache_file_name = 'no_collision_at_target_obj_poses_cache_state_data_mode_%s_action_data_mode_%s_loading_region_only.pkl' % (
    #    state_data_mode, action_data_mode)
    if action_type == 'pick':
        action_data_mode = 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base'
        cache_file_name = 'cache_smode_%s_amode_%s_atype_%s.pkl' % (state_data_mode, action_data_mode, action_type)
    else:
        action_data_mode = 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
        cache_file_name = 'cache_smode_%s_amode_%s_atype_%s_region_%s.pkl' % (state_data_mode,
                                                                              action_data_mode, action_type, desired_region)
    if os.path.isfile(traj_dir + cache_file_name):
        print "Loading the cache file", traj_dir + cache_file_name
        # return pickle.load(open(traj_dir + cache_file_name, 'r'))

    print 'caching file...%s' % cache_file_name
    all_states = []
    all_actions = []
    all_sum_rewards = []
    all_poses = []
    all_konf_relevance = []
    all_paths = []

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)

    for traj_file_idx, traj_file in enumerate(traj_files):
        if 'pidx' not in traj_file:
            continue
        if 'no_collision_at' in traj_file:
            continue
        traj = pickle.load(open(traj_dir + traj_file, 'r'))
        if len(traj.states) == 0:
            continue

        pidx = int(traj_file.split('_')[3])
        # problem_env, openrave_env = create_environment(pidx)
        # utils.viewer()
        # utils.set_robot_config([0, 0, 0])

        states = []
        poses = []
        actions = []
        traj_konfs = []
        konf_relevance = []
        place_paths = []
        rewards = np.array(traj.hvalues)[:-1] - np.array(traj.hvalues[1:])
        #rewards = np.array(traj.num_papable_to_goal[1:]) - np.array(traj.num_papable_to_goal[:-1])
        #rewards = np.array(traj.hvalues)[:-1] - np.array(traj.hvalues[1:])
        for s, a, reward in zip(traj.states, traj.actions, rewards):
            #print s, a
            if action_type == 'pick':
                state_vec = s.pick_collision_vector
            elif action_type == 'place':
                state_vec = s.place_collision_vector
            else:
                raise NotImplementedError

            n_key_configs = state_vec.shape[1]

            is_goal_obj = utils.convert_binary_vec_to_one_hot(np.array([s.obj in s.goal_entities]))
            is_goal_obj = np.tile(is_goal_obj, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))
            is_goal_region = utils.convert_binary_vec_to_one_hot(np.array([s.region in s.goal_entities]))
            is_goal_region = np.tile(is_goal_region, (n_key_configs, 1)).reshape((1, n_key_configs, 2, 1))

            is_move_to_goal_region = s.region in s.goal_entities
            if desired_region == 'home_region' and not is_move_to_goal_region:
                #utils.set_obj_xytheta(a['place_obj_abs_pose'], a['object_name'])
                continue
            if desired_region == 'loading_region' and is_move_to_goal_region:
                #utils.set_obj_xytheta(a['place_obj_abs_pose'], a['object_name'])
                continue

            if reward <= 0:
                #utils.set_obj_xytheta(a['place_obj_abs_pose'], a['object_name'])
                continue

            #utils.visualize_placements(a['place_obj_abs_pose'], a['object_name'])
            #utils.set_obj_xytheta(a['place_obj_abs_pose'], a['object_name'])

            state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)
            states.append(state_vec)
            poses.append(get_processed_poses_from_state(s, a))
            actions.append(get_processed_poses_from_action(s, a))

            place_motion = a['place_motion']
            place_paths.append(place_motion)
            pick_motion = a['pick_motion']
            binary_collision_vector = state_vec.squeeze()[:, 0]

            place_relevance = None # data_processing_utils.get_relevance_info(key_configs, binary_collision_vector, place_motion)
            pick_relevance = None # data_processing_utils.get_relevance_info(key_configs, binary_collision_vector, pick_motion)
            konf_relevance.append(place_relevance)

        states = np.array(states)
        poses = np.array(poses)
        actions = np.array(actions)

        rewards = traj.rewards
        sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])
        if len(states) == 0:
            continue
        all_poses.append(poses)
        all_states.append(states)
        all_actions.append(actions)
        all_sum_rewards.append(sum_rewards)
        all_konf_relevance.append(konf_relevance)
        all_paths.append(place_paths)

        print 'n_data %d progress %d/%d' % (len(np.vstack(all_actions)), traj_file_idx, len(traj_files))

        n_data = len(np.vstack(all_actions))
        assert len(np.vstack(all_states)) == n_data
        if n_data >= 5000:
            break

    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions).squeeze()
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    all_poses = np.vstack(all_poses).squeeze()
    all_konf_relevance = np.vstack(all_konf_relevance).squeeze()
    pickle.dump((all_states, all_konf_relevance, all_poses, all_actions, all_sum_rewards, all_paths),
                open(traj_dir + cache_file_name, 'wb'))
    return all_states, all_konf_relevance, all_poses, all_actions, all_sum_rewards[:, None]


def get_data(datatype, action_type, region):
    if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/planning_experience/processed/'

    if datatype == 'n_objs_pack_4':
        data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_4/sahs/sampler_trajectory_data/'
    else:
        data_dir = '/planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/sampler_trajectory_data/'
    print "Loading data from", data_dir
    states, konf_relelvance, poses, actions, sum_rewards = load_data(root_dir + data_dir, action_type, region)
    is_goal_flag = states[:, :, 2:, :]
    states = states[:, :, :2, :]  # collision vector

    n_data = 5000
    states = states[:5000, :]
    poses = poses[:n_data, :]
    actions = actions[:5000, :]
    sum_rewards = sum_rewards[:5000]
    is_goal_flags = is_goal_flag[:5000, :]
    konf_relelvance = konf_relelvance[:5000, :]

    return states, konf_relelvance, poses, is_goal_flags, actions, sum_rewards


def train(config):
    states, konf_relevance, poses, goal_flags, actions, sum_rewards = get_data(config.dtype, config.atype,
                                                                                          config.region)
    import pdb;pdb.set_trace()

    if config.atype == 'pick':
        actions = actions[:, :-4]
    elif config.atype == 'place':
        must_get_q0_from_pick_abs_pose = action_data_mode == 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
        assert must_get_q0_from_pick_abs_pose
        pick_abs_poses = actions[:, 3:7]  # must swap out the q0 with the pick base pose
        poses[:, -4:] = pick_abs_poses
        actions = actions[:, -4:]
    else:
        raise NotImplementedError

    #### This perhaps needs to be refactored ####
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)

    indices_to_delete = sampler_utils.get_indices_to_delete(config.region, key_configs)
    key_configs = np.delete(key_configs, indices_to_delete, axis=0)
    states = np.delete(states, indices_to_delete, axis=1)
    konf_relevance = np.delete(konf_relevance, indices_to_delete, axis=1)
    goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)
    ############

    # key_configs = [utils.decode_pose_with_sin_and_cos_angle(a) for a in actions]
    # key_configs = filter_configs_that_are_too_close(key_configs)
    # pickle.dump(key_configs, open('placements_%s.pkl' %(config.region), 'wb'))
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    n_key_configs = len(key_configs)
    key_configs = key_configs.reshape((1, n_key_configs, 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)
    print "Number of data", len(states)
    n_collisions = states.shape[1]
    policy = create_policy(config, n_collisions, n_key_configs)
    policy.policy_model.summary()
    policy.train_policy(states, konf_relevance, poses, key_configs, goal_flags, actions, sum_rewards)


def main():
    train(configs)


if __name__ == '__main__':
    main()
