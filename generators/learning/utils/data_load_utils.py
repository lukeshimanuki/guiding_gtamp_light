import os
import numpy as np
import pickle
import os

from data_processing_utils import get_processed_poses_from_state, get_processed_poses_from_action, \
    state_data_mode

import socket
from gtamp_utils import utils


def one_hot_encode(vec):
    n_elements = len(np.unique(vec))
    n_data = np.shape(vec)[0]
    encoded = np.zeros((n_data, n_elements))
    for i in range(n_data):
        encoded[i, vec[i, :]] = 1
    return encoded


def create_bit_encoding_of_konf(n_konf):
    n_helper = 10
    k_data = np.ones((n_konf, n_helper)) * -1
    idnumb = 1
    for idx in range(n_konf):
        # binstr=bin(idnumb)[2:]
        binstr = '{0:010b}'.format(idnumb)
        binidx = range(len(binstr))[::-1]
        for k in binidx:
            if int(binstr[k]) == 1:
                k_data[idx, k] = 1
            else:
                k_data[idx, k] = -1
        idnumb += 1
    k_data = k_data.reshape((n_konf, n_helper))
    return k_data


def convert_collision_vec_to_one_hot(c_data):
    n_konf = c_data.shape[1]
    onehot_cdata = []
    for cvec in c_data:
        one_hot_cvec = np.zeros((n_konf, 2))
        for boolean_collision, onehot_collision in zip(cvec, one_hot_cvec):
            onehot_collision[boolean_collision] = 1
        assert (np.all(np.sum(one_hot_cvec, axis=1) == 1))
        onehot_cdata.append(one_hot_cvec)

    """
    # test code
    for onehot_vec,cvec in zip(onehot_cdata,c_data):
      for onehot_collision,boolean_collision in zip(onehot_vec,cvec):
        assert( onehot_collision[boolean_collision] == 1 )
        assert( np.sum(onehot_collision) == 1 )
    """
    onehot_cdata = np.array(onehot_cdata)
    return onehot_cdata


def load_data(traj_dir, action_type, desired_region, use_filter):
    traj_files = os.listdir(traj_dir)
    # cache_file_name = 'no_collision_at_target_obj_poses_cache_state_data_mode_%s_action_data_mode_%s_loading_region_only.pkl' % (
    #    state_data_mode, action_data_mode)
    if action_type == 'pick':
        action_data_mode = 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base'
        cache_file_name = 'cache_smode_%s_amode_%s_atype_%s.pkl' % (state_data_mode, action_data_mode, action_type)
    else:
        action_data_mode = 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
        if use_filter:
            cache_file_name = 'cache_smode_%s_amode_%s_atype_%s_region_%s_filtered.pkl' % (state_data_mode,
                                                                                           action_data_mode,
                                                                                           action_type,
                                                                                           desired_region)
        else:
            cache_file_name = 'cache_smode_%s_amode_%s_atype_%s_region_%s_unfiltered.pkl' % (state_data_mode,
                                                                                             action_data_mode,
                                                                                             action_type,
                                                                                             desired_region)
    if os.path.isfile(traj_dir + cache_file_name):
        print "Loading the cache file", traj_dir + cache_file_name
        d = pickle.load(open(traj_dir + cache_file_name, 'r'))
        print "Cache data loaded"
        return d

    print 'caching file...%s' % cache_file_name
    all_states = []
    all_actions = []
    all_sum_rewards = []
    all_poses = []
    all_konf_relevance = []

    for traj_file_idx, traj_file in enumerate(traj_files):
        if 'pidx' not in traj_file:
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
        # rewards = np.array(traj.num_papable_to_goal[1:]) - np.array(traj.num_papable_to_goal[:-1])
        # rewards = np.array(traj.hvalues)[:-1] - np.array(traj.hvalues[1:])
        rewards = np.array(traj.num_papable_to_goal[1:]) - np.array(traj.num_papable_to_goal[:-1])
        rewards = np.array(traj.hcounts)[:-1] - np.array(traj.hcounts[1:])
        rewards = np.array(traj.hvalues)[:-1] - np.array(traj.hvalues[1:])

        if use_filter:
            rewards = np.array(traj.prev_n_in_way) - np.array(traj.n_in_way) >= 0

        for s, a, reward in zip(traj.states, traj.actions, rewards):
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

            if 'pick' not in action_type:
                is_move_to_goal_region = s.region in s.goal_entities
                if desired_region == 'home_region' and not is_move_to_goal_region:
                    continue

                if desired_region == 'loading_region' and is_move_to_goal_region:
                    continue

            if reward <= 0 and use_filter:
                continue

            state_vec = np.concatenate([state_vec, is_goal_obj, is_goal_region], axis=2)
            states.append(state_vec)
            poses.append(get_processed_poses_from_state(s, a))
            actions.append(get_processed_poses_from_action(s, a, action_data_mode))

            place_relevance = None  # data_processing_utils.get_relevance_info(key_configs, binary_collision_vector, place_motion)
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
        # all_paths.append(place_paths)

        print 'n_data %d progress %d/%d' % (len(np.vstack(all_actions)), traj_file_idx, len(traj_files))

        n_data = len(np.vstack(all_actions))
        assert len(np.vstack(all_states)) == n_data
        if traj_file_idx >= 5000:
            break

    all_states = np.vstack(all_states).squeeze(axis=1)
    all_actions = np.vstack(all_actions).squeeze()
    all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
    all_poses = np.vstack(all_poses).squeeze()
    pickle.dump((all_states, all_poses, all_actions, all_sum_rewards),
                open(traj_dir + cache_file_name, 'wb'))
    return all_states, all_poses, all_actions, all_sum_rewards[:, None]


def get_data(datatype, action_type, region, filtered):
    if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/planning_experience/processed/'

    if datatype == 'n_objs_pack_4':
        data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_4/sahs/uses_rrt/sampler_trajectory_data/'
    else:
        if filtered:
            data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
        else:
            data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/sampler_trajectory_data/'

    print "Loading data from", data_dir
    try:
        states, poses, actions, sum_rewards = load_data(root_dir + data_dir, action_type, region, filtered)
    except ValueError:
        states, poses, actions, sum_rewards, _ = load_data(root_dir + data_dir, action_type, region, filtered)

    is_goal_flags = states[:, :, 2:, :]
    states = states[:, :, :2, :]  # collision vector

    """
    n_data = 5000
    states = states[:5000, :]
    poses = poses[:n_data, :]
    actions = actions[:5000, :]
    sum_rewards = sum_rewards[:5000]
    is_goal_flags = is_goal_flag[:5000, :]
    """

    return states, poses, is_goal_flags, actions, sum_rewards
