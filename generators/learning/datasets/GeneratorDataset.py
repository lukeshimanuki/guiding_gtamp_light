from torch.utils.data import Dataset

import pickle
import numpy as np
import torch
from generators.learning.utils.data_processing_utils import get_processed_poses_from_state, \
    get_processed_poses_from_action

from gtamp_utils import utils
import os


class GeneratorDataset(Dataset):
    def __init__(self, config, use_filter, is_testing):
        self.use_filter = use_filter
        self.is_testing = is_testing
        self.config = config
        self.konf_obsts, self.poses, self.actions = self.get_data()

    def get_cache_file_name(self, action_data_mode):
        state_data_mode = 'absolute'
        action_type = self.config.atype
        desired_region = self.config.region
        use_filter = self.use_filter
        if 'pick' in action_type:
            cache_file_name = 'cache_smode_%s_amode_%s_atype_%s.pkl' % (state_data_mode, action_data_mode, action_type)
        else:
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
        return cache_file_name

    def get_data_dir(self):
        if self.use_filter:
            if 'one_arm' in self.config.domain:
                data_dir = 'planning_experience/processed/one_arm_mover/n_objs_pack_1/sahs/uses_rrt/' \
                           'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
            else:
                data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/' \
                           'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
        else:
            raise NotImplementedError
        return data_dir

    def we_should_skip_this_state_and_action(self, s, reward):
        action_type = self.config.atype
        desired_region = self.config.region
        use_filter = self.use_filter
        if 'place' in action_type:
            is_move_to_goal_region = s.region in s.goal_entities
            if reward <= 0 and use_filter:
                return True

            if 'two_arm' in self.config.domain:
                if desired_region == 'home_region' and not is_move_to_goal_region:
                    return True

                if desired_region == 'loading_region' and is_move_to_goal_region:
                    return True
            else:
                if desired_region == 'rectangular_packing_box1_region' and not is_move_to_goal_region:
                    return True

                if desired_region == 'center_shelf_region' and is_move_to_goal_region:
                    return True

        return False

    def load_data_from_files(self, action_data_mode):
        traj_dir = self.get_data_dir()
        print "Loading data from", traj_dir
        traj_files = os.listdir(traj_dir)
        cache_file_name = self.get_cache_file_name(action_data_mode)
        if os.path.isfile(traj_dir + cache_file_name):
            print "Loading the cache file", traj_dir + cache_file_name
            f = pickle.load(open(traj_dir + cache_file_name, 'r'))
            print "Cache data loaded"
            return f

        print 'caching file...%s' % cache_file_name
        all_states = []
        all_actions = []
        all_sum_rewards = []
        all_poses_ids = []
        all_konf_relevance = []
        n_episodes = 0
        for traj_file_idx, traj_file in enumerate(traj_files):
            if 'pidx' not in traj_file:
                print 'not pkl file'
                continue
            try:
                traj = pickle.load(open(traj_dir + traj_file, 'r')) 
            except:
                print traj_file
                continue

            if len(traj.states) == 0:
                print 'failed instance'
                continue

            states = []
            poses_ids = []
            actions = []
            konf_relevance = []

            if self.use_filter:
                rewards = np.array(traj.prev_n_in_way) - np.array(traj.n_in_way) > 0
            else:
                rewards = 1

            for s, a, reward, v_manip_goal in zip(traj.states, traj.actions, rewards, traj.prev_v_manip_goal):
                if self.we_should_skip_this_state_and_action(s, reward):
                    continue

                collision_vec = s.pick_collision_vector
                if 'two_arm' in self.config.domain:
                    v_manip_vec = utils.convert_binary_vec_to_one_hot(v_manip_goal.squeeze()).reshape((1, 618, 2, 1))
                else:
                    v_manip_vec = utils.convert_binary_vec_to_one_hot(v_manip_goal.squeeze()).reshape((1, 355, 2, 1))
                state_vec = np.concatenate([collision_vec, v_manip_vec], axis=2)

                states.append(state_vec)

                # note that this is not used in one arm domain
                if 'rectangular' in a['object_name']:
                    object_id = [1, 0]
                else:
                    object_id = [0, 1]
                poses_from_state = get_processed_poses_from_state(s, 'absolute')
                poses_from_state_and_id = np.hstack([poses_from_state, object_id])
                poses_ids.append(poses_from_state_and_id)
                actions.append(get_processed_poses_from_action(s, a, action_data_mode))

            states = np.array(states)
            poses_ids = np.array(poses_ids)
            actions = np.array(actions)

            rewards = traj.rewards
            sum_rewards = np.array([np.sum(traj.rewards[t:]) for t in range(len(rewards))])
            if len(states) == 0:
                print "no state"
                continue
            all_poses_ids.append(poses_ids)
            all_states.append(states)
            all_actions.append(actions)
            all_sum_rewards.append(sum_rewards)
            all_konf_relevance.append(konf_relevance)
            n_episodes += 1

            print 'n_data %d progress %d/%d' % (len(np.vstack(all_actions)), traj_file_idx, len(traj_files))
            print "N episodes", n_episodes
            n_data = len(np.vstack(all_actions))
            assert len(np.vstack(all_states)) == n_data

        all_states = np.vstack(all_states).squeeze(axis=1)
        all_actions = np.vstack(all_actions).squeeze()
        all_poses_ids = np.vstack(all_poses_ids).squeeze()
        pickle.dump((all_states, all_poses_ids, all_actions), open(traj_dir + cache_file_name, 'wb'))

        return all_states, all_poses_ids, all_actions

    def get_data(self):
        if self.config.atype == 'pick':
            action_data_mode = 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base'
        else:
            action_data_mode = 'PICK_grasp_params_and_abs_base_PLACE_abs_base'

        states, poses, actions = self.load_data_from_files(action_data_mode)
        if self.config.atype == 'pick':
            actions = actions[:, :-4]
        elif self.config.atype == 'place':
            pick_abs_poses = actions[:, 3:7]  # must swap out the q0 with the pick base pose
            poses[:, -6:-2] = pick_abs_poses
            actions = actions[:, -4:]
        else:
            raise NotImplementedError

        return states, poses, actions

    def __len__(self):
        return len(self.konf_obsts)

    def __getitem__(self, idx):
        raise NotImplementedError


class StandardDataset(GeneratorDataset):
    def __init__(self, config, use_filter, is_testing):
        super(StandardDataset, self).__init__(config, use_filter, is_testing)

    def __getitem__(self, idx):
        data = {
            'konf_obsts': self.konf_obsts[idx],
            'poses': self.poses[idx],
            'actions': self.actions[idx],
        }
        return data
