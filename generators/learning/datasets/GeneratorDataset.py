from torch.utils.data import Dataset

import pickle
import numpy as np
import torch
from generators.learning.utils.data_processing_utils import get_processed_poses_from_state, \
    get_processed_poses_from_action

from gtamp_utils import utils
import os


class GeneratorDataset(Dataset):
    def __init__(self, action_type, desired_region, use_filter):
        self.use_filter = use_filter
        self.desired_region = desired_region
        self.konf_obsts, self.poses, self.actions = self.get_data(action_type, desired_region)

    @staticmethod
    def get_cache_file_name(action_data_mode, action_type, desired_region, use_filter):
        state_data_mode = 'absolute'

        if action_type == 'pick':
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

    @staticmethod
    def get_data_dir(filtered):
        if filtered:
            data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/' \
                       'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'
        else:
            data_dir = 'planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/' \
                       'sampler_trajectory_data/'
        return data_dir

    @staticmethod
    def we_should_skip_this_state_and_action(s, desired_region, reward, action_type, use_filter):
        if 'pick' not in action_type:
            is_move_to_goal_region = s.region in s.goal_entities
            if desired_region == 'home_region' and not is_move_to_goal_region:
                return True

            if desired_region == 'loading_region' and is_move_to_goal_region:
                return True

        if reward <= 0 and use_filter:
            return True

        return False

    def load_data_from_files(self, action_type, desired_region, use_filter, action_data_mode):
        traj_dir = self.get_data_dir(use_filter)
        print "Loading data from", traj_dir
        traj_files = os.listdir(traj_dir)
        cache_file_name = self.get_cache_file_name(action_data_mode, action_type, desired_region, use_filter)
        if os.path.isfile(traj_dir + cache_file_name):
            print "Loading the cache file", traj_dir + cache_file_name
            f = pickle.load(open(traj_dir + cache_file_name, 'r'))
            print "Cache data loaded"
            return f

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

            states = []
            poses = []
            actions = []
            konf_relevance = []

            if use_filter:
                rewards = np.array(traj.prev_n_in_way) - np.array(traj.n_in_way) >= 0
            else:
                rewards = 0

            for s, a, reward, v_manip_goal in zip(traj.states, traj.actions, rewards, traj.prev_v_manip_goal):
                if self.we_should_skip_this_state_and_action(s, desired_region, reward, action_type, use_filter):
                    continue

                if action_type == 'pick':
                    collision_vec = s.pick_collision_vector
                elif action_type == 'place':
                    collision_vec = s.place_collision_vector
                else:
                    raise NotImplementedError

                v_manip_vec = utils.convert_binary_vec_to_one_hot(v_manip_goal.squeeze()).reshape((1, 618, 2, 1))
                state_vec = np.concatenate([collision_vec,v_manip_vec],axis=2)


                states.append(state_vec)
                poses.append(get_processed_poses_from_state(s, 'absolute'))
                actions.append(get_processed_poses_from_action(s, a, action_data_mode))

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

            print 'n_data %d progress %d/%d' % (len(np.vstack(all_actions)), traj_file_idx, len(traj_files))
            n_data = len(np.vstack(all_actions))
            assert len(np.vstack(all_states)) == n_data

        all_states = np.vstack(all_states).squeeze(axis=1)
        all_actions = np.vstack(all_actions).squeeze()
        all_sum_rewards = np.hstack(np.array(all_sum_rewards))[:, None]  # keras requires n_data x 1
        all_poses = np.vstack(all_poses).squeeze()
        pickle.dump((all_states, all_poses, all_actions), open(traj_dir + cache_file_name, 'wb'))

        return all_states, all_poses, all_actions

    def get_data(self, action_type, region):
        atype = action_type
        filtered = True
        if atype == 'pick':
            action_data_mode = 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base'
        else:
            action_data_mode = 'PICK_grasp_params_and_abs_base_PLACE_abs_base'

        states, poses, actions = self.load_data_from_files(atype, region, filtered, action_data_mode)
        if atype == 'pick':
            actions = actions[:, :-4]
        elif atype == 'place':
            pick_abs_poses = actions[:, 3:7]  # must swap out the q0 with the pick base pose
            poses[:, -4:] = pick_abs_poses
            actions = actions[:, -4:]
        else:
            raise NotImplementedError

        return konf_obsts, poses, actions

    def __len__(self):
        return len(self.konf_obsts)

    def __getitem__(self, idx):
        raise NotImplementedError


class StandardDataset(GeneratorDataset):
    def __init__(self, action_type, desired_region, use_filter):
        super(StandardDataset, self).__init__(action_type, desired_region, use_filter)

    def __getitem__(self, idx):
        data = {
            'konf_obsts': self.konf_obsts[idx],
            'poses': self.poses[idx],
            'actions': self.actions[idx],
        }
        return data


class GNNDataset(GeneratorDataset):
    def __init__(self, action_type, desired_region, use_filter):
        super(GNNDataset, self).__init__(action_type, desired_region, use_filter)
        self.prm_vertices, self.prm_edges = pickle.load(open('prm.pkl', 'r'))
        self.gnn_vertices = self.prm_vertices

        edges = [[], []]
        for src_idx, _ in enumerate(self.prm_vertices):
            neighbors = list(self.prm_edges[src_idx])
            n_edges = len(neighbors)
            edges[0] += [src_idx] * n_edges
            edges[1] += neighbors
            # How to make sure it is bidrectional?
        self.edges = np.array(edges)

    def __getitem__(self, idx):
        if type(idx) is int:
            # where is the pick robot pose?
            prm_vertices = self.prm_vertices
            n_konfs = len(prm_vertices)
            dim_q = self.q0s.shape[-1]
            q0 = np.array(self.q0s).reshape((len(self.q0s), dim_q))[idx][None, :]
            repeat_q0 = np.repeat(q0, n_konfs, axis=0)
            goal_obj_poses = self.goal_obj_poses[idx].reshape((1, 4 * 3))
            repeat_goal_obj_poses = np.repeat(goal_obj_poses, n_konfs, axis=0)
            v = np.hstack([prm_vertices, repeat_q0, repeat_goal_obj_poses, self.cols[idx]])
        else:
            raise NotImplementedError

        v = torch.from_numpy(v)

        return {'vertex': v, 'edges': self.edges, 'actions': self.actions[idx]}


"""
class GNNRelativeReachabilityDataset(GNNReachabilityDataset):
    def __init__(self, action_type):
        super(GNNRelativeReachabilityDataset, self).__init__(action_type)

    @staticmethod
    def compute_relative_config(src_config, end_config):
        src_config = np.array(src_config)
        end_config = np.array(end_config)

        assert len(src_config.shape) == 2, \
            'Put configs in shapes (n_config,dim_config)'

        rel_config = end_config - src_config
        neg_idxs_to_fix = rel_config[:, -1] < -np.pi
        pos_idxs_to_fix = rel_config[:, -1] > np.pi

        # making unique rel angles; keep the range to [-pi,pi]
        rel_config[neg_idxs_to_fix, -1] = rel_config[neg_idxs_to_fix, -1] + 2 * np.pi
        rel_config[pos_idxs_to_fix, -1] = rel_config[pos_idxs_to_fix, -1] - 2 * np.pi

        return rel_config

    def __getitem__(self, idx):
        if type(idx) is int:
            dim_q = self.q0s.shape[-1]
            q0 = np.array(self.q0s).reshape((len(self.q0s), dim_q))[idx][None, :]
            qg = np.array(self.qgs).reshape((len(self.qgs), dim_q))[idx][None, :]

            rel_qg = self.compute_relative_config(q0, qg)
            rel_key_configs = self.compute_relative_config(q0, self.prm_vertices)
            repeat_qg = np.repeat(rel_qg, 615, axis=0)
            v = np.hstack([rel_key_configs, repeat_qg, self.collisions[idx]])
        else:
            raise NotImplementedError

        return {'vertex': v, 'edges': self.edges, 'y': self.labels[idx]}
"""
