from torch.autograd import Variable
from torch.utils.data import Dataset

import os
import pickle
import numpy as np
from gtamp_utils import utils
import torch


class GeneratorDataset(Dataset):
    def __init__(self, action_type, desired_region, use_filter):
        self.use_filter = use_filter
        self.desired_region = desired_region
        self.cols, self.q0s, self.actions, self.goal_obj_poses, self.manip_obj_poses = self.get_data(action_type)

    def get_data(self, action_type):
        plan_exp_dir = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/' \
                       'uses_rrt/sampler_trajectory_data/'
        cache_file_name = plan_exp_dir + '/' + action_type + '_cached_data.pkl'
        if os.path.isfile(cache_file_name):
            cols, q0s, actions, goal_obj_poses, manip_obj_poses = pickle.load(open(cache_file_name, 'r'))
            return cols, q0s, actions, goal_obj_poses, manip_obj_poses

        plan_exp_files = os.listdir(plan_exp_dir)
        np.random.shuffle(plan_exp_files)

        q0s = []
        goal_obj_poses = []
        cols = []
        actions = []
        manip_obj_poses = []
        for plan_exp_file in plan_exp_files:
            if 'pap_traj' not in plan_exp_file: continue
            traj = pickle.load(open(plan_exp_dir + plan_exp_file, 'r'))

            print plan_exp_file
            try:
                if len(traj.states) == 0:
                    continue
            except:
                import pdb;pdb.set_trace()

            file_cols = []
            file_q0s = []
            file_actions = []
            file_manip_obj_poses = []
            file_goal_obj_poses = []
            rewards = np.array(traj.hvalues)[:-1] - np.array(traj.hvalues[1:])
            for s, a, reward in zip(traj.states, traj.actions, rewards):
                is_move_to_goal_region = s.region in s.goal_entities
                if self.desired_region == 'home_region' and not is_move_to_goal_region:
                    continue
                if self.desired_region == 'loading_region' and is_move_to_goal_region:
                    continue
                if reward <= 0 and self.use_filter:
                    continue

                if action_type == 'pick':
                    collisions = s.pick_collision_vector
                elif action_type == 'place':
                    collisions = s.place_collision_vector
                else:
                    raise NotImplementedError

                file_cols.append(collisions)
                file_q0s.append(s.abs_robot_pose.squeeze())
                file_manip_obj_poses.append(s.abs_obj_pose.squeeze())
                file_goal_obj_poses.append([p.squeeze() for p in s.abs_goal_obj_poses])

                if action_type == 'pick':
                    action = a['pick_action_parameters']
                else:
                    action = a['place_abs_base_pose']
                file_actions.append(action)

            cols.append(file_cols)
            q0s.append(file_q0s)
            actions.append(file_actions)
            goal_obj_poses.append(file_goal_obj_poses)
            manip_obj_poses.append(file_manip_obj_poses)

        cols = np.vstack(cols).squeeze()
        q0s = np.vstack(q0s)
        actions = np.vstack(actions)
        goal_obj_poses = np.vstack(goal_obj_poses)
        manip_obj_poses = np.vstack(manip_obj_poses)

        cols = Variable(torch.from_numpy(cols))
        q0s = Variable(torch.from_numpy(q0s))
        actions = Variable(torch.from_numpy(actions))
        goal_obj_poses = Variable(torch.from_numpy(goal_obj_poses))
        manip_obj_poses = Variable(torch.from_numpy(manip_obj_poses))
        pickle.dump((cols, q0s, actions, goal_obj_poses, manip_obj_poses), open(cache_file_name, 'wb'))

        return cols, q0s, actions, goal_obj_poses, manip_obj_poses

    def __len__(self):
        return len(self.q0s)

    def __getitem__(self, idx):
        raise NotImplementedError


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
