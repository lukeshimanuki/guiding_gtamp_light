from torch.utils.data import Dataset

import pickle
import numpy as np
import torch
from generators.learning.data_load_utils import get_data
from generators.learning.utils.data_processing_utils import action_data_mode


class GeneratorDataset(Dataset):
    def __init__(self, action_type, desired_region, use_filter):
        self.use_filter = use_filter
        self.desired_region = desired_region
        self.konf_obsts, self.poses, self.actions = self.get_data(action_type)

    def get_data(self, action_type):
        data_type = 'n_objs_pack_1'
        atype = action_type
        region = 'loading_region'
        filtered = False
        #konf_obsts, poses, _, actions, _ = pickle.load(open('tmp_data_for_debug_train_sampler.pkl', 'r'))
        konf_obsts, poses, _, actions, _ = get_data(data_type, atype, region, filtered)

        if atype == 'pick':
            actions = actions[:, :-4]
        elif atype == 'place':
            must_get_q0_from_pick_abs_pose = action_data_mode == 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
            assert must_get_q0_from_pick_abs_pose
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
