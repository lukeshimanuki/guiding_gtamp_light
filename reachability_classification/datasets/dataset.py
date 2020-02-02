from torch.autograd import Variable
from torch.utils.data import Dataset
from torch_geometric.data import Data

import os
import pickle
import numpy as np
from gtamp_utils import utils
import torch


class ReachabilityDataset(Dataset):
    def __init__(self, action_type):
        self.q0s, self.qgs, self.collisions, self.labels = self.get_data(action_type)

    def get_data(self, action_type):
        plan_exp_dir = './planning_experience/processed/motion_plans/'
        cache_file_name = plan_exp_dir + '/' + action_type + '_cached_data.pkl'
        if os.path.isfile(cache_file_name):
            q0s, qgs, collisions, labels = pickle.load(open(cache_file_name, 'r'))
            return q0s, qgs, collisions, labels

        plan_exp_files = os.listdir(plan_exp_dir)
        np.random.shuffle(plan_exp_files)

        q0s = []
        qgs = []
        collisions = []
        labels = []
        n_episodes = 0
        for plan_exp_file in plan_exp_files:
            if 'cached' in plan_exp_file: continue
            plan = pickle.load(open(plan_exp_dir + plan_exp_file, 'r'))
            if len(plan[action_type + '_q0s']) == 0:
                continue
            file_q0s = plan[action_type + '_q0s']
            #for idx, q0 in enumerate(file_q0s):
            #    file_q0s[idx] = utils.encode_pose_with_sin_and_cos_angle(q0)

            file_qgs = plan[action_type + '_qgs']
            #for idx, qg in enumerate(file_qgs):
            #    file_qgs[idx] = utils.encode_pose_with_sin_and_cos_angle(qg)

            q0s.append(np.array(file_q0s, dtype=np.float32))
            qgs.append(np.array(plan[action_type + '_qgs'], dtype=np.float32))

            cols = []
            for c in plan[action_type + '_collisions']:
                col = utils.convert_binary_vec_to_one_hot(c.squeeze())
                col = col.reshape((1, 618, 2))
                cols.append(col)

            collisions.append(np.array(cols, dtype=np.float32))
            labels.append(np.array(plan[action_type + '_labels'], dtype=np.float32))

            n_episodes += 1
            #if n_episodes == 5000:
            #    break

        q0s = np.vstack(q0s)
        qgs = np.vstack(qgs)
        collisions = np.vstack(collisions)
        try:
            labels = np.vstack(labels)
        except ValueError:
            labels = np.hstack(labels)

        q0s = Variable(torch.from_numpy(q0s))
        qgs = Variable(torch.from_numpy(qgs))
        collisions = Variable(torch.from_numpy(collisions))
        labels = Variable(torch.from_numpy(labels))
        pickle.dump((q0s, qgs, collisions, labels), open(cache_file_name, 'wb'))
        return q0s, qgs, collisions, labels

    def __len__(self):
        return len(self.q0s)

    def __getitem__(self, idx):
        return {'x': [self.q0s[idx], self.qgs[idx], self.collisions[idx]], 'y': self.labels[idx]}


class GNNReachabilityDataset(ReachabilityDataset):
    def __init__(self, action_type):
        super(GNNReachabilityDataset, self).__init__(action_type)
        self.prm_vertices, self.prm_edges = pickle.load(open('prm.pkl', 'r'))
        #self.prm_vertices = np.zeros((len(self.tmp_prm_vertices), 4))
        #for pidx, p in enumerate(self.tmp_prm_vertices):
        #    self.prm_vertices[pidx] = utils.encode_pose_with_sin_and_cos_angle(p)
        self.gnn_vertices = self.prm_vertices
        self.collisions = self.collisions.squeeze()

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
            dim_q = self.q0s.shape[-1]
            q0 = np.array(self.q0s).reshape((len(self.q0s), dim_q))[idx][None, :]
            qg = np.array(self.qgs).reshape((len(self.qgs), dim_q))[idx][None, :]
            repeat_q0 = np.repeat(q0, 618, axis=0)
            repeat_qg = np.repeat(qg, 618, axis=0)
            v = np.hstack([prm_vertices, repeat_q0, repeat_qg, self.collisions[idx]])
        else:
            prm_vertices = np.repeat(np.array(self.prm_vertices)[None, :], len(idx), axis=0)
            q0s = np.repeat(np.array(self.q0s)[idx][:, None, :], 618, axis=1)
            qgs = np.repeat(np.array(self.qgs)[idx][:, None, :], 618, axis=1)
            v = np.concatenate([prm_vertices, q0s, qgs, self.collisions[idx]], axis=-1)

        return {'vertex': v, 'edges': self.edges, 'y': self.labels[idx]}


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
            repeat_qg = np.repeat(rel_qg, 618, axis=0)
            v = np.hstack([rel_key_configs, repeat_qg, self.collisions[idx]])
        else:
            raise NotImplementedError

        return {'vertex': v, 'edges': self.edges, 'y': self.labels[idx]}
