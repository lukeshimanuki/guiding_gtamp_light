from torch.autograd import Variable
from torch.utils.data import Dataset

import os
import pickle
import numpy as np
from gtamp_utils import utils
import torch


class ReachabilityDataset(Dataset):
    def __init__(self):
        self.q0s, self.qgs, self.collisions, self.labels = self.get_data()

    def get_data(self):
        plan_exp_dir = './planning_experience/processed/motion_plans/'
        cache_file_name = plan_exp_dir + './cached_data.pkl'
        if os.path.isfile(cache_file_name):
            q0s, qgs, collisions, labels = pickle.load(open(cache_file_name, 'r'))
            return q0s, qgs, collisions, labels

        plan_exp_files = os.listdir(plan_exp_dir)

        q0s = []
        qgs = []
        collisions = []
        labels = []
        n_episodes = 0
        for plan_exp_file in plan_exp_files:
            plan = pickle.load(open(plan_exp_dir + plan_exp_file, 'r'))
            if len(plan['q0s']) == 0:
                continue
            q0s.append(np.array(plan['q0s'], dtype=np.float32))
            qgs.append(np.array(plan['qgs'], dtype=np.float32))

            cols = []
            for c in plan['collisions']:
                col = utils.convert_binary_vec_to_one_hot(c.squeeze())
                col = col.reshape((1, 618, 2))
                cols.append(col)

            collisions.append(np.array(cols, dtype=np.float32))
            labels.append(np.array(plan['labels'], dtype=np.float32))

            n_episodes += 1
            if n_episodes == 1000:
                break

        q0s = np.vstack(q0s)
        qgs = np.vstack(qgs)
        collisions = np.vstack(collisions)
        labels = np.vstack(labels)

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
