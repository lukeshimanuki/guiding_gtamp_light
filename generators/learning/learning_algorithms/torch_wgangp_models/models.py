import torch
from torch import nn
from generators.learning.utils.sampler_utils import get_indices_to_delete
import pickle
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, dim_konf, atype, region, problem_name):
        nn.Module.__init__(self)
        self.dim_konf = dim_konf
        self.atype = atype
        self.region = region
        self.problem_name = problem_name
        if 'two_arm' in problem_name:
            key_configs, _ = pickle.load(open('prm.pkl'))
            if 'pick' in atype:
                self.dim_pose_ids = 8 + 2
                self.dim_konf = 2
                self.konf_indices = get_indices_to_delete('home_region', key_configs)
            else:
                self.dim_pose_ids = 8 + 2
                self.dim_konf = 4
                if 'home' in self.region:
                    # get home indices
                    self.konf_indices = get_indices_to_delete('loading_region', key_configs)
                elif 'loading' in self.region:
                    self.konf_indices = get_indices_to_delete('home_region', key_configs)
                else:
                    raise NotImplementedError
            if 'home' in self.region or 'pick' in self.atype:
                self.dim_cnn_features = 2688
            else:
                self.dim_cnn_features = 2624
        else:
            key_configs = np.array(pickle.load(open('one_arm_key_configs.pkl'))['konfs'])
            if 'pick' in atype:
                self.dim_pose_ids = 4
                self.dim_konf = 2
                self.konf_indices = np.array(range(len(key_configs)))
            else:
                self.dim_pose_ids = 4
                self.dim_konf = 4
                self.konf_indices = np.array(range(len(key_configs)))

        self.n_hidden = 32
        self.n_konfs = len(self.konf_indices)

    def forward(self, action, konf, pose):
        raise NotImplementedError

    def filter_data_according_to_cases(self, konf, pose_ids):
        if 'two_arm' in self.problem_name:
            if 'pick' in self.atype:
                target_obj_pose = pose_ids[:, 0:4]
                robot_curr_pose_and_id = pose_ids[:, -6:]
                pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)
                konf = konf[:, :, 0:2, :]
            else:
                target_obj_pose = pose_ids[:, 0:4]
                robot_curr_pose_and_id = pose_ids[:, -6:]
                pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)
        else:
            # Input: konf collisions + target object pose
            if 'pick' in self.atype:
                target_obj_pose = pose_ids[:, 0:4] # only use object_pose
                pose_ids = target_obj_pose
                konf = konf[:, :, 0:2, :]
            else:
                target_obj_pose = pose_ids[:, 0:4]
                pose_ids = target_obj_pose
        konf = konf[:, self.konf_indices, :, :]
        return konf, pose_ids
