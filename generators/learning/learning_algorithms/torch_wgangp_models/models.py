import torch
from torch import nn
from generators.learning.utils.sampler_utils import get_indices_to_delete
import pickle


class BaseModel(nn.Module):
    def __init__(self, dim_konf, atype, region):
        nn.Module.__init__(self)
        self.dim_konf = dim_konf
        self.atype = atype
        self.region = region
        if 'pick' in atype:
            self.dim_pose_ids = 8 + 2
            self.dim_konf = 2
        else:
            self.dim_pose_ids = 8 * 3 + 2
            self.dim_konf = 4
            key_configs, _ = pickle.load(open('prm.pkl'))
            if 'home' in self.region:
                # get home indices
                self.konf_indices = get_indices_to_delete('loading_region', key_configs)
            elif 'loading' in self.region:
                self.konf_indices = get_indices_to_delete('home_region', key_configs)
            else:
                raise NotImplementedError

        self.n_hidden = 32
        self.n_konfs = 618 * self.dim_konf

    def forward(self, action, konf, pose):
        raise NotImplementedError

    def filter_data_according_to_cases(self, konf, pose_ids):
        if 'pick' in self.atype:
            target_obj_pose = pose_ids[:, 0:4]
            robot_curr_pose_and_id = pose_ids[:, -6:]
            pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)
            konf = konf[:, :, 0:2, :]

        konf = konf[:, self.konf_indices, :, :]
        return konf, pose_ids