import torch
from models import BaseGenerator, BaseDiscriminator


class CNNDiscriminator(BaseDiscriminator):
    def __init__(self, dim_konf, dim_actions, atype):
        BaseDiscriminator.__init__(self, dim_konf, atype)
        n_hidden = 32
        self.features = \
            torch.nn.Sequential(
                torch.nn.Conv2d(1, n_hidden, kernel_size=(1, self.dim_konf + 4+4+4+2)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 1)),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 1))
            )
        self.value = \
            torch.nn.Sequential(
                torch.nn.Linear(32 * 154, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )

    def forward(self, action, konf, pose_ids):

        target_obj_pose = pose_ids[:, 0:4]
        robot_curr_pose_and_id = pose_ids[:, -6:]
        pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)
        robot_curr_pose_expanded = pose_ids.unsqueeze(1).repeat((1, 618, 1)).unsqueeze(-1)
        action_expanded = action.unsqueeze(1).repeat((1, 618, 1)).unsqueeze(-1)
        concat = torch.cat([action_expanded, robot_curr_pose_expanded, konf], dim=2)
        concat = concat.reshape((concat.shape[0], concat.shape[-1], concat.shape[1], concat.shape[2]))

        features = self.features(concat)
        features = features.view((features.shape[0], features.shape[1] * features.shape[2]))
        value = self.value(features)
        return value


class CNNGenerator(BaseGenerator):
    def __init__(self, dim_konf, dim_data, atype):
        BaseGenerator.__init__(self, dim_konf, atype)
        n_hidden = 32

        self.features = \
            torch.nn.Sequential(
                torch.nn.Conv2d(1, n_hidden, kernel_size=(1, self.dim_konf + 4+4+2)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 1)),
                torch.nn.Conv2d(n_hidden, n_hidden, kernel_size=(1, 1)),
                torch.nn.LeakyReLU(),
                torch.nn.MaxPool2d(kernel_size=(2, 1))
            )
        self.value = \
            torch.nn.Sequential(
                torch.nn.Linear(32 * 154 + dim_data, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, dim_data))

    def forward(self, konf, pose_ids, noise):
        #robot_curr_pose = pose[:, -4:]
        target_obj_pose = pose_ids[:, 0:4]
        robot_curr_pose_and_id = pose_ids[:, -6:]
        pose_ids = torch.cat([target_obj_pose, robot_curr_pose_and_id], -1)

        robot_curr_pose_expanded = pose_ids.unsqueeze(1).repeat((1, 618, 1)).unsqueeze(-1)
        concat = torch.cat([robot_curr_pose_expanded, konf], dim=2)
        concat = concat.reshape((concat.shape[0], concat.shape[-1], concat.shape[1], concat.shape[2]))

        features = self.features(concat)
        features = features.view((features.shape[0], features.shape[1] * features.shape[2]))
        features = torch.cat([features, noise], dim=-1)
        value = self.value(features)

        return value
