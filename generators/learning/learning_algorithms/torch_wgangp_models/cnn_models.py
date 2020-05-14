import torch
from models import BaseModel


class CNNDiscriminator(BaseModel):
    def __init__(self, dim_konf, dim_data, atype, region):
        BaseModel.__init__(self, dim_konf, atype, region)
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
                torch.nn.Linear(2688, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )

    def forward(self, action, konf, pose_ids):
        konf, pose_ids = self.filter_data_according_to_cases(konf, pose_ids)
        pose_ids = pose_ids.unsqueeze(1).repeat((1, self.n_konfs, 1)).unsqueeze(-1)
        action_expanded = action.unsqueeze(1).repeat((1, self.n_konfs, 1)).unsqueeze(-1)
        concat = torch.cat([action_expanded, pose_ids, konf], dim=2)
        concat = concat.reshape((concat.shape[0], concat.shape[-1], concat.shape[1], concat.shape[2]))

        features = self.features(concat)
        features = features.view((features.shape[0], features.shape[1] * features.shape[2]))
        value = self.value(features)
        return value


class CNNGenerator(BaseModel):
    def __init__(self, dim_konf, dim_data, atype, region):
        BaseModel.__init__(self, dim_konf, atype, region)
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
                torch.nn.Linear(2688 + dim_data, 32), # for noise
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, dim_data))

    def forward(self, konf, pose_ids, noise):
        konf, pose_ids = self.filter_data_according_to_cases(konf, pose_ids)
        pose_ids = pose_ids.unsqueeze(1).repeat((1, self.n_konfs, 1)).unsqueeze(-1)
        concat = torch.cat([pose_ids, konf], dim=2)
        concat = concat.reshape((concat.shape[0], concat.shape[-1], concat.shape[1], concat.shape[2]))

        features = self.features(concat)
        features = features.view((features.shape[0], features.shape[1] * features.shape[2]))
        features = torch.cat([features, noise], dim=-1)
        value = self.value(features)

        return value
