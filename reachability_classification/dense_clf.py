import os
import pickle
import numpy as np
import torch
import time

from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from gtamp_utils import utils

import torch.optim as optim


class ReachabilityNet(nn.Module):
    def __init__(self):
        super(ReachabilityNet, self).__init__()

        # takes as an input q0, qg, and Ck, and outputs a reachability
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)

        in_channels = 2
        out_channels = 8
        self.x_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())
        self.y_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())
        self.th_lin = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU())
        in_channels = 618
        out_channels = 64

        self.collision_lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(618*2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        """
        self.collision_lin = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(10, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 1)),
            nn.Flatten()
        )
        """

        self.output_lin = nn.Sequential(
            nn.Linear(64+8*3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, q0s, qgs, collisions):
        n_data = len(q0s)
        x_concat = torch.cat((q0s[:, 0], qgs[:, 0]), 0).reshape((n_data, 2))
        x_features = self.x_lin(x_concat)
        y_concat = torch.cat((q0s[:, 1], qgs[:, 1]), 0).reshape((n_data, 2))
        y_features = self.y_lin(y_concat)
        th_concat = torch.cat((q0s[:, -1], qgs[:, -1]), 0).reshape((n_data, 2))
        th_features = self.th_lin(th_concat)

        q0qg_features = torch.cat((x_features, y_features, th_features), -1).reshape(n_data, 8 * 3)
        collision_features = self.collision_lin(collisions)
        everything = torch.cat((q0qg_features, collision_features), -1)
        output = self.output_lin(everything)

        return output


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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print device

    net = ReachabilityNet()
    net.to(device)
    loss_fn = nn.BCELoss()

    dataset = ReachabilityDataset()
    trainset, testset = torch.utils.data.random_split(dataset, [3000, len(dataset) - 3000])

    test_q0s = testset.dataset.q0s[testset.indices]
    test_qgs = testset.dataset.qgs[testset.indices]
    test_cols = testset.dataset.collisions[testset.indices]
    test_labels = testset.dataset.labels[testset.indices]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    acc_list = []
    for epoch in range(100):  # loop over the dataset multiple times
        for i, batch in enumerate(trainloader, 0):
            q0s = batch['x'][0].to(device)
            qgs = batch['x'][1].to(device)
            cols = batch['x'][2].to(device)
            labels = batch['y'].to(device)
            if len(q0s)==0:
                continue
            pred = net(q0s, qgs, cols)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print loss.item()

        test_q0s.to(device)
        test_qgs.to(device)
        test_cols.to(device)
        test_pred = net(test_q0s, test_qgs, test_cols)
        clf_result = test_pred > 0.5
        acc = np.mean(clf_result.numpy() == test_labels.numpy())
        acc_list.append(acc)
        print epoch, acc_list


if __name__ == '__main__':
    main()
