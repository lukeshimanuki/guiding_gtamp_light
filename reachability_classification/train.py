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
from classifiers.dense_clf import DenseReachabilityNet
from datasets.dataset import ReachabilityDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print device

    net = DenseReachabilityNet()
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
        print epoch, np.max(acc_list)


if __name__ == '__main__':
    main()

