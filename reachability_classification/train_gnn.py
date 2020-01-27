import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from classifiers.gnn import GNNReachabilityNet
from datasets.dataset import GNNReachabilityDataset


def main():
    device = torch.device("cpu")
    print device

    dataset = GNNReachabilityDataset()
    n_train = int(len(dataset) * 0.9)
    print "N_train", n_train
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])

    net = GNNReachabilityNet(trainset[1]['edges'], n_key_configs=618)
    net.to(device)
    loss_fn = nn.BCELoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    acc_list = []
    for epoch in range(100):
        for i, batch in enumerate(trainloader, 0):
            labels = batch['y'].to(device)
            vertices = batch['vertex'].to(device).float()
            pred = net(vertices)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print i

        test_vertices =  testset.dataset[testset.indices]['vertex'].to(device).float()
        test_labels = testset.dataset[testset.indices]['y']
        test_pred = net(test_vertices)
        clf_result = test_pred > 0.5
        acc = np.mean(clf_result.numpy() == test_labels.numpy())
        acc_list.append(acc)
        print acc_list
        """
        test_q0s.to(device)
        test_qgs.to(device)
        test_cols.to(device)
        test_pred = net(test_q0s, test_qgs, test_cols)
        clf_result = test_pred > 0.5
        acc = np.mean(clf_result.numpy() == test_labels.numpy())
        acc_list.append(acc)
        print acc_list
        print epoch, np.max(acc_list)
        """


if __name__ == '__main__':
    main()

