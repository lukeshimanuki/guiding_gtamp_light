import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from classifiers.gnn import GNNReachabilityNet
from datasets.dataset import GNNReachabilityDataset
import socket
import time


def main():
    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print device

    dataset = GNNReachabilityDataset()
    n_train = int(len(dataset) * 0.9)
    n_train = 3000
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    print "N_train", len(trainset)

    net = GNNReachabilityNet(trainset[1]['edges'], n_key_configs=618, device=device)
    net.to(device)
    loss_fn = nn.BCELoss()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)
    acc_list = []

    n_test = min(500, len(testset))
    test_vertices = torch.from_numpy(testset.dataset[testset.indices[0:n_test]]['vertex']).float().to(device)
    test_labels = testset.dataset[testset.indices[0:n_test]]['y'].to(device)
    test_pred = net(test_vertices)
    clf_result = test_pred > 0.5
    acc = np.mean(clf_result.cpu().numpy() == test_labels.cpu().numpy())
    acc_list.append(acc)

    for epoch in range(100):
        print "Starting an epoch %d" % epoch
        stime_ = time.time()
        for i, batch in enumerate(trainloader, 0):
            labels = batch['y'].to(device)
            vertices = batch['vertex'].float().to(device).float()
            optimizer.zero_grad()

            pred = net(vertices)  # this is computing the forward pass
            loss = loss_fn(pred, labels)
            loss.backward()  # this is computing the dloss/dx for every layer
            optimizer.step()  # taking the gradient step

        print "Epoch took ", time.time() - stime_

        test_pred = net(test_vertices)
        clf_result = test_pred > 0.5
        acc = np.mean(clf_result.cpu().numpy() == test_labels.cpu().numpy())
        acc_list.append(acc)
        print acc_list


if __name__ == '__main__':
    main()
