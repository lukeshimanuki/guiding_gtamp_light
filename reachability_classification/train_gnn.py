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
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset)-n_train])
    print "N_train", len(trainset)

    net = GNNReachabilityNet(trainset[1]['edges'], n_key_configs=618, device=device)
    net.to(device)
    loss_fn = nn.BCELoss()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)
    acc_list = []
    
    test_vertices = torch.from_numpy(testset.dataset[testset.indices[0:100]]['vertex']).float().to(device)
    test_labels = testset.dataset[testset.indices[0:100]]['y'].to(device)
    test_pred = net(test_vertices)
    clf_result = test_pred > 0.5
    acc = np.mean(clf_result.cpu().numpy() == test_labels.cpu().numpy())
    acc_list.append(acc)
    for epoch in range(100):
        print "Starting an epoch %d" % epoch
        stime = time.time()
        for i, batch in enumerate(trainloader, 0):
            stime = time.time()
            labels = batch['y'].to(device)
            vertices = batch['vertex'].float().to(device).float()
            print time.time()-stime
            optimizer.zero_grad()

            stime = time.time()
            pred = net(vertices)
            print time.time()-stime

            stime = time.time()
            loss = loss_fn(pred, labels)
            print time.time()-stime

            stime = time.time()
            loss.backward()
            print time.time()-stime

            stime = time.time()
            optimizer.step()
            print time.time()-stime

            import pdb;pdb.set_trace()
        print "Epoch took ", time.time()-stime
        test_pred = net(test_vertices)
        clf_result = test_pred > 0.5
        acc = np.mean(clf_result.cpu().numpy() == test_labels.cpu().numpy())
        acc_list.append(acc)
        print acc_list


if __name__ == '__main__':
    main()

