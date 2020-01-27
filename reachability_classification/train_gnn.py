import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from classifiers.gnn import GNNReachabilityNet
from datasets.dataset import GNNReachabilityDataset
import socket
from torchsummary import summary


def main():
    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print device

    dataset = GNNReachabilityDataset()
    n_train = int(len(dataset) * 0.4)
    trainset, testset = torch.utils.data.random_split(dataset, [3433, len(dataset)-3433])
    print "N_train", len(trainset)

    net = GNNReachabilityNet(trainset[1]['edges'], n_key_configs=618, device=device)
    net.to(device)
    loss_fn = nn.BCELoss()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    acc_list = []

    test_vertices = torch.from_numpy(testset.dataset[testset.indices[0:100]]['vertex']).float().to(device)
    test_labels = testset.dataset[testset.indices[0:100]]['y'].to(device)
    import pdb;pdb.set_trace()
    test_pred = net(test_vertices)
    for epoch in range(100):
        for i, batch in enumerate(trainloader, 0):
            labels = batch['y'].to(device)
            vertices = batch['vertex'].float().to(device).float()
            optimizer.zero_grad()
            pred = net(vertices)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            print i, loss
        import pdb;pdb.set_trace()

        test_pred = net(test_vertices)
        clf_result = test_pred > 0.5
        acc = np.mean(clf_result.numpy() == test_labels.numpy())
        acc_list.append(acc)
        print acc_list


if __name__ == '__main__':
    main()

