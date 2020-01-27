import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from classifiers.gnn import GNNReachabilityNet
from datasets.dataset import GNNReachabilityDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print device

    net = GNNReachabilityNet()
    net.to(device)
    loss_fn = nn.BCELoss()

    dataset = GNNReachabilityDataset()
    n_train = int(len(dataset) * 0.9)
    print "N_train", n_train
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])

    test_q0s = testset.dataset.q0s[testset.indices]
    test_qgs = testset.dataset.qgs[testset.indices]
    test_cols = testset.dataset.collisions[testset.indices]
    test_labels = testset.dataset.labels[testset.indices]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    acc_list = []
    for epoch in range(100):  # loop over the dataset multiple times
        for i, batch in enumerate(trainloader, 0):
            import pdb;pdb.set_trace()


            pred = net(batch, batch)
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
        print acc_list
        print epoch, np.max(acc_list)


if __name__ == '__main__':
    main()

