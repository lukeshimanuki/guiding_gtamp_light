import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

from classifiers.gnn import SimpleGNNReachabilityNet as GNNReachabilityNet
from classifiers.separate_q0_qg_qk_ck_gnn import Separateq0qgqkckGNNReachabilityNet
from classifiers.gnn_multiple_passes import SimpleMultiplePassGNNReachabilityNet
from classifiers.separate_q0_qg_qk_ck_gnn_multiple_passes import \
    Separateq0qgqkckMultiplePassGNNReachabilityNet as GNNReachabilityNet

from datasets.dataset import GNNReachabilityDataset
import socket
import time
import sys


def get_test_acc(testloader, net, device, n_test):
    accuracies = []
    for testset in testloader:
        test_vertices = testset['vertex'].float().to(device)
        test_labels = testset['y'].to(device)
        test_pred = net(test_vertices)
        clf_result = test_pred > 0.5
        accuracies.append(clf_result.cpu().numpy() == test_labels.cpu().numpy()[:, None])
        #print np.mean(np.vstack(accuracies)), len(np.vstack(accuracies))
        if len(np.vstack(accuracies)) >= n_test:
            break

    return np.mean(np.vstack(accuracies))


def save_weights(net, epoch, action_type, seed, n_msg_passing):
    gnn_name = net.__class__._get_name(net)
    PATH = './reachability_classification/weights/atype_%s_n_msgs_%d_seed_%d_%s_epoch_%d.pt' \
           % (action_type, n_msg_passing, seed, gnn_name, epoch)
    torch.save(net.state_dict(), PATH)


def main():
    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print device

    action_type = sys.argv[1]
    seed = int(sys.argv[2])
    n_msg_passing = int(sys.argv[3])
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = GNNReachabilityDataset(action_type)

    n_train = int(len(dataset) * 0.9)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    print "N_train", len(trainset)
    net = GNNReachabilityNet(trainset[1]['edges'], n_key_configs=618, device=device, n_msg_passing=n_msg_passing)
    net.to(device)
    loss_fn = nn.BCELoss()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)
    acc_list = []

    n_test = min(5000, len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=20, pin_memory=True)
    print "N test", len(testloader)

    test_acc = get_test_acc(testloader, net, device, n_test)
    acc_list.append(test_acc)
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
        test_acc = get_test_acc(testloader, net, device, n_test)
        acc_list.append(test_acc)
        if test_acc == np.max(acc_list):
            save_weights(net, epoch, action_type, seed, n_msg_passing)
        print acc_list, np.max(acc_list)


if __name__ == '__main__':
    main()
