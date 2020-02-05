import socket
import torch

from datasets.GeneratorDataset import GNNDataset


def main():
    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print device
    seed = 0
    torch.cuda.manual_seed_all(seed)

    action_type = 'place'
    dataset = GNNDataset(action_type, 'home_region', True)

    n_train = int(len(dataset) * 0.8)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=20, pin_memory=True)
    n_test = min(5000, len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=20, pin_memory=True)
    asdf = trainset[0]
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
