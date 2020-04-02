import socket
import torch

from datasets.GeneratorDataset import StandardDataset
from generators.learning.PlaceWGANGP import PlaceWGANgp

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp_light/learned_weights/'


def save_weights(net, epoch, action_type, seed, region):
    net_name = net.__class__._get_name(net)
    PATH = './learning/torch_weights/atype_%s_%s_region_%s_seed_%d_epoch_%d.pt' % (
    action_type, net_name, region, seed, epoch)
    torch.save(net.state_dict(), PATH)


def get_data_generator(action_type, region):
    dataset = StandardDataset(action_type, region, True)
    n_train = int(len(dataset) * 0.9)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20,
                                              pin_memory=True)
    return trainloader, trainset, testset


def main():
    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print device
    seed = 0
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    action_type = 'place'
    region = 'loading_region'
    trainloader, trainset, testset = get_data_generator(action_type, region)
    model = PlaceWGANgp(region)
    n_train = len(trainset)
    model.train(trainloader, testset, n_train)


if __name__ == '__main__':
    main()
