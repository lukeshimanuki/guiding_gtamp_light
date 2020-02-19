import socket
import torch
from torch import nn

from datasets.GeneratorDataset import StandardDataset
from pytorch_implementations.SuggestionNetwork import SuggestionNetwork


def generate_k_smples_for_multiple_states(states, noise_smpls, net):
    # goal_flags, rel_konfs, collisions, poses = states
    k_smpls = []
    k = noise_smpls.shape[1]
    for j in range(k):
        actions = net(states, noise_smpls[:, j, :])
        k_smpls.append(actions[None, :])
    k_smpls = torch.cat(k_smpls)
    k_smpls = k_smpls.permute((1, 0, 2))
    return k_smpls


def find_the_idx_of_closest_point_to_x1(x1, database):
    l2_distances = (x1.float() - database.float()).norm(dim=-1)
    return database[torch.argmin(l2_distances)], torch.argmin(l2_distances)


def get_closest_noise_smpls_for_each_action(actions, generated_actions, noise_smpls):
    chosen_noise_smpls = []
    idx = 0
    for true_action, generated_action, noise_smpls_for_action in zip(actions, generated_actions, noise_smpls):
        closest_point, closest_point_idx = find_the_idx_of_closest_point_to_x1(true_action, generated_action)
        noise_that_generates_closest_point_to_true_action = noise_smpls_for_action[closest_point_idx]
        chosen_noise_smpls.append(noise_that_generates_closest_point_to_true_action[None, :])
        idx += 1
    return chosen_noise_smpls


def save_weights(net, epoch, action_type, seed):
    net_name = net.__class__._get_name(net)
    PATH = './generators/learning/torch_weights/atype_%s_%s_seed_%d_epoch_%d.pt' % (action_type, net_name, seed, epoch)
    torch.save(net.state_dict(), PATH)


def main():
    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SuggestionNetwork().to(device)

    print device
    seed = 0
    torch.cuda.manual_seed_all(seed)

    action_type = 'place'
    dataset = StandardDataset(action_type, 'home_region', True)

    n_train = int(len(dataset) * 100)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])

    batch_size = 32
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=20,
                                              pin_memory=True)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    num_smpl_per_state = 10
    dim_noise = 618
    for epoch in range(100):
        print "Starting an epoch %d" % epoch
        for i, batch in enumerate(trainloader, 0):
            target_actions = batch['actions'].to(device)
            import pdb;pdb.set_trace()
            """
            vertices = batch['vertex'].float().to(device).float()
            vertices = vertices[:, None, :, :]

            optimizer.zero_grad()

            noise_smpls = torch.randn(len(vertices), num_smpl_per_state, dim_noise, device=device)
            generated_actions = generate_k_smples_for_multiple_states(vertices, noise_smpls, net)
            chosen_noise_smpls = get_closest_noise_smpls_for_each_action(target_actions, generated_actions, noise_smpls)
            """

            chosen_noise_smpls = torch.cat(chosen_noise_smpls)
            pred = net(vertices, chosen_noise_smpls)  # this is computing the forward pass
            loss = loss_fn(pred, target_actions.float())
            loss.backward()  # this is computing the dloss/dx for every layer
            optimizer.step()  # taking the gradient step

        save_weights(net, epoch, action_type, seed)


if __name__ == '__main__':
    main()
