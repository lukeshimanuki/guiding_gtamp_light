import torch
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import os

from torch import nn


class Discriminator(nn.Module):
    def __init__(self, dim_data):
        nn.Module.__init__(self)
        n_hidden = 32
        n_konfs = 618 * 2
        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(n_konfs, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_poses = 24
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_poses, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        self.action_net = \
            nn.Sequential(
                torch.nn.Linear(dim_actions, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_input = n_hidden * 3
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, 1)
            )

    def forward(self, action, konf, pose):
        konf = konf.view((-1, 618*2))
        konf_val = self.konf_net(konf)
        pose_val = self.pose_net(pose)
        action_val = self.action_net(action)
        concat = torch.cat((konf_val, pose_val, action_val), -1)
        return self.output(concat)


class Generator(nn.Module):
    def __init__(self, dim_data):
        nn.Module.__init__(self)
        n_hidden = 32
        n_konfs = 618 * 2
        self.konf_net = \
            nn.Sequential(
                torch.nn.Linear(n_konfs, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_poses = 24
        self.pose_net = \
            nn.Sequential(
                torch.nn.Linear(dim_poses, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                nn.ReLU()
            )

        dim_actions = dim_data
        dim_input = n_hidden * 2
        self.output = \
            nn.Sequential(
                torch.nn.Linear(dim_input, n_hidden),
                nn.ReLU(),
                torch.nn.Linear(n_hidden, dim_actions)
            )

    def forward(self, konf, pose, noise):
        konf = konf.view((-1, 618*2))
        konf_val = self.konf_net(konf)
        pose_val = self.pose_net(pose)
        concat = torch.cat((konf_val, pose_val), -1)
        return self.output(concat)


def calc_gradient_penalty(discriminator, actions_v, konf_obsts_v, poses_v, fake_data, batch_size):
    lambda_val = .1  # Smaller lambda seems to help for toy tasks specifically

    alpha = torch.rand(len(actions_v), 1)
    alpha = alpha.expand(actions_v.size())
    alpha = alpha

    interpolates = alpha * actions_v + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates, konf_obsts_v, poses_v)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_val
    return gradient_penalty


class PlaceWGANgp:
    def __init__(self, problem_name):
        self.n_dim_actions = 4
        self.discriminator = Discriminator(self.n_dim_actions)
        self.generator = Generator(self.n_dim_actions)
        self.problem_name = problem_name
        self.weight_dir = './weights/wgangp/' + problem_name + '/'

        if not os.path.isdir(self.weight_dir):
            os.makedirs(self.weight_dir)

    def generate(self, n_data):
        noise = torch.randn(n_data, self.n_dim_actions)
        noisev = autograd.Variable(noise, volatile=True)
        samples = self.generator(noisev).cpu().data.numpy()
        return samples

    def load_weights(self, iteration, verbose=True):
        weight_file = self.weight_dir + '/gen_iter_%d.pt' % iteration
        if verbose:
            print "Loading weight file", weight_file
        self.generator.load_state_dict(torch.load(weight_file))
        weight_file = self.weight_dir + '/disc_iter_%d.pt' % iteration
        self.discriminator.load_state_dict(torch.load(weight_file))

    def train(self, data_loader, test_set, n_train):
        batch_size = 32  # Batch size

        optimizerD = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

        DIM = 512  # Model dimensionality
        MODE = 'wgan-gp'  # wgan or wgan-gp
        FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
        # Gaussian noise, as in the plots in the paper
        CRITIC_ITERS = 5  # How many critic iterations per generator iteration
        use_cuda = False

        one = torch.FloatTensor([1])
        mone = one * -1
        if use_cuda:
            one = one.cuda()
            mone = mone.cuda()

        n_data_dim = self.n_dim_actions
        total_n_data = n_train
        total_iterations = 1000 * (total_n_data + 1)

        def data_generator():
            while True:
                for d in data_loader:
                    yield d

        data_gen = data_generator()

        for iteration in xrange(total_iterations):
            ############################
            # (1) Update D network
            ###########################
            for p in self.discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in self.generator update

            for iter_d in xrange(CRITIC_ITERS):
                _data = data_gen.next()
                poses = _data['poses'].float()
                konf_obsts = _data['konf_obsts'].float()
                actions = _data['actions'].float()
                #real_data = torch.Tensor(_data)
                if use_cuda:
                    poses = poses.cuda()
                    konf_obsts = konf_obsts.cuda()
                    actions = actions.cuda()
                poses_v = autograd.Variable(poses)
                konf_obsts_v = autograd.Variable(konf_obsts)
                actions_v = autograd.Variable(actions)

                self.discriminator.zero_grad()

                # train with real
                D_real = self.discriminator(actions_v, konf_obsts_v, poses_v)
                D_real = D_real.mean()
                D_real.backward(mone)

                # train with fake
                noise = torch.randn(len(_data), n_data_dim)
                if use_cuda:
                    noise = noise.cuda()
                noisev = autograd.Variable(noise, volatile=True)  # totally freeze self.generator
                fake = autograd.Variable(self.generator(konf_obsts_v, poses_v, noisev).data)
                inputv = fake
                D_fake = self.discriminator(inputv, konf_obsts_v, poses_v)
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty(self.discriminator, actions_v.data, konf_obsts_v, poses_v, fake.data, batch_size)
                gradient_penalty.backward()

                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            _data = data_gen.next()
            poses = _data['poses'].float()
            konf_obsts = _data['konf_obsts'].float()
            actions = _data['actions'].float()
            # real_data = torch.Tensor(_data)
            if use_cuda:
                poses = poses.cuda()
                konf_obsts = konf_obsts.cuda()
                actions = actions.cuda()
            poses_v = autograd.Variable(poses)
            konf_obsts_v = autograd.Variable(konf_obsts)
            actions_v = autograd.Variable(actions)

            for p in self.discriminator.parameters():
                p.requires_grad = False  # to avoid computation
            self.generator.zero_grad()

            noise = torch.randn(batch_size, n_data_dim)
            if use_cuda:
                noise = noise.cuda()
            noisev = autograd.Variable(noise)
            fake = self.generator(konf_obsts_v, poses_v, noisev)
            G = self.discriminator(fake, konf_obsts_v, poses_v)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step()

            # Write logs and save samples
            if iteration % 100 == 0:
                print "Iteration %d / %d" % (iteration, total_iterations)
                path = self.weight_dir + '/disc_iter_%d.pt' % iteration
                torch.save(self.discriminator.state_dict(), path)
                path = self.weight_dir + '/gen_iter_%d.pt' % iteration
                torch.save(self.generator.state_dict(), path)

                test_data = test_set.dataset[:]
                poses = torch.from_numpy(test_data['poses']).float()
                konf_obsts = torch.from_numpy(test_data['konf_obsts']).float()
                actions = torch.from_numpy(test_data['actions']).float()
                noise = torch.randn(batch_size, n_data_dim)
                fake_actions = self.generator(konf_obsts, poses, noise)
                mse_loss = torch.nn.MSELoss()
                print mse_loss(fake_actions, actions).detach()

                # How can I evaluate what's going on?
                # Like, is mean MSE of 1.4818 good or bad?
                # Also, how big is its entropy?
