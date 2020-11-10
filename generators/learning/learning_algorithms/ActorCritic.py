import torch
import torch.autograd as autograd
import torch.optim as optim

from torch_wgangp_models.fc_models import Generator, Discriminator

import socket
import torch
import numpy as np
import time
from matplotlib import pyplot as plt

from gtamp_utils import utils
import torch.nn as nn
import os


class ActorCritic:
    def __init__(self, config):
        self.config = config
        self.action_type = config.atype
        self.n_dim_actions = self.get_dim_action(self.action_type)
        self.seed = config.seed
        self.architecture = config.architecture
        self.region_name = config.region
        if socket.gethostname() == 'lab' or 'office' in socket.gethostname():
            self.device = torch.device('cpu')  # somehow even if I delete CUDA_VISIBLE_DEVICES, it still detects it?
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.problem_name = config.domain
        self.discriminator, self.generator = self.create_models()
        self.weight_dir = self.get_weight_dir(self.action_type, self.region_name)
        self.domain = self.get_domain(self.action_type, self.region_name)
        
        if not os.path.isdir(self.weight_dir):
            os.makedirs(self.weight_dir)

    @staticmethod
    def get_dim_action(action_type):
        if 'pick' in action_type:
            return 7
        else:
            return 4

    def get_domain(self, action_type, region_name):
        if 'place' in action_type:
            if region_name == 'loading_region':
                domain = np.array(
                    [[-0.34469225, -8.14641946, -1., -0.99999925], [3.92354742, -5.25567767, 1., 0.99999993]])
            else:
                domain = np.array(
                    [[-1.28392928, -2.95494754, -0.99999998, -0.99999999], [5.01948716, 2.58819546, 1., 1.]])
        else:
            domain = utils.get_pick_domain()
            portion, base_angle, facing_angle_offset = domain[0, 3:]
            grasp_params = domain[0, 0:3]
            base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
            min_domain = np.hstack([grasp_params, portion, base_angle, facing_angle_offset])
            min_domain[4:6] = np.array([-1, -1])

            portion, base_angle, facing_angle_offset = domain[1, 3:]
            grasp_params = domain[1, 0:3]
            base_angle = utils.encode_angle_in_sin_and_cos(base_angle)
            max_domain = np.hstack([grasp_params, portion, base_angle, facing_angle_offset])
            max_domain[4:6] = np.array([1, 1])

            domain = np.vstack([min_domain, max_domain])

        return domain

    def create_models(self):
        if self.architecture == 'fc':
            discriminator = Discriminator(self.n_dim_actions, self.action_type, self.region_name, self.problem_name)
            generator = Generator(self.n_dim_actions, self.action_type, self.region_name, self.problem_name)
        else:
            raise NotImplementedError
        discriminator.to(self.device)
        generator.to(self.device)
        return discriminator, generator

    def get_weight_dir(self, action_type, region_name):
        if 'place' in action_type:
            dir = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/{}/{}/seed_{}'.format(
                self.problem_name,
                self.config.num_episode,
                action_type,
                region_name,
                self.config.train_type,
                self.architecture,
                self.seed)
        else:
            dir = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/{}/seed_{}'.format(self.problem_name,
                                                                                                     self.config.num_episode,
                                                                                                     action_type,
                                                                                                     self.config.train_type,
                                                                                                     self.architecture,
                                                                                                     self.seed)
        return dir

    def get_data_from_dataloader(self, dataloader):
        dataset = dataloader.dataset[:]
        poses = torch.from_numpy(dataset['poses']).float().to(self.device)
        konf_obsts = torch.from_numpy(dataset['konf_obsts']).float().to(self.device)
        actions = torch.from_numpy(dataset['actions']).float().to(self.device)
        dist_to_goal = torch.from_numpy(dataset['dists_to_goal']).float().to(self.device)
        return poses, konf_obsts, actions, dist_to_goal

    def train(self, data_loader, test_set, n_train):
        batch_size = 32  # Batch size

        optimizerD = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))

        CRITIC_ITERS = 5  # How many critic iterations per generator iteration
        use_cuda = 'cuda' in self.device.type

        one = torch.FloatTensor([1])
        mone = one * -1
        if use_cuda:
            one = one.cuda()
            mone = mone.cuda()

        n_data_dim = self.n_dim_actions
        total_n_data = n_train
        total_iterations = 10000 * (total_n_data + 1) / batch_size  # 100 epochs
        total_iterations = 500

        def data_generator():
            while True:
                for d in data_loader:
                    yield d

        data_gen = data_generator()
        there_exists_cond_satisfied = False

        te_poses, te_konf_obsts, te_actions, te_dist_to_goal = self.get_data_from_dataloader(test_set)
        tr_poses, tr_konf_obsts, tr_actions, tr_dist_to_goal = self.get_data_from_dataloader(data_loader)

        test_losses = []
        train_losses = []
        min_test_loss = np.inf
        patience = 0
        for iteration in xrange(total_iterations):
            ############################
            # (1) Update D network
            ###########################
            for p in self.discriminator.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in self.generator update

            # print "%d / %d" % (iteration, total_iterations)

            _data = data_gen.next()
            poses = _data['poses'].float()
            konf_obsts = _data['konf_obsts'].float()
            actions = _data['actions'].float()
            dist_to_goal = _data['dists_to_goal'].float()
            if use_cuda:
                poses = poses.cuda()
                konf_obsts = konf_obsts.cuda()
                actions = actions.cuda()
            poses_v = autograd.Variable(poses)
            konf_obsts_v = autograd.Variable(konf_obsts)
            actions_v = autograd.Variable(actions)
            #self.discriminator.zero_grad()
            optimizerD.zero_grad()

            # train with MSE loss
            # 15 minutes remaining - fix this bug!
            D_real = self.discriminator(actions_v, konf_obsts_v, poses_v)
            loss = nn.MSELoss()
            D_loss = loss(D_real, dist_to_goal.reshape((len(dist_to_goal), 1)))
            D_loss.backward()
            optimizerD.step()

            # Evaluation
            te_val = self.discriminator(te_actions, te_konf_obsts, te_poses).squeeze()
            test_loss = loss(te_val, te_dist_to_goal)

            if test_loss < min_test_loss:
                min_test_loss = test_loss
                patience = 0
            else:
                patience += 1

            if patience >= 50:
                break
            
            tr_val = self.discriminator(tr_actions, tr_konf_obsts, tr_poses).squeeze() 
            train_loss = loss(tr_val, tr_dist_to_goal)

            test_losses.append(test_loss.detach().numpy())
            train_losses.append(train_loss.detach().numpy())

            # stopping criteria using test data
            print "Test %.2f Train %.2f" % (test_losses[-1], train_losses[-1])
            plt.figure()
            plt.plot(test_losses, 'r', label='test')
            plt.plot(train_losses, 'b', label='train')
            plt.legend()
            plt.savefig('./losses.png')
            plt.close('all')

        path = self.weight_dir + '/disc.pt'
        torch.save(self.discriminator.state_dict(), path)

        ############################
        # (2) Update G network
        ###########################
        patience = 0
        min_test_loss = np.inf
        for iteration in xrange(total_iterations):
            _data = data_gen.next()
            poses = _data['poses'].float()
            konf_obsts = _data['konf_obsts'].float()
            if use_cuda:
                poses = poses.cuda()
                konf_obsts = konf_obsts.cuda()
            poses_v = autograd.Variable(poses)
            konf_obsts_v = autograd.Variable(konf_obsts)

            for p in self.discriminator.parameters():
                p.requires_grad = False  # to avoid computation
            self.generator.zero_grad()

            noise = torch.randn(len(poses), n_data_dim)
            if use_cuda:
                noise = noise.cuda()
            noisev = autograd.Variable(noise)

            generated = self.generator(konf_obsts_v, poses_v, noisev)
            G = self.discriminator(generated, konf_obsts_v, poses_v)  # keep in mind that I am minimizing
            G = G.mean()
            G.backward()
            optimizerG.step()

            noise = torch.randn(len(te_konf_obsts), n_data_dim)
            te_generated = self.generator(te_konf_obsts, te_poses, noise).squeeze()
            test_loss = torch.mean(self.discriminator(te_generated, te_konf_obsts, te_poses))
            print "generator test value", test_loss
            if test_loss < min_test_loss:
                min_test_loss = test_loss
                patience = 0
            else:
                patience += 1

            if patience >= 50:
                break

        path = self.weight_dir + '/gen.pt'
        torch.save(self.discriminator.state_dict(), path)

        return True