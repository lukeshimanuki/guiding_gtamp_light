from torch_importance_ratio_models.fc_models import FCImportanceRatioEstimator
import socket
import torch
import torch.optim as optim
import os

"""
def f_loss(y, w_pred):
    y_labels = y
    neg_mask = tf.equal(y_labels, 0)
    w_neg = tf.boolean_mask(w_pred, neg_mask)
    pos_mask = tf.equal(y_labels, 1)
    w_pos = tf.boolean_mask(w_pred, pos_mask)

    weight_sum_sqred_neg = tf.reduce_sum(tf.square(w_neg))
    loss1 = tf.cond(tf.equal(weight_sum_sqred_neg, 0), \
                    lambda: weight_sum_sqred_neg, \
                    lambda: (1 / 2.0) * tf.reduce_mean(tf.square(w_neg)))

    weight_sum_pos = tf.reduce_sum(w_pos)
    loss2 = tf.cond(tf.equal(weight_sum_pos, 0), \
                    lambda: weight_sum_pos, \
                    lambda: tf.reduce_mean(w_pos))

    loss3 = tf.maximum(-w_pred, 0)
    loss3 = tf.cond(tf.equal(tf.reduce_sum(loss3), 0), \
                    lambda: tf.reduce_sum(loss3), \
                    lambda: tf.reduce_mean(loss3))

    loss = loss1 - loss2
    return loss
"""


class ImportanceWeightEstimation:
    def __init__(self, config):
        if socket.gethostname() == 'lab':
            self.device = torch.device('cpu')  # somehow even if I delete CUDA_VISIBLE_DEVICES, it still detects it?
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = FCImportanceRatioEstimator(config)
        self.model.to(self.device)
        if config.atype == 'place':
            self.weight_dir = './generators/learning/learned_weights/{}/{}/{}/importance/{}/seed_{}'.format(
                config.domain,
                config.atype,
                config.region,
                config.architecture,
                config.seed)
        else:
            self.weight_dir = './generators/learning/learned_weights/{}/{}//importance/{}/seed_{}'.format(
                config.domain,
                config.atype,
                config.architecture,
                config.seed)

        if not os.path.isdir(self.weight_dir):
            os.makedirs(self.weight_dir)

    def evaluate_on_testset(self, iteration, te_poses, te_konf_obsts, te_actions, te_labels):
        w_values = self.model(te_actions, te_konf_obsts, te_poses)
        pos_w_values = w_values[te_labels == 1]
        neg_w_values = w_values[te_labels == 0]
        loss = torch.mean(neg_w_values ** 2) - 2 * torch.mean(pos_w_values)
        print('epoch {}, loss {}'.format(iteration, loss.item()))
        return loss.item()

    def load_weights(self):
        weight_file = self.weight_dir + '/best_weight_seed_%d.pt' % self.config.seed
        print "Loading weight", weight_file
        if 'cpu' in self.device.type:
            self.model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(weight_file))
            self.model.to(self.device)

    def predict(self, actions, konf_obsts, poses):
        return self.model(actions.to(self.device), konf_obsts.to(self.device), poses.to(self.device))

    def train(self, data_loader, test_data_loader, n_train):
        def data_generator():
            while True:
                for d in data_loader:
                    yield d
        data_gen = data_generator()

        # used to evaluate the w
        all_poses = torch.from_numpy(data_loader.dataset.poses).float().to(self.device)
        all_actions = torch.from_numpy(data_loader.dataset.actions).float().to(self.device)
        all_konf_obsts = torch.from_numpy(data_loader.dataset.konf_obsts).float().to(self.device)
        all_labels = torch.from_numpy(data_loader.dataset.labels).float().to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.9))
        testloss = None
        patience = 0
        patience_limit = 10
        best_loss = 99999
        use_cuda = 'cuda' in self.device.type

        for iteration in range(100000):
            _data = data_gen.next()
            poses = _data['poses'].float()
            konf_obsts = _data['konf_obsts'].float()
            actions = _data['actions'].float()
            labels = _data['labels'].float()
            if use_cuda:
                poses = poses.cuda()
                konf_obsts = konf_obsts.cuda()
                actions = actions.cuda()
                labels = labels.cuda()

            w_values = self.model(actions, konf_obsts, poses)
            pos_w_values = w_values[labels == 1]
            neg_w_values = w_values[labels == 0]

            loss = torch.mean(neg_w_values ** 2) - 2 * torch.mean(pos_w_values)
            self.model.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                testloss = self.evaluate_on_testset(iteration, all_poses, all_konf_obsts, all_actions, all_labels)
                if testloss < best_loss:
                    best_loss = testloss
                    path = self.weight_dir + '/best_weight_seed_%d.pt' % self.config.seed
                    torch.save(self.model.state_dict(), path)
                    patience = 0
                else:
                    patience += 1
                    if patience == patience_limit:
                        break
                print "Best loss so far", best_loss
