from keras import backend as K
from PlacePolicy import PlacePolicy

import socket
import numpy as np

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


class PlacePolicyMSE(PlacePolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicy.__init__(self, dim_action, dim_collision, save_folder, tau, config)

    def construct_policy_output(self):
        raise NotImplementedError

    def construct_policy_model(self):
        raise NotImplementedError

    def compute_policy_mse(self, data):
        pred = self.policy_model.predict(
            [data['goal_flags'], data['rel_konfs'], data['states'], data['poses']])
        return np.mean(np.power(pred - data['actions'], 2))

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_pretraining()

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        pre_mse = self.compute_policy_mse(test_data)
        self.policy_model.fit([goal_flags, rel_konfs, collisions, poses], actions,
                              batch_size=32,
                              epochs=epochs,
                              verbose=2,
                              callbacks=callbacks,
                              validation_split=0.1, shuffle=False)
        post_mse = self.compute_policy_mse(test_data)
        print "Pre-and-post test errors", pre_mse, post_mse

