from generators.learning.Policy import Policy
from keras.layers import *
from keras.callbacks import *

import numpy as np


# This class setups up input variables and defines the abstract functions that need to be implemented
class PlacePolicy(Policy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        Policy.__init__(self, dim_action, dim_collision, save_folder, tau)
        self.n_key_confs = 291


        self.dim_poses = 4
        self.dim_collision = dim_collision

        # setup inputs
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # pose
        self.key_config_input = Input(shape=(self.n_key_confs, 4, 1), name='konf', dtype='float32')  # relative key config
        self.goal_flag_input = Input(shape=(self.n_key_confs, 4, 1), name='goal_flag', dtype='float32')  # goal flag (is_goal_r, is_goal_obj)

        # setup inputs related to detecting whether a key config is relevant
        self.cg_input = Input(shape=(dim_action,), name='cg', dtype='float32')  # action
        self.ck_input = Input(shape=(dim_action,), name='ck', dtype='float32')  # action
        self.collision_at_each_ck = Input(shape=(2,), name='ck', dtype='float32')  # action

        self.seed = config.seed

        self.policy_output = self.construct_policy_output()
        self.policy_model = self.construct_policy_model()

    def construct_policy_output(self):
        raise NotImplementedError

    def construct_policy_model(self):
        raise NotImplementedError

    def compute_policy_mse(self, data):
        # a proxy function for planning performance
        pred = self.policy_model.predict(
            [data['goal_flags'], data['rel_konfs'], data['states'], data['poses']])
        return np.mean(np.power(pred - data['actions'], 2))

    def create_callbacks_for_training(self):
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10),
            ModelCheckpoint(filepath=self.save_folder + self.weight_file_name+'.h5',
                            verbose=False,
                            save_best_only=True,
                            save_weights_only=True),
        ]
        return callbacks

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        raise NotImplementedError


