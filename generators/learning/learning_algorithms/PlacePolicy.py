from generators.learning.Policy import Policy
from keras.layers import *
from keras.callbacks import *
from keras import Model

import numpy as np


# This class setups up input variables and defines the abstract functions that need to be implemented
class PlacePolicy(Policy):
    def __init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder, config):
        Policy.__init__(self, dim_action, dim_collision, dim_poses, n_key_configs, save_folder)

        # setup inputs
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # pose
        self.key_config_input = Input(shape=(self.n_key_confs, 4, 1), name='konf', dtype='float32')  # relative key config
        self.goal_flag_input = Input(shape=(self.n_collisions, 4, 1), name='goal_flag', dtype='float32')  # goal flag (is_goal_r, is_goal_obj)

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

    def construct_model(self, output, name):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input, self.noise_input],
                      outputs=[output],
                      name=name)
        return model

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

    def generate_k_smples_for_multiple_states(self, states, noise_smpls):
        goal_flags, rel_konfs, collisions, poses = states
        k_smpls = []
        k = noise_smpls.shape[1]
        for j in range(k):
            actions = self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls[:, j, :]])
            k_smpls.append(actions)
        new_k_smpls = np.array(k_smpls).swapaxes(0, 1)
        return new_k_smpls

    @staticmethod
    def find_the_idx_of_closest_point_to_x1(x1, database):
        l2_distances = np.linalg.norm(x1 - database, axis=-1)
        return database[np.argmin(l2_distances)], np.argmin(l2_distances)

    def get_closest_noise_smpls_for_each_action(self, actions, generated_actions, noise_smpls):
        chosen_noise_smpls = []
        for true_action, generated, noise_smpls_for_action in zip(actions, generated_actions, noise_smpls):
            closest_point, closest_point_idx = self.find_the_idx_of_closest_point_to_x1(true_action, generated)
            noise_that_generates_closest_point_to_true_action = noise_smpls_for_action[closest_point_idx]
            chosen_noise_smpls.append(noise_that_generates_closest_point_to_true_action)
        return np.array(chosen_noise_smpls)


