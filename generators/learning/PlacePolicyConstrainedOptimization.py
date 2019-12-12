from PlacePolicy import PlacePolicy
from keras.layers import *
from keras.models import Model
import keras

import tensorflow as tf
import numpy as np


def noise(z_size):
    return np.random.normal(size=z_size).astype('float32')


def inside_region_loss(region):
    def loss(_, action_prediction):
        return 1

    return loss


def collision_loss(key_config_obstacles):
    def loss(key_configs, action_prediction):
        return 1

    return loss


class PlacePolicyConstrainedOptimization(PlacePolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        self.dim_noise = dim_action
        self.noise_input = Input(shape=(self.dim_noise,), name='noise_input', dtype='float32')
        PlacePolicy.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.relevance_net_definition = self.construct_relevance_net()
        self.score_function_definition = self.construct_score_function()

        self.relevance_net = self.construct_model(self.relevance_net_definition, 'relnet')
        self.score_function = self.construct_model(self.score_function_definition, 'score')
        self.policy_model = self.construct_policy_model()
        self.loss_model = self.construct_loss_model()
        self.weight_file_name = 'constrained_optimization'

    def construct_loss_model(self):
        # loss_layer = [self.score_function_definition, self.policy_output]
        loss_layer = [self.policy_output, self.policy_output]
        loss_inputs = [self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                       self.noise_input]
        loss_model = Model(loss_inputs, loss_layer)

        def negative_of_score(_, pred):
            return -pred[:, 0]

        xmin = -8.13138832; xmax = 8.62905648
        ymin = -9.10388493; ymax = 4.36857242
        def out_of_region_loss(region_limits_relative_to_obj, predicted_actions):
            # todo: currently, actions are treated as if they are in absolute coordinate. Turn this into relative.
            action_x = predicted_actions[:, 0]
            action_y = predicted_actions[:, 1]
            xmin = region_limits_relative_to_obj[:, 0]
            xmax = region_limits_relative_to_obj[:, 1]
            ymin = region_limits_relative_to_obj[:, 2]
            ymax = region_limits_relative_to_obj[:, 3]
            smaller_than_xmin = tf.keras.backend.maximum(xmin - action_x, 0)
            bigger_than_xmax = tf.keras.backend.maximum(action_x - xmax, 0)
            smaller_than_ymin = tf.keras.backend.maximum(ymin - action_y, 0)
            bigger_than_ymax = tf.keras.backend.maximum(action_y - ymax, 0)
            return smaller_than_xmin + bigger_than_xmax + smaller_than_ymin + bigger_than_ymax

        min_dist_away_from_key_configs = 2
        def collision_loss(_, predicted_actions):
            in_collision = self.collision_input[:, :, 0]
            in_collision = tf.squeeze(in_collision, axis=-1)
            key_configs = tf.squeeze(self.key_config_input, axis=-1)
            predicted_actions = tf.expand_dims(predicted_actions, axis=1)
            predicted_actions = keras.backend.repeat_elements(predicted_actions, 615, axis=1)
            dists_to_key_configs = tf.norm(predicted_actions[:, :, 0:2] - key_configs[:, :, 0:2], axis=-1)
            distances = tf.keras.backend.maximum(min_dist_away_from_key_configs - dists_to_key_configs, 0)
            dists_to_key_configs_in_collision = distances * in_collision
            return tf.reduce_sum(dists_to_key_configs_in_collision, axis=-1) / tf.reduce_sum(in_collision, axis=-1)

        # output = Lambda(collision_loss)(self.policy_output)
        # self.collision_loss_model = Model(loss_inputs, output)
        # loss_model.compile(loss=[negative_of_score, out_of_region_loss], loss_weights=[1, 1],  optimizer='adam')

        loss_model.compile(loss=[out_of_region_loss, collision_loss], loss_weights=[1, 1], optimizer='adam')
        return loss_model

    def construct_policy_model(self):
        return self.construct_model(self.policy_output, 'policy_output')

    def construct_model(self, output, name):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[output],
                      name=name)
        return model

    def construct_relevance_net(self):
        candidate_qg = self.policy_output
        candidate_qg = RepeatVector(615)(candidate_qg)
        candidate_qg = Reshape((615, self.dim_poses, 1))(candidate_qg)
        key_config_input = self.key_config_input

        concat_input = Concatenate(axis=2, name='qg_pose')([candidate_qg, key_config_input])
        n_dim = concat_input.shape[2]._value
        n_filters = 32
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_input)
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        n_pose_features = 1
        relnet = Conv2D(filters=n_pose_features,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        activation='linear',
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        name='q0_qg_eval')(H)

        def compute_softmax(x):
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x, axis=-1)

        relnet = Lambda(compute_softmax, name='softmax_q0_qg')(relnet)
        relnet = Reshape((615, 1))(relnet)
        return relnet

    def construct_score_function(self):
        collision_input = self.collision_input

        def get_first_column(x):
            return x[:, :, 0]

        def get_second_column(x):
            return x[:, :, 1]

        col_flags = Lambda(get_first_column)(collision_input)
        col_free_flags = Lambda(get_second_column)(collision_input)
        collision_diff = Subtract()([col_free_flags, col_flags])
        self.collision_diff = self.construct_model(collision_diff, 'collision_diff')
        score = Dot(axes=1)([collision_diff, self.relevance_net_definition])
        score = Reshape((1,))(score)
        return score

    def construct_policy_output(self):
        dense_num = 32
        key_config_input = Flatten()(self.collision_input)
        concat_input = Concatenate()([key_config_input, self.noise_input])
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(concat_input)
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(hidden_action)
        action_output = Dense(self.dim_action,
                              activation='linear',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer,
                              name='policy_output')(hidden_action)
        return action_output

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_training()
        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        n_data = len(actions)

        noise_smpls = noise(z_size=(n_data, self.dim_noise))
        self.loss_model.fit([goal_flags, rel_konfs, collisions, poses, noise_smpls], [actions, actions],
                            batch_size=32,
                            epochs=epochs * 2,
                            verbose=2,
                            callbacks=callbacks,
                            shuffle=False)
        self.save_weights()
        """
        pre_mse = self.compute_policy_mse(test_data)
        self.policy_model.fit([goal_flags, rel_konfs, collisions, poses], actions,
                              batch_size=32,
                              epochs=epochs,
                              verbose=2,
                              callbacks=callbacks,
                              validation_split=0.1, shuffle=False)
        # load the best model
        self.load_weights()
        post_mse = self.compute_policy_mse(test_data)
        print "Pre-and-post test errors", pre_mse, post_mse
        """
