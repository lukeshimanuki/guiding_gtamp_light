from PlacePolicyAdMon import PlacePolicyAdMon
from keras.models import Model
from keras.layers import *
from keras import backend as K


class PlacePolicyAdMonSelfAttention(PlacePolicyAdMon):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyAdMon.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_sa_admon_seed_%d' % config.seed

    def construct_critic(self):
        collision_input = Flatten()(self.collision_input)
        critic_input = Concatenate(axis=1, name='q0_ck')([self.pose_input, self.collision_input])
        raise NotImplementedError

    def construct_policy_output(self):
        raise NotImplementedError

