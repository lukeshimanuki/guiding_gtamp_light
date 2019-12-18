from PlacePolicyAdMon import PlacePolicyAdMon
from keras.models import Model
from keras.layers import *
from keras import backend as K


class PlacePolicyAdMonSelfAttention(PlacePolicyAdMon):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyAdMon.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'place_sa_admon_seed_%d' % config.seed

    def construct_critic(self):
        a_input = RepeatVector(self.n_key_confs)(self.action_input)
        p_input = RepeatVector(self.n_key_confs)(self.pose_input)
        critic_input = Concatenate(axis=1, name='q0_qg_qk_ck')([a_input, p_input, self.collision_input])

    def construct_policy_output(self):
        p_input = RepeatVector(self.n_key_confs)(self.pose_input)
        p_input = Reshape((self.n_key_confs, self.dim_poses, 1))(p_input)
        import pdb;pdb.set_trace()
        critic_input = Concatenate(axis=2, name='q0_qk_ck')([p_input, self.collision_input])
        import pdb;pdb.set_trace()

