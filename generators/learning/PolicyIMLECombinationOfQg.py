from keras.layers import *
from keras import backend as K
from keras.models import Model

from PlacePolicyIMLECombinationOfQg import PlacePolicyIMLECombinationOfQg


class PolicyIMLECombinationOfQg(PlacePolicyIMLECombinationOfQg):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlacePolicyIMLECombinationOfQg.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.weight_file_name = 'policy_qg_combination_seed_%d' % config.seed  # todo change the name later
