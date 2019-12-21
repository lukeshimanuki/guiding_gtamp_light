# from generators.learning.PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput import \
#    PlacePolicyMSESelfAttentionEvalNetWithCandidateGoalAndCollisionInput
from generators.learning.PlacePolicyIMLESelfAttention import PlacePolicyIMLESelfAttention
from generators.learning.PlacePolicyIMLEFeedForward import PlacePolicyIMLEFeedForward
from generators.learning.PlacePolicyMSESelfAttentionAbsolutePoses import PlacePolicyMSESelfAttentionAbsolutePoses
from generators.learning.PlacePolicyConstrainedOptimization import PlacePolicyConstrainedOptimization
from generators.learning.PlacePolicyMSEScoreBased import PlacePolicyMSEScoreBased
from generators.learning.PlacePolicyMSECombinationOfQg import PlacePolicyMSECombinationOfQg
from generators.learning.PlacePolicyIMLECombinationOfQg import PlacePolicyIMLECombinationOfQg
from generators.learning.PlacePolicyAdMonSelfAttention import PlacePolicyAdMonSelfAttention
from generators.learning.PlacePolicyMSESelfAttention import PlacePolicyMSESelfAttention
from generators.learning.PlacePolicyMSETau import PlacePolicyMSETau
from generators.learning.PlacePolicyAdMonCombinationOfQg import PlacePolicyAdMonCombinationOfQg
from generators.learning.PlacePolicyMSETransformer import PlacePolicyMSETransformer

from generators.learning.PolicyIMLECombinationOfQg import PolicyIMLECombinationOfQg

from data_processing_utils import state_data_mode, action_data_mode

import socket

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra' or socket.gethostname() == 'dell-XPS-15-9560':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp_light/learned_weights/'


def load_weights(policy, seed, use_unregularized):
    if use_unregularized:
        weight_fname = 'without_regularizer/imle_pose_seed_%d.h5' % seed
    else:
        weight_fname = 'imle_pose_seed_%d.h5' % seed
    print "Loading weights at ", policy.save_folder + weight_fname
    policy.policy_model.load_weights(policy.save_folder + weight_fname)


def create_policy(config):
    n_key_configs = 291
    dim_collision = (n_key_configs, 2, 1)

    if config.atype == 'pick' or config.atype == 'place':
        dim_action = 4
    else:
        dim_action = 8

    if config.atype == 'pick':
        dim_pose = 20
    else:
        dim_pose = 20

    if ROOTDIR == './':
        savedir = './generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/%s/' % \
                  (config.dtype, state_data_mode, action_data_mode, config.algo)
    else:
        savedir = ''

    if config.algo == 'mse_qg_combination':
        policy = PlacePolicyMSECombinationOfQg(dim_action=dim_action, dim_collision=dim_collision, dim_pose=dim_pose,
                                               save_folder=savedir, config=config)
    elif config.algo == 'imle_qg_combination':
        policy = PlacePolicyIMLECombinationOfQg(dim_action=dim_action, dim_collision=dim_collision, dim_pose=dim_pose,
                                                save_folder=savedir, config=config)
    else:
        raise NotImplementedError
    return policy
