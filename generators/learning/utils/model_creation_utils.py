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
from generators.learning.PlacePolicyMSESelfAttentionDenseEvalNet import PlacePolicyMSEBestqkTransformation
from data_processing_utils import state_data_mode, action_data_mode

import collections


def load_weights(policy, seed, use_unregularized):
    if use_unregularized:
        weight_fname = 'without_regularizer/imle_pose_seed_%d.h5' % seed
    else:
        weight_fname = 'imle_pose_seed_%d.h5' % seed
    print "Loading weights at ", policy.save_folder + weight_fname
    policy.policy_model.load_weights(policy.save_folder + weight_fname)


def create_imle_model(seed, use_unregularized=False):
    n_key_configs = 615  # indicating whether it is a goal obj and goal region
    savedir = 'generators/learning/learned_weights/state_data_mode_%s_action_data_mode_%s/rel_konf_place_admon/' % (
        state_data_mode, action_data_mode)

    mconfig_type = collections.namedtuple('mconfig_type',
                                          'tau seed')

    config = mconfig_type(
        tau=1.0,
        seed=seed
    )

    dim_action = 4
    dim_state = (n_key_configs, 2, 1)
    policy = RelKonfIMLEPose(dim_action, dim_state, savedir, 1.0, config)
    load_weights(policy, seed, use_unregularized)
    return policy


def create_policy(config):
    n_key_configs = 291
    dim_state = (n_key_configs, 2, 1)
    dim_action = 4
    savedir = 'generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/%s/' % \
              (config.dtype, state_data_mode, action_data_mode, config.algo)
    if config.algo == "sa_mse":
        policy = PlacePolicyMSEBestqkTransformation(dim_action=dim_action,
                                                    dim_collision=dim_state,
                                                    save_folder=savedir,
                                                    tau=config.tau,
                                                    config=config)
    elif config.algo == "sa_mse_abs":
        policy = PlacePolicyMSESelfAttentionAbsolutePoses(dim_action=dim_action,
                                                          dim_collision=dim_state,
                                                          save_folder=savedir,
                                                          tau=config.tau,
                                                          config=config)
    elif config.algo == 'sa_imle':
        policy = PlacePolicyIMLESelfAttention(dim_action=dim_action, dim_collision=dim_state, save_folder=savedir,
                                              tau=config.tau, config=config)
    elif config.algo == 'ff_imle':
        policy = PlacePolicyIMLEFeedForward(dim_action=dim_action, dim_collision=dim_state, save_folder=savedir,
                                            tau=config.tau, config=config)
    elif config.algo == 'constrained':
        policy = PlacePolicyConstrainedOptimization(dim_action=dim_action, dim_collision=dim_state, save_folder=savedir,
                                                    tau=config.tau, config=config)

    elif config.algo == 'scorebased':
        policy = PlacePolicyMSEScoreBased(dim_action=dim_action, dim_collision=dim_state,
                                          save_folder=savedir, tau=config.tau, config=config)
    elif config.algo == 'qg_combination':
        policy = PlacePolicyMSECombinationOfQg(dim_action=dim_action, dim_collision=dim_state,
                                               save_folder=savedir, tau=config.tau, config=config)
    elif config.algo == 'imle_qg_combination':
        policy = PlacePolicyIMLECombinationOfQg(dim_action=dim_action, dim_collision=dim_state,
                                                save_folder=savedir, tau=config.tau, config=config)
    elif config.algo == 'sa_admon':
        policy = PlacePolicyAdMonSelfAttention(dim_action=dim_action, dim_collision=dim_state,
                                               save_folder=savedir, tau=config.tau, config=config)
    else:
        raise NotImplementedError
    return policy
