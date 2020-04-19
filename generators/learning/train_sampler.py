import os
import pickle
import numpy as np
import random
import argparse

from generators.learning.utils.model_creation_utils import create_policy


def parse_args():
    parser = argparse.ArgumentParser(description='Process configurations')
    parser.add_argument('-n_data', type=int, default=100)
    parser.add_argument('-d_lr', type=float, default=1e-3)
    parser.add_argument('-g_lr', type=float, default=1e-4)
    parser.add_argument('-algo', type=str, default='mse')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-tau', type=float, default=0.0)
    parser.add_argument('-dtype', type=str, default='n_objs_pack_1')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-region', type=str, default='loading_region')
    parser.add_argument('-filtered', action='store_true')
    parser.add_argument('-debug', action='store_true')
    args = parser.parse_args()
    return args


configs = parse_args()
np.random.seed(configs.seed)
random.seed(configs.seed)
os.environ['PYTHONHASHSEED'] = str(configs.seed)

import tensorflow as tf
from gtamp_utils import utils
from generators.learning.utils.data_load_utils import get_data
from utils.data_processing_utils import action_data_mode
tf.set_random_seed(configs.seed)


def train(config):
    if config.debug:
        states, poses, goal_flags, actions, sum_rewards = pickle.load(open('tmp_data_for_debug_train_sampler.pkl', 'r'))

    else:
        states, poses, goal_flags, actions, sum_rewards = get_data(config.dtype, config.atype, config.region,
                                                                   config.filtered)
        pickle.dump([states[0:256, :], poses[0:256, :], goal_flags[0:256, :], actions[0:256, :], sum_rewards[0:256, :]],
                    open('tmp_data_for_debug_train_sampler.pkl', 'wb'))

    if config.atype == 'pick':
        actions = actions[:, :-4]
    elif config.atype == 'place':
        must_get_q0_from_pick_abs_pose = action_data_mode == 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
        assert must_get_q0_from_pick_abs_pose
        pick_abs_poses = actions[:, 3:7]  # must swap out the q0 with the pick base pose
        poses[:, -4:] = pick_abs_poses
        actions = actions[:, -4:]
    else:
        raise NotImplementedError

    #### This perhaps needs to be refactored ####
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    """
    if config.region != 'home_region':
        indices_to_delete = sampler_utils.get_indices_to_delete(config.region, key_configs)
        key_configs = np.delete(key_configs, indices_to_delete, axis=0)
        states = np.delete(states, indices_to_delete, axis=1)
        goal_flags = np.delete(goal_flags, indices_to_delete, axis=1)
    """
    ############

    # key_configs = [utils.decode_pose_with_sin_and_cos_angle(a) for a in actions]
    # key_configs = filter_configs_that_are_too_close(key_configs)
    # pickle.dump(key_configs, open('placements_%s.pkl' %(config.region), 'wb'))
    key_configs = np.array([utils.encode_pose_with_sin_and_cos_angle(p) for p in key_configs])
    n_key_configs = len(key_configs)
    key_configs = key_configs.reshape((1, n_key_configs, 4, 1))
    key_configs = key_configs.repeat(len(poses), axis=0)

    print "Number of data", len(states)
    n_collisions = states.shape[1]
    assert n_key_configs == n_collisions
    assert n_key_configs == 618
    policy = create_policy(config, n_collisions, n_key_configs)
    # policy.policy_model.summary()
    policy.train_policy(states, poses, key_configs, goal_flags, actions, sum_rewards)


def main():
    train(configs)


if __name__ == '__main__':
    main()
