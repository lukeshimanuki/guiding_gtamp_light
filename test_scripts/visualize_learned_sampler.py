from generators.learned_generator import LearnedGenerator
from generators.learning.utils.model_creation_utils import create_policy
from generators.learning.utils import sampler_utils
from generators.learning.PlacePolicyIMLE import noise
from generators.learning.utils import data_processing_utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from gtamp_problem_environments.mover_env import Mover

from gtamp_utils import utils
import numpy as np
import random
import pickle
import sys
import collections


def gaussian_noise(z_size):
    return np.random.normal(size=z_size, scale=0.5).astype('float32')


def uniform_noise(z_size):
    noise_dim = z_size[-1]
    return np.random.uniform([0] * noise_dim, [1] * noise_dim, size=z_size).astype('float32')


def get_pick_base_poses(action, smples):
    pick_base_poses = []
    for smpl in smples:
        smpl = smpl[0:4]
        sin_cos_encoding = smpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        smpl = np.hstack([smpl[0:2], decoded_angle])
        abs_base_pose = utils.get_absolute_pick_base_pose_from_ir_parameters(smpl, action.discrete_parameters['object'])
        pick_base_poses.append(abs_base_pose)
    return pick_base_poses


def get_place_base_poses(action, smples, mover):
    place_base_poses = smples[:, 4:]
    to_return = []
    for bsmpl in place_base_poses:
        sin_cos_encoding = bsmpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        bsmpl = np.hstack([bsmpl[0:2], decoded_angle])
        to_return.append(bsmpl)
    to_return = np.array(to_return)
    to_return[:, 0:2] += mover.regions[action.discrete_parameters['region']].box[0]
    return to_return


def compute_state(obj, region, problem_env):
    goal_entities = ['square_packing_box1', 'square_packing_box2', 'square_packing_box3', 'square_packing_box4',
                     'home_region']
    goal_entities = ['rectangular_packing_box1', 'rectangular_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4', 'home_region']
    return ConcreteNodeState(problem_env, obj, region, goal_entities)


def create_environment(problem_idx):
    problem_env = Mover(problem_idx)
    openrave_env = problem_env.env
    return problem_env, openrave_env


def visualize_in_training_env(problem_env, learned_sampler, plan):
    key_configs = pickle.load(open('prm.pkl', 'r'))[0]

    state = None
    utils.viewer()
    for action_idx, action in enumerate(plan):
        if 'pick' in action.type:
            associated_place = plan[action_idx + 1]
            state = compute_state(action.discrete_parameters['object'],
                                  associated_place.discrete_parameters['region'],
                                  problem_env)
            smpler = LearnedGenerator(action, problem_env, learned_sampler, state)
            smples = np.vstack([smpler.generate() for _ in range(10)])
            action.discrete_parameters['region'] = associated_place.discrete_parameters['region']
            pick_base_poses = get_pick_base_poses(action, smples)
            place_base_poses = get_place_base_poses(action, smples, problem_env)
            utils.visualize_path(place_base_poses)
        action.execute()


def visualize(problem_env, learned_sampler, target_obj_name, policy_mode):
    utils.viewer()
    target_obj = problem_env.env.GetKinBody(target_obj_name)

    orig_color = utils.get_color_of(target_obj)
    utils.set_color(target_obj, [1, 0, 0])

    target_obj.Enable(False)
    state = compute_state(target_obj_name, 'loading_region', problem_env)
    z_smpls = uniform_noise(z_size=(80, 4))

    # todo it would be informative to see how many of the samples are feasible
    if policy_mode == 'full':
        picks, places = sampler_utils.generate_pick_and_place_batch(state, learned_sampler, z_smpls)
        utils.visualize_path(picks[0:20, :])
        utils.visualize_path(picks[20:40, :])
        utils.visualize_path(picks[40:60, :])
        utils.visualize_path(picks[60:80, :])
    elif policy_mode == 'pick':
        picks = sampler_utils.generate_pick_or_place_batch(state, learned_sampler, z_smpls)
        if 'mse' in learned_sampler.weight_file_name:
            utils.visualize_path(picks[0:1, :])
        else:
            utils.visualize_path(picks[0:10, :])
            utils.visualize_path(picks[20:40, :])
            utils.visualize_path(picks[40:60, :])
            utils.visualize_path(picks[60:80, :])
    else:
        smpls = sampler_utils.generate_pick_or_place_batch(state, learned_sampler, z_smpls)
        utils.visualize_placements(smpls, target_obj_name)
    utils.set_color(target_obj, orig_color)


def main():
    seed = int(sys.argv[1])
    epoch = sys.argv[2]
    algo = str(sys.argv[3])
    placeholder_config_definition = collections.namedtuple('config', 'algo dtype tau seed atype')
    placeholder_config = placeholder_config_definition(
        algo=algo,
        tau=1.0,
        dtype='n_objs_pack_4',
        seed=seed,
        atype='pick'
    )

    problem_seed = 0
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)
    sampler = create_policy(placeholder_config)
    if 'mse' in algo:
        sampler.load_weights()
    else:
        if epoch == 'best':
            sampler.load_best_weights()
        else:
            sampler.load_weights('epoch_' + str(epoch))
    target_obj_name = 'rectangular_packing_box3'
    visualize(problem_env, sampler, target_obj_name, placeholder_config.atype)
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
