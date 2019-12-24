from generators.learned_generator import LearnedGenerator
from generators.learning.utils import model_creation_utils
from generators.feasibility_checkers import two_arm_pick_feasibility_checker
from trajectory_representation.operator import Operator
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils.data_processing_utils import action_data_mode
from gtamp_problem_environments.mover_env import Mover
from generators.learning.utils import sampler_utils
from gtamp_utils import utils
from generators.learning.PlacePolicyIMLECombinationOfQg import PlacePolicyIMLECombinationOfQg

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


def visualize_samples(samples, problem_env, target_obj_name, policy_mode):
    target_obj = problem_env.env.GetKinBody(target_obj_name)

    orig_color = utils.get_color_of(target_obj)
    utils.set_color(target_obj, [1, 0, 0])

    if policy_mode == 'full':
        picks, places = samples[0], samples[1]
        utils.visualize_path(picks[0:20, :])
        utils.visualize_path(picks[20:40, :])
        utils.visualize_path(picks[40:60, :])
        utils.visualize_path(picks[60:80, :])
    elif policy_mode == 'pick':
        if 'PICK_grasp_params_and_ir_parameters' in action_data_mode:
            pick_base_poses = []
            for p in samples:
                _, pose = utils.get_pick_base_pose_and_grasp_from_pick_parameters(target_obj, p)
                pick_base_poses.append(pose)
            pick_base_poses = np.array(pick_base_poses)
        elif 'PICK_grasp_params_and_abs_base' in action_data_mode:
            pick_base_poses = samples[:, 3:]
        else:
            raise NotImplementedError
        utils.visualize_path(pick_base_poses[0:10, :])
    else:
        utils.visualize_placements(samples, target_obj_name)
    utils.set_color(target_obj, orig_color)


def unprocess_pick_smpls(smpls):
    unprocessed = []
    if 'PICK_grasp_params_and_ir_parameters' in action_data_mode:
        for smpl in smpls:
            grasp_params = smpl[0:3]
            ir_parameters = smpl[3:]
            portion = ir_parameters[0]
            base_angle = utils.decode_sin_and_cos_to_angle(ir_parameters[1:3])
            facing_angle_offset = ir_parameters[3]
            unprocessed.append(np.hstack([grasp_params, portion, base_angle, facing_angle_offset]))
    elif 'PICK_grasp_params_and_abs_base' in action_data_mode:
        for smpl in smpls:
            grasp_params = smpl[0:3]
            abs_base_pose = utils.decode_pose_with_sin_and_cos_angle(smpl[3:])
            unprocessed.append(np.hstack([grasp_params, abs_base_pose]))
    else:
        raise NotImplementedError
    return np.array(unprocessed)


def generate_smpls(problem_env, sampler, target_obj_name, config):
    state = compute_state(target_obj_name, 'loading_region', problem_env)
    z_smpls = gaussian_noise(z_size=(200, 7))

    if config.atype == 'place':
        samples = sampler_utils.generate_pick_and_place_batch(state, sampler, z_smpls)
    elif config.atype == 'pick':
        samples = sampler_utils.make_predictions(state, sampler, z_smpls)
    else:
        raise NotImplementedError
    return samples


def get_pick_feasibility_rate(smpls, target_obj, problem_env):
    feasibility_checker = two_arm_pick_feasibility_checker.TwoArmPickFeasibilityChecker(problem_env)
    op = Operator('two_arm_pick', {"object": target_obj})

    if 'PICK_grasp_params_and_ir_parameters' in action_data_mode:
        parameter_mode = 'ir_params'
    else:
        parameter_mode = 'absolute_pose'

    n_success = 0
    for param in smpls:
        _, status = feasibility_checker.check_feasibility(op, param, parameter_mode=parameter_mode)
        n_success += status == 'HasSolution'
        print status

    total_samples = len(smpls)
    return n_success / float(total_samples) * 100


def get_place_feasibility_rate(pick_smpls, place_smpls, target_obj, problem_env):
    total_samples = len(pick_smpls)
    raise NotImplementedError


def create_policy(place_holder_config):
    if place_holder_config.atype == 'pick':
        policy = model_creation_utils.create_policy(place_holder_config)
    elif place_holder_config.atype == 'place':
        pick_savedir = './generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/%s/' % \
                       ('n_objs_pack_4', 'absolute', 'PICK_grasp_params_and_ir_parameters_PLACE_abs_base',
                        place_holder_config.algo)
        dim_action = 7
        n_key_configs = 291
        dim_collision = (n_key_configs, 2, 1)
        dim_pose = 24
        pick_place_holder_config = place_holder_config._replace(atype='pick')
        pick_policy = PlacePolicyIMLECombinationOfQg(dim_action=dim_action, dim_collision=dim_collision,
                                                     dim_pose=dim_pose,
                                                     save_folder=pick_savedir, config=pick_place_holder_config)
        dim_action = 4
        place_place_holder_config = place_holder_config._replace(atype='place')
        place_savedir = './generators/learning/learned_weights/dtype_%s_state_data_mode_%s_action_data_mode_%s/%s/' % \
                        ('n_objs_pack_4', 'absolute', 'PICK_grasp_params_and_abs_base_PLACE_abs_base',
                         place_holder_config.algo)
        place_policy = PlacePolicyIMLECombinationOfQg(dim_action=dim_action, dim_collision=dim_collision,
                                                      dim_pose=dim_pose,
                                                      save_folder=place_savedir, config=place_place_holder_config)
        policy = {'pick': pick_policy, 'place': place_policy}
    else:
        raise NotImplementedError
    return policy


def load_sampler_weights(sampler, config):
    if config.atype == 'pick':
        if 'mse' in config.algo:
            sampler.load_weights()
        else:
            if config.epoch == 'best':
                sampler.load_best_weights()
            else:
                sampler.load_weights('epoch_' + str(config.epoch))
    elif config.atype == 'place':
        if 'mse' in config.algo:
            sampler['pick'].load_weights()
            sampler['place'].load_weights()
        else:
            if config.epoch == 'best':
                sampler['pick'].load_best_weights()
                sampler['place'].load_best_weights()
            else:
                sampler['pick'].load_weights('epoch_' + str(config.epoch))
                sampler['place'].load_weights('epoch_' + str(config.epoch))
    else:
        raise NotImplementedError


def main():
    seed = int(sys.argv[1])
    epoch = sys.argv[2]
    algo = str(sys.argv[3])

    atype = 'place'
    placeholder_config_definition = collections.namedtuple('config', 'algo dtype tau seed atype epoch')
    placeholder_config = placeholder_config_definition(
        algo=algo,
        tau=1.0,
        dtype='n_objs_pack_4',
        seed=seed,
        atype=atype,
        epoch=epoch
    )

    problem_seed = 0
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)
    sampler = create_policy(placeholder_config)
    load_sampler_weights(sampler, placeholder_config)
    utils.viewer()

    target_obj_name = 'square_packing_box4'
    use_uniform = False
    if use_uniform:
        pick_domain = utils.get_pick_domain()
        dim_parameters = pick_domain.shape[-1]
        domain_min = pick_domain[0]
        domain_max = pick_domain[1]
        smpls = np.random.uniform(domain_min, domain_max, (200, dim_parameters)).squeeze()
    else:
        smpls = generate_smpls(problem_env, sampler, target_obj_name, placeholder_config)
        smpls = unprocess_pick_smpls(smpls)

    if atype == 'pick':
        feasibility_rate = get_pick_feasibility_rate(smpls, target_obj_name, problem_env)
        print 'Feasibility rate %.5f' % feasibility_rate
        raw_input("Press a button to visualize smpls")
    elif atype == 'place':
        # I need to generate the pick samples first
        feasibility_rate = get_place_feasibility_rate(smpls, target_obj_name, problem_env)

    visualize_samples(smpls, problem_env, target_obj_name, placeholder_config.atype)


if __name__ == '__main__':
    main()
