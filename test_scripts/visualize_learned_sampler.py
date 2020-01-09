from generators.learned_generator import LearnedGenerator
from generators.learning.utils import model_creation_utils
from generators.feasibility_checkers import two_arm_pick_feasibility_checker
from generators.feasibility_checkers import two_arm_place_feasibility_checker
from generators.feasibility_checkers import two_arm_pap_feasiblity_checker
from trajectory_representation.operator import Operator
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils.data_processing_utils import action_data_mode
from gtamp_problem_environments.mover_env import Mover
from generators.learning.utils import sampler_utils
from gtamp_utils import utils
from generators.learning.PlacePolicyIMLECombinationOfQg import PlacePolicyIMLECombinationOfQg
from planners.subplanners.motion_planner import BaseMotionPlanner
from generators.learning.utils import model_creation_utils

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
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
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
        # assumes that pick sampler is predicting ir parameters
        pick_base_poses = []
        for p in samples:
            _, pose = utils.get_pick_base_pose_and_grasp_from_pick_parameters(target_obj, p)
            pick_base_poses.append(pose)
        pick_base_poses = np.array(pick_base_poses)
        utils.visualize_path(pick_base_poses[0:10, :], )
    else:
        utils.visualize_placements(samples, target_obj_name)
    utils.set_color(target_obj, orig_color)


def unprocess_pick_smpls(smpls):
    unprocessed = []
    for smpl in smpls:
        grasp_params = smpl[0:3]
        ir_parameters = smpl[3:]
        portion = ir_parameters[0]
        base_angle = utils.decode_sin_and_cos_to_angle(ir_parameters[1:3])
        facing_angle_offset = ir_parameters[3]
        unprocessed.append(np.hstack([grasp_params, portion, base_angle, facing_angle_offset]))
    return np.array(unprocessed)


def unprocess_place_smpls(smpls):
    unprocessed = []
    for smpl in smpls:
        abs_base_pose = utils.decode_pose_with_sin_and_cos_angle(smpl)
        unprocessed.append(abs_base_pose)
    return np.array(unprocessed)


def generate_smpls(problem_env, sampler, target_obj_name, config):
    state = compute_state(target_obj_name, config.region, problem_env)
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

    n_success = 0
    for param in smpls:
        pick_param, status = feasibility_checker.check_feasibility(op, param, parameter_mode='ir_params')
        if status == 'HasSolution':
            motion, status = problem_env.motion_planner.get_motion_plan([pick_param['q_goal']], cached_collisions=None)
        n_success += status == 'HasSolution'
        # print status

    total_samples = len(smpls)
    return n_success / float(total_samples) * 100


def get_place_feasibility_rate(pick_smpls, place_smpls, target_obj, problem_env):
    # todo start from here
    pick_feasibility_checker = two_arm_pick_feasibility_checker.TwoArmPickFeasibilityChecker(problem_env)
    parameter_mode = 'ir_params'
    place_feasibility_checker = two_arm_place_feasibility_checker.TwoArmPlaceFeasibilityChecker(problem_env)
    n_success = 0
    for pick_smpl, place_smpl in zip(pick_smpls, place_smpls):
        pick_op = Operator('two_arm_pick', {"object": target_obj})
        pick_smpl, status = pick_feasibility_checker.check_feasibility(pick_op, pick_smpl,
                                                                       parameter_mode=parameter_mode)
        pick_op.continuous_parameters = pick_smpl
        if status == 'HasSolution':
            pick_op.execute()
            op = Operator('two_arm_place', {"object": target_obj, "place_region": 'loading_region'},
                          continuous_parameters=place_smpl)
            place_smpl, status = place_feasibility_checker.check_feasibility(op, place_smpl, parameter_mode='obj_pose')
            n_success += status == 'HasSolution'
            utils.two_arm_place_object(pick_op.continuous_parameters)

    total_samples = len(pick_smpls)
    return n_success / float(total_samples) * 100


def get_uniform_sampler_place_feasibility_rate(pick_smpls, place_smpls, target_obj, problem_env):
    feasibility_checker = two_arm_pap_feasiblity_checker.TwoArmPaPFeasibilityChecker(problem_env)
    op = Operator('two_arm_place', {"object": target_obj, "place_region": 'loading_region'})
    n_success = 0
    orig_xytheta = utils.get_robot_xytheta(problem_env.robot)
    for pick_smpl, place_smpl in zip(pick_smpls, place_smpls):
        parameters = np.hstack([pick_smpl, place_smpl])
        param, status = feasibility_checker.check_feasibility(op, parameters, swept_volume_to_avoid=None,
                                                              parameter_mode='obj_pose')
        """
        if status == 'HasSolution':
            motion, status = problem_env.motion_planner.get_motion_plan([param['pick']['q_goal']],
                                                                        cached_collisions=None)
            if status == 'HasSolution':
                utils.two_arm_pick_object(target_obj, param['pick'])
                motion, status = problem_env.motion_planner.get_motion_plan([param['place']['q_goal']],
                                                                            cached_collisions=None)
                utils.two_arm_place_object(param['pick'])
                utils.set_robot_config(orig_xytheta, problem_env.robot)
        """

        n_success += status == 'HasSolution'
    total_samples = len(pick_smpls)
    return n_success / float(total_samples) * 100


def create_policy(place_holder_config):
    if place_holder_config.atype == 'pick':
        pick_place_holder_config = place_holder_config._replace(atype='pick')
        pick_place_holder_config = pick_place_holder_config._replace(region='loading_region')
        pick_place_holder_config = pick_place_holder_config._replace(seed=0)
        pick_policy = model_creation_utils.create_policy(pick_place_holder_config, 291, 291,
                                                         given_action_data_mode='PICK_grasp_params_and_ir_parameters_PLACE_abs_base')
        policy = pick_policy
    elif place_holder_config.atype == 'place':
        pick_place_holder_config = place_holder_config._replace(atype='pick')
        pick_place_holder_config = pick_place_holder_config._replace(region='loading_region')
        pick_place_holder_config = pick_place_holder_config._replace(seed=0)
        pick_policy = model_creation_utils.create_policy(pick_place_holder_config, 291, 291,
                                                         given_action_data_mode='PICK_grasp_params_and_ir_parameters_PLACE_abs_base')
        place_place_holder_config = place_holder_config._replace(atype='place')
        place_place_holder_config = place_place_holder_config._replace(region=place_holder_config.region)

        if place_holder_config.region == 'loading_region':
            n_collisions = 291
            key_configs = pickle.load(open('placements_%s.pkl' % (place_holder_config.region), 'r'))
            n_key_configs = len(key_configs)
        else:
            n_collisions = 284
            key_configs = pickle.load(open('placements_%s.pkl' % (place_holder_config.region), 'r'))
            n_key_configs = len(key_configs)
        place_policy = model_creation_utils.create_policy(place_place_holder_config, n_collisions, n_key_configs,
                                                          given_action_data_mode='PICK_grasp_params_and_abs_base_PLACE_abs_base')
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
                sampler['pick'].load_best_weights()
                sampler['place'].load_weights('epoch_' + str(config.epoch))
    else:
        raise NotImplementedError


def get_smpls(problem_env, atype, sampler, target_obj_name, placeholder_config, use_uniform):
    if use_uniform:
        pick_domain = utils.get_pick_domain()
        dim_parameters = pick_domain.shape[-1]
        domain_min = pick_domain[0]
        domain_max = pick_domain[1]
        if atype == 'pick':
            smpls = np.random.uniform(domain_min, domain_max, (500, dim_parameters)).squeeze()
        else:
            pick_smpls = np.random.uniform(domain_min, domain_max, (500, dim_parameters)).squeeze()
            place_domain = utils.get_place_domain(region=problem_env.regions['loading_region'])
            dim_parameters = place_domain.shape[-1]
            domain_min = place_domain[0]
            domain_max = place_domain[1]
            place_smpls = np.random.uniform(domain_min, domain_max, (500, dim_parameters)).squeeze()
            smpls = (pick_smpls, place_smpls)
    else:
        smpls = generate_smpls(problem_env, sampler, target_obj_name, placeholder_config)
        if atype == 'pick':
            smpls = unprocess_pick_smpls(smpls)
        else:
            pick_smpls = unprocess_pick_smpls(smpls[0])
            place_smpls = unprocess_place_smpls(smpls[1])
            """
            place_domain = utils.get_place_domain(region=problem_env.regions['loading_region'])
            dim_parameters = place_domain.shape[-1]
            domain_min = place_domain[0]
            domain_max = place_domain[1]
            place_smpls = np.random.uniform(domain_min, domain_max, (200, dim_parameters)).squeeze()
            """
            smpls = (pick_smpls, place_smpls)

    return smpls


def check_feasibility_rate(problem_env, atype, sampler, placeholder_config):
    obj_names = ['square_packing_box1', 'square_packing_box2', 'square_packing_box3', 'square_packing_box4',
                 'rectangular_packing_box1', 'rectangular_packing_box2', 'rectangular_packing_box3',
                 'rectangular_packing_box4']
    use_uniform = False
    for target_obj_name in obj_names:
        smpls = get_smpls(problem_env, atype, sampler, target_obj_name, placeholder_config, use_uniform)

        if atype == 'pick':
            feasibility_rate = get_pick_feasibility_rate(smpls, target_obj_name, problem_env)
        elif atype == 'place':
            pick_smpls = smpls[0]
            place_smpls = smpls[1]
            if use_uniform:
                feasibility_rate = get_uniform_sampler_place_feasibility_rate(pick_smpls, place_smpls, target_obj_name,
                                                                              problem_env)
            else:
                feasibility_rate = get_uniform_sampler_place_feasibility_rate(pick_smpls, place_smpls, target_obj_name,
                                                                              problem_env)
            smpls = smpls[1]
        else:
            raise NotImplementedError

        print feasibility_rate


def main():
    seed = int(sys.argv[1])
    epoch = sys.argv[2]
    algo = str(sys.argv[3])

    atype = 'pick'
    placeholder_config_definition = collections.namedtuple('config', 'algo dtype tau seed atype epoch region')
    placeholder_config = placeholder_config_definition(
        algo=algo,
        tau=1.0,
        dtype='n_objs_pack_4',
        seed=seed,
        atype=atype,
        epoch=epoch,
        region='loading_region'
    )

    problem_seed = 0
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)
    sampler = create_policy(placeholder_config)
    load_sampler_weights(sampler, placeholder_config)

    check_feasibility_rate(problem_env, atype, sampler, placeholder_config)

    use_uniform = False
    utils.viewer()
    obj_to_visualize = 'rectangular_packing_box2'
    smpls = get_smpls(problem_env, atype, sampler, obj_to_visualize, placeholder_config, use_uniform)
    visualize_samples(smpls[1], problem_env, obj_to_visualize, atype)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
