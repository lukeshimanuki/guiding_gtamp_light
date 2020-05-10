import argparse
import numpy as np
import random
import pickle
import time
import os
import torch
import socket

from gtamp_problem_environments.mover_env import PaPMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner
from generators.learning.learning_algorithms.WGANGP import WGANgp

from generators.samplers.uniform_sampler import UniformSampler
from generators.samplers.sampler import PlaceOnlyLearnedSampler, PickPlaceLearnedSampler
from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.samplers.voo_sampler import VOOSampler
from generators.voo import TwoArmVOOGenerator

from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState

from gtamp_utils import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-pidx', type=int, default=20000)  # used for threaded runs

    # epoch 18700 for loading region, 86400 for home region, 43700 for pick
    # home 96000 for fc
    # home 98100 for cnn
    # loading 34500 for cnn
    # loading 9700 for fc
    #     result_dir = weight dir + /results_summary
    #     result_files = os.listdir(result_dir + '/')
    #  for i, kde, mse, entropy in zip(iterations, results[:, 1], results[:, 0], results[:, 2]):
    #     print i, kde, mse, entropy
    # parser.add_argument('-epoch_home', type=int, default=98100)
    # parser.add_argument('-epoch_loading', type=int, default=9700)
    # parser.add_argument('-epoch_pick', type=int, default=100700)
    parser.add_argument('-pick_architecture', type=str, default='fc')
    parser.add_argument('-place_architecture', type=str, default='fc')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-sampling_strategy', type=str, default='unif')  # used for threaded runs
    parser.add_argument('-use_learning', action='store_true', default=False)  # used for threaded runs
    parser.add_argument('-n_mp_limit', type=int, default=5)  # used for threaded runs
    config = parser.parse_args()
    return config


def create_environment(problem_idx):
    problem_env = PaPMoverEnv(problem_idx)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    return problem_env


def get_learned_smpler(config):
    region = 'home_region'
    action_type = 'place'
    home_place_model = WGANgp(action_type, region, architecture=config.place_architecture)
    home_place_model.load_best_weights()

    region = 'loading_region'
    action_type = 'place'
    loading_place_model = WGANgp(action_type, region, architecture=config.place_architecture)
    loading_place_model.load_best_weights()

    """
    pick_model = None
    if epoch_pick is not None:
        action_type = 'pick'
        pick_model = None  # WGANgp(action_type, region)
        # pick_model.load_weights(epoch_pick)
    """

    model = {'place_home': home_place_model, 'place_loading': loading_place_model, 'pick': None}
    return model


def load_planning_experience_data(problem_seed):
    raw_dir = './planning_experience/for_testing_generators/'
    fname = 'pidx_%d_planner_seed_0_gnn_seed_0.pkl' % problem_seed

    plan_data = pickle.load(open(raw_dir + fname, 'r'))
    plan = plan_data['plan']

    np.random.seed(problem_seed)
    random.seed(problem_seed)

    problem_env = create_environment(problem_seed)

    return plan, problem_env


class DummyAbstractState:
    def __init__(self, problem_env, goal_entities):
        self.prm_vertices, self.prm_edges = pickle.load(open('./prm.pkl', 'rb'))
        self.problem_env = problem_env
        self.goal_entities = goal_entities


def visualize_samplers_along_plan(plan, problem_env, goal_entities, config):
    #    abstract_state = DummyAbstractState(problem_env, goal_entities)
    # abstract_state = pickle.load(open('temp.pkl', 'r'))
    abstract_state = ShortestPathPaPState(problem_env, goal_entities)
    abstract_state.make_plannable(problem_env)

    for action in plan:
        abstract_action = action
        generator = get_generator(config, abstract_state, action, 5, problem_env)
        sampler = generator.sampler

        # getting the samples
        samples = [sampler.sample() for _ in range(100)]
        place_poses = [smpl[-3:] for smpl in samples]

        pick_samples = [smpl[:-3] for smpl in samples]
        pick_abs_poses = np.array(
            [utils.get_pick_base_pose_and_grasp_from_pick_parameters(abstract_action.discrete_parameters['object'], s)[
                 1] for s in pick_samples])
        # utils.visualize_path(pick_abs_poses[0:50, :])

        obj = action.discrete_parameters['object']
        color = utils.get_color_of(obj)
        utils.set_color(obj, [0, 0, 1])
        utils.visualize_path(place_poses[0:20])
        utils.set_color(obj, color)
        print action.discrete_parameters
        import pdb;
        pdb.set_trace()
        action.execute()
        abstract_state = ShortestPathPaPState(problem_env, goal_entities,
                                              parent_state=abstract_state, parent_action=action)


def get_generator(config, abstract_state, action, n_mp_limit, problem_env):
    if not config.use_learning:
        print "Using {} sampler".format(config.sampling_strategy)
        if config.sampling_strategy == 'unif':
            sampler = UniformSampler(action.discrete_parameters['place_region'])
            sampler.infeasible_action_value = -9999
            generator = TwoArmPaPGenerator(abstract_state, action, sampler,
                                           n_parameters_to_try_motion_planning=n_mp_limit,
                                           n_iter_limit=2000, problem_env=problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
        else:
            sampler = VOOSampler(action.discrete_parameters['object'],
                                 action.discrete_parameters['place_region'], 0.3, -9999)
            generator = TwoArmVOOGenerator(abstract_state, action, sampler,
                                           n_parameters_to_try_motion_planning=n_mp_limit,
                                           n_iter_limit=2000, problem_env=problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
    else:
        print "Using learned sampler"
        sampler_model = get_learned_smpler(config)
        sampler = PlaceOnlyLearnedSampler(sampler_model, abstract_state, action)
        sampler.infeasible_action_value = -9999
        generator = TwoArmPaPGenerator(abstract_state, action, sampler,
                                       n_parameters_to_try_motion_planning=n_mp_limit,
                                       n_iter_limit=2000, problem_env=problem_env,
                                       pick_action_mode='ir_parameters',
                                       place_action_mode='robot_base_pose')
    return generator


def visualize_samples(action, sampler):
    samples = [sampler.sample() for _ in range(30)]
    place_poses = [smpl[-3:] for smpl in samples]
    pick_samples = [smpl[:-3] for smpl in samples]
    pick_abs_poses = np.array(
        [utils.get_pick_base_pose_and_grasp_from_pick_parameters(action.discrete_parameters['object'], s)[1] for s in
         pick_samples])
    # utils.visualize_path(pick_abs_poses)
    utils.visualize_path(place_poses)


def execute_policy(plan, problem_env, goal_entities, config):
    # init_abstract_state = DummyAbstractState(problem_env, goal_entities)
    # init_abstract_state = pickle.load(open('temp.pkl', 'r'))
    init_abstract_state = ShortestPathPaPState(problem_env, goal_entities)
    abstract_state = init_abstract_state
    abstract_state.make_plannable(problem_env)
    goal_objs = goal_entities[:-1]

    total_ik_checks = 0
    total_mp_checks = 0
    total_pick_mp_checks = 0
    total_place_mp_checks = 0

    total_pick_mp_infeasible = 0
    total_place_mp_infeasible = 0

    total_infeasible_mp = 0
    plan_idx = 0
    n_total_actions = 0
    goal_reached = False
    stime = time.time()
    samples_tried = {i: [] for i in range(len(plan))}
    sample_values = {i: [] for i in range(len(plan))}
    n_mp_limit = config.n_mp_limit
    while plan_idx < len(plan):
        goal_reached = problem_env.is_goal_reached()
        if goal_reached:
            break
        if n_total_actions >= 50:
            break

        action = plan[plan_idx]

        generator = get_generator(config, abstract_state, action, n_mp_limit, problem_env)

        stime = time.time()
        cont_smpl = generator.sample_next_point(samples_tried[plan_idx], sample_values[plan_idx])
        print time.time() - stime
        total_ik_checks += generator.n_ik_checks
        total_mp_checks += generator.n_mp_checks
        total_infeasible_mp += generator.n_mp_infeasible
        total_pick_mp_checks += generator.n_pick_mp_checks
        total_place_mp_checks += generator.n_place_mp_checks
        total_pick_mp_infeasible += generator.n_pick_mp_infeasible
        total_place_mp_infeasible += generator.n_place_mp_infeasible

        n_total_actions += 1

        if cont_smpl['is_feasible']:
            print "Action executed"
            action.continuous_parameters = cont_smpl
            action.execute()
            plan_idx += 1
            abstract_state = ShortestPathPaPState(problem_env, goal_entities,
                                                  parent_state=abstract_state, parent_action=action)
        else:
            print "No feasible action"
            problem_env.init_saver.Restore()
            plan_idx = 0
            for s in cont_smpl['samples']:
                samples_tried[plan_idx].append(s)
                sample_values[plan_idx].append(generator.sampler.infeasible_action_value)
            abstract_state = init_abstract_state
        goal_reached = plan_idx == len(plan)
        print "Total IK checks {} Total actions {}".format(total_ik_checks, n_total_actions)

    print time.time() - stime
    return total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible, total_place_mp_checks, \
           total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_logfile_name(config):
    logfile_dir = 'generators/sampler_performances/{}/'.format(socket.gethostname())
    if not os.path.isdir(logfile_dir):
        os.makedirs(logfile_dir)

    if config.use_learning:
        logfile = open(logfile_dir + 'epoch_home_%d_epoch_loading_%d.txt' % (config.epoch_home, config.epoch_loading),
                       'a')
    else:
        logfile = open(logfile_dir + config.sampling_strategy + '_sqrt_pap_mps_n_mp_limit_%d.txt' % config.n_mp_limit,
                       'a')
    return logfile


def main():
    config = parse_arguments()

    plan, problem_env = load_planning_experience_data(config.pidx)
    goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3', 'rectangular_packing_box4']
    goal_region = 'home_region'
    problem_env.set_goal(goal_objs, goal_region)

    if config.v:
        utils.viewer()
        visualize_samplers_along_plan(plan, problem_env, goal_objs + [goal_region], config)

    set_seeds(config.seed)

    total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible, total_place_mp_checks, \
    total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached = \
        execute_policy(plan, problem_env, goal_objs + [goal_region], config)

    logfile = get_logfile_name(config)
    result_log = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % (
        config.pidx, config.seed, total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible,
        total_place_mp_checks,
        total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached
    )
    logfile.write(result_log)


if __name__ == '__main__':
    main()
