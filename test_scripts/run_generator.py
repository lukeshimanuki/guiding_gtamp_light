import argparse
import numpy as np
import random
import pickle
import time
import os
import torch

from gtamp_problem_environments.mover_env import PaPMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner
from generators.learning.learning_algorithms.WGANGP import WGANgp

from generators.samplers.uniform_sampler import UniformSampler
from generators.samplers.sampler import PlaceOnlyLearnedSampler, PickPlaceLearnedSampler
from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.samplers.voo_sampler import VOOSampler
from generators.voo import TwoArmVOOGenerator

from gtamp_utils import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-pidx', type=int, default=20000)  # used for threaded runs

    # epoch 41900 for loading region, 98400 for home region, 43700 for pick
    parser.add_argument('-epoch_home', type=int, default=111200)
    parser.add_argument('-epoch_loading', type=int, default=41900)
    parser.add_argument('-epoch_pick', type=int, default=242000)
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


def get_learned_smpler(epoch_home=None, epoch_loading=None, epoch_pick=None):
    region = 'home_region'
    if epoch_home is not None:
        action_type = 'place'
        home_place_model = WGANgp(action_type, region)
        home_place_model.load_weights(epoch_home)

    region = 'loading_region'
    if epoch_loading is not None:
        action_type = 'place'
        loading_place_model = WGANgp(action_type, region)
        loading_place_model.load_weights(epoch_loading)

    pick_model = None
    if epoch_pick is not None:
        action_type = 'pick'
        pick_model = WGANgp(action_type, region)
        pick_model.load_weights(epoch_pick)

    model = {'place_home': home_place_model, 'place_loading': loading_place_model, 'pick': pick_model}
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
    abstract_state = DummyAbstractState(problem_env, goal_entities)

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
        # utils.visualize_path(place_poses)
        utils.set_color(obj, color)
        import pdb;
        pdb.set_trace()
        action.execute()


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
        sampler_model = get_learned_smpler(epoch_home=config.epoch_home,
                                           epoch_loading=config.epoch_loading, epoch_pick=config.epoch_pick)
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
    pick_abs_poses = np.array([utils.get_pick_base_pose_and_grasp_from_pick_parameters(action.discrete_parameters['object'], s)[1] for s in pick_samples])
    #utils.visualize_path(pick_abs_poses)
    utils.visualize_path(place_poses)


def execute_policy(plan, problem_env, goal_entities, config):
    abstract_state = DummyAbstractState(problem_env, goal_entities)

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

        print plan_idx, len(samples_tried[plan_idx])
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
        else:
            print "No feasible action"
            problem_env.init_saver.Restore()
            plan_idx = 0
            for s in cont_smpl['samples']:
                samples_tried[plan_idx].append(s)
                sample_values[plan_idx].append(generator.sampler.infeasible_action_value)
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
    logfile_dir = 'generators/sampler_performances/'
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
