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

from generators.sampler import UniformSampler, PlaceOnlyLearnedSampler, LearnedSampler, PickPlaceLearnedSampler
from generators.TwoArmPaPGeneratory import TwoArmPaPGenerator

from gtamp_utils import utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-pidx', type=int, default=0)  # used for threaded runs
    parser.add_argument('-epoch_home', type=int, default=None)  # used for threaded runs

    # epoch 41900 for loading region, 98400 for home region, 43700 for pick
    parser.add_argument('-epoch_loading', type=int, default=None)  # used for threaded runs
    parser.add_argument('-epoch_pick', type=int, default=None)  # used for threaded runs
    parser.add_argument('-seed', type=int, default=0)  # used for threaded runs
    config = parser.parse_args()
    return config


def create_environment(problem_idx):
    problem_env = PaPMoverEnv(problem_idx)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    return problem_env


def get_learned_smpler(problem_env, epoch_home=None, epoch_loading=None, epoch_pick=None):
    region = 'home_region'
    if epoch_home is not None:
        action_type = 'place'
        home_place_model = WGANgp(action_type, region)
        home_place_model.load_weights(epoch_home)
    else:
        region = problem_env.regions[region]
        home_place_model = UniformSampler(region)

    region = 'loading_region'
    if epoch_loading is not None:
        action_type = 'place'
        loading_place_model = WGANgp(action_type, region)
        loading_place_model.load_weights(epoch_loading)
    else:
        region = problem_env.regions[region]
        loading_place_model = UniformSampler(region)

    pick_model = None
    if epoch_pick is not None:
        action_type = 'pick'
        pick_model = WGANgp(action_type, region)
        pick_model.load_weights(epoch_pick)

    model = {'place_home': home_place_model, 'place_loading': loading_place_model, 'pick': pick_model}
    return model


def load_planning_experience_data(problem_seed):
    #raw_dir = './planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/' \
    #          'qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_' \
    #          'use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
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


def visualize_samplers_along_plan(plan, sampler_model, problem_env, goal_entities):
    abstract_state = DummyAbstractState(problem_env, goal_entities)
    utils.viewer()

    for action in plan:
        abstract_action = action
        action.execute_pick()

        if 'loading' in action.discrete_parameters['place_region']:
            chosen_sampler = sampler_model['place_loading']
        else:
            chosen_sampler = sampler_model['place_home']
        is_uniform_sampler = "Uniform" in chosen_sampler.__class__.__name__
        if is_uniform_sampler:
            sampler = chosen_sampler
            obj_placements = []
            for s in sampler.samples:
                s = s[-3:]
                utils.set_robot_config(s)
                obj_placements.append(utils.get_body_xytheta(sampler.obj))
        else:
            sampler = PlaceOnlyLearnedSampler(sampler_model, abstract_state, abstract_action,
                                              pick_abs_base_pose=action.continuous_parameters['pick']['q_goal'])
            obj_placements = [sampler.sample()[-3:] for _ in range(100)]

        utils.set_robot_config(abstract_action.continuous_parameters['pick']['q_goal'])
        utils.visualize_placements(obj_placements, sampler.obj)
        action.execute()


def execute_policy(plan, sampler_model, problem_env, goal_entities):
    abstract_state = DummyAbstractState(problem_env, goal_entities)

    total_ik_checks = 0
    total_mp_checks = 0
    total_infeasible_mp = 0
    plan_idx = 0
    n_total_actions = 0
    goal_reached = False
    stime = time.time()
    while plan_idx < len(plan):
        goal_reached = problem_env.is_goal_reached()
        if goal_reached:
            break
        if n_total_actions >= 200:
            break

        action = plan[plan_idx]
        if 'loading' in action.discrete_parameters['place_region']:
            chosen_sampler = sampler_model['place_loading']
        else:
            chosen_sampler = sampler_model['place_home']

        is_uniform_sampler = "Uniform" in chosen_sampler.__class__.__name__
        if is_uniform_sampler:
            print "Using uniform sampler"
            sampler = chosen_sampler
            generator = TwoArmPaPGenerator(abstract_state, action, sampler,
                                           n_parameters_to_try_motion_planning=5,
                                           n_iter_limit=200, problem_env=problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
        else:
            print "Using learned sampler"
            sampler = PlaceOnlyLearnedSampler(sampler_model, abstract_state, action)
            generator = TwoArmPaPGenerator(abstract_state, action, sampler,
                                           n_parameters_to_try_motion_planning=5,
                                           n_iter_limit=200, problem_env=problem_env,
                                           pick_action_mode='robot_base_pose',
                                           place_action_mode='robot_base_pose')
        cont_smpl = generator.sample_next_point()
        total_ik_checks += generator.n_ik_checks
        total_mp_checks += generator.n_mp_checks
        total_infeasible_mp += generator.n_mp_infeasible

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
        goal_reached = plan_idx == len(plan)
    print time.time() - stime
    return total_ik_checks, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_logfile_name(config):
    logfile_dir = 'generators/sampler_performances/'
    if not os.path.isdir(logfile_dir):
        os.makedirs(logfile_dir)

    is_uniform = config.epoch_home is None and config.epoch_loading is None
    if is_uniform:
        logfile = open(logfile_dir + 'uniform.txt', 'a')
    else:
        logfile = open(logfile_dir + 'epoch_home_%d_epoch_loading_%d.txt' % (config.epoch_home, config.epoch_loading),
                       'a')
    return logfile


def main():
    config = parse_arguments()
    np.random.seed(config.pidx)
    random.seed(config.pidx)

    plan, problem_env = load_planning_experience_data(config.pidx)
    goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3', 'rectangular_packing_box4']
    goal_region = 'home_region'
    problem_env.set_goal(goal_objs, goal_region)
    smpler = get_learned_smpler(problem_env, config.epoch_home, config.epoch_loading, config.epoch_pick)

    if config.v:
        utils.viewer()
        visualize_samplers_along_plan(plan, smpler, problem_env, goal_objs + [goal_region])

    set_seeds(config.seed)

    total_ik_checks, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached = \
        execute_policy(plan, smpler, problem_env, goal_objs + [goal_region])

    logfile = get_logfile_name(config)
    result_log = "%d,%d,%d,%d,%d,%d,%d\n" % (config.pidx, config.seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions)
    logfile.write(result_log)


if __name__ == '__main__':
    main()
