from trajectory_representation.concrete_node_state import ConcreteNodeState
from gtamp_utils import utils

from gtamp_problem_environments.mover_env import Mover
from planners.subplanners.motion_planner import BaseMotionPlanner

from generators.learning.pytorch_implementations.SuggestionNetwork import SuggestionNetwork
from generators.feasibility_checkers import two_arm_pick_feasibility_checker
from trajectory_representation.operator import Operator

import numpy as np
import random
import torch
import socket


def create_environment(problem_idx):
    problem_env = Mover(problem_idx)
    openrave_env = problem_env.env
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    return problem_env, openrave_env


def compute_state(obj, region, problem_env):
    goal_entities = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4', 'home_region']
    return ConcreteNodeState(problem_env, obj, region, goal_entities)


def visualize_samples(samples, problem_env, target_obj_name):
    target_obj = problem_env.env.GetKinBody(target_obj_name)

    orig_color = utils.get_color_of(target_obj)
    utils.set_color(target_obj, [1, 0, 0])

    utils.visualize_placements(samples, target_obj_name)
    utils.set_color(target_obj, orig_color)


def get_feasible_pick(problem_env, target_obj):
    pick_domain = utils.get_pick_domain()
    dim_parameters = pick_domain.shape[-1]
    domain_min = pick_domain[0]
    domain_max = pick_domain[1]
    smpls = np.random.uniform(domain_min, domain_max, (500, dim_parameters)).squeeze()

    feasibility_checker = two_arm_pick_feasibility_checker.TwoArmPickFeasibilityChecker(problem_env)
    op = Operator('two_arm_pick', {"object": target_obj})

    for smpl in smpls:
        pick_param, status = feasibility_checker.check_feasibility(op, smpl, parameter_mode='ir_params')
        if status == 'HasSolution':
            op.continuous_parameters = pick_param
            return op


def generate_smpls(problem_env, sampler, target_obj_name):
    # sample a feasible pick

    dim_noise = 3
    noise_smpls = torch.randn(100, dim_noise, device=sampler.device)

    # todo
    #   generate a feasible pick
    #   generate places based on this

    feasible_pick = get_feasible_pick(problem_env, target_obj)

    # x_vals = [batch['goal_poses'],  batch['obj_pose'], batch['q0'], batch['collision']]
    # pred = sampler(x_vals, chosen_noise_smpls)
    # return samples


def main():
    problem_seed = 0
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)

    if socket.gethostname() == 'lab':
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = SuggestionNetwork().to(device)

    obj_to_visualize = 'square_packing_box4'
    smpls = generate_smpls()
    visualize_samples(smpls, problem_env, obj_to_visualize)


if __name__ == '__main__':
    main()
