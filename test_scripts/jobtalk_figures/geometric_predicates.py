from generators.feasibility_checkers import two_arm_pick_feasibility_checker
from gtamp_problem_environments.mover_env import Mover
from gtamp_utils import utils
from trajectory_representation.operator import Operator
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState

from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.sahs import helper

import os
import numpy as np
import random
import pickle


def create_sampler(problem_env):
    pick_domain = utils.get_pick_domain()
    pick_dim_parameters = pick_domain.shape[-1]
    pick_domain_min = pick_domain[0]
    place_domain_max = pick_domain[1]

    def pick_smpler(n): return np.random.uniform(pick_domain_min, place_domain_max, (n, pick_dim_parameters)).squeeze()

    place_domain = utils.get_place_domain(problem_env.regions['home_region'])
    dim_parameters = place_domain.shape[-1]
    domain_min = place_domain[0]
    domain_max = place_domain[1]

    def place_smpler(n): return np.random.uniform(domain_min, domain_max, (n, dim_parameters)).squeeze()

    return pick_smpler, place_smpler


target_obj_name = 'rectangular_packing_box3'
target_obj_name = 'square_packing_box1'
target_obj_name = 'rectangular_packing_box2'


def visualize_pick(pick_smpler, problem_env):
    global target_obj_name
    target_obj = problem_env.env.GetKinBody(target_obj_name)

    orig_color = utils.get_color_of(target_obj)
    # utils.set_color(target_obj, [1, 0, 0])
    utils.set_robot_config(np.array([[2.10064864, 0.93038344, 0.28421403]]))

    raw_input('Take a picture of c0')

    pick_base_poses = []
    pick_smpls = pick_smpler(2000)
    for p in pick_smpls:
        _, pose = utils.get_pick_base_pose_and_grasp_from_pick_parameters(target_obj, p)
        pick_base_poses.append(pose)
    pick_base_poses = np.array(pick_base_poses)

    # utils.visualize_path(pick_base_poses[8:20, :])
    # utils.set_color(target_obj, orig_color)
    return pick_smpls


def get_feasible_pick(problem_env, smpls):
    global target_obj_name

    feasibility_checker = two_arm_pick_feasibility_checker.TwoArmPickFeasibilityChecker(problem_env)
    op = Operator('two_arm_pick', {"object": target_obj_name})

    for idx, param in enumerate(smpls):
        feasible_pick, status = feasibility_checker.check_feasibility(op, param, parameter_mode='ir_params')
        if status == 'HasSolution':
            print idx
            if idx == 98: continue  # idx=110
            op.continuous_parameters = feasible_pick
            break
    return op


def main():
    problem_idx = 0

    env_name = 'two_arm_mover'
    if env_name == 'one_arm_mover':
        problem_env = PaPOneArmMoverEnv(problem_idx)
        goal = ['rectangular_packing_box1_region'] + [obj.GetName() for obj in problem_env.objects[:3]]
        state_fname = 'one_arm_state_gpredicate.pkl'
    else:
        problem_env = PaPMoverEnv(problem_idx)
        goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4']
        goal_region = 'home_region'
        goal = [goal_region] + goal_objs
        state_fname = 'two_arm_state_gpredicate.pkl'

    problem_env.set_goal(goal)
    if os.path.isfile(state_fname):
        state = pickle.load(open(state_fname, 'r'))
    else:
        statecls = helper.get_state_class(env_name)
        state = statecls(problem_env, problem_env.goal)

    utils.viewer()

    if env_name == 'one_arm_mover':
        obj_name = 'c_obst9'
        pick_op = Operator(operator_type='one_arm_pick',
                           discrete_parameters={'object': problem_env.env.GetKinBody(obj_name)},
                           continuous_parameters=state.pick_params[obj_name][0])
        pick_op.execute()
        problem_env.env.Remove(problem_env.env.GetKinBody('computer_table'))
        utils.set_color(obj_name, [0.9, 0.8, 0.0])
        utils.set_color('c_obst0', [0, 0, 0.8])
        utils.set_obj_xytheta(np.array([[4.47789478, -0.01477591, 4.76236795]]), 'c_obst0')
        T_viewer = np.array([[-0.69618481, -0.41674492, 0.58450867, 3.62774134],
                             [-0.71319601, 0.30884202, -0.62925993, 0.39102399],
                             [0.08172004, -0.85495045, -0.51223194, 1.70261502],
                             [0., 0., 0., 1.]])

        viewer = problem_env.env.GetViewer()
        viewer.SetCamera(T_viewer)

        import pdb;
        pdb.set_trace()
        utils.release_obj()
    else:
        T_viewer = np.array([[0.99964468, -0.00577897, 0.02602139, 1.66357124],
                             [-0.01521307, -0.92529857, 0.37893419, -7.65383244],
                             [0.02188771, -0.37919541, -0.92505771, 6.7393589],
                             [0., 0., 0., 1.]])
        viewer = problem_env.env.GetViewer()
        viewer.SetCamera(T_viewer)

        import pdb;
        pdb.set_trace()
        # prefree and occludespre
        target = 'rectangular_packing_box4'
        utils.set_obj_xytheta(np.array([[0.1098148, -6.33305931, 0.22135689]]), target)
        utils.get_body_xytheta(target)
        utils.visualize_path(state.cached_pick_paths['rectangular_packing_box4'])
        import pdb;
        pdb.set_trace()

        # manipfree and occludesmanip
        pick_obj = 'square_packing_box2'
        pick_used = state.pick_used[pick_obj]
        utils.two_arm_pick_object(pick_obj, pick_used.continuous_parameters)
        utils.visualize_path(state.cached_place_paths[(u'square_packing_box2', 'home_region')])

    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
