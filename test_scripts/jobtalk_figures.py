from generators.feasibility_checkers import two_arm_pick_feasibility_checker
from gtamp_problem_environments.mover_env import Mover
from gtamp_utils import utils
from trajectory_representation.operator import Operator
from planners.subplanners.motion_planner import BaseMotionPlanner

import numpy as np
import random
import pickle


def create_sampler(problem_env):
    pick_domain = utils.get_pick_domain()
    pick_dim_parameters = pick_domain.shape[-1]
    pick_domain_min = pick_domain[0]
    place_domain_max = pick_domain[1]

    def pick_smpler(n): return np.random.uniform(pick_domain_min, place_domain_max, (n, pick_dim_parameters)).squeeze()

    place_domain = utils.get_place_domain(problem_env.regions['loading_region'])
    dim_parameters = place_domain.shape[-1]
    domain_min = place_domain[0]
    domain_max = place_domain[1]

    def place_smpler(n): return np.random.uniform(domain_min, domain_max, (n, dim_parameters)).squeeze()

    return pick_smpler, place_smpler


target_obj_name = 'rectangular_packing_box3'
target_obj_name = 'square_packing_box1'


def visualize_pick(pick_smpler, problem_env):
    global target_obj_name
    target_obj = problem_env.env.GetKinBody(target_obj_name)

    orig_color = utils.get_color_of(target_obj)
    # utils.set_color(target_obj, [1, 0, 0])

    raw_input('Take a picture of c0')

    pick_base_poses = []
    pick_smpls = pick_smpler(200)
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

    problem_env = Mover(problem_idx, problem_type='jobtalk')
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'rrt'))
    utils.viewer()

    seed = 1
    np.random.seed(seed)
    random.seed(seed)

    pick_smpler, place_smpler = create_sampler(problem_env)

    # Set camera view - to get camera transform: viewer.GetCameraTransform()
    cam_transform = np.array([[-0.54866337, -0.70682829, 0.44650004, -1.45953619],
                              [-0.83599448, 0.45806221, -0.30214604, 2.02016926],
                              [0.00904058, -0.53904803, -0.8422265, 4.88620949],
                              [0., 0., 0., 1.]])
    cam_transform = np.array([[0.76808539, 0.51022899, -0.38692533, 2.7075901],
                              [0.63937823, -0.57785198, 0.50723029, -2.0267117],
                              [0.03521803, -0.6369878, -0.77006898, 4.52542162],
                              [0., 0., 0., 1.]])

    init_goal_cam_transform = np.array(
        [[0.99941927, -0.00186311, 0.03402425, 1.837726], [-0.02526303, -0.71058334, 0.70315937, -5.78141165],
         [0.022867, -0.70361058, -0.71021775, 6.03373909], [0., 0., 0., 1.]])
    goal_obj_poses = pickle.load(open('./test_scripts/jobtalk_figure_cache_files/goal_obj_poses.pkl', 'r'))
    for o in problem_env.objects: utils.set_obj_xytheta(goal_obj_poses[o.GetName()], o)
    viewer = problem_env.env.GetViewer()
    viewer.SetCamera(init_goal_cam_transform)
    """
    # how do you go from intrinsic params to [fx, fy, cx, cy]? What are these anyways?
    # cam_intrinsic_params = viewer.GetCameraIntrinsics()
    I = problem_env.env.GetViewer().GetCameraImage(1920, 1080, cam_transform, cam_intrinsic_params)
    scipy.misc.imsave('test.png', I)
    """

    # Visualize c0 and pick samples
    pick_smpls = visualize_pick(pick_smpler, problem_env)
    feasible_pick_op = get_feasible_pick(problem_env, pick_smpls)
    import pdb;
    pdb.set_trace()

    # Visualize path
    paths = pickle.load(open('./test_scripts/jobtalk_figure_cache_files/paths.pkl', 'r'))
    pick_path = paths['pick_path']
    place_path = paths['place_path']
    # path_to_feasible_pick, status = problem_env.motion_planner.get_motion_plan(
    #    feasible_pick_op.continuous_parameters['q_goal'])
    # utils.visualize_path(pick_path, transparency=0.7)

    # Visualize a feasible pick
    feasible_pick_op.execute()
    raw_input("Continue?")

    # Visualize a feasible place
    q_goal = np.array([3.84895635, -0.30975723, 3.14 / 2])
    # status = "NoSolution"
    # while status == "NoSolution":
    #    path_to_feasible_place, status = problem_env.motion_planner.get_motion_plan(q_goal)
    # pickle.dump({"pick_path":path_to_feasible_pick, "place_path": path_to_feasible_place}),
    # open('./test_scripts/jobtalk_figure_cache_files/paths.pkl','wb'))
    utils.set_robot_config(q_goal)
    utils.visualize_path(place_path, transparency=0.9, is_last_config_thick=True)


if __name__ == '__main__':
    main()
