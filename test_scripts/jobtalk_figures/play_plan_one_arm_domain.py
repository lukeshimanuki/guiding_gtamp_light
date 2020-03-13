from gtamp_utils import utils
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.subplanners.motion_planner import ArmBaseMotionPlanner
from openravepy import DOFAffine, RaveCreateKinBody, RaveCreateRobot

import os
import numpy as np
import random
import pickle
import time


def make_environment(pidx):
    np.random.seed(pidx)
    random.seed(pidx)
    problem_env = PaPOneArmMoverEnv(pidx)
    goal = ['rectangular_packing_box1_region'] + [obj.GetName() for obj in problem_env.objects[:1]]
    problem_env.set_goal(goal)
    return problem_env


def load_plan():
    # solution_file_name = './/test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal//q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0//n_mp_limit_10_n_iter_limit_200/pidx_0_planner_seed_3_gnn_seed_0.pkl'
    plan_file_dir = '/home/beomjoon/Dropbox (MIT)/cloud_results/test_results/greedy_results_on_mover_domain/domain_one_arm_mover/n_objs_pack_1/gnn/loss_largemargin/num_train_5000/'
    max_action_file = None
    n_max_actions = 0
    max_action_plan = None
    soln_file = ['pidx_59_planner_seed_1_train_seed_2_domain_one_arm_mover.pkl']
    soln_file = os.listdir(plan_file_dir)

    plans = []
    pidxs = []
    max_action_plan_length = 0
    for soln_file in soln_file:
        plan = pickle.load(open(plan_file_dir + soln_file, 'rb')).actions
        pidx = int(soln_file.split('_')[1])
        if pidx == 59:
            # if len(plan) >= 5 and len(plan) <= 10:
            max_action_plan = plan
            pidxs.append(pidx)
            plans.append(plan)
            if len(plans) == 3:
                break

    return plans, pidxs


def play_plan(plan, motion_plans, robot):
    pick_motions = motion_plans[::2]
    place_motions = motion_plans[1::2]

    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    raw_input('Play?')
    time.sleep(5)
    for action, pick_motion, place_motion in zip(plan, pick_motions, place_motions):
        play_motion(pick_motion, 0.09)
        action.execute_pick()

        play_motion(place_motion, 0.09)
        action.execute()


def play_motion(motion, t_sleep):

    for c in motion:
        utils.set_active_config(c)
        time.sleep(t_sleep)


def get_motion_plans(plan, problem_env, fname):
    if os.path.isfile(fname):
        mp_plans = pickle.load(open(fname, 'r'))
        return mp_plans

    robot = problem_env.robot
    utils.open_gripper(robot)
    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])
    mp = ArmBaseMotionPlanner(problem_env, 'rrt')
    mp_plans = []
    for action in plan:
        pick_goal = action.continuous_parameters['pick']['q_goal']
        mp_plan, status = mp.get_motion_plan(pick_goal, region_name='home_region')
        if status == 'HasSolution':
            mp_plans.append(mp_plan)
            action.execute_pick()
        else:
            import pdb;
            pdb.set_trace()
        place_goal = action.continuous_parameters['place']['q_goal']
        mp_plan, status = mp.get_motion_plan(place_goal, region_name='home_region')
        if status == 'HasSolution':
            mp_plans.append(mp_plan)
            action.execute()
        else:
            import pdb;
            pdb.set_trace()
    pickle.dump(mp_plans, open(fname, 'wb'))
    return mp_plans


def main():
    plans, pidxs = load_plan()
    plan = plans[2]
    problem_env = make_environment(59)
    utils.set_body_transparency('top_wall_1', 0.5)
    #utils.set_robot_config(np.array([[ 3.6153236 ,  0.93829982,  5.63509206]]))
    utils.viewer()
    [utils.set_color(obj, [0.0, 0.0, 0.7]) for obj in problem_env.objects]
    utils.set_color('c_obst1', [1.0, 1.0, 0])
    viewer = problem_env.env.GetViewer()
    cam_transform = np.array([[0.58774001, -0.61021391, 0.53122562, 0.93478185],
                              [-0.80888257, -0.45655478, 0.37049525, -1.98781455],
                              [0.01645225, -0.64745402, -0.76192691, 4.26729631],
                              [0., 0., 0., 1.]])
    viewer.SetCamera(cam_transform)

    robot = problem_env.robot
    utils.open_gripper(robot)
    manip = robot.GetManipulator('rightarm_torso')
    robot.SetActiveDOFs(manip.GetArmIndices(), DOFAffine.X | DOFAffine.Y | DOFAffine.RotationAxis, [0, 0, 1])

    motion_plan_file = './test_scripts/jobtalk_figures/one_arm_domain_motion_plans_for_job_talk.pkl'
    mp_plans = get_motion_plans(plan, problem_env, motion_plan_file)

    import pdb;pdb.set_trace()

    play_plan(plan, mp_plans, robot)
    import pdb;
    pdb.set_trace()
    problem_env.init_saver.Restore()


if __name__ == '__main__':
    main()
