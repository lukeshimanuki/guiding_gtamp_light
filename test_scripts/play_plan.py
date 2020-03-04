from run_greedy import parse_arguments, set_problem_env_config, get_problem_env, get_solution_file_name
from gtamp_utils import utils
import numpy as np
import random
import pickle
import time


def make_environment(config):
    np.random.seed(config.pidx)
    random.seed(config.pidx)

    goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3', 'rectangular_packing_box4']
    goal_region = 'home_region'
    problem_env = get_problem_env(config, goal_region, goal_objs)
    set_problem_env_config(problem_env, config)
    [utils.set_color(o, [0, 0, 0.8]) for o in goal_objs]
    return problem_env


def load_plan(config):
    solution_file_name = get_solution_file_name(config)
    solution_file_name = './/test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal//q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0//n_mp_limit_10_n_iter_limit_200/pidx_0_planner_seed_3_gnn_seed_0.pkl'
    f = open(solution_file_name, 'rb')
    trajectory = pickle.load(f)
    plan = trajectory['plan']
    return plan


def play_action(action, t_sleep):
    pick = action.continuous_parameters['pick']
    place = action.continuous_parameters['place']

    for c in pick['motion']:
        utils.set_robot_config(c)
        time.sleep(t_sleep)
    action.execute_pick()

    time.sleep(t_sleep)
    for c in place['motion']:
        utils.set_robot_config(c)
        time.sleep(t_sleep)
    action.execute()


def play_plan(plan):
    t_sleep = 0.05
    raw_input("Begin recording?")
    time.sleep(3)
    for action in plan:
        play_action(action, t_sleep)


def main():
    config = parse_arguments()
    problem_env = make_environment(config)
    plan = load_plan(config)
    utils.viewer()
    T_viewer = np.array([[-0.02675345, 0.86098854, -0.50792025, 6.43201399],
                         [0.99806971, -0.00548069, -0.06186132, -2.36350584],
                         [-0.05604564, -0.50859482, -0.85917995, 8.97449112],
                         [0., 0., 0., 1.]])
    viewer = problem_env.env.GetViewer()
    viewer.SetCamera(T_viewer)
    import pdb;pdb.set_trace()
    play_plan(plan)


if __name__ == '__main__':
    main()
