import numpy as np
import random
import argparse
import socket
import os
import sys
import collections
import tensorflow as tf
import pickle
import time
import subprocess

from learn.pap_gnn import PaPGNN
from gtamp_problem_environments.mover_env import Mover, PaPMoverEnv
from planners.subplanners.motion_planner import OperatorBaseMotionPlanner
from gtamp_problem_environments.reward_functions.reward_function import GenericRewardFunction
from gtamp_problem_environments.reward_functions.shaped_reward_function import ShapedRewardFunction
from planners.flat_mcts.mcts import MCTS
from planners.flat_mcts.mcts_with_leaf_strategy import MCTSWithLeafStrategy
from planners.heuristics import compute_hcount_with_action, get_objects_to_move


def make_and_get_save_dir(parameters, filename, commit_hash):
    hostname = socket.gethostname()
    if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab' or \
            hostname == 'glaucus':
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp/' + commit_hash

    save_dir = root_dir + '/test_results/'+commit_hash+ '/mcts_results_with_q_bonus/' \
               + 'domain_' + str(parameters.domain) + '/' \
               + 'n_objs_pack_' + str(parameters.n_objs_pack) + '/' \
               + 'sampling_strategy_' + str(parameters.sampling_strategy) + '/' \
               + 'n_mp_trials_' + str(parameters.n_motion_plan_trials) + '/' \
               + 'n_feasibility_checks_' + str(parameters.n_feasibility_checks) + '/' \
               + 'widening_' + str(parameters.widening_parameter) + '/' \
               + 'uct_' + str(parameters.ucb_parameter) + '/' \
               + 'switch_frequency_' + str(parameters.switch_frequency) + '/' \
               + 'reward_shaping_' + str(parameters.use_shaped_reward) + '/' \
               + 'learned_q_' + str(parameters.use_learned_q) + '/' \
               + 'use_pw_' + str(parameters.pw) + '/' \
               + 'use_ucb_at_cont_nodes_' + str(parameters.use_ucb) + '/'

    if 'uniform' not in parameters.sampling_strategy:
        save_dir += 'explr_p_' + str(parameters.explr_p) + '/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def parse_mover_problem_parameters():
    parser = argparse.ArgumentParser(description='planner parameters')

    # Problem-specific parameters
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-planner', type=str, default='mcts_with_leaf_strategy')
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-planner_seed', type=int, default=0)

    # Planner-agnostic parameters
    parser.add_argument('-timelimit', type=int, default=np.inf)
    parser.add_argument('-dont_use_learned_q', action='store_false', default=True)
    parser.add_argument('-n_feasibility_checks', type=int, default=2000)
    parser.add_argument('-n_motion_plan_trials', type=int, default=5)
    parser.add_argument('-planning_horizon', type=int, default=10000)

    # Learning-related parameters
    parser.add_argument('-train_seed', type=int, default=0)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-num_train', type=int, default=7000)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)

    # MCTS parameters
    parser.add_argument('-switch_frequency', type=int, default=100)
    parser.add_argument('-ucb_parameter', type=float, default=0.1)
    parser.add_argument('-widening_parameter', type=float, default=10)  # number of re-evals
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-mcts_iter', type=int, default=1000)
    parser.add_argument('-use_learned_q', action='store_true', default=False)
    parser.add_argument('-use_ucb', action='store_true', default=False)
    parser.add_argument('-pw', action='store_true', default=False)
    parser.add_argument('-f', action='store_true', default=False)  # what was this?
    parser.add_argument('-sampling_strategy', type=str, default='uniform')
    parser.add_argument('-use_shaped_reward', action='store_true', default=False)

    parameters = parser.parse_args()
    return parameters


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def load_learned_q(config, problem_env):
    mconfig_type = collections.namedtuple('mconfig_type',
                                          'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes '
                                          'top_k optimizer lr use_mse batch_size seed num_train val_portion '
                                          'mse_weight diff_weight_msg_passing same_vertex_model '
                                          'weight_initializer loss use_region_agnostic')

    pap_mconfig = mconfig_type(
        operator='two_arm_pick_two_arm_place',
        n_msg_passing=1,
        n_layers=2,
        num_fc_layers=2,
        n_hidden=32,
        no_goal_nodes=False,

        top_k=1,
        optimizer='adam',
        lr=1e-4,
        use_mse=True,

        batch_size='32',
        seed=config.train_seed,
        num_train=5000,
        val_portion=.1,
        mse_weight=1.0,
        diff_weight_msg_passing=False,
        same_vertex_model=False,
        weight_initializer='glorot_uniform',
        loss=config.loss,
        use_region_agnostic=config.use_region_agnostic
    )
    if config.domain == 'two_arm_mover':
        num_entities = 11
        n_regions = 2
    elif config.domain == 'one_arm_mover':
        num_entities = 12
        n_regions = 2
    else:
        raise NotImplementedError
    num_node_features = 10
    num_edge_features = 44
    entity_names = problem_env.entity_names

    with tf.variable_scope('pap'):
        pap_model = PaPGNN(num_entities, num_node_features, num_edge_features, pap_mconfig, entity_names, n_regions)
    pap_model.load_weights()

    return pap_model


def get_commit_hash():
    syscmd = 'git ls-remote ./ refs/heads/master | cut -f 1'
    hash = subprocess.check_output(['git', 'log',  '--pretty=format:%h', '-n', '1'])

    return hash


def main():
    commit_hash = get_commit_hash()
    parameters = parse_mover_problem_parameters()
    filename = 'pidx_%d_planner_seed_%d.pkl' % (parameters.pidx, parameters.planner_seed)
    save_dir = make_and_get_save_dir(parameters, filename, commit_hash)
    solution_file_name = save_dir + filename
    is_problem_solved_before = os.path.isfile(solution_file_name)
    print solution_file_name
    if is_problem_solved_before and not parameters.f:
        print "***************Already solved********************"
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            tottime = trajectory['search_time_to_reward'][-1][2]
            print 'Time: %.2f ' % tottime
        sys.exit(-1)

    set_seed(parameters.pidx)
    problem_env = PaPMoverEnv(parameters.pidx)

    goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3', 'rectangular_packing_box4']
    goal_region = 'home_region'
    problem_env.set_goal(goal_objs, goal_region)
    goal_entities = goal_objs + [goal_region]
    if parameters.use_shaped_reward:
        # uses the reward shaping per Ng et al.
        reward_function = ShapedRewardFunction(problem_env, goal_objs, goal_region,
                                               parameters.planning_horizon)
    else:
        reward_function = GenericRewardFunction(problem_env, goal_objs, goal_region,
                                                parameters.planning_horizon)

    motion_planner = OperatorBaseMotionPlanner(problem_env, 'prm')

    problem_env.set_reward_function(reward_function)
    problem_env.set_motion_planner(motion_planner)

    learned_q = None
    prior_q = None
    if parameters.use_learned_q:
        learned_q = load_learned_q(parameters, problem_env)

    v_fcn = lambda state: -len(get_objects_to_move(state, problem_env))

    if parameters.planner == 'mcts':
        planner = MCTS(parameters, problem_env, goal_entities, prior_q, learned_q)
    elif parameters.planner == 'mcts_with_leaf_strategy':
        planner = MCTSWithLeafStrategy(parameters, problem_env, goal_entities, v_fcn, learned_q)
    else:
        raise NotImplementedError

    set_seed(parameters.planner_seed)
    stime = time.time()
    search_time_to_reward, n_feasibility_checks, plan = planner.search(max_time=parameters.timelimit)
    tottime = time.time() - stime
    print 'Time: %.2f ' % (tottime)

    # todo
    #   save the entire tree

    pickle.dump({"search_time_to_reward": search_time_to_reward,
                 'plan': plan,
                 'commit_hash': commit_hash,
                 'n_feasibility_checks': n_feasibility_checks,
                 'n_nodes': len(planner.tree.get_discrete_nodes())}, open(save_dir + filename, 'wb'))


if __name__ == '__main__':
    main()
