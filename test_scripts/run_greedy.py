import argparse
import pickle
import time
import numpy as np
import socket
import random
import os
import torch
import tensorflow as tf
import collections
import sys

from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner
# from generators.learning.utils.model_creation_utils import create_policy
from gtamp_utils import utils
from test_scripts.run_mcts import get_commit_hash

# from test_scripts.visualize_learned_sampler import create_policy
from planners.sahs.greedy_new import search
from learn.pap_gnn import PaPGNN
from generators.learning.learning_algorithms.WGANGP import WGANgp


def get_problem_env(config, goal_region, goal_objs):
    n_objs_pack = config.n_objs_pack
    np.random.seed(config.pidx)
    random.seed(config.pidx)
    if config.domain == 'two_arm_mover':
        problem_env = PaPMoverEnv(config.pidx)
        # goal = ['home_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        # for obj in problem_env.objects[:n_objs_pack]:
        #    utils.set_color(obj, [0, 1, 0])
        [utils.set_color(o, [0, 0, 0.8]) for o in goal_objs]

        # goal = ['home_region'] + ['rectangular_packing_box1', 'rectangular_packing_box2', 'rectangular_packing_box3',
        #                 'rectangular_packing_box4']
        problem_env.set_goal(goal_objs, goal_region)
    elif config.domain == 'one_arm_mover':
        problem_env = PaPOneArmMoverEnv(config.pidx)
        goal = ['rectangular_packing_box1_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        problem_env.set_goal(goal)
    else:
        raise NotImplementedError
    return problem_env


def get_solution_file_name(config):
    root_dir = './'
    # if hostname in {'dell-XPS-15-9560', 'phaedra', 'shakey', 'lab', 'glaucus', 'luke-laptop-1'}:
    #    root_dir = './'
    # else:
    #    root_dir = '/data/public/rw/pass.port/guiding_gtamp_light/'
    commit_hash = get_commit_hash()

    if config.gather_planning_exp:
        root_dir = root_dir + '/planning_experience/raw/uses_rrt/'
        solution_file_dir = root_dir + '/%s/n_objs_pack_%d' \
                            % (config.domain, config.n_objs_pack)
    else:
        solution_file_dir = root_dir + '/test_results/%s/sahs_results/uses_rrt/uses_reachability_clf_%s/domain_%s/n_objs_pack_%d' \
                            % (commit_hash, config.use_reachability_clf, config.domain, config.n_objs_pack)
    solution_file_dir += '/' + config.h_option + '/'

    q_config = '/q_config_num_train_' + str(config.num_train) + \
               '_mse_weight_' + str(config.mse_weight) + \
               '_use_region_agnostic_' + str(config.use_region_agnostic) + \
               '_mix_rate_' + str(config.mixrate) + '/'
    solution_file_dir += q_config

    if config.use_learning:
        sampler_config = '/smpler_num_train_' + str(config.num_train) + '/'
        solution_file_dir += '/using_learned_sampler/'
        solution_file_dir += sampler_config

    solution_file_dir += '/n_mp_limit_%d_n_iter_limit_%d/' % (config.n_mp_limit, config.n_iter_limit)

    if config.use_learning:
        solution_file_name = 'pidx_' + str(config.pidx) + \
                             '_planner_seed_' + str(config.planner_seed) + \
                             '_gnn_seed_' + str(config.absq_seed) + \
                             '_smpler_seed_' + str(config.sampler_seed) + '.pkl'
    else:
        solution_file_name = 'pidx_' + str(config.pidx) + \
                             '_planner_seed_' + str(config.planner_seed) + \
                             '_gnn_seed_' + str(config.absq_seed) + '.pkl'

    if not os.path.isdir(solution_file_dir):
        os.makedirs(solution_file_dir)
    solution_file_dir += '/sampling_strategy_' + config.sampling_strategy
    solution_file_name = solution_file_dir + solution_file_name
    return solution_file_name


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')

    parser.add_argument('-v', action='store_true', default=False)

    # problem definition
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=4)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-problem_type', type=str, default='normal')  # was used for non-monotonic planning case
    parser.add_argument('-gather_planning_exp', action='store_true', default=False)  # sets the allowed time to infinite

    # planning budget setup
    parser.add_argument('-num_node_limit', type=int, default=3000)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-timelimit', type=float, default=300)
    parser.add_argument('-mse_weight', type=float, default=1.0)
    parser.add_argument('-n_mp_limit', type=int, default=5)
    parser.add_argument('-n_iter_limit', type=int, default=2000)

    # abstract Q setup
    parser.add_argument('-dontsimulate', action='store_true', default=False)
    parser.add_argument('-loss', type=str, default='largemargin')
    parser.add_argument('-absq_seed', type=int, default=0)
    parser.add_argument('-mixrate', type=float, default=1.0)
    parser.add_argument('-use_region_agnostic', action='store_true', default=False)

    # abstract heuristic function setup
    parser.add_argument('-h_option', type=str, default='qlearned_hcount_old_number_in_goal')

    # Sampler setup
    parser.add_argument('-sampler_seed', type=int, default=0)
    parser.add_argument('-sampling_strategy', type=str, default='uniform')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-use_learning', action='store_true', default=False)
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-pick_architecture', type=str, default='fc')
    parser.add_argument('-place_architecture', type=str, default='fc')
    parser.add_argument('-pick_sampler_seed', type=int, default=2)  # used for threaded runs
    parser.add_argument('-loading_sampler_seed', type=int, default=1)  # used for threaded runs
    parser.add_argument('-home_sampler_seed', type=int, default=0)  # used for threaded runs

    # whether to use the learned sampler and the reachability
    parser.add_argument('-use_reachability_clf', action='store_true', default=False)

    config = parser.parse_args()
    return config


def set_problem_env_config(problem_env, config):
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    problem_env.seed = config.pidx
    problem_env.init_saver = DynamicEnvironmentStateSaver(problem_env.env)


def get_pap_gnn_model(mover, config):
    is_use_gnn = 'qlearned' in config.h_option
    if is_use_gnn:
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
            seed=config.absq_seed,
            num_train=config.num_train,
            val_portion=.1,
            mse_weight=config.mse_weight,
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
        entity_names = mover.entity_names

        with tf.variable_scope('pap'):
            pap_model = PaPGNN(num_entities, num_node_features, num_edge_features, pap_mconfig, entity_names, n_regions)
        pap_model.load_weights()
    else:
        pap_model = None

    return pap_model


def make_pklable(plan):
    for p in plan:
        obj = p.discrete_parameters['object']
        region = p.discrete_parameters['place_region']
        if not isinstance(region, str):
            p.discrete_parameters['place_region'] = region.name
        if not (isinstance(obj, unicode) or isinstance(obj, str)):
            p.discrete_parameters['object'] = obj.GetName()


def make_node_pklable(node):
    node.state.make_pklable()
    node.tried_samples = {}
    node.tried_sample_feasibility_labels = {}
    """
    for k in node.generators.keys():
        node.tried_samples[k] = node.generators[k].tried_samples
        node.tried_sample_feasibility_labels[k] = node.generators[k].tried_sample_labels
    """
    node.generators = None


def get_total_n_feasibility_checks(nodes):
    total_ik_checks = 0
    total_mp_checks = 0
    for n in nodes:
        for generator in n.generators.values():
            total_ik_checks += generator.n_ik_checks
            total_mp_checks += generator.n_mp_checks
    return {'mp': total_mp_checks, 'ik': total_ik_checks}


def get_learned_sampler_models(config):
    home_place_model = None
    loading_place_model = None
    pick_model = None

    if not config.use_learning:
        return None

    if 'place' in config.atype:
        region = 'home_region'
        action_type = 'place'
        home_place_model = WGANgp(action_type, region, architecture=config.place_architecture, seed=config.home_sampler_seed)
        home_place_model.load_best_weights()

        region = 'loading_region'
        action_type = 'place'
        loading_place_model = WGANgp(action_type, region, architecture=config.place_architecture,
                                     seed=config.loading_sampler_seed)
        loading_place_model.load_best_weights()

    if 'pick' in config.atype:
        action_type = 'pick'
        region = ''
        pick_model = WGANgp(action_type, region, architecture=config.pick_architecture, seed=config.pick_sampler_seed)
        pick_model.load_best_weights()

    model = {'place_home': home_place_model, 'place_loading': loading_place_model, 'pick': pick_model}
    return model


def main():
    config = parse_arguments()
    solution_file_name = get_solution_file_name(config)
    is_problem_solved_before = os.path.isfile(solution_file_name)
    if is_problem_solved_before and not config.f:
        print "***************Already solved********************"
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            success = trajectory['success']
            tottime = trajectory['tottime']
            num_nodes = trajectory['num_nodes']
            plan_length = len(trajectory['plan']) if success else 0
            print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d N_feasible: %d' % (
                tottime, success, plan_length, num_nodes, trajectory['n_feasibility_checks']['ik'])
        sys.exit(-1)

    if config.gather_planning_exp:
        config.timelimit = np.inf

    if config.domain == 'two_arm_mover':
        goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4']
        goal_region = 'home_region'
    elif config.domain == 'one_arm_mover':
        goal_objs = ['c_obst0', 'c_obst1', 'c_obst2', 'c_obst3']
        goal_region = 'rectangular_packing_box1_region'
    else:
        raise NotImplementedError

    problem_env = get_problem_env(config, goal_region, goal_objs)
    set_problem_env_config(problem_env, config)
    if config.v:
        utils.viewer()

    is_use_gnn = 'qlearned' in config.h_option
    if is_use_gnn:
        pap_model = get_pap_gnn_model(problem_env, config)
    else:
        pap_model = None

    [utils.set_color(o, [1, 0, 0]) for o in goal_objs]
    t = time.time()
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    learned_sampler_model = get_learned_sampler_models(config)
    nodes_to_goal, plan, num_nodes, nodes = search(problem_env, config, pap_model, goal_objs,
                                                   goal_region, learned_sampler_model)
    tottime = time.time() - t
    n_feasibility_checks = get_total_n_feasibility_checks(nodes)

    success = plan is not None
    plan_length = len(plan) if success else 0
    if success and config.domain == 'one_arm_mover':
        make_pklable(plan)

    if config.gather_planning_exp:
        [make_node_pklable(nd) for nd in nodes]
    else:
        nodes = None

    h_for_sampler_training = []
    if success:
        h_for_sampler_training = []
        for n in nodes_to_goal:
            h_for_sampler_training.append(n.h_for_sampler_training)

    data = {
        'n_objs_pack': config.n_objs_pack,
        'tottime': tottime,
        'success': success,
        'plan_length': plan_length,
        'num_nodes': num_nodes,
        'plan': plan,
        'nodes': nodes,
        'hvalues': h_for_sampler_training,
        'n_feasibility_checks': n_feasibility_checks
    }
    with open(solution_file_name, 'wb') as f:
        pickle.dump(data, f)
    print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d' % (tottime, success, plan_length, num_nodes)


if __name__ == '__main__':
    main()
