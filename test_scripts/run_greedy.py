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
from gtamp_utils import utils
from test_scripts.run_mcts import get_commit_hash

from planners.sahs.greedy_new import search
from learn.pap_gnn import PaPGNN
from generators.learning.learning_algorithms.WGANGP import WGANgp
from generators.samplers.uniform_sampler import UniformSampler


def get_problem_env(config, goal_region, goal_objs):
    np.random.seed(config.pidx)
    random.seed(config.pidx)
    if config.domain == 'two_arm_mover':
        problem_env = PaPMoverEnv(config.pidx)
        [utils.set_color(o, [0.8, 0, 0]) for o in goal_objs]
        problem_env.set_goal(goal_objs, goal_region)
    elif config.domain == 'one_arm_mover':
        problem_env = PaPOneArmMoverEnv(config.pidx)
        [utils.set_color(obj, [0.0, 0.0, 0.7]) for obj in problem_env.objects]
        [utils.set_color(o, [1.0, 1.0, 0]) for o in goal_objs]
        problem_env.set_goal(goal_objs, goal_region)
    else:
        raise NotImplementedError
    return problem_env


def get_solution_file_name(config, pick_seed, home_seed, loading_seed):
    root_dir = './'
    # if hostname in {'dell-XPS-15-9560', 'phaedra', 'shakey', 'lab', 'glaucus', 'luke-laptop-1'}:
    #    root_dir = './'
    # else:
    #    root_dir = '/data/public/rw/pass.port/guiding_gtamp_light/'
    commit_hash = get_commit_hash()

    if config.gather_planning_exp:
        root_dir = root_dir + '/planning_experience/raw/'
        solution_file_dir = root_dir + '/%s/n_objs_pack_%d' \
                            % (config.domain, config.n_objs_pack)
    else:
        solution_file_dir = root_dir + '/test_results/%s/sahs_results/domain_%s/n_objs_pack_%d' \
                            % (commit_hash, config.domain, config.n_objs_pack)
    solution_file_dir += '/' + config.h_option + '/'

    q_config = '/q_config_num_train_' + str(config.num_train) + \
               '_mse_weight_' + str(config.mse_weight) + \
               '_use_region_agnostic_' + str(config.use_region_agnostic) + '/'
    solution_file_dir += q_config

    if config.use_learning:
        solution_file_dir += '/using_learned_sampler/{}/sampler_seed_{}_{}_{}/{}'.format(config.num_episode,
                                                                                   pick_seed, home_seed, loading_seed,
                                                                                   config.train_type)

    solution_file_dir += '/n_mp_limit_%d_n_iter_limit_%d/' % (config.n_mp_limit, config.n_iter_limit)

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
    parser.add_argument('-timelimit', type=float, default=2000)
    parser.add_argument('-mse_weight', type=float, default=0.0)
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
    parser.add_argument('-sampling_strategy', type=str, default='uniform')
    parser.add_argument('-atype', type=str, default='place')
    parser.add_argument('-use_learning', action='store_true', default=False)
    parser.add_argument('-explr_p', type=float, default=0.3)
    parser.add_argument('-architecture', type=str, default='fc')
    parser.add_argument('-train_type', type=str, default='wgandi')
    parser.add_argument('-sampler_seed', type=int, default=0)  # used for threaded runs
    parser.add_argument('-num_episode', type=int, default=1000)

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
            num_entities = 11  # 8
            n_regions = 2
        elif config.domain == 'one_arm_mover':
            num_entities = 12
            n_regions = 2
        else:
            raise NotImplementedError
        num_node_features = 20
        num_edge_features = 28
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
        if 'region' in p.discrete_parameters:
            region = p.discrete_parameters['region']
        else:
            region = p.discrete_parameters['place_region']
        if not isinstance(region, str):
            p.discrete_parameters['place_region'] = region.name
        if not (isinstance(obj, unicode) or isinstance(obj, str)):
            p.discrete_parameters['object'] = obj.GetName()


def make_node_pklable(node):
    node.state.make_pklable()
    node.tried_samples = {}
    node.tried_sample_feasibility_labels = {}
    if 'heuristic_vals' in dir(node):
        node.heuristic_vals = None
    node.generators = None


def get_total_n_feasibility_checks(nodes):
    total_ik_checks = 0
    total_mp_checks = 0
    for n in nodes:
        for generator in n.generators.values():
            total_ik_checks += generator.n_ik_checks
            total_mp_checks += generator.n_mp_checks
    return {'mp': total_mp_checks, 'ik': total_ik_checks}


def make_sampler_model_and_load_weights(config):
    model = WGANgp(config)
    model.load_best_weights()
    return model


def get_best_seeds(atype, region, config):
    if atype == 'pick':
        sampler_weight_path = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/fc/'.format(config.domain,
                                                                                                          config.num_episode,
                                                                                                          atype,
                                                                                                          config.train_type)
    else:
        sampler_weight_path = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/{}/fc/'.format(
            config.domain,
            config.num_episode,
            atype,
            region,
            config.train_type)

    seed_dirs = os.listdir(sampler_weight_path)
    max_kde = -np.inf
    candidate_seeds = []
    candidate_seed_kdes = []
    for sd_dir in seed_dirs:
        logfiles = [p for p in os.listdir(sampler_weight_path + sd_dir) if '.pt' not in p]
        best_kde_for_sd = np.max([float(logfile.split('_kde_')[1].split('_')[0]) for logfile in logfiles])
        print best_kde_for_sd
        is_pick = 'pick' in sampler_weight_path
        if 'two_arm_mover' in sampler_weight_path:
            if is_pick:
                target_kde = -150
            else:
                if 'home_region' in sampler_weight_path:
                    target_kde = -40
                else:
                    target_kde = -70
        else:
            raise NotImplementedError

        if best_kde_for_sd > target_kde:
            candidate_seeds.append(int(sd_dir.split('_')[1]))
            candidate_seed_kdes.append(best_kde_for_sd)
    print "N qualified seeds for {} {}".format(atype, region), len(candidate_seeds)
    print "Qualified seeds for {} {}".format(atype, region), candidate_seeds[config.sampler_seed], \
        candidate_seed_kdes[config.sampler_seed]
    print "Selected KDE",candidate_seed_kdes[config.sampler_seed]

    # ordering on the cloud
    return candidate_seeds[config.sampler_seed]


def get_learned_sampler_models(config):
    if not config.use_learning:
        return None
    if 'two_arm' in config.domain:
        train_type = config.train_type

        # place home region
        config.atype = 'place'
        config.region = 'home_region'
        config.train_type = train_type
        best_seed = get_best_seeds('place', 'home_region', config)
        config.seed = best_seed
        home_seed = best_seed
        goal_region_place_model = make_sampler_model_and_load_weights(config)

        # place load region
        config.atype = 'place'
        config.region = 'loading_region'
        config.train_type = train_type
        best_seed = get_best_seeds('place', 'loading_region', config)
        loading_seed = best_seed
        config.seed = best_seed
        obj_region_place_model = make_sampler_model_and_load_weights(config)

        # pick
        config.atype = 'pick'
        config.region = ''
        best_seed = get_best_seeds('pick', '', config)
        pick_seed = best_seed
        config.seed = best_seed
        pick_model = make_sampler_model_and_load_weights(config)
    else:
        goal_region_place_model = UniformSampler(target_region='rectangular_packing_box1_region',
                                                 atype='one_arm_place')  # I don't think we need to learn sampler for this
        config.atype = 'place';
        config.region = 'center_shelf_region';
        config.seed = config.sampler_seed
        obj_region_place_model = make_sampler_model_and_load_weights(config)
        config.atype = 'pick';
        config.region = '';
        config.seed = config.sampler_seed
        pick_model = make_sampler_model_and_load_weights(config)
    model = {'place_goal_region': goal_region_place_model, 'place_obj_region': obj_region_place_model,
             'pick': pick_model}
    return model, pick_seed, home_seed, loading_seed


def get_goal_obj_and_region(config):
    if config.domain == 'two_arm_mover':
        if config.n_objs_pack == 4:
            goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3', 'rectangular_packing_box4']
            goal_region = 'home_region'
        else:
            goal_objs = ['square_packing_box1']
            goal_region = 'home_region'
    elif config.domain == 'one_arm_mover':
        assert config.n_objs_pack == 1
        goal_objs = ['c_obst1']
        goal_region = 'rectangular_packing_box1_region'
    else:
        raise NotImplementedError
    return goal_objs, goal_region


def main():
    config = parse_arguments()
    learned_sampler_model, pick_seed, home_seed, loading_seed = get_learned_sampler_models(config)
    solution_file_name = get_solution_file_name(config, pick_seed, home_seed, loading_seed)
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
        # assert config.h_option == 'hcount_old_number_in_goal'
        config.timelimit = np.inf
        pass

    goal_objs, goal_region = get_goal_obj_and_region(config)
    print "Goal:", goal_objs, goal_region
    problem_env = get_problem_env(config, goal_region, goal_objs)
    set_problem_env_config(problem_env, config)
    if config.v:
        utils.viewer()

    is_use_gnn = 'qlearned' in config.h_option
    if is_use_gnn:
        pap_model = get_pap_gnn_model(problem_env, config)
    else:
        pap_model = None

    t = time.time()
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
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
