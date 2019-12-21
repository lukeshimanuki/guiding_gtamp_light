import argparse
import pickle
import time
import numpy as np
import socket
import random
import os
import tensorflow as tf
import collections

from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner
from generators.learning.utils.model_creation_utils import create_policy
from gtamp_utils import utils

from planners.sahs.greedy_new import search
from learn.pap_gnn import PaPGNN


def get_problem_env(config):
    n_objs_pack = config.n_objs_pack
    if config.domain == 'two_arm_mover':
        problem_env = PaPMoverEnv(config.pidx)
        goal = ['home_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        for obj in problem_env.objects[:n_objs_pack]:
            utils.set_color(obj, [0,1,0])
        #goal = ['home_region'] + ['rectangular_packing_box1', 'rectangular_packing_box2', 'rectangular_packing_box3',
        #                 'rectangular_packing_box4']
        problem_env.set_goal(goal)
    elif config.domain == 'one_arm_mover':
        problem_env = PaPOneArmMoverEnv(config.pidx)
        goal = ['rectangular_packing_box1_region'] + [obj.GetName() for obj in problem_env.objects[:n_objs_pack]]
        problem_env.set_goal(goal)
    else:
        raise NotImplementedError
    return problem_env


def get_solution_file_name(config):
    hostname = socket.gethostname()
    if hostname in {'dell-XPS-15-9560', 'phaedra', 'shakey', 'lab', 'glaucus', 'luke-laptop-1'}:
        root_dir = './'
    else:
        root_dir = '/data/public/rw/pass.port/guiding_gtamp_light/'

    if config.gather_planning_exp:
        root_dir = root_dir + '/planning_experience/raw/'
        solution_file_dir = root_dir + '/%s/n_objs_pack_%d' \
                            % (config.domain, config.n_objs_pack)
    else:
        solution_file_dir = root_dir + '/test_results/sahs_results/using_weights_for_submission/domain_%s/n_objs_pack_%d' \
                            % (config.domain, config.n_objs_pack)
    is_use_gnn = 'qlearned' in config.h_option
    solution_file_dir += '/' + config.h_option + '/'
    if is_use_gnn:
        q_config = '/q_config_num_train_' + str(config.num_train) + \
                   '_mse_weight_' + str(config.mse_weight) + \
                   '_use_region_agnostic_' + str(config.use_region_agnostic) + \
                   '_mix_rate_' + str(config.mixrate) + '/'
        solution_file_dir += q_config

    if config.integrated:
        sampler_config = '/smpler_num_train_' + str(config.num_train) + '/'
        solution_file_dir += '/integrated_500_smpls_per_batch_timelimit_1200/shortest_irsc/'
        solution_file_dir += sampler_config
    elif config.integrated_unregularized_sampler:
        sampler_config = '/unregularized_smpler_num_train_' + str(config.num_train) + '/'
        solution_file_dir += '/integrated/shortest_irsc/'
        solution_file_dir += sampler_config

    if config.integrated or config.integrated_unregularized_sampler:
        solution_file_name = 'pidx_' + str(config.pidx) + \
                             '_planner_seed_' + str(config.planner_seed) + \
                             '_train_seed_' + str(config.absq_seed) + \
                             '_smpler_seed_' + str(config.sampler_seed) + \
                             '_domain_' + str(config.domain) + '.pkl'
    else:
        solution_file_name = 'pidx_' + str(config.pidx) + \
                             '_planner_seed_' + str(config.planner_seed) + \
                             '_train_seed_' + str(config.absq_seed) + \
                             '_domain_' + str(config.domain) + '.pkl'
    if not os.path.isdir(solution_file_dir):
        os.makedirs(solution_file_dir)
    solution_file_name = solution_file_dir + solution_file_name
    return solution_file_name


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')

    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-pidxs', nargs=2, type=int, default=[0, 1])  # used for threaded runs

    # problem definition
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-problem_type', type=str, default='normal')  # was used for non-monotonic planning case
    parser.add_argument('-gather_planning_exp', action='store_true', default=False) # sets the allowed time to infinite

    # planning budget setup
    parser.add_argument('-num_node_limit', type=int, default=3000)
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-timelimit', type=float, default=300)
    parser.add_argument('-mse_weight', type=float, default=1.0)

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
    parser.add_argument('-integrated_unregularized_sampler', action='store_true', default=False)
    parser.add_argument('-sampler_algo', type=str, default='imle_qg_combination')

    # whether to use the sampler
    parser.add_argument('-integrated', action='store_true', default=False)

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
                                              'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes top_k optimizer lr use_mse batch_size seed num_train val_portion mse_weight diff_weight_msg_passing same_vertex_model weight_initializer loss use_region_agnostic')

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


def get_learned_smpler(sampler_seed, algo):
    print "Creating the learned sampler.."
    placeholder_config_definition = collections.namedtuple('config', 'algo dtype tau seed')
    placeholder_config = placeholder_config_definition(
        algo=algo,
        tau=1.0,
        dtype='n_objs_pack_4',
        seed=sampler_seed
    )
    epoch = 700
    sampler = create_policy(placeholder_config)
    #sampler.load_weights('epoch_' + str(epoch))
    sampler.load_best_weights()
    return sampler


def make_pklable(plan):
    for p in plan:
        obj = p.discrete_parameters['object']
        region = p.discrete_parameters['place_region']
        if not isinstance(region, str):
            p.discrete_parameters['place_region'] = region.name
        if not (isinstance(obj, unicode) or isinstance(obj, str)):
            p.discrete_parameters['object'] = obj.GetName()


def main():
    config = parse_arguments()
    if config.gather_planning_exp:
        config.timelimit = np.inf
    np.random.seed(config.pidx)
    random.seed(config.pidx)

    problem_env = get_problem_env(config)
    set_problem_env_config(problem_env, config)

    is_use_gnn = 'qlearned' in config.h_option
    if is_use_gnn:
        pap_model = get_pap_gnn_model(problem_env, config)
    else:
        pap_model = None
    if config.integrated or config.integrated_unregularized_sampler:
        smpler = get_learned_smpler(config.sampler_seed, config.sampler_algo)
    else:
        smpler = None

    solution_file_name = get_solution_file_name(config)
    is_problem_solved_before = os.path.isfile(solution_file_name)
    plan_length = 0
    if is_problem_solved_before and not config.f:
        print "***************Already solved********************"
        with open(solution_file_name, 'rb') as f:
            trajectory = pickle.load(f)
            success = trajectory['success']
            tottime = trajectory['tottime']
            num_nodes = trajectory['num_nodes']
    else:
        t = time.time()
        plan, num_nodes, nodes = search(problem_env, config, pap_model, smpler)
        tottime = time.time() - t
        success = plan is not None
        plan_length = len(plan) if success else 0
        if success and config.domain == 'one_arm_mover':
            make_pklable(plan)

        for n in nodes:
            n.state.make_pklable()

        nodes = None

        data = {
            'n_objs_pack': config.n_objs_pack,
            'tottime': tottime,
            'success': success,
            'plan_length': plan_length,
            'num_nodes': num_nodes,
            'plan': plan,
            'nodes': nodes
        }
        with open(solution_file_name, 'wb') as f:
            pickle.dump(data, f)
    print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d' % (tottime, success, plan_length, num_nodes)


if __name__ == '__main__':
    main()
