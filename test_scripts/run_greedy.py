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
from generators.learning.learning_algorithms.ActorCritic import ActorCritic

from planners.sahs.greedy_new import search
from learn.pap_gnn import PaPGNN
from learn.pose_based_models.fc import FullyConnected as PoseBasedRankFunction

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


def get_solution_file_name(config):
    root_dir = './'
    if config.timelimit == np.inf:
        commit_hash = ''
    else:
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
               '_use_region_agnostic_' + str(config.use_region_agnostic) + \
               '_loss_' + str(config.loss) + '/'
    solution_file_dir += q_config

    if config.use_learning:
        solution_file_dir += '/using_learned_sampler/{}/{}/sampler_seed_{}_{}_{}/sampler_epoch_{}_{}_{}/'.format(
            config.num_episode,
            config.train_type,
            config.pick_seed,
            config.place_goal_region_seed,
            config.place_obj_region_seed,
            config.pick_epoch,
            config.place_goal_region_epoch,
            config.place_obj_region_epoch
        )
    solution_file_dir += '/n_mp_limit_%d_n_iter_limit_%d/' % (config.n_mp_limit, config.n_iter_limit)

    solution_file_name = 'pidx_' + str(config.pidx) + \
                         '_planner_seed_' + str(config.planner_seed) + \
                         '_gnn_seed_' + str(config.absq_seed) + '.pkl'

    if not os.path.isdir(solution_file_dir):
        os.makedirs(solution_file_dir)
    solution_file_dir += '/sampling_strategy_' + config.sampling_strategy
    solution_file_name = solution_file_dir + solution_file_name
    print "Solution file name", solution_file_name
    print "Solution file name", solution_file_name
    print "Solution file name", solution_file_name
    return solution_file_name


def convert_seed_epoch_idxs_to_seed_and_epoch(atype, region, config):
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
    candidate_seeds = []
    for sd_dir in seed_dirs:
        weight_files = [f for f in os.listdir(sampler_weight_path + sd_dir) if 'epoch' in f and '.pt' in f]
        if len(weight_files) > 1:
            seed = int(sd_dir.split('_')[1])
            candidate_seeds.append(seed)
    # todo sort the candidate seeds in order
    candidate_seeds = np.sort(candidate_seeds)
    seed = int(candidate_seeds[config.sampler_seed_idx])

    if config.sampler_epoch_idx == -1:
        if atype == 'pick':
            epoch = config.pick_epoch
        elif atype == 'place':
            if 'home' in region:
                epoch = config.place_goal_region_epoch
            elif 'loading' in region:
                epoch = config.place_obj_region_epoch
            else:
                raise NotImplementedError
        epochs = []
    else:
        epochs = [f for f in os.listdir(sampler_weight_path + 'seed_{}'.format(seed)) if 'epoch' in f and '.pt' in f]
        epoch = int(epochs[config.sampler_epoch_idx].split('_')[-1].split('.pt')[0])
    print sampler_weight_path
    print "Candidate seeds {}".format(candidate_seeds)
    print "Selected seed {} epoch {}".format(seed, epoch)

    return seed, epoch, epochs


def setup_seed_and_epoch(config):
    if 'pick' in config.learned_sampler_atype:
        pick_seed, pick_epoch, _ = convert_seed_epoch_idxs_to_seed_and_epoch('pick', '', config)
        if config.pick_epoch != -1:
            pick_epoch = config.pick_epoch
    else:
        pick_seed, pick_epoch = -1, -1

    if 'place_loading' in config.learned_sampler_atype:
        place_obj_region_seed, place_obj_region_epoch, _ = convert_seed_epoch_idxs_to_seed_and_epoch('place',
                                                                                                     'loading_region',
                                                                                                     config)
        if config.place_obj_region_epoch != -1:
            place_obj_region_epoch = config.place_obj_region_epoch

    else:
        place_obj_region_seed, place_obj_region_epoch = -1, -1

    if 'place_home' in config.learned_sampler_atype:
        place_goal_region_seed, place_goal_region_epoch, _ = convert_seed_epoch_idxs_to_seed_and_epoch('place',
                                                                                                       'home_region',
                                                                                                       config)
        if config.place_goal_region_epoch != -1:
            place_goal_region_epoch = config.place_goal_region_epoch
    else:
        place_goal_region_seed, place_goal_region_epoch = -1, -1

    config.pick_seed = pick_seed
    config.pick_epoch = pick_epoch
    config.place_obj_region_seed = place_obj_region_seed
    config.place_obj_region_epoch = place_obj_region_epoch
    config.place_goal_region_seed = place_goal_region_seed
    config.place_goal_region_epoch = place_goal_region_epoch


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
    parser.add_argument('-timelimit', type=float, default=2000)
    parser.add_argument('-mse_weight', type=float, default=0.0)
    parser.add_argument('-n_mp_limit', type=int, default=5)
    parser.add_argument('-n_iter_limit', type=int, default=2000)

    # abstract Q setup
    parser.add_argument('-num_train', type=int, default=5000)
    parser.add_argument('-num_trains_to_run', nargs='+')
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
    parser.add_argument('-state_mode', type=str, default='keyconfigs')
    parser.add_argument('-num_episode', type=int, default=1500)

    parser.add_argument('-pick_seed', type=int, default=0)  # used for threaded runs
    parser.add_argument('-place_obj_region_seed', type=int, default=0)  # used for threaded runs
    parser.add_argument('-place_goal_region_seed', type=int, default=0)  # used for threaded runs
    parser.add_argument('-pick_epoch', type=int, default=-1)  # used for threaded runs
    parser.add_argument('-place_obj_region_epoch', type=int, default=-1)  # used for threaded runs
    parser.add_argument('-place_goal_region_epoch', type=int, default=-1)  # used for threaded runs
    parser.add_argument('-learned_sampler_atype', type=str,
                        default='pick_place_loading_place_home')  # used for threaded runs
    parser.add_argument('-use_best_kde_sampler', action='store_true', default=False)
    parser.add_argument('-sampler_seed_idx', type=int, default=-1)
    parser.add_argument('-sampler_epoch_idx', type=int, default=-1)
    parser.add_argument('-test_multiple_epochs', action='store_true', default=False)

    # whether to use the learned sampler and the reachability
    parser.add_argument('-use_reachability_clf', action='store_true', default=False)

    ## used for evaluating samplers
    parser.add_argument('-use_test_pidxs', action='store_true', default=False)
    parser.add_argument('-use_best_epochs', action='store_true', default=False)
    parser.add_argument('-epochs_to_evaluate', nargs='+')
    parser.add_argument('-planner_seeds_to_run', nargs='+')

    config = parser.parse_args()
    return config


def set_problem_env_config(problem_env, config):
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    problem_env.seed = config.pidx
    problem_env.init_saver = DynamicEnvironmentStateSaver(problem_env.env)


def get_pap_model(mover, config):
    is_use_gnn = 'qlearned' in config.h_option and 'pose' not in config.h_option
    mconfig_type = collections.namedtuple('mconfig_type',
                                          'operator n_msg_passing n_layers num_fc_layers n_hidden no_goal_nodes '
                                          'top_k optimizer lr use_mse batch_size seed num_train val_portion '
                                          'mse_weight diff_weight_msg_passing same_vertex_model '
                                          'weight_initializer loss use_region_agnostic')

    mconfig = mconfig_type(
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

    if is_use_gnn:
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
            pap_model = PaPGNN(num_entities, num_node_features, num_edge_features, mconfig, entity_names, n_regions)
        pap_model.load_weights()
    else:
        pap_model = PoseBasedRankFunction(mconfig)
        pap_model.load_state_dict(torch.load(pap_model.weight_file_name))

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
    if config.train_type == 'actorcritic':
        model = ActorCritic(config)
        model.load_weights()
    elif config.train_type == 'wgangp':
        model = WGANgp(config)
        if config.use_best_kde_sampler:
            model.load_best_weights()
        else:
            model.load_weights()
    else:
        raise NotImplementedError
    return model


def get_learned_sampler_models(config):
    if not config.use_learning:
        return None, None, None, None
    if 'two_arm' in config.domain:
        train_type = config.train_type
        if 'place_home' in config.learned_sampler_atype:
            config.atype = 'place'
            config.region = 'home_region'
            config.train_type = train_type
            config.seed = config.place_goal_region_seed
            config.epoch = config.place_goal_region_epoch
            goal_region_place_model = make_sampler_model_and_load_weights(config)
        else:
            goal_region_place_model = None

        if 'place_loading' in config.learned_sampler_atype:
            config.atype = 'place'
            config.region = 'loading_region'
            config.train_type = train_type
            config.seed = config.place_obj_region_seed
            config.epoch = config.place_obj_region_epoch
            obj_region_place_model = make_sampler_model_and_load_weights(config)
        else:
            obj_region_place_model = None

        if 'pick' in config.learned_sampler_atype:
            config.atype = 'pick'
            config.region = ''
            config.seed = config.pick_seed
            config.epoch = config.pick_epoch
            pick_model = make_sampler_model_and_load_weights(config)
        else:
            pick_model = None
    else:
        goal_region_place_model = UniformSampler(target_region='rectangular_packing_box1_region',
                                                 atype='one_arm_place')  # how does this actually get used?
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
    return model


def get_goal_obj_and_region(config):
    if config.domain == 'two_arm_mover':
        if config.n_objs_pack == 4:
            goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                         'rectangular_packing_box4']
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
    if config.use_learning:
        if config.sampler_seed_idx != -1:
            setup_seed_and_epoch(config)
    learned_sampler_model = get_learned_sampler_models(config)

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
        # assert config.h_option == 'hcount_old_number_in_goal'
        config.timelimit = np.inf
        pass

    goal_objs, goal_region = get_goal_obj_and_region(config)
    print "Goal:", goal_objs, goal_region
    problem_env = get_problem_env(config, goal_region, goal_objs)
    set_problem_env_config(problem_env, config)
    if config.v:
        utils.viewer()

    pap_model = get_pap_model(problem_env, config)

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
    print(data['n_feasibility_checks'])


if __name__ == '__main__':
    main()
