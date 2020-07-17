import argparse
import numpy as np
import random
import pickle
import time
import os
import torch
import socket

from gtamp_problem_environments.mover_env import PaPMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner

from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from planners.sahs.greedy_new import get_generator
from test_scripts.run_greedy import get_learned_sampler_models
from run_greedy import get_problem_env, get_goal_obj_and_region

from gtamp_utils import utils


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
    seed = int(candidate_seeds[config.sampler_seed_idx])
    epochs = [f for f in os.listdir(sampler_weight_path + 'seed_{}'.format(seed)) if 'epoch' in f and '.pt' in f]
    epoch = int(epochs[config.sampler_epoch_idx].split('_')[-1].split('.pt')[0])
    print sampler_weight_path
    print "Candidate seeds {}".format(candidate_seeds)
    print "Selected seed {} epoch {}".format(seed, epoch)
    return seed, epoch, epochs


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-architecture', type=str, default='fc')
    parser.add_argument('-planner_seed', type=int, default=0)
    parser.add_argument('-sampling_strategy', type=str, default='uniform')
    parser.add_argument('-use_learning', action='store_true', default=False)
    parser.add_argument('-n_mp_limit', type=int, default=5)
    parser.add_argument('-n_iter_limit', type=int, default=2000)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-train_type', type=str, default='wgandi')
    parser.add_argument('-num_episode', type=int, default=1000)
    parser.add_argument('-target_pidx_idx', type=int, default=0)
    parser.add_argument('-atype', type=str, default='UsedOnlyByWGANGP')

    parser.add_argument('-learned_sampler_atype', type=str, default='pick_place_home_place_loading')
    parser.add_argument('-sampler_seed_idx', type=int, default=0)
    parser.add_argument('-sampler_epoch_idx', type=int, default=0)

    config = parser.parse_args()
    return config


def load_planning_experience_data(config):
    if 'two_arm' in config.domain:
        raw_dir = 'planning_experience/for_testing_generators//' \
                  'sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
                  'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/' \
                  'n_mp_limit_5_n_iter_limit_2000/'
        target_pidxs = [40200,40201,40202,40204,40205,40206,40207,40208,40209]
    else:
        raise NotImplementedError

    pidx = target_pidxs[config.target_pidx_idx]
    config.pidx = pidx

    fname = 'pidx_%d_planner_seed_0_gnn_seed_2.pkl' % pidx
    try:
        plan_data = pickle.load(open(raw_dir + fname, 'r'))
    except:
        plan_data = pickle.load(open(raw_dir + 'sampling_strategy_uniform' + fname, 'r'))

    plan = plan_data['plan']
    goal_objs, goal_region = get_goal_obj_and_region(config)
    problem_env = get_problem_env(config, goal_region, goal_objs)
    return plan, problem_env


class DummyAbstractState:
    def __init__(self, problem_env, goal_entities):
        self.prm_vertices, self.prm_edges = pickle.load(open('./prm.pkl', 'rb'))
        self.problem_env = problem_env
        self.goal_entities = goal_entities


def make_abstract_state(problem_env, goal_entities, parent_state=None, parent_action=None):
    if 'two_arm' in problem_env.name:
        abstract_state = ShortestPathPaPState(problem_env, goal_entities,
                                              parent_state=parent_state,
                                              parent_action=parent_action)
    else:
        abstract_state = OneArmPaPState(problem_env, goal_entities,
                                        parent_state=parent_state,
                                        parent_action=parent_action)
    return abstract_state


def setup_seed_and_epoch(config):
    if config.learned_sampler_atype == 'pick':
        pick_seed, pick_epoch, _ = convert_seed_epoch_idxs_to_seed_and_epoch('pick', '', config)  
    else:
        pick_seed, pick_epoch = -1, -1

    if config.learned_sampler_atype == 'place_loading':
        place_obj_region_seed, place_obj_region_epoch, _ = convert_seed_epoch_idxs_to_seed_and_epoch('place', 'loading_region', config) 
    else:
        place_obj_region_seed, place_obj_region_epoch = -1, -1

    if config.learned_sampler_atype == 'place_home':
        place_goal_region_seed, place_goal_region_epoch, _ = convert_seed_epoch_idxs_to_seed_and_epoch('place', 'home_region', config)
    else:
        place_goal_region_seed, place_goal_region_epoch = -1, -1

    config.pick_seed = pick_seed
    config.pick_epoch = pick_epoch
    config.place_obj_region_seed = place_obj_region_seed
    config.place_obj_region_epoch = place_obj_region_epoch
    config.place_goal_region_seed = place_goal_region_seed
    config.place_goal_region_epoch = place_goal_region_epoch


def execute_policy(plan, learned_sampler_model, problem_env, goal_entities, config):
    init_abstract_state = make_abstract_state(problem_env, goal_entities)

    abstract_state = init_abstract_state
    total_ik_checks = 0
    total_mp_checks = 0
    total_pick_mp_checks = 0
    total_place_mp_checks = 0
    total_pick_mp_infeasible = 0
    total_place_mp_infeasible = 0

    total_infeasible_mp = 0
    plan_idx = 0
    n_total_actions = 0
    goal_reached = False
    stime = time.time()
    samples_tried = {i: [] for i in range(len(plan))}
    sample_values = {i: [] for i in range(len(plan))}

    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'rrt'))
    while plan_idx < len(plan):
        goal_reached = problem_env.is_goal_reached()
        if goal_reached:
            break
        if n_total_actions >= 100:
            break

        action = plan[plan_idx]
        generator = get_generator(abstract_state, action, learned_sampler_model, config)
        print action.discrete_parameters
        if 'two_arm' in config.domain:
            cont_smpl = generator.sample_next_point(samples_tried[plan_idx], sample_values[plan_idx])
        else:
            pick_smpl, place_smpl, status = generator.sample_next_point(samples_tried[plan_idx],
                                                                        sample_values[plan_idx])
            cont_smpl = {'pick': pick_smpl, 'place': place_smpl, 'is_feasible': status == "HasSolution"}

        total_ik_checks += generator.n_ik_checks
        total_mp_checks += generator.n_mp_checks
        total_infeasible_mp += generator.n_mp_infeasible
        total_pick_mp_checks += generator.n_pick_mp_checks
        total_place_mp_checks += generator.n_place_mp_checks
        total_pick_mp_infeasible += generator.n_pick_mp_infeasible
        total_place_mp_infeasible += generator.n_place_mp_infeasible

        n_total_actions += 1

        if cont_smpl['is_feasible']:
            print "Action executed"
            action.continuous_parameters = cont_smpl
            action.execute()
            plan_idx += 1
            abstract_state = make_abstract_state(problem_env, goal_entities,
                                                 parent_state=abstract_state,
                                                 parent_action=action)
        else:
            print "No feasible action"
            problem_env.init_saver.Restore()
            plan_idx = 0
            abstract_state = init_abstract_state
        goal_reached = plan_idx == len(plan)
        print "Total IK checks {} Total actions {}".format(total_ik_checks, n_total_actions)
    print time.time() - stime
    return total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible, total_place_mp_checks, \
           total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_logfile_name(config):
    logfile_dir = 'generators/sampler_performances/{}/{}/'.format(socket.gethostname(), config.learned_sampler_atype)
    if not os.path.isdir(logfile_dir):
        os.makedirs(logfile_dir)

    # todo save the log file using true epoch and true seed, instead of indices

    if config.use_learning:
        if config.learned_sampler_atype == 'pick':
            sampler_seed = config.pick_seed
            sampler_epoch = config.pick_epoch
        elif config.learned_sampler_atype == 'place_loading':
            sampler_seed = config.place_obj_region_seed
            sampler_epoch = config.place_obj_region_epoch
        else:
            sampler_seed = config.place_goal_region_seed
            sampler_epoch = config.place_goal_region_epoch

        logfile_dir += '/sampler_seed_{}/{}'.format(sampler_seed, config.train_type)
        if not os.path.isdir(logfile_dir):
            os.makedirs(logfile_dir)
        logfile = open(logfile_dir + '/epoch_{}.txt'.format(sampler_epoch), 'a')
    else:
        logfile = open(logfile_dir + config.sampling_strategy + '.txt', 'a')

    return logfile


def main():
    config = parse_arguments()

    plan, problem_env = load_planning_experience_data(config)
    goal_objs = problem_env.goal_objects
    goal_region = problem_env.goal_region
    planning_seed = config.planner_seed

    if config.v:
        utils.viewer()
    set_seeds(config.planner_seed)

    setup_seed_and_epoch(config)
    learned_sampler_model = get_learned_sampler_models(config)
    logfile = get_logfile_name(config)

    total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible, total_place_mp_checks, \
    total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached = \
        execute_policy(plan, learned_sampler_model, problem_env, goal_objs + [goal_region], config)

    result_log = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % (
        config.target_pidx_idx, planning_seed, total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible,
        total_place_mp_checks,
        total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached
    )
    logfile.write(result_log)
    logfile.close()


if __name__ == '__main__':
    main()
