import numpy as np
import random
import pickle
import time
import os
import torch
import sys
from planners.sahs.helper import get_state_class

from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from planners.sahs.greedy_new import get_generator
from test_scripts.run_greedy import get_learned_sampler_models

from gtamp_utils import utils
from run_greedy import parse_arguments, get_pap_gnn_model, get_problem_env, get_goal_obj_and_region, \
    set_problem_env_config
from planners.sahs.helper import get_actions, compute_heuristic
from test_scripts.run_mcts import get_commit_hash


def get_max_action_using(pap_model, abstract_state, problem_env, actions_tried=[]):
    actions = get_actions(problem_env, None, None)
    best_action = None
    best_hval = -np.inf
    if len(actions_tried) >= len(actions):
        actions_tried = []
    candidate_actions = [a for a in actions if
                         (a.discrete_parameters['object'], a.discrete_parameters['place_region']) not in actions_tried]
    for a in candidate_actions:
        hval = compute_heuristic(abstract_state, a, pap_model, 'qlearned', 1)
        print a.discrete_parameters, hval
        if hval > best_hval:
            params = (a.discrete_parameters['object'], a.discrete_parameters['place_region'])
            if params not in actions_tried:
                best_hval = hval
                best_action = a

    return best_action


def execute_learned_predictors(pap_model, learned_sampler, problem_env, goal_entities, config):
    statecls = get_state_class(config.domain)

    if os.path.isfile('tmp.pkl'):
        init_abstract_state = pickle.load(open('tmp.pkl', 'r'))
    else:
        init_abstract_state = statecls(problem_env, goal_entities)
        init_abstract_state.make_pklable()
        pickle.dump(init_abstract_state, open('tmp.pkl', 'wb'))

    abstract_state = init_abstract_state
    abstract_state.make_plannable(problem_env)

    plan_idx = 0
    n_total_actions = 0
    goal_reached = False
    stime = time.time()

    infeasible_abstract_actions = []
    while time.time() - stime < config.timelimit:
        goal_reached = problem_env.is_goal_reached()
        if goal_reached:
            break

        action = get_max_action_using(pap_model, abstract_state, problem_env, infeasible_abstract_actions)
        generator = get_generator(abstract_state, action, learned_sampler, config)
        if 'two_arm' in config.domain:
            cont_smpl = generator.sample_next_point()
            is_cont_param_feasible = cont_smpl['is_feasible']
            action.continuous_parameters = cont_smpl
        else:
            pick_params, place_params, status = generator.sample_next_point()
            is_cont_param_feasible = status == 'HasSolution'
            action.continuous_parameters = {
                'pick': pick_params,
                'place': place_params,
            }

        n_total_actions += 1

        if is_cont_param_feasible:
            print "Action executed"
            action.execute()
            plan_idx += 1
            abstract_state = statecls(problem_env, goal_entities, parent_state=abstract_state, parent_action=action)
        else:
            print "No feasible action"
            if plan_idx == 0:
                infeasible_abstract_actions.append(
                    (action.discrete_parameters['object'], action.discrete_parameters['place_region']))
            problem_env.init_saver.Restore()
            plan_idx = 0

    total_time = time.time() - stime
    return total_time, n_total_actions, goal_reached


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def get_solution_file_name(config):
    root_dir = './'
    commit_hash = get_commit_hash()

    solution_file_dir = root_dir + '/test_results/%s/pure_learning/domain_%s/n_objs_pack_%d' \
                        % (commit_hash, config.domain, config.n_objs_pack)
    solution_file_dir += '/' + config.h_option + '/'

    q_config = '/q_config_num_train_' + str(config.num_train) + \
               '_mse_weight_' + str(config.mse_weight) + \
               '_use_region_agnostic_' + str(config.use_region_agnostic) + \
               '_mix_rate_' + str(config.mixrate) + '/'
    solution_file_dir += q_config

    assert config.use_learning
    solution_file_dir += '/using_learned_sampler/'
    solution_file_dir += '/n_mp_limit_%d_n_iter_limit_%d/' % (config.n_mp_limit, config.n_iter_limit)

    solution_file_name = 'pidx_' + str(config.pidx) + \
                         '_planner_seed_' + str(config.planner_seed) + \
                         '_gnn_seed_' + str(config.absq_seed) + '.pkl'

    if not os.path.isdir(solution_file_dir):
        os.makedirs(solution_file_dir)
    solution_file_dir += '/sampling_strategy_' + config.sampling_strategy
    solution_file_name = solution_file_dir + solution_file_name
    return solution_file_name


def main():
    config = parse_arguments()
    config.use_learning = True
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
            print 'Time: %.2f Success: %d Plan length: %d Num nodes: %d ' % (
                tottime, success, plan_length, num_nodes)
        sys.exit(-1)

    goal_objs, goal_region = get_goal_obj_and_region(config)
    problem_env = get_problem_env(config, goal_region, goal_objs)
    set_problem_env_config(problem_env, config)

    pap_model = get_pap_gnn_model(problem_env, config)
    np.random.seed(config.planner_seed)
    random.seed(config.planner_seed)
    learned_sampler_model = get_learned_sampler_models(config)

    total_time, n_total_actions, goal_reached = \
        execute_learned_predictors(pap_model, learned_sampler_model, problem_env, goal_objs + [goal_region], config)
    print goal_reached
    data = {
        'n_objs_pack': config.n_objs_pack,
        'tottime': total_time,
        'success': goal_reached,
        'num_nodes': n_total_actions,
    }

    with open(solution_file_name, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
