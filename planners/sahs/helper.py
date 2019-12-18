from learn.data_traj import extract_individual_example
from planners.heuristics import compute_hcount_with_action, compute_hcount
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from learn.data_traj import get_actions as convert_action_to_predictable_form
import numpy as np


def get_actions(mover, goal, config):
    actions = mover.get_applicable_ops()
    permuted_actions = np.random.permutation(actions).tolist()
    return permuted_actions


def compute_bonus_val(pap_model, nodes, edges, a_raw_form):
    q_val = pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...], a_raw_form[None, ...])[0]
    if abs(q_val) > 10:
        bonus_val = np.exp(q_val / 100.0)
    else:
        bonus_val = np.exp(q_val)
    return bonus_val


def compute_q_bonus(state, nodes, edges, actions, pap_model, problem_env):
    all_actions = get_actions(problem_env, None, None)
    entity_names = list(state.nodes.keys())[::-1]
    q_vals = []
    for a in all_actions:
        a_raw_form = convert_action_to_predictable_form(a, entity_names)
        if np.all(a_raw_form == actions):
            continue
        bonus_val = compute_bonus_val(pap_model, nodes, edges, a_raw_form)
        q_vals.append(bonus_val)

    bonus_val_on_curr_a = compute_bonus_val(pap_model, nodes, edges, actions)
    q_bonus = bonus_val_on_curr_a / (np.sum(q_vals) + bonus_val_on_curr_a)
    return q_bonus


def get_state_class(domain):
    if domain == 'two_arm_mover':
        statecls = ShortestPathPaPState
    elif domain == 'one_arm_mover':
        def create_one_arm_pap_state(*args, **kwargs):
            while True:
                state = OneArmPaPState(*args, **kwargs)
                if len(state.nocollision_place_op) > 0:
                    return state
                else:
                    print('failed to find any paps, trying again')

        statecls = create_one_arm_pap_state
    else:
        raise NotImplementedError
    return statecls


def compute_heuristic(state, action, pap_model, problem_env, config):
    is_two_arm_domain = 'two_arm_' in action.type
    if is_two_arm_domain:
        target_o = action.discrete_parameters['object']
        target_r = action.discrete_parameters['place_region']
    else:
        target_o = action.discrete_parameters['object'].GetName()
        target_r = action.discrete_parameters['region'].name

    nodes, edges, actions, _ = extract_individual_example(state, action)
    nodes = nodes[..., 6:]

    region_is_goal = state.nodes[target_r][8]

    if 'two_arm' in problem_env.name:
        goal_objs = [tmp_o for tmp_o in state.goal_entities if 'box' in tmp_o]
        goal_region = 'home_region'
    else:
        goal_objs = [tmp_o for tmp_o in state.goal_entities if 'region' not in tmp_o]
        goal_region = 'rectangular_packing_box1_region'

    h_option = config.h_option

    if h_option == 'hcount':
        hval = compute_hcount_with_action(state, action, problem_env)
    elif h_option == 'hcount_old_number_in_goal':
        number_in_goal = compute_number_in_goal(state, target_o, problem_env, region_is_goal)
        hcount = compute_hcount_with_action(state, action, problem_env)
        hval = hcount - number_in_goal
    elif h_option == 'state_hcount':
        hval = compute_hcount(state, problem_env)
    elif h_option == 'qlearned_hcount_new_number_in_goal':
        number_in_goal = compute_new_number_in_goal(state, goal_objs, goal_region)
        q_bonus = compute_q_bonus(state, nodes, edges, actions, pap_model, problem_env)
        hcount = compute_hcount(state, problem_env)
        obj_already_in_goal = state.binary_edges[(target_o, goal_region)][0]
        hval = -number_in_goal + obj_already_in_goal + hcount - config.mixrate * q_bonus
    elif h_option == 'qlearned_hcount_old_number_in_goal':
        number_in_goal = compute_number_in_goal(state, target_o, problem_env, region_is_goal)

        nodes, edges, actions, _ = extract_individual_example(state, action) # why do I call this again?
        nodes = nodes[..., 6:]

        q_bonus = compute_q_bonus(state, nodes, edges, actions, pap_model, problem_env)
        hcount = compute_hcount(state, problem_env)
        obj_already_in_goal = state.binary_edges[(target_o, goal_region)][0]
        hval = -number_in_goal + obj_already_in_goal + hcount - config.mixrate * q_bonus
    elif h_option == 'qlearned_old_number_in_goal':
        number_in_goal = compute_number_in_goal(state, target_o, problem_env, region_is_goal)
        q_val_on_curr_a = pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...],
                                                                  actions[None, ...])
        obj_already_in_goal = state.binary_edges[(target_o, goal_region)][0]
        hval = -number_in_goal - q_val_on_curr_a + obj_already_in_goal
    elif h_option == 'config.qlearned_new_number_in_goal':
        number_in_goal = compute_new_number_in_goal(state, goal_objs, goal_region)
        q_val_on_curr_a = pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...],
                                                                  actions[None, ...])
        obj_already_in_goal = state.binary_edges[(target_o, goal_region)][0]
        hval = -number_in_goal - q_val_on_curr_a + obj_already_in_goal
    elif h_option == 'config.pure_learned_q':
        q_val_on_curr_a = pap_model.predict_with_raw_input_format(nodes[None, ...], edges[None, ...],
                                                                  actions[None, ...])
        hval = -q_val_on_curr_a
    else:
        raise NotImplementedError

    return hval


def compute_number_in_goal(state, target_o, problem_env, region_is_goal):
    number_in_goal = 0
    for i in state.nodes:
        if i == target_o:
            continue
        for tmpr in problem_env.regions:
            if tmpr in state.nodes:
                is_r_goal_region = state.nodes[tmpr][8]
                if is_r_goal_region:
                    is_i_in_r = state.binary_edges[(i, tmpr)][0]
                    if is_r_goal_region:
                        number_in_goal += is_i_in_r
    number_in_goal += int(region_is_goal)  # encourage moving goal obj to goal region
    return number_in_goal


def compute_new_number_in_goal(state, goal_objs, goal_region):
    number_in_goal = 0
    for obj_name in goal_objs:
        is_obj_in_goal_region = state.binary_edges[(obj_name, goal_region)][0]
        if is_obj_in_goal_region:
            number_in_goal += 1
    return number_in_goal


def update_search_queue(state, actions, node, action_queue, pap_model, mover, config):
    print "Enqueuing..."
    for a in actions:
        hval = compute_heuristic(state, a, pap_model, mover, config)
        discrete_params = (a.discrete_parameters['object'], a.discrete_parameters['place_region'])
        node.set_heuristic(discrete_params, hval)
        action_queue.put((hval, float('nan'), a, node))  # initial q
        print "%35s %35s  hval %.4f" % (a.discrete_parameters['object'], a.discrete_parameters['place_region'], hval)

