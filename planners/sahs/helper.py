from learn.data_traj import extract_individual_example
from planners.heuristics import compute_hcount_with_action, compute_hcount, get_goal_objs_not_in_goal_region
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from learn.data_traj import get_actions as convert_action_to_predictable_form
import numpy as np
import openravepy


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
    exp_q_vals = []
    for a in all_actions:
        a_raw_form = convert_action_to_predictable_form(a, entity_names)
        if np.all(a_raw_form == actions):
            continue
        bonus_val = compute_bonus_val(pap_model, nodes, edges, a_raw_form)
        exp_q_vals.append(bonus_val)

    bonus_val_on_curr_a = compute_bonus_val(pap_model, nodes, edges, actions)
    q_bonus = bonus_val_on_curr_a / (np.sum(exp_q_vals) + 1e-5)
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


def compute_hcount_old_number_in_goal(state, action):
    problem_env = state.problem_env
    target_o = action.discrete_parameters['object']
    if type(target_o) != str and type(target_o) != unicode:
        target_o = target_o.GetName()
    target_r = action.discrete_parameters['place_region']
    if type(target_r) != str and type(target_r) != unicode:
        target_r = target_r.name
    region_is_goal = state.nodes[target_r][8]
    goal_region = problem_env.goal_region
    hcount = compute_hcount(state)
    given_obj_already_in_goal = state.binary_edges[(target_o, goal_region)][0]  # The target object is already in goal
    number_in_goal = len(problem_env.goal_objects) - len(get_goal_objs_not_in_goal_region(state))
    analytical_heuristic = -number_in_goal + given_obj_already_in_goal + hcount
    # print target_o, target_r
    # print "HCount %d number_in_goal %d given_objs_already_in_goal %d" % (hcount, number_in_goal, given_obj_already_in_goal)
    return analytical_heuristic


def compute_heuristic(state, action, pap_model, h_option, mixrate):
    # parameters used for CoRL
    assert h_option == 'qlearned_hcount_old_number_in_goal' or \
           h_option == 'hcount_old_number_in_goal' or\
           h_option == 'qlearned'
    assert mixrate == 1

    is_two_arm_domain = 'two_arm_' in action.type
    problem_env = state.problem_env
    if is_two_arm_domain:
        target_o = action.discrete_parameters['object']
        target_r = action.discrete_parameters['place_region']
    else:
        target_o = action.discrete_parameters['object'].GetName()
        target_r = action.discrete_parameters['place_region'].name

    nodes, edges, actions, _ = extract_individual_example(state, action)
    nodes = nodes[..., 6:]

    region_is_goal = state.nodes[target_r][8]

    if 'two_arm' in problem_env.name:
        goal_objs = [tmp_o for tmp_o in state.goal_entities if 'box' in tmp_o]
        goal_region = 'home_region'
    else:
        goal_objs = [tmp_o for tmp_o in state.goal_entities if 'region' not in tmp_o]
        goal_region = 'rectangular_packing_box1_region'

    if h_option == 'qlearned_hcount_old_number_in_goal':
        nodes, edges, actions, _ = extract_individual_example(state, action)  # why do I call this again?
        nodes = nodes[..., 6:]
        q_bonus = compute_q_bonus(state, nodes, edges, actions, pap_model, problem_env)
        analytical_heuristic = compute_hcount_old_number_in_goal(state, action)
        hval = analytical_heuristic - mixrate * q_bonus
    elif h_option == 'hcount_old_number_in_goal':
        analytical_heuristic = compute_hcount_old_number_in_goal(state, action)
        hval = analytical_heuristic
    elif h_option == 'qlearned':
        nodes, edges, actions, _ = extract_individual_example(state, action)  # why do I call this again?
        nodes = nodes[..., 6:]
        hval = compute_q_bonus(state, nodes, edges, actions, pap_model, problem_env)

    return hval


def count_pickable_goal_objs_and_placeable_to_goal_region_not_yet_in_goal_region(state):
    goal_r = [entity for entity in state.goal_entities if 'region' in entity][0]
    goal_objs = [entity for entity in state.goal_entities if not 'region' in entity]
    n_pickable_and_placeable = 0
    for goal_obj in goal_objs:
        goal_obj_pickable = state.nodes[goal_obj][-2]
        goal_obj_placeable = state.binary_edges[(goal_obj, goal_r)][-1]
        n_pickable_and_placeable += goal_obj_pickable and goal_obj_placeable
    return n_pickable_and_placeable


def update_search_queue(state, actions, node, action_queue, pap_model, mover, config):
    print "Enqueuing..."
    for a in actions:
        hval = compute_heuristic(state, a, pap_model, config.h_option, config.mixrate)
        if config.gather_planning_exp:
            h_for_sampler_training = compute_hcount(state)
            num_in_goal = len(state.problem_env.goal_objects) - len(get_goal_objs_not_in_goal_region(state))

            # hcount recursively counts the number of objects obstructing the way to the goal objs not in the goal reigon
            # This can potentially have error in estimating the cost-to-go, because even a single object not in a goal
            # can have all objects in its way, since our motion planner is not optimal in MCR sense.
            # So, I have to track the number of objects in goal.
            node.h_for_sampler_training = h_for_sampler_training - num_in_goal

        discrete_params = (a.discrete_parameters['object'], a.discrete_parameters['place_region'])
        node.set_heuristic(discrete_params, hval)
        action_queue.put((hval, float('nan'), a, node))  # initial q

        obj = a.discrete_parameters['object']
        if not (isinstance(obj, str) or  isinstance(obj, unicode)):
            obj = obj.GetName()
        region = a.discrete_parameters['place_region']
        if not (isinstance(region, str)):
            region = region.name
        print "%35s %35s  hval %.4f" % (obj, region, hval)
