import time
import pickle
import Queue
import numpy as np
import os

from node import Node
from gtamp_utils import utils
from trajectory_representation.operator import Operator

# generators and samplers
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.voo import TwoArmVOOGenerator
from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.one_arm_generators.one_arm_pap_generator import OneArmPaPGenerator
from generators.samplers.pick_place_learned_sampler import PickPlaceLearnedSampler
from generators.samplers.pick_only_learned_sampler import PickOnlyLearnedSampler
from generators.samplers.place_only_learned_sampler import PlaceOnlyLearnedSampler

from generators.samplers.uniform_sampler import UniformSampler
from generators.samplers.voo_sampler import VOOSampler

from helper import get_actions, get_state_class, update_search_queue

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0


def get_sampler(config, abstract_state, abstract_action, learned_sampler_model):
    if not config.use_learning:
        if 'uniform' in config.sampling_strategy:
            target_region = abstract_state.problem_env.regions[abstract_action.discrete_parameters['place_region']]
            if 'two_arm' in config.domain:
                sampler = UniformSampler(atype='two_arm_pick_and_place', target_region=target_region)
            else:
                sampler = {'pick': UniformSampler(target_region=None, atype='one_arm_pick'),
                           'place': UniformSampler(target_region=target_region, atype='one_arm_place')}
        elif 'voo' in config.sampling_strategy:
            raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        if 'two_arm' in config.domain:
            sampler = PickPlaceLearnedSampler('two_arm_pick_and_place', learned_sampler_model, abstract_state, abstract_action)
        else:
            pick_sampler = PickOnlyLearnedSampler('one_arm_pick', learned_sampler_model, abstract_state, abstract_action)
            place_sampler = PlaceOnlyLearnedSampler('one_arm_place', learned_sampler_model, abstract_state, abstract_action)
            sampler = {'pick': pick_sampler, 'place': place_sampler}
    return sampler


def get_generator(abstract_state, action, sampler_model, config):
    sampler = get_sampler(config, abstract_state, action, sampler_model)
    if 'unif' in config.sampling_strategy:
        if 'two_arm' in config.domain:
            sampler.infeasible_action_value = -9999
            generator = TwoArmPaPGenerator(abstract_state, action, sampler,
                                           n_parameters_to_try_motion_planning=config.n_mp_limit,
                                           n_iter_limit=config.n_iter_limit, problem_env=abstract_state.problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='robot_base_pose')
        else:
            generator = OneArmPaPGenerator(action, n_iter_limit=config.n_iter_limit,
                                           problem_env=abstract_state.problem_env,
                                           pick_sampler=sampler['pick'], place_sampler=sampler['place'])

    elif 'voo' in config.sampling_strategy:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return generator


def sample_continuous_parameters(abstract_state, abstract_action, abstract_node, sampler_model, config):
    disc_param = (abstract_action.discrete_parameters['object'], abstract_action.discrete_parameters['place_region'])

    we_dont_have_generator_for_this_discrete_action_yet = disc_param not in abstract_node.generators
    if we_dont_have_generator_for_this_discrete_action_yet:
        generator = get_generator(abstract_state, abstract_action, sampler_model, config)
        abstract_node.generators[disc_param] = generator
    smpled_param = abstract_node.generators[disc_param].sample_next_point()

    if config.sampling_strategy == 'voo' and not smpled_param['is_feasible']:
        abstract_node.generators[disc_param].update_mp_infeasible_samples(smpled_param['samples'])

    return smpled_param


def search(mover, config, pap_model, goal_objs, goal_region_name, learned_sampler_model):
    tt = time.time()
    print "Greedy search began"
    goal_region = mover.placement_regions[goal_region_name]
    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack
    statecls = get_state_class(config.domain)
    goal = mover.goal_entities
    mover.reset_to_init_state_stripstream()
    depth_limit = 60
    # lowest valued items are retrieved first in PriorityQueue
    search_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory);a
    print "State computation..."
    state = statecls(mover, goal)

    initnode = Node(None, None, state)
    actions = get_actions(mover, goal, config)

    nodes = [initnode]
    update_search_queue(state, actions, initnode, search_queue, pap_model, mover, config)

    iter = 0
    # beginning of the planner
    print "Beginning of the while-loop"
    while True:
        iter += 1
        curr_time = time.time() - tt
        print "Time %.2f / %.2f " % (curr_time, config.timelimit)
        print "Iter %d / %d" % (iter, config.num_node_limit)
        if curr_time > config.timelimit or iter > config.num_node_limit:
            return None, None, iter, nodes

        # persistency
        if search_queue.empty():
            actions = get_actions(mover, goal, config)
            for a in actions:
                discrete_params = (a.discrete_parameters['object'], a.discrete_parameters['place_region'])
                hval = initnode.heuristic_vals[discrete_params]
                search_queue.put((hval, float('nan'), a, initnode))  # initial q

        # can a node be associated with different action? I think so.
        # For example, the init node can be associated with many actions
        curr_hval, _, action, node = search_queue.get()
        print "Chosen abstract action", action.discrete_parameters['object'], action.discrete_parameters['place_region']
        state = node.state
        print "Curr hval", curr_hval

        if node.depth > depth_limit:
            print('skipping because depth limit', node.action.discrete_parameters.values())

        # reset to state
        state.restore(mover)

        if action.type == 'two_arm_pick_two_arm_place':
            print("Sampling for {}".format(action.discrete_parameters.values()))
            smpled_param = sample_continuous_parameters(state, action, node, learned_sampler_model, config)
            if smpled_param['is_feasible']:
                action.continuous_parameters = smpled_param
                action.execute()
                print "Action executed"
            else:
                print "Failed to sample an action"
                continue
            is_goal_achieved = np.all([goal_region.contains(mover.env.GetKinBody(o).ComputeAABB()) for o in goal_objs])
            if is_goal_achieved:
                print("found successful plan: {}".format(n_objs_pack))
                node.is_goal_traj = True
                nodes_to_goal = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                plan = [nd.parent_action for nd in nodes_to_goal[1:]] + [action]
                for plan_action, nd in zip(plan, nodes_to_goal):
                    nd.is_goal_traj = True
                    nd.executed_action = plan_action

                return nodes_to_goal, plan, iter, nodes
            else:
                stime = time.time()
                newstate = statecls(mover, goal, node.state, action)
                print "New state computation time ", time.time() - stime
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                update_search_queue(newstate, newactions, newnode, search_queue, pap_model, mover, config)
                nodes.append(newnode)

        elif action.type == 'one_arm_pick_one_arm_place':
            print("Sampling for {}".format(action.discrete_parameters.values()))
            success = False

            o = action.discrete_parameters['object']
            r = action.discrete_parameters['place_region']

            if (o, r) in state.nocollision_place_op:
                print "Already no collision place op"
                pick_op, place_op = node.state.nocollision_place_op[(o, r)]
                pap_params = pick_op.continuous_parameters, place_op.continuous_parameters
            else:
                mover.enable_objects()
                #papg = OneArmPaPUniformGenerator(action, mover,
                #                                 cached_picks=None)
                #pick_params, place_params, status = papg.sample_next_point(200)
                pick_params, place_params, status = sample_continuous_parameters(state, action, node,
                                                                                 learned_sampler_model, config)
                if status == 'HasSolution':
                    pap_params = pick_params, place_params
                else:
                    pap_params = None

            if pap_params is not None:
                pick_params, place_params = pap_params
                action = Operator(
                    operator_type='one_arm_pick_one_arm_place',
                    discrete_parameters={
                        'object': o,
                        'region': mover.regions[r],
                    },
                    continuous_parameters={
                        'pick': pick_params,
                        'place': place_params,
                    }
                )
                action.execute()

                success = True

                is_goal_achieved = \
                    np.all([mover.regions['rectangular_packing_box1_region'].contains(
                        mover.env.GetKinBody(o).ComputeAABB()) for o in obj_names[:n_objs_pack]])

                if is_goal_achieved:
                    print("found successful plan: {}".format(n_objs_pack))
                    node.is_goal_traj = True
                    nodes_to_goal = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                    plan = [nd.parent_action for nd in nodes_to_goal[1:]] + [action]
                    return nodes_to_goal, plan, iter, nodes
                else:
                    newstate = statecls(mover, goal, node.state, action)
                    newnode = Node(node, action, newstate)
                    newactions = get_actions(mover, goal, config)
                    update_search_queue(newstate, newactions, newnode, search_queue, pap_model, mover, config)

            if not success:
                print('failed to execute action')
            else:
                print('action successful')

        else:
            raise NotImplementedError
