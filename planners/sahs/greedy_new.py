import time
import pickle
import Queue
import numpy as np
from node import Node
from gtamp_utils import utils

from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.uniform import PaPUniformGenerator
from generators.learned_generator import LearnedGenerator

from trajectory_representation.operator import Operator

from helper import get_actions, compute_heuristic, get_state_class, update_search_queue

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0
counter = 1
sum_smpl_time = 0


def sample_continuous_parameters(abstract_action, abstract_state, abstract_node, mover, learned_sampler):
    global counter
    global sum_smpl_time
    target_obj = abstract_action.discrete_parameters['object']
    place_region = abstract_action.discrete_parameters['place_region']
    if learned_sampler is None or 'loading' not in place_region:
        smpler = PaPUniformGenerator(abstract_action, mover, max_n_iter=200)

        smpled_param = smpler.sample_next_point(abstract_action, n_parameters_to_try_motion_planning=3,
                                                cached_collisions=abstract_state.collides,
                                                cached_holding_collisions=None)
    else:
        """
        if abstract_node.smpler is None:
            smpler = LearnedGenerator(abstract_action, mover, learned_sampler, abstract_state, max_n_iter=200)
            abstract_node.smpler = smpler
        else:
            smpler = abstract_node.smpler
        """

        stime = time.time()
        """
        if (target_obj, place_region) in abstract_node.smplers_for_each_action:
            smpler = abstract_node.smplers_for_each_action[(target_obj, place_region)]
        else:
            smpler = LearnedGenerator(abstract_action, mover, learned_sampler, abstract_state, max_n_iter=200)
            abstract_node.smplers_for_each_action[(target_obj, place_region)] = smpler
        """
        smpler = LearnedGenerator(abstract_action, mover, learned_sampler, abstract_state, max_n_iter=200)

        smpled_param = smpler.sample_next_point(abstract_action, n_parameters_to_try_motion_planning=3,
                                                cached_collisions=abstract_state.collides,
                                                cached_holding_collisions=None)
        sum_smpl_time += time.time() - stime
        counter += 1

        print 'smpling time', time.time() - stime
        print "avgs smapling time", sum_smpl_time / counter

    return smpled_param


def search(mover, config, pap_model, learned_smpler=None):
    tt = time.time()

    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack
    statecls = get_state_class(config.domain)
    goal = mover.goal
    mover.reset_to_init_state_stripstream()
    depth_limit = 60

    # lowest valued items are retrieved first in PriorityQueue
    search_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory);
    state = statecls(mover, goal)
    initnode = Node(None, None, state)
    actions = get_actions(mover, goal, config)

    nodes = [initnode]
    update_search_queue(state, actions, initnode, search_queue, pap_model, mover, config)

    iter = 0
    # beginning of the planner
    while True:
        iter += 1
        curr_time = time.time() - tt
        print "Time %.2f / %.2f " % (curr_time, config.timelimit)
        print "Iter %d / %d" % (iter, config.num_node_limit)
        if curr_time > config.timelimit or iter > config.num_node_limit:
            return None, iter, nodes

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
        state = node.state
        print "Curr hval", curr_hval

        if node.depth > depth_limit:
            print('skipping because depth limit', node.action.discrete_parameters.values())

        # reset to state
        state.restore(mover)

        if action.type == 'two_arm_pick_two_arm_place':
            print("Sampling for {}".format(action.discrete_parameters.values()))
            # todo save the smpler with the abstract node
            smpled_param = sample_continuous_parameters(action, state, node, mover, learned_smpler)

            if smpled_param['is_feasible']:
                action.continuous_parameters = smpled_param
                action.execute()
                print "Action executed"
            else:
                print "Failed to sample an action"
                continue

            is_goal_achieved = \
                np.all([mover.regions['home_region'].contains(mover.env.GetKinBody(o).ComputeAABB()) for o in
                        obj_names[:n_objs_pack]])
            if is_goal_achieved:
                print("found successful plan: {}".format(n_objs_pack))
                plan = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                plan = [nd.action for nd in plan[1:]] + [action]
                return plan, iter, nodes
            else:
                newstate = statecls(mover, goal, node.state, action)
                print "New state computed"
                newnode = Node(node, action, newstate)
                newactions = get_actions(mover, goal, config)
                update_search_queue(newstate, newactions, newnode, search_queue, pap_model, mover, config)
                nodes.append(newnode)

        elif action.type == 'one_arm_pick_one_arm_place':
            print("Sampling for {}".format(action.discrete_parameters.values()))
            success = False

            obj = action.discrete_parameters['object']
            region = action.discrete_parameters['place_region']
            o = obj.GetName()
            r = region.name

            if (o, r) in state.nocollision_place_op:
                pick_op, place_op = node.state.nocollision_place_op[(o, r)]
                pap_params = pick_op.continuous_parameters, place_op.continuous_parameters
            else:
                mover.enable_objects()
                current_region = mover.get_region_containing(obj).name
                papg = OneArmPaPUniformGenerator(action, mover, cached_picks=(
                    node.state.iksolutions[current_region], node.state.iksolutions[r]))
                pick_params, place_params, status = papg.sample_next_point(500)
                if status == 'HasSolution':
                    pap_params = pick_params, place_params
                else:
                    pap_params = None

            if pap_params is not None:
                pick_params, place_params = pap_params
                action = Operator(
                    operator_type='one_arm_pick_one_arm_place',
                    discrete_parameters={
                        'object': obj,
                        'place_region': mover.regions[r],
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
                    plan = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                    plan = [nd.action for nd in plan[1:]] + [action]
                    return plan, iter, nodes
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
