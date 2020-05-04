import time
import pickle
import Queue
import numpy as np
from node import Node
from gtamp_utils import utils

from generators.samplers.uniform_sampler import UniformSampler
from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator

from trajectory_representation.operator import Operator
from generators.samplers.voo_sampler import VOOSampler
from generators.voo import TwoArmVOOGenerator

from helper import get_actions, get_state_class, update_search_queue

prm_vertices, prm_edges = pickle.load(open('prm.pkl', 'rb'))
prm_vertices = list(prm_vertices)  # TODO: needs to be a list rather than ndarray

connected = np.array([len(s) >= 2 for s in prm_edges])
prm_indices = {tuple(v): i for i, v in enumerate(prm_vertices)}
DISABLE_COLLISIONS = False
MAX_DISTANCE = 1.0
counter = 1
sum_smpl_time = 0


def sample_continuous_parameters(abstract_action, abstract_state, abstract_node, mover, learned_sampler,
                                 reachability_clf, config):
    global counter
    global sum_smpl_time
    problem_env = abstract_state.problem_env
    place_region = problem_env.regions[abstract_action.discrete_parameters['place_region']]
    disc_param = (abstract_action.discrete_parameters['object'], abstract_action.discrete_parameters['place_region'])

    assert type(abstract_action.discrete_parameters['object']) == unicode \
           or type(abstract_action.discrete_parameters['object']) == str

    assert type(abstract_action.discrete_parameters['place_region']) == unicode \
           or type(abstract_action.discrete_parameters['place_region']) == str

    # todo save the generator into the abstract node
    we_dont_have_generator_for_this_discrete_action_yet = disc_param not in abstract_node.generators
    if we_dont_have_generator_for_this_discrete_action_yet:
        if config.sampling_strategy == 'uniform':
            sampler = UniformSampler(place_region)
            generator = TwoArmPaPGenerator(abstract_state, abstract_action, sampler,
                                           n_parameters_to_try_motion_planning=config.n_mp_limit,
                                           n_iter_limit=config.n_iter_limit, problem_env=problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
        elif config.sampling_strategy == 'voo':
            target_obj = abstract_action.discrete_parameters['object']
            sampler = VOOSampler(target_obj, place_region, config.explr_p, -np.inf)
            generator = TwoArmVOOGenerator(abstract_state, abstract_action, sampler,
                                           n_parameters_to_try_motion_planning=config.n_mp_limit,
                                           n_iter_limit=config.n_iter_limit, problem_env=problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
        else:
            raise NotImplementedError
        abstract_node.generators[disc_param] = generator
    smpled_param = abstract_node.generators[disc_param].sample_next_point()

    if config.sampling_strategy == 'voo' and not smpled_param['is_feasible']:
        abstract_node.generators[disc_param].update_mp_infeasible_samples(smpled_param['samples'])
    return smpled_param


def search(mover, config, pap_model, goal_objs, goal_region_name, learned_smpler=None, reachability_clf=None):
    tt = time.time()
    goal_region = mover.placement_regions[goal_region_name]
    obj_names = [obj.GetName() for obj in mover.objects]
    n_objs_pack = config.n_objs_pack
    statecls = get_state_class(config.domain)
    goal = mover.goal
    mover.reset_to_init_state_stripstream()
    depth_limit = 60

    # lowest valued items are retrieved first in PriorityQueue
    search_queue = Queue.PriorityQueue()  # (heuristic, nan, operator skeleton, state. trajectory);
    state = statecls(mover, goal)
    [utils.set_color(o, [1, 0, 0]) for o in goal_objs]
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
            smpled_param = sample_continuous_parameters(action, state, node, mover, learned_smpler, reachability_clf, config)

            if smpled_param['is_feasible']:
                action.continuous_parameters = smpled_param
                action.execute()
                executed_action = utils.get_body_xytheta(action.discrete_parameters['object']).squeeze()
                intended_action = action.continuous_parameters['place']['object_pose'].squeeze()
                placement_poses_match = np.all(np.isclose(executed_action[0:2], intended_action[0:2]))
                try:
                    assert placement_poses_match
                except:
                    import pdb;
                    pdb.set_trace()
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
                    node.is_goal_traj = True
                    nodes_to_goal = list(node.backtrack())[::-1]  # plan of length 0 is possible I think
                    plan = [nd.action for nd in nodes_to_goal[1:]] + [action]
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
