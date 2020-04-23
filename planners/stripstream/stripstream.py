from __future__ import print_function

import os
import time
import pickle
import random
import copy
import Queue
import sys
import cProfile
import pstats
import argparse

import pddlstream.algorithms.instantiate_task
pddlstream.algorithms.instantiate_task.FD_INSTANTIATE = False

from pddlstream.algorithms.focused import solve_focused
from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.language.constants import print_solution
from pddlstream.utils import read
from pddlstream.language.generator import from_gen_fn, from_list_fn, from_test, fn_from_constant, from_fn
from pddlstream.algorithms.search import SERIALIZE
#import pdb;pdb.set_trace()

from gtamp_problem_environments.mover_env import Mover
from generators.uniform import UniformGenerator, PaPUniformGenerator
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator

from trajectory_representation.operator import Operator
from planners.subplanners.motion_planner import BaseMotionPlanner

from mover_library.utils import set_robot_config, set_obj_xytheta, visualize_path, two_arm_pick_object, two_arm_place_object, \
    get_body_xytheta, CustomStateSaver, set_color

from mover_library.motion_planner import rrt_region

from openravepy import RaveSetDebugLevel, DebugLevel
from trajectory_representation.trajectory import Trajectory

import numpy as np
import openravepy

from collections import Counter
from manipulation.primitives.display import set_viewer_options, draw_line, draw_point
from manipulation.primitives.savers import DynamicEnvironmentStateSaver

from generators.sampler import UniformSampler, LearnedSampler
from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator

PRM_VERTICES, PRM_EDGES = pickle.load(open('prm.pkl', 'rb'))
PRM_VERTICES = list(PRM_VERTICES)  # TODO: needs to be a list rather than ndarray

def gen_pap(problem, config):
    # cache ik solutions
    ikcachename = './ikcache.pkl'
    iksolutions = {}
    iksolutions = pickle.load(open(ikcachename, 'r'))

    def fcn(o, r, s):
        if problem.name == 'two_arm_mover':
            abstract_state = None # TODO: figure out what goes here
            abstract_action = Operator('two_arm_pick_two_arm_place', {'object': problem.env.GetKinBody(o), 'place_region': problem.regions[r]})
            sampler = UniformSampler(problem.regions[r])
            generator = TwoArmPaPGenerator(abstract_state, abstract_action, sampler,
                                           n_parameters_to_try_motion_planning=config.n_mp_limit,
                                           n_iter_limit=config.n_iter_limit, problem_env=problem,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
            while True:
                s.Restore()
                params = generator.sample_next_point()
                if params['is_feasible']:
                    abstract_action.continuous_parameters = params
                    abstract_action.execute()
                    t = CustomStateSaver(problem.env)
                    yield params, t
                else:
                    yield None

                if params['is_feasible']:
                    action.continuous_parameters = params
                    action.execute()
                    t = CustomStateSaver(problem.env)
                    yield params, t
                else:
                    yield None
        elif problem.name == 'one_arm_mover':
            while True:
                s.Restore()
                action = Operator('one_arm_pick_one_arm_place', {'object': problem.env.GetKinBody(o), 'place_region': problem.regions[r]})
                current_region = problem.get_region_containing(problem.env.GetKinBody(o)).name
                sampler = OneArmPaPUniformGenerator(action, problem, cached_picks=(iksolutions[current_region], iksolutions[r]))
                pick_params, place_params, status = sampler.sample_next_point(500)

                if status == 'HasSolution':
                    action.continuous_parameters = {'pick': pick_params, 'place': place_params}
                    action.execute()
                    t = CustomStateSaver(problem.env)
                    yield action.continuous_parameters, t
                else:
                    yield None
        else:
            raise NotImplementedError

    return fcn


def get_problem(mover, goal_objects, goal_region_name, config):
    directory = os.path.dirname(os.path.abspath(__file__))
    domain_pddl = read(os.path.join(directory, 'domain.pddl'))
    stream_pddl = read(os.path.join(directory, 'stream.pddl'))

    constant_map = {}
    stream_map = {
        'gen-pap': from_gen_fn(gen_pap(mover, config)),
    }

    obj_names = [obj.GetName() for obj in mover.objects]
    obj_poses = [get_body_xytheta(mover.env.GetKinBody(obj_name)).squeeze() for obj_name in obj_names]

    initial_robot_conf = get_body_xytheta(mover.robot).squeeze()

    if mover.name == 'two_arm_mover':
        goal_region = 'home_region'
        nongoal_regions = ['loading_region']
    elif mover.name == 'one_arm_mover':
        goal_region = mover.target_box_region.name
        nongoal_regions = ['center_shelf_region']#list(mover.shelf_regions)
    else:
        raise NotImplementedError

    print(goal_region, nongoal_regions, mover.regions.keys())

    init = [('Pickable', obj_name) for obj_name in obj_names]
    init += [('InRegion', obj_name, mover.get_region_containing(mover.env.GetKinBody(obj_name)).name) for obj_name in obj_names]
    init += [('Region', region) for region in nongoal_regions + [goal_region]]

    init += [('GoalObject', obj_name) for obj_name in goal_objects]
    init += [('NonGoalRegion', region) for region in nongoal_regions]

    init_state = CustomStateSaver(mover.env)
    init += [('State', init_state)]
    init += [('AtState', init_state)]

    # robot initialization
    init += [('EmptyArm',)]
    init += [('AtConf', initial_robot_conf)]
    init += [('BaseConf', initial_robot_conf)]

    # object initialization
    init += [('Pose', obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]
    init += [('PoseInRegion', obj_pose, 'loading_region') for obj_name, obj_pose in zip(obj_names, obj_poses)]
    init += [('AtPose', obj_name, obj_pose) for obj_name, obj_pose in zip(obj_names, obj_poses)]

    goal = ['and'] + [('InRegion', obj_name, goal_region_name)
                      for obj_name in goal_objects]

    print('Num init:', Counter(fact[0] for fact in init))
    print('Goal:', goal)
    print('Streams:', sorted(stream_map))

    return domain_pddl, constant_map, stream_pddl, stream_map, init, goal


def search(mover, config, pap_model, goal_objs, goal_region_name, learned_smpler=None, reachability_clf=None):
    if learned_smpler is not None:
        raise NotImplementedError

    goal_objs = goal_objs[:config.n_objs_pack]

    print('Vertices:', len(PRM_VERTICES))
    print('Edges:', sum(len(edges) for edges in PRM_EDGES))

    pddlstream_problem = get_problem(mover, goal_objs, goal_region_name, config)
    stime = time.time()
    pr = cProfile.Profile()
    pr.enable()
    # planner = 'ff-lazy' # -tiebreak
    # planner = 'ff-eager-tiebreak' # -tiebreak
    planner = 'ff-wastar5'
    # planner = 'cea-wastar5' # Performs worse than ff-wastar
    # planner = 'ff-ehc' # Worse

    set_color(mover.objects[0], [1, 0, 0])
    solution = solve_focused(pddlstream_problem, unit_costs=True, max_time=10 * 60,
    #solution = solve_incremental(pddlstream_problem, unit_costs=True, max_time=10 * 60,
                                 planner=planner, debug=True, verbose=True)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)
    search_time = time.time() - stime
    plan, cost, evaluations = solution
    print("time: {}".format(search_time))
    if plan is not None:
        print('Success')

        if config.domain == 'two_arm_mover':
            actions = [
                Operator('two_arm_pick_two_arm_place', {
                    'object': str(action.args[0]),
                    'place_region': str(action.args[1]),
                }, action.args[3])
                for action in plan
            ]
        elif config.domain == 'one_arm_mover':
            actions = [
                Operator('one_arm_pick_one_arm_place', {
                    'object': str(action.args[0]),
                    'place_region': str(action.args[1]),
                }, action.args[3])
                for action in plan
            ]
        else:
            raise NotImplementedError
        print(actions)
        return [], actions, 0, []
    else:
        print("Plan not found")
        return [], None, 0, []

