from trajectory_representation.operator import Operator
from generators.uniform import UniformGenerator
from planners.subplanners.motion_planner import BaseMotionPlanner
from gtamp_utils.utils import CustomStateSaver

from gtamp_utils import utils

from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.one_arm_pap_uniform_generator import OneArmPaPUniformGenerator
from generators.samplers.uniform_sampler import UniformSampler

import pickle
import numpy as np


class PlannerWithoutReachability:
    def __init__(self, problem_env, goal_object_names, goal_region, config):
        self.problem_env = problem_env
        self.goal_objects = [problem_env.env.GetKinBody(o) for o in goal_object_names]
        self.goal_region = self.problem_env.regions[goal_region]

        self.config = config

        # cache ik solutions
        ikcachename = './ikcache.pkl'
        self.iksolutions = pickle.load(open(ikcachename, 'r'))

        self.n_mp = 0
        self.n_ik = 0

    def sample_cont_params(self, operator_skeleton, n_iter):
        target_object = operator_skeleton.discrete_parameters['object']
        self.problem_env.disable_objects_in_region('entire_region')
        generator = UniformGenerator(operator_skeleton, self.problem_env, None)
        target_object.Enable(True)
        print "Generating goals for ", target_object
        param = generator.sample_next_point(operator_skeleton,
                                            n_iter=n_iter,
                                            n_parameters_to_try_motion_planning=1,
                                            dont_check_motion_existence=True)
        self.problem_env.enable_objects_in_region('entire_region')
        return param

    def get_goal_config_used(self, motion_plan, potential_goal_configs):
        which_goal = np.argmin(np.linalg.norm(motion_plan[-1] - potential_goal_configs, axis=-1))
        return potential_goal_configs[which_goal]

    def find_pap(self, curr_obj):
        if self.problem_env.name.find("one_arm") != -1:
            op = Operator(operator_type='one_arm_pick_one_arm_place',
                                discrete_parameters={'object': curr_obj, 'place_region': self.goal_region.name})
            current_region = problem.get_region_containing(problem.env.GetKinBody(o)).name
            sampler = OneArmPaPUniformGenerator(action, problem, cached_picks=(iksolutions[current_region], self.iksolutions[self.goal_region.name]))
            pick_params, place_params, status = sampler.sample_next_point(self.config.n_iter_limit)
            self.n_ik += generator.n_ik_checks
            if status == 'HasSolution':
                params = {'pick': pick_params, 'place': place_params}
            else:
                params = None
        else:
            op = Operator(operator_type='two_arm_pick_two_arm_place', discrete_parameters={'object': curr_obj,
                                                                                    'place_region': self.goal_region.name})
            state = None

            sampler = UniformSampler(self.goal_region)
            generator = TwoArmPaPGenerator(state, op, sampler,
                                           n_parameters_to_try_motion_planning=self.config.n_mp_limit,
                                           n_iter_limit=self.config.n_iter_limit, problem_env=self.problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
            params = generator.sample_next_point()
            self.n_mp += generator.n_mp_checks
            self.n_ik += generator.n_ik_checks
        # it must be because sampling a feasible pick can be done by trying as many as possible,
        # but placements cannot be made feasible  by sampling more
        # also, it takes longer to check feasibility on place?
        # I just have to check the IK solution once
        if not params['is_feasible']:
            return None

        op.continuous_parameters = params
        return op

    def search(self):
        # returns the order of objects that respects collision at placements
        # todo if I cannot find a grasp or placement in the goal region, then I should declare infeasible problem

        init_state = CustomStateSaver(self.problem_env.env)
        # self.problem_env.set_exception_objs_when_disabling_objects_in_region(self.goal_objects)
        idx = 0
        plan = []
        goal_obj_move_plan = []

        while True:
            curr_obj = self.goal_objects[idx]

            self.problem_env.disable_objects_in_region('entire_region')
            print [o.IsEnabled() for o in self.problem_env.objects]
            curr_obj.Enable(True)
            pap = self.find_pap(curr_obj)
            if pap is None:
                plan = []  # reset the whole thing?
                goal_obj_move_plan = []
                idx += 1
                idx = idx % len(self.goal_objects)
                init_state.Restore()
                print "Pap sampling failed"
                continue
            pap.execute()

            plan.append(pap)
            goal_obj_move_plan.append(curr_obj)

            idx += 1
            idx = idx % len(self.goal_objects)
            print "Plan length: ", len(plan)
            if len(plan) / 2.0 == len(self.goal_objects):
                break

        init_state.Restore()
        return goal_obj_move_plan, plan
