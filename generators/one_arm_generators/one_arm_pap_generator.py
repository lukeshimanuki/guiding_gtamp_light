import numpy as np
import random
import copy

from gtamp_utils.utils import get_pick_base_pose_and_grasp_from_pick_parameters, get_body_xytheta, set_robot_config, \
    release_obj
from gtamp_utils import utils
from generators.one_arm_generators.one_arm_pick_generator import OneArmPickGenerator
from generators.uniform import UniformGenerator
from trajectory_representation.operator import Operator
from gtamp_utils.operator_utils import grasp_utils
from generators.samplers.uniform_sampler import UniformSampler

from generators.feasibility_checkers.one_arm_pick_feasibility_checker import OneArmPickFeasibilityChecker
from generators.feasibility_checkers.one_arm_place_feasibility_checker import OneArmPlaceFeasibilityChecker


class OneArmPaPGenerator:
    def __init__(self, operator_skeleton, n_iter_limit, problem_env, pick_sampler=None, place_sampler=None):
        self.problem_env = problem_env
        self.n_iter_limit = n_iter_limit
        target_region = None
        if 'place_region' in operator_skeleton.discrete_parameters:
            target_region = operator_skeleton.discrete_parameters['place_region']
            if type(target_region) == str:
                target_region = self.problem_env.regions[target_region]
        target_obj = operator_skeleton.discrete_parameters['object']
        self.robot = problem_env.robot
        self.target_region = target_region
        self.target_obj = target_obj
        if type(self.target_obj) is str  or type(self.target_obj) is unicode:
            self.target_obj = self.problem_env.env.GetKinBody(self.target_obj)

        # todo change this to use the sampler passed in
        self.pick_op = Operator(operator_type='one_arm_pick',
                                discrete_parameters={'object': target_obj})
        self.pick_sampler = pick_sampler

        self.place_op = Operator(operator_type='one_arm_place',
                                 discrete_parameters={'object': target_obj, 'place_region': target_region},
                                 continuous_parameters={})
        self.place_sampler = place_sampler
        self.pick_feasibility_checker = OneArmPickFeasibilityChecker(problem_env)
        self.place_feasibility_checker = OneArmPlaceFeasibilityChecker(problem_env)
        self.operator_skeleton = operator_skeleton

        self.n_pick_mp_checks = 0
        self.n_mp_checks = 0
        self.n_mp_infeasible = 0
        self.n_place_mp_checks = 0
        self.n_pick_mp_infeasible = 0
        self.n_place_mp_infeasible = 0

        self.n_ik_checks = 0

    def sample_next_point(self, samples_tried=None, sample_values=None):
        # n_iter refers to the max number of IK attempts on pick
        n_ik_attempts = 0
        while True:
            pick_cont_params, place_cont_params, status = self.sample_from_continuous_space()
            if status == 'InfeasibleIK' or status == 'InfeasibleBase':
                n_ik_attempts += 1
                self.n_ik_checks += 1
                if n_ik_attempts == self.n_iter_limit:
                    break
            elif status == 'HasSolution':
                return pick_cont_params, place_cont_params, 'HasSolution'
        return None, None, 'NoSolution'

    def is_base_feasible(self, base_pose):
        utils.set_robot_config(base_pose, self.robot)
        robot_aabb = self.robot.ComputeAABB()
        inside_region = self.problem_env.regions['home_region'].contains(robot_aabb)
        # or self.problem_env.regions['loading_region'].contains(robot_aabb)
        no_collision = not self.problem_env.env.CheckCollision(self.robot)
        if (not inside_region) or (not no_collision):
            return False
        else:
            return True

    def sample_pick_cont_parameters(self):
        op_parameters = self.pick_sampler.sample()
        grasp_params, pick_base_pose = get_pick_base_pose_and_grasp_from_pick_parameters(self.target_obj, op_parameters)
        if not self.is_base_feasible(pick_base_pose):
            return None, 'InfeasibleBase'

        utils.open_gripper()
        grasps = grasp_utils.compute_one_arm_grasp(depth_portion=grasp_params[2],
                                                   height_portion=grasp_params[1],
                                                   theta=grasp_params[0],
                                                   obj=self.target_obj,
                                                   robot=self.robot)
        grasp_config, grasp = grasp_utils.solveIKs(self.problem_env.env, self.robot, grasps)

        param = {'q_goal': np.hstack([grasp_config, pick_base_pose]),
                 'grasp_params': grasp_params,
                 'g_config': grasp_config,
                 'action_parameters': op_parameters}

        if grasp_config is None:
            return None, 'InfeasibleIK'
        else:
            return param, 'HasSolution'

    def sample_place_cont_parameters(self, pick_params):
        obj_place_pose = self.place_sampler.sample()
        self.place_op.continuous_parameters['grasp_params'] = pick_params['grasp_params']
        cont_params, status = self.place_feasibility_checker.check_feasibility(self.place_op, obj_place_pose)
        if status != 'HasSolution':
            return None, status
        else:
            return cont_params, status

    def sample_from_continuous_space(self):
        assert len(self.robot.GetGrabbed()) == 0

        # sample pick
        pick_cont_params = None
        n_ik_attempts = 0
        n_base_attempts = 0
        status = "NoSolution"
        robot_pose = utils.get_body_xytheta(self.robot)
        robot_config = self.robot.GetDOFValues()

        # sample pick parameters
        while pick_cont_params is None:
            pick_cont_params, status = self.sample_pick_cont_parameters()
            if status == 'InfeasibleBase':
                n_base_attempts += 1
            elif status == 'InfeasibleIK':
                n_ik_attempts += 1
            elif status == 'HasSolution':
                n_ik_attempts += 1
                break
            if n_ik_attempts == 1 or n_base_attempts == 4:
                break
        if status != 'HasSolution':
            utils.set_robot_config(robot_pose)
            return None, None, status

        self.pick_op.continuous_parameters = pick_cont_params
        self.pick_op.execute()

        # sample place
        n_ik_attempts = 0
        n_base_attempts = 0
        status = "NoSolution"
        place_cont_params = None
        while place_cont_params is None:
            place_cont_params, status = self.sample_place_cont_parameters(pick_cont_params)
            if status == 'InfeasibleBase':
                n_base_attempts += 1
            elif status == 'InfeasibleIK':
                n_ik_attempts += 1
            elif status == 'HasSolution':
                n_ik_attempts += 1
                break
            if n_ik_attempts == 1 or n_base_attempts == 1:
                break

        # reverting back to the state before sampling
        utils.one_arm_place_object(pick_cont_params)
        self.robot.SetDOFValues(robot_config)
        utils.set_robot_config(robot_pose)

        if status != 'HasSolution':
            return None, None, status
        else:
            self.place_op.continuous_parameters = place_cont_params
            return pick_cont_params, place_cont_params, status

