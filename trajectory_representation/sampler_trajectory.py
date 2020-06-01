from gtamp_problem_environments.mover_env import Mover
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from trajectory_representation.one_arm_pap_state import OneArmPaPState


import numpy as np
import random


def get_pick_base_poses(action, smples):
    pick_base_poses = []
    for smpl in smples:
        smpl = smpl[0:4]
        sin_cos_encoding = smpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        smpl = np.hstack([smpl[0:2], decoded_angle])
        abs_base_pose = utils.get_absolute_pick_base_pose_from_ir_parameters(smpl, action.discrete_parameters['object'])
        pick_base_poses.append(abs_base_pose)
    return pick_base_poses


def get_place_base_poses(action, smples, mover):
    place_base_poses = smples[:, 4:]
    to_return = []
    for bsmpl in place_base_poses:
        sin_cos_encoding = bsmpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        bsmpl = np.hstack([bsmpl[0:2], decoded_angle])
        to_return.append(bsmpl)
    to_return = np.array(to_return)
    to_return[:, 0:2] += mover.regions[action.discrete_parameters['region']].box[0]
    return to_return


class SamplerTrajectory:
    def __init__(self, problem_idx, n_objs_pack):
        self.problem_idx = problem_idx
        self.paps_used = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.hvalues = []
        self.hcounts = []
        self.n_in_way = []
        self.num_in_goal = []
        self.num_papable_to_goal = []
        self.prev_n_in_way = []
        self.state_prime = []
        self.v_manip_goal = []
        self.prev_v_manip_goal = []
        self.goal_objs_not_in_goal = []
        self.seed = None  # this defines the initial state
        self.problem_env = None
        self.n_objs_pack = n_objs_pack

    def add_state_prime(self):
        self.state_prime = self.states[1:]

    def add_sah_tuples(self, s, a, prev_n_in_way, n_in_way, prev_v_manip, v_manip):
        self.states.append(s)
        self.actions.append(a)
        self.prev_n_in_way.append(prev_n_in_way)
        self.n_in_way.append(n_in_way)
        self.prev_v_manip_goal.append(prev_v_manip)
        self.v_manip_goal.append(v_manip)

    def create_environment(self):
        problem_env = Mover(self.problem_idx)
        openrave_env = problem_env.env
        return problem_env, openrave_env

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    def add_trajectory(self, plan):
        raise NotImplementedError

    @staticmethod
    def get_action_info(action):
        # Information regarding actions
        pick_action_info = action.continuous_parameters['pick']
        place_action_info = action.continuous_parameters['place']

        pick_parameters = pick_action_info['action_parameters']
        place_obj_abs_pose = utils.clean_pose_data(place_action_info['object_pose'])
        if 'one_arm' in  action.type:
            pick_base_pose = utils.clean_pose_data(pick_action_info['q_goal'][-3:])
            place_base_pose = utils.clean_pose_data(place_action_info['q_goal'][-3:])
            pick_motion = None
            place_motion = None
        else:
            pick_base_pose = utils.clean_pose_data(pick_action_info['q_goal'])
            place_base_pose = utils.clean_pose_data(place_action_info['q_goal'])
            pick_motion = [utils.clean_pose_data(q) for q in pick_action_info['motion']]
            place_motion = [utils.clean_pose_data(q) for q in place_action_info['motion']]
        action_info = {
            'object_name': action.discrete_parameters['object'],
            'region_name': action.discrete_parameters['place_region'],
            'pick_action_parameters': pick_parameters,
            'pick_abs_base_pose': pick_base_pose,
            'place_abs_base_pose': place_base_pose,
            'place_obj_abs_pose': place_obj_abs_pose,
            'pick_motion': pick_motion,
            'place_motion': place_motion
        }
        return action_info

    def compute_n_in_way_for_object_moved(self, object_moved, abs_state, goal_objs):
        raise NotImplementedError

