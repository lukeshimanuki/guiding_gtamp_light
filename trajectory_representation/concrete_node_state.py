from gtamp_utils import utils
import numpy as np
import pickle
import time


class ConcreteNodeState:
    def __init__(self, abstract_state, abstract_action):
        self.abstract_state = abstract_state
        self.abstract_action = abstract_action
        self.obj = abstract_action.discrete_parameters['object']
        self.region = abstract_action.discrete_parameters['place_region']
        self.goal_entities = abstract_state.goal_entities
        self.problem_env = abstract_state.problem_env


class TwoArmConcreteNodeState(ConcreteNodeState):
    def __init__(self, abstract_state, abstract_action):
        ConcreteNodeState.__init__(self, abstract_state, abstract_action)
        self.pick_collision_vector = self.convert_collision_at_prm_indices_to_col_vec(abstract_state.current_collides)
        self.place_collision_vector = None
        self.abs_robot_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.robot))
        self.abs_obj_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.env.GetKinBody(self.obj)))
        self.abs_goal_obj_poses = [np.array([0, 0, 0])]

    def convert_collision_at_prm_indices_to_col_vec(self, obj_pose_and_obj_name_to_prm_indices_in_collision):
        prm_indices_in_collision = list(set.union(*obj_pose_and_obj_name_to_prm_indices_in_collision.values()))
        key_configs = self.abstract_state.prm_vertices
        collision_vector = np.zeros((len(key_configs)))
        collision_vector[prm_indices_in_collision] = 1
        collision_vector = utils.convert_binary_vec_to_one_hot(collision_vector)
        collision_vector = collision_vector.reshape((1, len(collision_vector), 2, 1))
        return collision_vector


class OneArmConcreteNodeState(ConcreteNodeState):
    def __init__(self, abstract_state, abstract_action, key_configs):
        ConcreteNodeState.__init__(self, abstract_state, abstract_action)
        self.key_configs = key_configs
        self.pick_collision_vector = self.get_konf_obstacles()
        self.place_collision_vector = None

        # following three variables are used in get_processed_poses_from_state
        self.abs_robot_pose = utils.get_rightarm_torso_config(self.problem_env.robot)
        self.abs_obj_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.env.GetKinBody(self.obj)))
        self.abs_goal_obj_poses = [np.array([0, 0, 0])]

    def get_konf_obstacles(self):
        # todo implement this guy
        return -1
