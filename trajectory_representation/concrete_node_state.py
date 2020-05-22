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

        self.pick_collision_vector = self.convert_collision_at_prm_indices_to_col_vec(abstract_state.current_collides)
        self.place_collision_vector = None
        self.abs_robot_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.robot))
        self.abs_obj_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.env.GetKinBody(self.obj)))
        self.abs_goal_obj_poses = [np.array([0, 0, 0])]
        #self.abs_goal_obj_poses = [utils.clean_pose_data(utils.get_body_xytheta(o))
        #                           for o in self.problem_env.goal_entities if 'region' not in o]

    def convert_collision_at_prm_indices_to_col_vec(self, obj_pose_and_obj_name_to_prm_indices_in_collision):
        prm_indices_in_collision = list(set.union(*obj_pose_and_obj_name_to_prm_indices_in_collision.values()))
        key_configs = self.abstract_state.prm_vertices
        collision_vector = np.zeros((len(key_configs)))
        collision_vector[prm_indices_in_collision] = 1
        collision_vector = utils.convert_binary_vec_to_one_hot(collision_vector)

        """
        # testing if the conversion is working correctly
        self.problem_env.set_robot_to_default_dof_values()
        collision_vector2 = utils.compute_occ_vec(key_configs)
        collision_vector2 = utils.convert_binary_vec_to_one_hot(collision_vector2)
        print np.sum(np.sum(collision_vector2 == collision_vector,axis=-1)==0)
        import pdb;pdb.set_trace()
        """
        collision_vector = collision_vector.reshape((1, len(collision_vector), 2, 1))
        return collision_vector
