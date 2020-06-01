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
        self.key_configs = None

    def get_konf_obstacles(self, obj_pose_and_obj_name_to_prm_indices_in_collision):
        prm_indices_in_collision = list(set.union(*obj_pose_and_obj_name_to_prm_indices_in_collision.values()))
        key_configs = self.key_configs
        collision_vector = np.zeros((len(key_configs)))
        collision_vector[prm_indices_in_collision] = 1
        collision_vector = utils.convert_binary_vec_to_one_hot(collision_vector)
        collision_vector = collision_vector.reshape((1, len(collision_vector), 2, 1))
        return collision_vector


class TwoArmConcreteNodeState(ConcreteNodeState):
    def __init__(self, abstract_state, abstract_action):
        ConcreteNodeState.__init__(self, abstract_state, abstract_action)
        self.key_configs = self.abstract_state.prm_vertices
        self.pick_collision_vector = self.get_konf_obstacles(abstract_state.current_collides)
        self.place_collision_vector = None
        self.abs_robot_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.robot))
        self.abs_obj_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.env.GetKinBody(self.obj)))
        self.abs_goal_obj_poses = [np.array([0, 0, 0])]


class OneArmConcreteNodeState(ConcreteNodeState):
    def __init__(self, abstract_state, abstract_action, key_configs, parent_state=None):
        ConcreteNodeState.__init__(self, abstract_state, abstract_action)
        self.parent_state = parent_state
        self.key_configs = key_configs
        self.konf_collisions_with_obj_poses = self.get_object_pose_collisions()
        self.pick_collision_vector = self.get_konf_obstacles(self.konf_collisions_with_obj_poses)
        self.place_collision_vector = None

        # following three variables are used in get_processed_poses_from_state
        self.abs_robot_pose = utils.get_rightarm_torso_config(self.problem_env.robot)
        self.abs_obj_pose = utils.clean_pose_data(utils.get_body_xytheta(self.problem_env.env.GetKinBody(self.obj)))
        self.abs_goal_obj_poses = [np.array([0, 0, 0])]

    def get_object_pose_collisions(self):
        def in_collision_with_obj(q, obj):
            utils.set_rightarm_torso(q, self.problem_env.robot)
            col = self.problem_env.env.CheckCollision(self.problem_env.robot, obj)
            return col

        obj_name_to_pose = {
            obj.GetName(): tuple(utils.get_body_xytheta(obj)[0].round(6))
            for obj in self.problem_env.objects
        }
        parent_collides = None if self.parent_state is None else self.parent_state.konf_collisions_with_obj_poses
        collisions_at_all_obj_pose_pairs = {}
        for obj in self.problem_env.objects:
            obj_name_pose_tuple = (obj.GetName(), obj_name_to_pose[obj.GetName()])
            collisions_with_obj_did_not_change = parent_collides is not None and obj_name_pose_tuple in parent_collides
            if collisions_with_obj_did_not_change:
                konfs_in_collision_with_obj = parent_collides[obj_name_pose_tuple]
            else:
                konfs_in_collision_with_obj = {i for i, q in enumerate(self.key_configs) if
                                               in_collision_with_obj(q, obj)}
            collisions_at_all_obj_pose_pairs[obj_name_pose_tuple] = konfs_in_collision_with_obj
        return collisions_at_all_obj_pose_pairs
