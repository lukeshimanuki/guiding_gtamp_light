from gtamp_utils import utils
import numpy as np
import torch


class ReachabilityPredictor:
    def __init__(self, net):
        self.net = net

    def make_vertices(self, qg, key_configs, collisions):
        q0 = utils.get_robot_xytheta().squeeze()
        repeat_q0 = np.repeat(np.array(q0)[None, :], 618, axis=0)
        repeat_qg = np.repeat(np.array(qg)[None, :], 618, axis=0)
        v = np.hstack([key_configs, repeat_q0, repeat_qg, collisions])
        v = v[None, :]
        v = torch.from_numpy(v).float()
        return v

    def predict(self, qg, abstract_state):
        collisions = self.process_abstract_state_collisions_into_key_config_obstacles(abstract_state)
        key_configs = abstract_state.prm_vertices
        v = self.make_vertices(qg, key_configs, collisions)

        return self.net(v)

    def get_colliding_prm_idxs(self, abstract_state):
        colliding_vtx_idxs = []
        for obj_name_pose_pair in abstract_state.collides.keys():
            obj_name = obj_name_pose_pair[0]
            assert type(obj_name) == str or type(obj_name) == unicode
            colliding_vtx_idxs.append(abstract_state.collides[obj_name_pose_pair])
        colliding_vtx_idxs = list(set().union(*colliding_vtx_idxs))
        return colliding_vtx_idxs

    def process_abstract_state_collisions_into_key_config_obstacles(self, abstract_state):
        n_vtxs = len(abstract_state.prm_vertices)
        collision_vector = np.zeros((n_vtxs))
        colliding_vtx_idxs = self.get_colliding_prm_idxs(abstract_state)
        collision_vector[colliding_vtx_idxs] = 1
        collision_vector = utils.convert_binary_vec_to_one_hot(collision_vector)
        collision_vector = collision_vector.reshape((len(collision_vector), 2))
        return collision_vector
