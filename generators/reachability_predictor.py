from gtamp_utils import utils
import numpy as np
import torch


def compute_relative_config(src_config, end_config):
    src_config = np.array(src_config)
    end_config = np.array(end_config)

    assert len(src_config.shape) == 2, \
        'Put configs in shapes (n_config,dim_config)'

    rel_config = end_config - src_config
    neg_idxs_to_fix = rel_config[:, -1] < -np.pi
    pos_idxs_to_fix = rel_config[:, -1] > np.pi

    # making unique rel angles; keep the range to [-pi,pi]
    rel_config[neg_idxs_to_fix, -1] = rel_config[neg_idxs_to_fix, -1] + 2 * np.pi
    rel_config[pos_idxs_to_fix, -1] = rel_config[pos_idxs_to_fix, -1] - 2 * np.pi

    return rel_config


class ReachabilityPredictor:
    def __init__(self, pick_net, place_net):
        self.pick_net = pick_net
        self.place_net = place_net

    def make_vertices(self, qg, key_configs, collisions):
        q0 = utils.get_robot_xytheta().squeeze()
        if 'Relative' in self.pick_net.__class__.__name__:
            rel_qg = compute_relative_config(q0[None, :], qg[None, :])
            rel_qk = compute_relative_config(q0[None, :], key_configs)
            repeat_qg = np.repeat(np.array(rel_qg), 618, axis=0)
            v = np.hstack([rel_qk, repeat_qg, collisions])
        else:
            repeat_q0 = np.repeat(np.array(q0)[None, :], 618, axis=0)
            repeat_qg = np.repeat(np.array(qg)[None, :], 618, axis=0)
            v = np.hstack([key_configs, repeat_q0, repeat_qg, collisions])

        v = v[None, :]
        v = torch.from_numpy(v).float()
        return v

    def predict(self, op_parameters, abstract_state, abstract_action):
        collisions = self.process_abstract_state_collisions_into_key_config_obstacles(abstract_state)
        key_configs = abstract_state.prm_vertices

        pick_qg = op_parameters['pick']['q_goal']
        v = self.make_vertices(pick_qg, key_configs, collisions)
        is_pick_reachable = ((self.pick_net(v) > 0.5).cpu().numpy() == True)[0, 0]

        if is_pick_reachable:
            # I have to hold the object and check collisions
            orig_config = utils.get_robot_xytheta()
            target_obj = abstract_action.discrete_parameters['object']
            utils.two_arm_pick_object(target_obj, {'q_goal': pick_qg})
            collisions = utils.compute_occ_vec(key_configs)
            collisions = utils.convert_binary_vec_to_one_hot(collisions)
            utils.two_arm_place_object({'q_goal': pick_qg})
            utils.set_robot_config(orig_config)
            place_qg = op_parameters['place']['q_goal']
            v = self.make_vertices(place_qg, key_configs, collisions)
            pred = self.place_net(v)
            is_place_reachable = ((pred > 0.5).cpu().numpy() == True)[0, 0]
            return is_place_reachable
        else:
            return False

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
