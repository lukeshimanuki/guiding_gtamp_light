from uniform import PaPUniformGenerator
from generators.learning.utils.data_processing_utils import action_data_mode
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils

from gtamp_utils import utils
from generators.learning.utils.sampler_utils import generate_pick_and_place_batch
from generators.learning.utils.sampler_utils import unprocess_pick_and_place_smpls
from generators.learning.PlacePolicyIMLE import uniform_noise

import time
import numpy as np
import pickle


class LearnedGenerator(PaPUniformGenerator):
    def __init__(self, operator_skeleton, problem_env, sampler, abstract_state, max_n_iter,
                 swept_volume_constraint=None):
        PaPUniformGenerator.__init__(self, operator_skeleton, problem_env, max_n_iter, swept_volume_constraint)
        self.feasible_pick_params = {}
        self.sampler = sampler
        self.abstract_state = abstract_state
        self.obj = operator_skeleton.discrete_parameters['object']
        self.region = operator_skeleton.discrete_parameters['place_region']

        goal_entities = self.abstract_state.goal_entities
        key_config_obstacles = self.process_abstract_state_collisions_into_key_config_obstacles(abstract_state)
        self.key_configs = np.delete(abstract_state.prm_vertices, [415, 586, 615, 618, 619], axis=0)
        self.smpler_state = ConcreteNodeState(self.problem_env, self.obj, self.region,
                                              goal_entities,
                                              key_configs=self.key_configs,
                                              collision_vector=key_config_obstacles)
        self.noises_used = []
        self.tried_smpls = []

        # to do generate 1000 smpls here
        n_total_iters = sum(range(10, self.max_n_iter, 10))

        z_smpls = uniform_noise(z_size=(500, 7))
        stime=time.time()
        smpls = generate_pick_and_place_batch(self.smpler_state, self.sampler, z_smpls)
        self.policy_smpl_batch = unprocess_pick_and_place_smpls(smpls)
        print "Prediction time", time.time() - stime

        """
        utils.viewer()
        orig_color = utils.get_color_of(self.obj)
        utils.set_color(self.obj, [1, 0, 0])
        utils.visualize_placements(self.policy_smpl_batch[:, -3:], self.obj)
        utils.set_color(self.obj, orig_color)
        import pdb;pdb.set_trace()
        """
        self.policy_smpl_idx = 0

    def process_abstract_state_collisions_into_key_config_obstacles(self, abstract_state):
        # todo prevent collision with the target object
        n_vtxs = len(abstract_state.prm_vertices)
        collision_vector = np.zeros((n_vtxs))
        colliding_vtx_idxs = self.get_colliding_prm_idxs(abstract_state)
        collision_vector[colliding_vtx_idxs] = 1
        collision_vector = np.delete(collision_vector, [415, 586, 615, 618, 619], axis=0)

        """
        key_configs = pickle.load(open('prm.pkl', 'r'))[0]
        key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
        self.problem_env.env.GetKinBody(self.obj).Enable(False)
        collision_vector2 = utils.compute_occ_vec(key_configs)
        self.problem_env.env.GetKinBody(self.obj).Enable(True)

        assert np.all(collision_vector == collision_vector2)
        """

        return collision_vector

    def get_colliding_prm_idxs(self, abstract_state):
        colliding_vtx_idxs = []
        target_obj_name = self.problem_env.env.GetKinBody(self.obj).GetName()
        for obj_name_pose_pair in abstract_state.collides.keys():
            obj_name = obj_name_pose_pair[0]
            assert type(obj_name) == str or type(obj_name) == unicode
            print obj_name, target_obj_name
            if obj_name == target_obj_name:
                continue
            colliding_vtx_idxs.append(abstract_state.collides[obj_name_pose_pair])
        colliding_vtx_idxs = list(set().union(*colliding_vtx_idxs))

        return colliding_vtx_idxs

    def sample_from_learned_samplers(self):
        # Wait..
        # Sample pick first, and only if that is feasible, use place
        # Right now, I am discarding the entire sample
        # But really, I should learn to predict both the pick and place,
        # so that if the place paired with the pick is infeasible, then I can discard both samples
        """
        if action_data_mode == 'pick_parameters_place_relative_to_object':
            place_smpl = self.policy_smpl_batch[self.policy_smpl_idx]
            place_smpl = data_processing_utils.get_absolute_placement_from_relative_placement(place_smpl,
                                                                                              self.smpler_state.abs_obj_pose)
            self.policy_smpl_idx += 1
        elif action_data_mode == 'pick_abs_base_pose_place_abs_obj_pose':
            place_smpl = self.policy_smpl_batch[self.policy_smpl_idx]
            self.policy_smpl_idx += 1
            if self.policy_smpl_idx >= len(self.policy_smpl_batch):
                z_smpls = uniform_noise(z_size=(500, 4))
                stime = time.time()
                self.policy_smpl_batch = generate_smpl_batch(self.smpler_state, self.sampler, z_smpls, self.key_configs)
                print "Prediction time for further sampling", time.time() - stime
                self.policy_smpl_idx = 0
        else:
            raise NotImplementedError
        """
        smpl = self.policy_smpl_batch[self.policy_smpl_idx]
        self.policy_smpl_idx += 1
        if self.policy_smpl_idx >= len(self.policy_smpl_batch):
            z_smpls = uniform_noise(z_size=(500, 7))
            stime = time.time()
            smpls = generate_pick_and_place_batch(self.smpler_state, self.sampler, z_smpls)
            self.policy_smpl_batch = unprocess_pick_and_place_smpls(smpls)
            print "Prediction time for further sampling", time.time() - stime
            self.policy_smpl_idx = 0
        self.tried_smpls.append(smpl)
        #parameters = self.sample_from_uniform()
        #parameters[6:] = place_smpl
        return smpl

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        obj = operator_skeleton.discrete_parameters['object']

        orig_color = utils.get_color_of(obj)
        # utils.set_color(obj, [1, 0, 0])
        # utils.viewer()
        for i in range(n_iter):
            stime = time.time()
            op_parameters = self.sample_from_learned_samplers()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton, op_parameters,
                                                                                  self.swept_volume_constraint,
                                                                                  parameter_mode='obj_pose')
            smpling_time = time.time() - stime
            self.smpling_time.append(smpling_time)
            if status == 'HasSolution':
                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= n_parameters_to_try_motion_planning:
                    break

        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            # import pdb; pdb.set_trace()
            status = "HasSolution"

        utils.set_color(obj, orig_color)
        return feasible_op_parameters, status
