from uniform import PaPUniformGenerator
from generators.learning.utils.data_processing_utils import action_data_mode
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils

from gtamp_utils import utils

from generators.learning.utils.sampler_utils import generate_pick_and_place_batch, prepare_input_except_noise
from generators.learning.utils.sampler_utils import unprocess_pick_and_place_smpls, get_indices_to_delete
from generators.learning.PlacePolicyIMLE import uniform_noise
from generators.learning.PlacePolicyIMLE import gaussian_noise

import time
import numpy as np
import copy
import pickle

# noise = uniform_noise
noise = gaussian_noise


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
        # todo There is a mismatch between the true key configs and the key config obstacles from the abstract state.
        #   I will fix this error later. For now, check the number of nodes explored
        self.smpler_state = ConcreteNodeState(self.problem_env, self.obj, self.region,
                                              goal_entities,
                                              key_configs=self.key_configs)

        self.noises_used = []
        self.tried_smpls = []

        """
        self.pick_input = prepare_input_except_noise(self.smpler_state, delete=True, region='loading_region',
                                                     filter_konfs=False)
        self.place_input = prepare_input_except_noise(self.smpler_state, delete=True,
                                                      region=operator_skeleton.discrete_parameters['place_region'],
                                                      filter_konfs=False)
        """

        # to do generate 1000 smpls here
        z_smpls = noise(z_size=(201, 7))
        stime = time.time()
        smpls = generate_pick_and_place_batch(self.smpler_state, self.sampler, z_smpls)
        self.policy_smpl_batch = unprocess_pick_and_place_smpls(smpls)
        print "Prediction time", time.time() - stime
        """
        orig_color = utils.get_color_of(self.obj)
        utils.set_color(self.obj, [0, 1, 0])
        utils.visualize_placements(self.policy_smpl_batch[0:100, -3:], self.obj)
        utils.set_color(self.obj, orig_color)
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
        smpl = self.policy_smpl_batch[self.policy_smpl_idx]
        self.policy_smpl_idx += 1
        if self.policy_smpl_idx >= len(self.policy_smpl_batch):
            z_smpls = noise(z_size=(100, 7))
            stime = time.time()
            smpls = generate_pick_and_place_batch(self.smpler_state, self.sampler, z_smpls)
            self.policy_smpl_batch = unprocess_pick_and_place_smpls(smpls)
            #print "Prediction time for further sampling", time.time() - stime
            self.policy_smpl_idx = 0
        self.tried_smpls.append(smpl)
        # parameters = self.sample_from_uniform()
        # parameters[6:] = place_smpl
        return smpl

    def sample_from_pick_sampler(self):
        pick_noise = noise(z_size=(1, 7))
        # todo fix the pick_input delete option
        inp = [self.pick_input['goal_flags'], self.pick_input['key_configs'],
               self.pick_input['collisions'], self.pick_input['poses'],
               pick_noise]
        pick_pred = self.sampler['pick'].policy_model.predict(inp).squeeze()
        ir_parameters = pick_pred[3:]
        portion = ir_parameters[0]
        base_angle = utils.decode_sin_and_cos_to_angle(ir_parameters[1:3])
        facing_angle_offset = ir_parameters[3]
        pick_param = np.hstack([pick_pred[:3], portion, base_angle, facing_angle_offset])
        return pick_param

    def sample_from_place_sampler(self, operator_skeleton, pick_base_pose):
        place_noise = noise(z_size=(1, 4))

        # setup the poses from pick
        poses = self.place_input['poses']
        pick_base_pose = utils.encode_pose_with_sin_and_cos_angle(pick_base_pose)
        poses[:, -4:] = pick_base_pose
        self.place_input['poses'] = poses

        inp = [self.place_input['goal_flags'], self.place_input['key_configs'], self.place_input['collisions'], self.place_input['poses'], place_noise]
        place_pred = self.sampler['place'].policy_model.predict(inp)
        place_pred = utils.decode_pose_with_sin_and_cos_angle(place_pred)
        return place_pred

    def sample_feasible_op_parameters(self, operator_skeleton, n_iter, n_parameters_to_try_motion_planning):
        assert n_iter > 0
        feasible_op_parameters = []
        obj = operator_skeleton.discrete_parameters['object']

        orig_color = utils.get_color_of(obj)
        feasibility_check_time = 0
        stime = time.time()
        for i in range(n_iter):
            op_parameters = self.sample_from_learned_samplers()
            stime2 = time.time()
            op_parameters, status = self.op_feasibility_checker.check_feasibility(operator_skeleton,
                                                                                  op_parameters,
                                                                                  self.swept_volume_constraint,
                                                                                  parameter_mode='obj_pose')
            feasibility_check_time += time.time()-stime2

            if status == 'HasSolution':
                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= n_parameters_to_try_motion_planning:
                    break
            """

            pick_op_parameters = self.sample_from_pick_sampler()
            pick_parameters, pick_status = self.op_feasibility_checker.check_pick_feasible(pick_op_parameters,
                                                                                           operator_skeleton)
            if pick_status != 'HasSolution':
                continue

            place_op_parameters = self.sample_from_place_sampler(operator_skeleton, pick_parameters['q_goal'])
            place_parameters, place_status = self.op_feasibility_checker.check_place_feasible(pick_parameters,
                                                                                              place_op_parameters,
                                                                                              operator_skeleton,
                                                                                              parameter_mode='obj_pose')
            if place_status != 'HasSolution':
                continue

            status = "HasSolution"
            op_parameters = {'pick': pick_parameters, 'place': place_parameters}
            if status == 'HasSolution':
                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= n_parameters_to_try_motion_planning:
                    #break
                    pass
            """

        smpling_time = time.time() - stime
        print "Total sampling time", smpling_time
        print "Feasibilty time", feasibility_check_time

        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        utils.set_color(obj, orig_color)
        return feasible_op_parameters, status
