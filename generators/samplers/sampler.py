import numpy as np
import time

from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils
from gtamp_utils.utils import get_pick_domain, get_place_domain, get_pick_base_pose_and_grasp_from_pick_parameters

import torch


class Sampler:
    def __init__(self, policy, target_region):
        self.policies = policy
        pick_min = get_pick_domain()[0]
        pick_max = get_pick_domain()[1]
        place_min = get_place_domain(target_region)[0]
        place_max = get_place_domain(target_region)[1]
        mins = np.hstack([pick_min, place_min])
        maxes = np.hstack([pick_max, place_max])
        self.domain = np.vstack([mins, maxes])


class LearnedSampler(Sampler):
    def __init__(self, sampler, abstract_state, abstract_action):
        Sampler.__init__(self, sampler, abstract_action.discrete_parameters['place_region'])
        self.key_configs = abstract_state.prm_vertices
        self.abstract_state = abstract_state
        self.obj = abstract_action.discrete_parameters['object']
        self.region = abstract_action.discrete_parameters['place_region']

        self.smpler_state = ConcreteNodeState(abstract_state, abstract_action)
        self.n_smpl_per_iter = 2000

    def sample_new_points(self, n_smpls):
        # Here, it would be much more accurate if I use place collision vector, but at this point
        # I don't know if the pick is feasible. Presumably, we can check the feasbility based on pick first, and
        # only if that is feasible, move onto a place. But this gets ugly as to how to "count" the number of samples
        # tried. I guess if we count the pick trials, it is same as now?
        raise NotImplementedError

    def sample(self):
        # prepare input to the network
        if self.curr_smpl_idx >= len(self.samples):
            self.samples = self.sample_new_points(self.n_smpl_per_iter)
            self.curr_smpl_idx = 0
        new_sample = self.samples[self.curr_smpl_idx]
        self.curr_smpl_idx += 1
        return new_sample

    def decode_base_angle_encoding(self, pick_samples):
        base_angles = pick_samples[:, 4:6]
        base_angles = [utils.decode_sin_and_cos_to_angle(base_angle) for base_angle in base_angles]
        pick_samples[:, 4] = base_angles
        pick_samples = np.delete(pick_samples, 5, 1)  # remove unnecessary 2-dim encoded base angle
        return pick_samples


def compute_v_manip(abs_state, goal_objs):
    goal_objs_not_in_goal = [goal_obj for goal_obj in goal_objs if
                             not abs_state.binary_edges[(goal_obj, 'home_region')][0]]
    v_manip = np.zeros((len(abs_state.prm_vertices), 1))
    # todo optimize this code
    init_end_times = 0
    path_times = 0
    stime = time.time()
    for goal_obj in goal_objs_not_in_goal:
        prm_path = abs_state.cached_place_paths[(goal_obj, 'home_region')]
        stime2 = time.time()
        # todo possible optimization:
        #   I can use the prm_path[1]'s edge to compute the neighbors, instead of using all of them.
        distances = [utils.base_pose_distance(prm_path[0], prm_vtx) for prm_vtx in abs_state.prm_vertices]
        init_end_times += time.time() - stime2
        closest_prm_idx = np.argmin(distances)
        prm_path.pop(0)
        prm_path.insert(0, abs_state.prm_vertices[closest_prm_idx, :])

        stime2 = time.time()
        distances = [utils.base_pose_distance(prm_path[-1], prm_vtx) for prm_vtx in abs_state.prm_vertices]
        init_end_times += time.time() - stime2
        closest_prm_idx = np.argmin(distances)
        prm_path[-1] = abs_state.prm_vertices[closest_prm_idx, :]

        stime2 = time.time()
        for p in prm_path:
            boolean_matching_prm_vertices = np.all(np.isclose(abs_state.prm_vertices[:, :2], p[:2]), axis=-1)
            if np.any(boolean_matching_prm_vertices):
                idx = np.argmax(boolean_matching_prm_vertices)
                v_manip[idx] = 1
        path_times += time.time() - stime2
    print 'v_manip creation time', time.time() - stime
    return v_manip

class PlaceOnlyLearnedSampler(LearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        LearnedSampler.__init__(self, sampler, abstract_state, abstract_action)
        if pick_abs_base_pose is not None:
            self.pick_abs_base_pose = pick_abs_base_pose.reshape((1, 3))
        else:
            self.pick_abs_base_pose = None
        self.v_manip = None
        self.samples = self.sample_new_points(self.n_smpl_per_iter)
        self.curr_smpl_idx = 0

    def sample_picks(self, n_smpls):
        pick_min = get_pick_domain()[0]
        pick_max = get_pick_domain()[1]
        pick_samples = np.random.uniform(pick_min, pick_max, (n_smpls, 6)).squeeze()
        return pick_samples

    def sample_placements(self, pose_ids, collisions, pick_samples, n_smpls):
        stttt = time.time()
        obj_kinbody = self.abstract_state.problem_env.env.GetKinBody(self.obj)

        stime = time.time()
        obj_xyth = utils.get_body_xytheta(obj_kinbody)
        # print 'objxytheta time',time.time()-stime
        stime = time.time()
        pick_abs_poses = []
        for s in pick_samples:
            _, poses = utils.get_pick_base_pose_and_grasp_from_pick_parameters(obj_kinbody, s, obj_xyth)
            pick_abs_poses.append(poses)
        # print "Pick abs pose time", time.time()-stime

        stime = time.time()
        encoded_pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])
        # print "Pick pose encoding time", time.time() - stime

        pose_ids[:, -6:-2] = encoded_pick_abs_poses
        if self.v_manip is None:
            v_manip = compute_v_manip(self.abstract_state, self.abstract_state.goal_entities[:-1])
            stime = time.time()
            v_manip = utils.convert_binary_vec_to_one_hot(v_manip.squeeze()).reshape((1, 618, 2, 1))
            # print 'vmanip conversion time', time.time()-stime
            v_manip = np.tile(v_manip, (n_smpls, 1, 1, 1))

            self.v_manip = v_manip

        stime = time.time()
        state_vec = np.concatenate([collisions, self.v_manip], axis=2)
        # print 'concat time', time.time()-stime

        if 'home' in self.region:
            chosen_sampler = self.policies['place_home']
        else:
            chosen_sampler = self.policies['place_loading']

        stime = time.time()
        place_samples = chosen_sampler.generate(state_vec, pose_ids)
        # print "prediction time", time.time()-stime

        stime = time.time()
        place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])
        # print "place decoding time", time.time()-stime
        return place_samples

    def sample_new_points(self, n_smpls):
        # note: this function outputs absolute pick base pose and absolute place base pose
        print "Generating new points"
        stime = time.time()

        pick_samples = self.sample_picks(n_smpls)

        collisions = self.smpler_state.pick_collision_vector
        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]
        poses = np.tile(poses, (n_smpls, 1))
        if 'rectangular' in self.obj:
            object_id = [1, 0]
        else:
            object_id = [0, 1]
        object_id = np.tile(np.array(object_id)[None, :], (n_smpls, 1))
        pose_ids = np.hstack([poses, object_id])

        place_samples = self.sample_placements(pose_ids, collisions, pick_samples, n_smpls)
        samples = np.hstack([pick_samples, place_samples])
        print 'place prediction time', time.time() - stime
        return samples


class PickOnlyLearnedSampler(LearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        LearnedSampler.__init__(self, sampler, abstract_state, abstract_action)
        self.samples = self.sample_new_points(2000)
        self.curr_smpl_idx = 0

    def sample_picks(self, poses, collisions):
        pick_samples = self.policies['pick'].generate(collisions, poses)
        pick_samples = self.decode_base_angle_encoding(pick_samples)
        return pick_samples

    def sample_placements(self, n_smpls):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        pickplace_smpls = np.random.uniform(domain_min, domain_max, (n_smpls, dim_parameters)).squeeze()
        place_smpls = pickplace_smpls[:, -3:]
        return place_smpls

    def sample_new_points(self, n_smpls):
        print "Generating new points"
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]
        poses = np.tile(poses, (n_smpls, 1))
        if 'rectangular' in self.obj:
            object_id = [1, 0]
        else:
            object_id = [0, 1]
        object_id = np.tile(np.array(object_id)[None, :], (n_smpls, 1))
        poses = np.hstack([poses, object_id])
        collisions = self.smpler_state.pick_collision_vector
        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))

        pick_samples = self.sample_picks(poses, collisions)
        place_samples = self.sample_placements(n_smpls)
        return np.hstack([pick_samples, place_samples])
