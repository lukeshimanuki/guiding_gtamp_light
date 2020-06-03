from sampler import LearnedSampler

import numpy as np
import time
from gtamp_utils import utils
from generators.learning.utils import data_processing_utils


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


class PickPlaceLearnedSampler(LearnedSampler):
    def __init__(self, atype, sampler, abstract_state, abstract_action):
        LearnedSampler.__init__(self,  atype, sampler, abstract_state, abstract_action)
        self.v_manip = None
        self.state_vec = None
        self.samples = self.sample_new_points(self.n_smpl_per_iter)
        self.curr_smpl_idx = 0

    def sample_picks(self, poses, collisions):
        pick_samples = self.samplers['pick'].generate(collisions, poses)
        pick_samples = self.decode_base_angle_encoding(pick_samples)
        return pick_samples

    def sample_placements(self, pose_ids, collisions, pick_samples, n_smpls):
        stttt = time.time()
        obj_kinbody = self.abstract_state.problem_env.env.GetKinBody(self.obj)

        stime = time.time()
        obj_xyth = utils.get_body_xytheta(obj_kinbody)
        print 'objxytheta time',time.time()-stime
        stime = time.time()
        pick_abs_poses = []
        for s in pick_samples:
            _, poses = utils.get_pick_base_pose_and_grasp_from_pick_parameters(obj_kinbody, s, obj_xyth)
            pick_abs_poses.append(poses)
        print "Pick abs pose time", time.time()-stime

        stime = time.time()
        encoded_pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])
        print "Pick pose encoding time", time.time() - stime

        pose_ids[:, -6:-2] = encoded_pick_abs_poses
        if 'home' in self.region:
            chosen_sampler = self.samplers['place_goal_region']
        else:
            chosen_sampler = self.samplers['place_obj_region']

        stime = time.time()
        place_samples = chosen_sampler.generate(self.state_vec, pose_ids)
        print "prediction time", time.time()-stime

        stime = time.time()
        place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])
        print "place decoding time", time.time()-stime
        # print time.time() - stttt
        return place_samples

    def sample_new_points(self, n_smpls):
        stime11 = time.time()
        stime = time.time()
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]
        if 'rectangular' in self.obj:
            object_id = [1, 0]
        else:
            object_id = [0, 1]
        object_id = np.array(object_id)[None, :]
        pose_ids = np.hstack([poses, object_id])
        pose_ids = np.tile(pose_ids, (n_smpls, 1))
        collisions = self.smpler_state.pick_collision_vector

        if self.state_vec is None:
            stime = time.time()
            v_manip = compute_v_manip(self.abstract_state, self.abstract_state.goal_entities[:-1])
            v_manip = utils.convert_binary_vec_to_one_hot(v_manip.squeeze()).reshape((1, 618, 2, 1))
            self.state_vec = np.concatenate([collisions, v_manip], axis=2)
            self.state_vec = np.tile(self.state_vec, (n_smpls, 1, 1, 1))
            print 'state_vec creation time', time.time()-stime

        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
        print "input processing time", time.time()-stime

        stime = time.time()
        pick_samples = self.sample_picks(pose_ids, collisions)
        print "Pick sampling time", time.time() - stime
        stime = time.time()
        place_samples = self.sample_placements(pose_ids, collisions, pick_samples, n_smpls)
        print "Place sampling time", time.time() - stime
        samples = np.hstack([pick_samples, place_samples])
        print "Total sampling time", time.time() - stime11
        return samples
