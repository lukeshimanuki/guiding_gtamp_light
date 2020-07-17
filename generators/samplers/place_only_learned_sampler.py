from sampler import LearnedSampler
from generators.learning.utils import data_processing_utils
from gtamp_utils.utils import get_pick_domain
import numpy as np
import time
from gtamp_utils import utils
from trajectory_representation.one_arm_sampler_trajectory import compute_v_manip as one_arm_compute_v_manip



def two_arm_compute_v_manip(abs_state, goal_objs):
    goal_objs_not_in_goal = [goal_obj for goal_obj in goal_objs if
                             not abs_state.binary_edges[(goal_obj, 'home_region')][0]]
    v_manip = np.zeros((len(abs_state.prm_vertices), 1))
    # todo optimize this code
    init_end_times = 0
    path_times = 0
    original_config = utils.get_robot_xytheta(abs_state.problem_env.robot)
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
    utils.set_robot_config(original_config, robot=abs_state.problem_env.robot)
    return v_manip


class PlaceOnlyLearnedSampler(LearnedSampler):
    def __init__(self, atype, sampler, abstract_state, abstract_action, smpler_state, pick_sampler):
        LearnedSampler.__init__(self, atype, sampler, abstract_state, abstract_action, smpler_state)
        self.atype = atype
        self.v_manip = None
        self.pick_sampler = pick_sampler
        self.samples = self.sample_new_points(self.n_smpl_per_iter)

    def sample_placements(self, pose_ids, collisions, n_smpls):
        if self.v_manip is None:
            if 'one_arm' in self.atype:
                v_manip = one_arm_compute_v_manip(self.abstract_state, self.abstract_state.goal_entities[:-1], self.key_configs)
            else:
                v_manip = two_arm_compute_v_manip(self.abstract_state, self.abstract_state.goal_entities[:-1])
            v_manip = utils.convert_binary_vec_to_one_hot(v_manip.squeeze()).reshape((1, len(self.key_configs), 2, 1))
            v_manip = np.tile(v_manip, (n_smpls, 1, 1, 1))
            self.v_manip = v_manip
        state_vec = np.concatenate([collisions, self.v_manip], axis=2)

        if 'center_shelf' in self.region:
            chosen_sampler = self.samplers['place_obj_region']
            place_samples = chosen_sampler.generate(state_vec, pose_ids)
            place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])
        elif 'rectangular_packing_box1_region' in self.region:
            chosen_sampler = self.samplers['place_goal_region']
            place_samples = np.array([chosen_sampler.sample() for _ in range(n_smpls)])
        else:
            if 'home' in self.region:
                chosen_sampler = self.samplers['place_goal_region']
            else:
                chosen_sampler = self.samplers['place_obj_region']
            place_samples = chosen_sampler.generate(state_vec, pose_ids)
            place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])
        return place_samples

    def sample_new_points(self, n_smpls):
        # note: this function outputs absolute pick base pose and absolute place base pose
        print "Generating new place points"

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

        ###  Process pose_ids using pick samples
        obj_kinbody = self.abstract_state.problem_env.env.GetKinBody(self.obj)
        obj_xyth = utils.get_body_xytheta(obj_kinbody)
        pick_samples = self.pick_sampler.samples # can we make sure that pick is called first?
        pick_abs_poses = []
        for s in pick_samples:
            _, poses = utils.get_pick_base_pose_and_grasp_from_pick_parameters(obj_kinbody, s, obj_xyth)
            pick_abs_poses.append(poses)
        encoded_pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])
        pose_ids[:, -6:-2] = encoded_pick_abs_poses
        ###

        place_samples = self.sample_placements(pose_ids, collisions, n_smpls)
        return place_samples
