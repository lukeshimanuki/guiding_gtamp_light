import numpy as np
import time

from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils
from gtamp_utils.utils import get_pick_domain, get_place_domain


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

        goal_entities = self.abstract_state.goal_entities
        stime = time.time()
        self.smpler_state = ConcreteNodeState(abstract_state.problem_env, self.obj, self.region, goal_entities,
                                              key_configs=self.key_configs)
        print "Concre node creation time", time.time() - stime

    def sample_new_points(self, n_smpls):
        # Here, it would be much more accurate if I use place collision vector, but at this point
        # I don't know if the pick is feasible. Presumably, we can check the feasbility based on pick first, and
        # only if that is feasible, move onto a place. But this gets ugly as to how to "count" the number of samples
        # tried. I guess if we count the pick trials, it is same as now?
        raise NotImplementedError

    def sample(self):
        # prepare input to the network
        if self.curr_smpl_idx >= len(self.samples):
            self.samples = self.sample_new_points(200)
            self.curr_smpl_idx = 0
        new_sample = self.samples[self.curr_smpl_idx]
        self.curr_smpl_idx += 1
        return new_sample


def compute_v_manip(abs_state, goal_objs):
    goal_objs_not_in_goal = [goal_obj for goal_obj in goal_objs if
                             not abs_state.binary_edges[(goal_obj, 'home_region')][0]]
    v_manip_values = []
    v_manip = np.zeros((len(abs_state.prm_vertices), 1))
    for goal_obj in goal_objs_not_in_goal:
        prm_path = abs_state.cached_place_paths[(goal_obj, 'home_region')]
        distances = [utils.base_pose_distance(prm_path[0], prm_vtx) for prm_vtx in abs_state.prm_vertices]
        closest_prm_idx = np.argmin(distances)
        prm_path.pop(0)
        prm_path.insert(0, abs_state.prm_vertices[closest_prm_idx, :])

        distances = [utils.base_pose_distance(prm_path[-1], prm_vtx) for prm_vtx in abs_state.prm_vertices]
        closest_prm_idx = np.argmin(distances)
        prm_path[-1] = abs_state.prm_vertices[closest_prm_idx, :]

        v_manip_values.append(prm_path)
        for p in prm_path:
            boolean_matching_prm_vertices = np.all(np.isclose(abs_state.prm_vertices[:, :2], p[:2]), axis=-1)
            if np.any(boolean_matching_prm_vertices):
                idx = np.argmax(boolean_matching_prm_vertices)
                v_manip[idx] = 1
    return v_manip


class PlaceOnlyLearnedSampler(LearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        LearnedSampler.__init__(self, sampler, abstract_state, abstract_action)
        if pick_abs_base_pose is not None:
            self.pick_abs_base_pose = pick_abs_base_pose.reshape((1, 3))
        else:
            self.pick_abs_base_pose = None
        self.samples = self.sample_new_points(2000)
        self.curr_smpl_idx = 0

    def sample_picks(self, poses, collisions, n_smpls):
        pick_min = get_pick_domain()[0]
        pick_max = get_pick_domain()[1]
        pick_samples = np.random.uniform(pick_min, pick_max, (n_smpls, 6)).squeeze()
        return pick_samples

    def sample_placements(self, collisions, poses, encoded_pick_abs_poses, n_smpls):
        poses[:, -6:-2] = encoded_pick_abs_poses
        v_manip = compute_v_manip(self.abstract_state, self.abstract_state.goal_entities[:-1])
        v_manip = utils.convert_binary_vec_to_one_hot(v_manip.squeeze()).reshape((1, 618, 2, 1))
        v_manip = np.tile(v_manip, (n_smpls, 1, 1, 1))
        state_vec = np.concatenate([collisions, v_manip], axis=2)

        if 'home' in self.region:
            chosen_sampler = self.policies['place_home']
        else:
            chosen_sampler = self.policies['place_loading']

        place_samples = chosen_sampler.generate(state_vec, poses)
        place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])
        return place_samples

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

        if self.pick_abs_base_pose is None:
            pick_samples = self.sample_picks(poses, collisions, n_smpls)
            pick_abs_poses = np.array(
                [utils.get_pick_base_pose_and_grasp_from_pick_parameters(self.obj, s)[1] for s in pick_samples])
            encoded_pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])
        else:
            encoded_pick_abs_poses = np.array(utils.encode_pose_with_sin_and_cos_angle(self.pick_abs_base_pose))
            encoded_pick_abs_poses = np.tile(encoded_pick_abs_poses, (n_smpls, 1))
            pick_samples = self.sample_picks(poses, collisions, n_smpls)

        place_samples = self.sample_placements(collisions, poses, encoded_pick_abs_poses, n_smpls)
        samples = np.hstack([pick_samples, place_samples])
        return samples


class PickPlaceLearnedSampler(PlaceOnlyLearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        PlaceOnlyLearnedSampler.__init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose)
        self.samples = self.sample_new_points(2000)
        self.curr_smpl_idx = 0

    def sample_picks(self, poses, collisions, n_smpls):
        stime = time.time()
        pick_samples = self.policies['pick'].generate(collisions, poses)
        print "pick sampling time", time.time()-stime
        base_angles = pick_samples[:, 4:6]
        base_angles = [utils.decode_sin_and_cos_to_angle(base_angle) for base_angle in base_angles]
        pick_samples[:, 4] = base_angles
        pick_samples = np.delete(pick_samples, 5, 1)
        return pick_samples
