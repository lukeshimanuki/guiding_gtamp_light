import numpy as np
import time

from gtamp_utils.utils import get_pick_domain, get_place_domain
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils


class Sampler:
    def __init__(self, policy):
        self.policies = policy

    def sample(self):
        raise NotImplementedError


class UniformSampler(Sampler):
    def __init__(self, target_region, policy=None):
        Sampler.__init__(self, policy)
        pick_min = get_pick_domain()[0]
        pick_max = get_pick_domain()[1]
        place_min = get_place_domain(target_region)[0]
        place_max = get_place_domain(target_region)[1]
        mins = np.hstack([pick_min, place_min])
        maxes = np.hstack([pick_max, place_max])
        self.domain = np.vstack([mins, maxes])

    def sample(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()


class LearnedSampler(Sampler):
    def __init__(self, sampler, abstract_state, abstract_action):
        Sampler.__init__(self, sampler)
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


class PickPlaceLearnedSampler(LearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        LearnedSampler.__init__(self, sampler, abstract_state, abstract_action)
        self.samples = self.sample_new_points(200)
        self.curr_smpl_idx = 0

    def sample_picks(self, poses, collisions, n_smpls):
        poses = np.tile(poses, (n_smpls, 1))
        chosen_sampler = self.policies['pick']
        pick_samples = chosen_sampler.generate(collisions, poses)
        # I need to encode it in the theta
        base_angles = pick_samples[:, 4:6]
        base_angles = [utils.decode_sin_and_cos_to_angle(base_angle) for base_angle in base_angles]
        pick_samples[:, 4] = base_angles
        pick_samples = np.delete(pick_samples, 5, 1)
        return pick_samples

    def sample_placements(self, poses, collisions, pick_samples, n_smpls):
        pick_abs_poses = np.array(
            [utils.get_pick_base_pose_and_grasp_from_pick_parameters(self.obj, s)[1] for s in pick_samples])
        import pdb;pdb.set_trace()
        pick_samples[:, -3:] = pick_abs_poses
        encoded_pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])

        poses = np.tile(poses, (n_smpls, 1))
        poses[:, -4:] = encoded_pick_abs_poses

        if 'home' in self.region:
            chosen_sampler = self.policies['place_home']
        else:
            chosen_sampler = self.policies['place_loading']

        place_samples = chosen_sampler.generate(collisions, poses)
        return place_samples

    def sample_new_points(self, n_smpls):
        # note: this function outputs absolute pick base pose and absolute place base pose
        print "Generating new points"
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]
        # making predictions using the sampler
        collisions = self.smpler_state.pick_collision_vector
        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))

        pick_samples = self.sample_picks(poses, collisions, n_smpls)
        place_samples = self.sample_placements(poses, collisions, pick_samples, n_smpls)

        # This is in robot's pose. I need to convert it to object poses
        place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])

        samples = np.hstack([pick_samples, place_samples])
        return samples


class PlaceOnlyLearnedSampler(LearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        LearnedSampler.__init__(self, sampler, abstract_state, abstract_action)
        if pick_abs_base_pose is not None:
            self.pick_abs_base_pose = pick_abs_base_pose.reshape((1, 3))
        else:
            self.pick_abs_base_pose = None
        self.samples = self.sample_new_points(200)
        self.curr_smpl_idx = 0

    def sample_picks(self, n_smpls):
        pick_min = get_pick_domain()[0]
        pick_max = get_pick_domain()[1]
        pick_samples = np.random.uniform(pick_min, pick_max, (n_smpls, 6)).squeeze()
        return pick_samples

    def sample_new_points(self, n_smpls):
        # note: this function outputs absolute pick base pose and absolute place base pose
        print "Generating new points"
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]

        # sample picks
        if self.pick_abs_base_pose is None:
            pick_samples = self.sample_picks(n_smpls)
            pick_abs_poses = np.array(
                [utils.get_pick_base_pose_and_grasp_from_pick_parameters(self.obj, s)[1] for s in pick_samples])
            pick_samples[:, -3:] = pick_abs_poses
            encoded_pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])
        else:
            encoded_pick_abs_poses = np.array(utils.encode_pose_with_sin_and_cos_angle(self.pick_abs_base_pose))
            encoded_pick_abs_poses = np.tile(encoded_pick_abs_poses, (n_smpls, 1))
            # some dummy variable. if we use this case besides the visualization, then save the pick samples
            pick_samples = self.sample_picks(n_smpls)

        poses = np.tile(poses, (n_smpls, 1))
        poses[:, -4:] = encoded_pick_abs_poses

        # making predictions using the sampler
        collisions = self.smpler_state.pick_collision_vector
        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
        if 'home' in self.region:
            chosen_sampler = self.policies['place_home']
        else:
            chosen_sampler = self.policies['place_loading']
        place_samples = chosen_sampler.generate(collisions, poses)

        # This is in robot's pose. I need to convert it to object poses
        place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])

        samples = np.hstack([pick_samples, place_samples])
        return samples
