import numpy as np
import time

from gtamp_utils.utils import get_pick_domain, get_place_domain
from gtamp_utils import utils

from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.utils import data_processing_utils
from generators.learning.utils.sampler_utils import generate_pick_and_place_batch
from generators.learning.utils.sampler_utils import unprocess_pick_and_place_smpls
from generators.learning.utils.data_processing_utils import action_data_mode


class Sampler:
    def __init__(self, policy):
        self.policy = policy

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

        self.samples = self.sample_new_points(100)
        utils.viewer()

        self.curr_smpl_idx = 0

    def sample_new_points(self, n_smpls):
        # Here, it would be much more accurate if I use place collision vector, but at this point
        # I don't know if the pick is feasible. Presumably, we can check the feasbility based on pick first, and
        # only if that is feasible, move onto a place. But this gets ugly as to how to "count" the number of samples
        # tried. I guess if we count the pick trials, it is same as now?
        raise NotImplementedError

    def sample(self):
        # prepare input to the network
        if self.curr_smpl_idx >= len(self.samples):
            self.samples = self.sample_new_points(100)

        new_sample = self.samples[self.curr_smpl_idx]
        return new_sample


class PlaceOnlyLearnedSampler(LearnedSampler):
    def __init__(self, sampler, abstract_state, abstract_action):
        LearnedSampler.__init__(self, sampler, abstract_state, abstract_action)

    def sample_picks(self, n_smpls):
        pick_min = get_pick_domain()[0]
        pick_max = get_pick_domain()[1]
        pick_samples = np.random.uniform(pick_min, pick_max, (n_smpls, 6)).squeeze()

        must_get_q0_from_pick_abs_pose = action_data_mode == 'PICK_grasp_params_and_abs_base_PLACE_abs_base'
        assert must_get_q0_from_pick_abs_pose
        pick_abs_poses = pick_samples[:, 3:7]
        return pick_abs_poses

    def sample_new_points(self, n_smpls):
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]

        # sample picks
        pick_abs_poses = self.sample_picks(n_smpls)
        pick_abs_poses = np.array([utils.encode_pose_with_sin_and_cos_angle(s) for s in pick_abs_poses])
        poses = np.tile(poses, (n_smpls, 1))
        poses[:, -4:] = pick_abs_poses

        collisions = self.smpler_state.pick_collision_vector
        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
        samples = self.policy.generate(collisions, poses)
        samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in samples])
        return samples
