import numpy as np
import time

from gtamp_utils.utils import get_pick_domain, get_place_domain
from gtamp_utils import utils

from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.learning.PlacePolicyIMLE import gaussian_noise
from generators.learning.utils.sampler_utils import generate_pick_and_place_batch
from generators.learning.utils.sampler_utils import unprocess_pick_and_place_smpls

noise = gaussian_noise


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
    def __init__(self, policy, abstract_state, abstract_action):
        Sampler.__init__(self, policy)
        self.key_configs = abstract_state.prm_vertices
        self.abstract_state = abstract_state
        self.obj = abstract_action.discrete_parameters['object']
        self.region = abstract_action.discrete_parameters['place_region']

        goal_entities = self.abstract_state.goal_entities
        self.smpler_state = ConcreteNodeState(abstract_state.problem_env, self.obj, self.region,
                                              goal_entities,
                                              key_configs=self.key_configs)

        self.noises_used = []
        self.tried_smpls = []

        z_smpls = noise(z_size=(101, 7))
        smpls = generate_pick_and_place_batch(self.smpler_state, self.policy, z_smpls)
        self.policy_smpl_batch = unprocess_pick_and_place_smpls(smpls)
        self.policy_smpl_idx = 0

    def sample(self):
        smpl = self.policy_smpl_batch[self.policy_smpl_idx]
        self.policy_smpl_idx += 1
        if self.policy_smpl_idx >= len(self.policy_smpl_batch):
            z_smpls = noise(z_size=(100, 7))
            smpls = generate_pick_and_place_batch(self.smpler_state, self.policy, z_smpls)
            self.policy_smpl_batch = unprocess_pick_and_place_smpls(smpls)
            self.policy_smpl_idx = 0
        self.tried_smpls.append(smpl)
        return smpl
