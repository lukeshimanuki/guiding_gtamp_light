import numpy as np
from gtamp_utils.utils import get_pick_domain, get_place_domain


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
