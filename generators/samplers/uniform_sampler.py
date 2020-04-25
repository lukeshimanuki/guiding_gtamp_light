from sampler import Sampler
import numpy as np


class UniformSampler(Sampler):
    def __init__(self, target_region, policy=None):
        Sampler.__init__(self, policy, target_region)

    def sample(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
