from sampler import Sampler
import numpy as np


class UniformSampler(Sampler):
    def __init__(self, atype, target_region):
        Sampler.__init__(self, atype, target_region, sampler=None)

    def sample(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
