from sampler import Sampler


class VOOSampler(Sampler):
    def __init__(self, target_region, policy=None):
        Sampler.__init__(self, policy, target_region)

    def sample(self, actions, q_values):
        # todo paste the code from adversarial_voo
        pass
