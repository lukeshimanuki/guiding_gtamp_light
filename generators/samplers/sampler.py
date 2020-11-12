import numpy as np
import time

from gtamp_utils import utils
from trajectory_representation.concrete_node_state import TwoArmConcreteNodeState, OneArmConcreteNodeState
from gtamp_utils.utils import get_pick_domain, get_place_domain, get_pick_base_pose_and_grasp_from_pick_parameters
import pickle


class Sampler:
    def __init__(self, atype, target_region, sampler):
        self.samplers = sampler
        if 'pick' in atype and 'place' in atype:
            pick_min = get_pick_domain()[0]
            pick_max = get_pick_domain()[1]
            place_min = get_place_domain(target_region)[0]
            place_max = get_place_domain(target_region)[1]
            mins = np.hstack([pick_min, place_min])
            maxes = np.hstack([pick_max, place_max])
        elif 'pick' in atype and 'place' not in atype:
            pick_min = get_pick_domain()[0]
            pick_max = get_pick_domain()[1]
            mins = pick_min
            maxes = pick_max
        elif 'pick' not in atype and 'place' in atype:
            place_min = get_place_domain(target_region)[0]
            place_max = get_place_domain(target_region)[1]
            mins = place_min
            maxes = place_max
        else:
            raise NotImplementedError
        self.domain = np.vstack([mins, maxes])
        if 'one_arm' in atype:
            self.n_smpl_per_iter = 200
        else:
            self.n_smpl_per_iter = 2000
        self.curr_smpl_idx = 0

    def sample(self):
        # prepare input to the network
        if self.curr_smpl_idx >= len(self.samples):
            self.samples = self.sample_new_points(self.n_smpl_per_iter)
            self.curr_smpl_idx = 0
        new_sample = self.samples[self.curr_smpl_idx]
        self.curr_smpl_idx += 1
        return new_sample

    def sample_new_points(self, n_smpls):
        raise NotImplementedError

class LearnedSampler(Sampler):
    def __init__(self, atype, sampler, abstract_state, abstract_action, config, smpler_state=None):
        target_region = abstract_state.problem_env.regions[abstract_action.discrete_parameters['place_region']]
        self.config = config
        Sampler.__init__(self, atype,  target_region, sampler)
        self.abstract_state = abstract_state
        self.obj = abstract_action.discrete_parameters['object']
        self.region = abstract_action.discrete_parameters['place_region']
        stime = time.time()
        if 'one_arm' in atype:
            self.key_configs = pickle.load(open('one_arm_key_configs.pkl', 'r'))['konfs']
            if smpler_state is None:
                self.smpler_state = OneArmConcreteNodeState(abstract_state, abstract_action, self.key_configs)
            else:
                # this is because we use separate pick and place samplers in one arm case
                self.smpler_state = smpler_state
        else:
            self.key_configs = abstract_state.prm_vertices
            self.smpler_state = TwoArmConcreteNodeState(abstract_state, abstract_action)
        print 'Concrete node creation', time.time()-stime

    def decode_base_angle_encoding(self, pick_samples):
        base_angles = pick_samples[:, 4:6]
        base_angles = [utils.decode_sin_and_cos_to_angle(base_angle) for base_angle in base_angles]
        pick_samples[:, 4] = base_angles
        pick_samples = np.delete(pick_samples, 5, 1)  # remove unnecessary 2-dim encoded base angle
        return pick_samples
