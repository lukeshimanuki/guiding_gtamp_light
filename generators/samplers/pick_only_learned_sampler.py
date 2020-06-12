from sampler import LearnedSampler
import numpy as np
from generators.learning.utils import data_processing_utils
from gtamp_utils import utils
import time


class PickOnlyLearnedSampler(LearnedSampler):
    def __init__(self, atype, sampler, abstract_state, abstract_action, pick_abs_base_pose=None):
        LearnedSampler.__init__(self, atype, sampler, abstract_state, abstract_action)
        self.samples = self.sample_new_points(200)
        """
        ### Debugging purpose
        obj_kinbody = self.abstract_state.problem_env.env.GetKinBody(self.obj)
        obj_xyth = utils.get_body_xytheta(obj_kinbody)
        pick_abs_poses = []
        for s in self.samples:
            _, poses = utils.get_pick_base_pose_and_grasp_from_pick_parameters(obj_kinbody, s, obj_xyth)
            pick_abs_poses.append(poses)
        ####
        import pdb;pdb.set_trace()
        """
        self.curr_smpl_idx = 0

    def sample_picks(self, poses, collisions):
        pick_samples = self.samplers['pick'].generate(collisions, poses)
        pick_samples = self.decode_base_angle_encoding(pick_samples)
        return pick_samples

    def sample_placements(self, n_smpls):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        pickplace_smpls = np.random.uniform(domain_min, domain_max, (n_smpls, dim_parameters)).squeeze()
        place_smpls = pickplace_smpls[:, -3:]
        return place_smpls

    def sample_new_points(self, n_smpls):
        print "Generating new pick points"
        stime = time.time()
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
        pick_samples = self.sample_picks(poses, collisions)
        print 'pick prediction time', time.time() - stime
        return pick_samples
