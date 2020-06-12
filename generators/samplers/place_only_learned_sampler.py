from sampler import LearnedSampler
from generators.learning.utils import data_processing_utils
from gtamp_utils.utils import get_pick_domain
import numpy as np
import time
from gtamp_utils import utils
from trajectory_representation.one_arm_sampler_trajectory import compute_v_manip


class PlaceOnlyLearnedSampler(LearnedSampler):
    def __init__(self, atype, sampler, abstract_state, abstract_action, smpler_state):
        LearnedSampler.__init__(self, atype, sampler, abstract_state, abstract_action, smpler_state)
        self.v_manip = None
        self.samples = self.sample_new_points(200)
        self.curr_smpl_idx = 0

    def sample_placements(self, pose_ids, collisions, n_smpls):
        if self.v_manip is None:
            v_manip = compute_v_manip(self.abstract_state, self.abstract_state.goal_entities[:-1], self.key_configs)
            v_manip = utils.convert_binary_vec_to_one_hot(v_manip.squeeze()).reshape((1, len(self.key_configs), 2, 1))
            v_manip = np.tile(v_manip, (n_smpls, 1, 1, 1))
            self.v_manip = v_manip
        state_vec = np.concatenate([collisions, self.v_manip], axis=2)

        if 'center_shelf' in self.region:
            chosen_sampler = self.samplers['place_obj_region']
            place_samples = chosen_sampler.generate(state_vec, pose_ids)
            place_samples = np.array([utils.decode_pose_with_sin_and_cos_angle(s) for s in place_samples])
        elif 'rectangular_packing_box1_region' in self.region:
            chosen_sampler = self.samplers['place_goal_region']
            place_samples = np.array([chosen_sampler.sample() for _ in range(n_smpls)])
        else:
            raise NotImplementedError
        return place_samples

    def sample_new_points(self, n_smpls):
        # note: this function outputs absolute pick base pose and absolute place base pose
        print "Generating new place points"
        stime = time.time()

        collisions = self.smpler_state.pick_collision_vector
        collisions = np.tile(collisions, (n_smpls, 1, 1, 1))
        poses = data_processing_utils.get_processed_poses_from_state(self.smpler_state, None)[None, :]
        poses = np.tile(poses, (n_smpls, 1))
        if 'rectangular' in self.obj:
            object_id = [1, 0]
        else:
            object_id = [0, 1]
        object_id = np.tile(np.array(object_id)[None, :], (n_smpls, 1))
        pose_ids = np.hstack([poses, object_id])

        place_samples = self.sample_placements(pose_ids, collisions, n_smpls)
        print 'place prediction time', time.time() - stime
        return place_samples
