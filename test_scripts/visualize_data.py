from test_scripts.visualize_learned_sampler import create_environment
from generators.learning.train_sampler import get_data
from generators.learning.utils.data_processing_utils import filter_configs_that_are_too_close
from gtamp_utils import utils
from generators.learning.utils import sampler_utils

import numpy as np
import pickle




def main():
    problem_seed = 1
    states, konf_relevance, poses, rel_konfs, goal_flags, actions, sum_rewards = \
        get_data('n_objs_pack_4', 'place', 'loading_region')
    actions = actions[:, -4:]
    actions = [utils.decode_pose_with_sin_and_cos_angle(a) for a in actions]

    problem_env, openrave_env = create_environment(problem_seed)

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]
    key_configs = np.delete(key_configs, [415, 586, 615, 618, 619], axis=0)
    indices_to_delete = sampler_utils.get_indices_to_delete('loading_region', key_configs)
    key_configs = np.delete(key_configs, indices_to_delete, axis=0)

    filtered_konfs = filter_configs_that_are_too_close(actions)
    import pdb;pdb.set_trace()

    utils.viewer()
    utils.visualize_placements(filtered_konfs, 'square_packing_box1')
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
