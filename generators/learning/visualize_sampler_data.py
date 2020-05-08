import os
import pickle
from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_utils import utils
import numpy as np


def get_first_placement_in_home(data):
    flag = True
    for action_idx, action in enumerate(data.actions):
        if action['region_name'] == 'loading_region':
            obj = action['object_name']
            #do_i_move_it_more = np.sum([obj==a['object_name'] and 'home_region'==a['region_name'] for a in data.actions])==2
            #if not do_i_move_it_more:
            if True:
                if data.n_in_way[action_idx] - data.prev_n_in_way[action_idx] <= 0:
                    return action['place_abs_base_pose']
                else:
                    return None
            else:
                return None

def main():
    data_dir = './planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/sampler_trajectory_data/includes_n_in_way/'
    #data_dir = 'planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    traj_files = [f for f in os.listdir(data_dir) if 'pidx' in f]
    placements = []
    for traj_file in traj_files:
        print traj_file
        data = pickle.load(open(data_dir + traj_file, 'r'))
        placement_pose = get_first_placement_in_home(data)
        if placement_pose is not None:
            placements.append(placement_pose)
        if len(placements) > 100:
            break
    print (np.array(placements) == 4).mean()
    import pdb;pdb.set_trace()
    problem_env = PaPMoverEnv(0)
    utils.viewer()
    utils.visualize_path(placements)

    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
