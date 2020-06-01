from trajectory_representation.two_arm_sampler_trajectory import TwoArmSAHSSamplerTrajectory
from trajectory_representation.one_arm_sampler_trajectory import OneArmSAHSSamplerTrajectory

import pickle
import os
import argparse
import socket
import sys

hostname = socket.gethostname()
ROOTDIR = './'

def get_save_dir(parameters):
    if 'two_arm' in parameters.domain:
        n_objs_pack = 1
    else:
        n_objs_pack = 1

    save_dir = ROOTDIR + '/planning_experience/processed/{}/n_objs_pack_{}/sahs/uses_rrt/' \
                         'sampler_trajectory_data/includes_n_in_way/includes_vmanip/'.format(parameters.domain,
                                                                                             n_objs_pack)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_raw_dir(parameters):
    if parameters.domain == 'two_arm_mover':
        raw_dir = ROOTDIR + 'planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    elif parameters.domain == 'one_arm_mover':
        raw_dir = ROOTDIR + 'planning_experience/raw/one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    else:
        raise NotImplementedError
    return raw_dir


def get_p_idx(fname):
    return int(fname.split('.pkl')[0])


def save_traj(traj, save_fname):
    # making picklable
    for state in traj.states:
        state.problem_env = None
        state.abstract_state = None
    traj.problem_env = None
    pickle.dump(traj, open(save_fname, 'wb'))


def process_plan_file(filename, pidx):
    print "Plan file name", filename
    plan_data = pickle.load(open(filename, 'r'))
    plan = plan_data['plan']
    if not plan_data['success']:
        return None
    if 'one_arm_mover' in filename:
        traj = OneArmSAHSSamplerTrajectory(pidx, plan_data['n_objs_pack'])
    else:
        traj = TwoArmSAHSSamplerTrajectory(pidx, plan_data['n_objs_pack'])
    traj.add_trajectory(plan)
    return traj


def parse_parameters():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parameters = parser.parse_args()

    return parameters


def get_processed_fname(raw_fname):
    traj_fname = 'pap_traj_' + raw_fname
    return traj_fname


def get_raw_fname(raw_dir, parameters):
    # fname = 'pidx_%d_planner_seed_0_train_seed_0_domain_two_arm_mover.pkl' % parameters.pidx
    # fname = 'seed_0_pidx_' + str(parameters.pidx) + '.pkl'
    if 'one_arm_mover' in raw_dir:
        fname = 'sampling_strategy_uniformpidx_{}_planner_seed_0_gnn_seed_0.pkl'.format(parameters.pidx)
    else:
        fname = 'pidx_%d_planner_seed_0_gnn_seed_0.pkl' % parameters.pidx
    return fname


def quit_if_already_done(fpath, config):
    if os.path.isfile(fpath) and not config.f:
        print "Already done"
        sys.exit(-1)


def main():
    parameters = parse_parameters()
    raw_dir = get_raw_dir(parameters)
    # make_key_configs(parameters, raw_dir)
    raw_fname = get_raw_fname(raw_dir, parameters)
    save_dir = get_save_dir(parameters)
    processed_fname = get_processed_fname(raw_fname)
    print "Raw fname", raw_dir + raw_fname
    print "Processed fname ", save_dir + processed_fname
    quit_if_already_done(save_dir + processed_fname, parameters)

    # Every second element in the prm - it does not have to be, because state computation checks the collisions
    # at all configs anyways. todo: reprocess the data using the full prm
    # key_configs = np.delete(key_configs, 293, axis=0)
    traj = process_plan_file(raw_dir + raw_fname, parameters.pidx)
    if traj is not None:
        save_traj(traj, save_dir + processed_fname)


if __name__ == '__main__':
    main()
