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
        if parameters.planner == 'greedy':
            raw_dir = 'planning_experience/raw/two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/n_mp_limit_5_n_iter_limit_2000/'
        elif parameters.planner == 'rsc':
            raise NotImplementedError
            #raw_dir = ROOTDIR + 'planning_experience/raw/irsc/n_objs_pack_1/'
    elif parameters.domain == 'one_arm_mover':
        raw_dir = ROOTDIR + 'planning_experience/raw/one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
                            'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                            'n_mp_limit_5_n_iter_limit_2000/'
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


def get_sampler_traj_instance(filename, pidx, plan_data):
    print "Plan file name", filename
    if 'one_arm_mover' in filename:
        traj = OneArmSAHSSamplerTrajectory(pidx, plan_data['n_objs_pack'])
    else:
        traj = TwoArmSAHSSamplerTrajectory(pidx, plan_data['n_objs_pack'])
    return traj


def parse_parameters():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-f', action='store_true', default=False)
    parser.add_argument('-domain', type=str, default='two_arm_mover')
    parser.add_argument('-n_objs_pack', type=int, default=1)
    parser.add_argument('-planner', type=str, default='rsc')
    parameters = parser.parse_args()

    return parameters


def get_processed_fname(raw_fname):
    traj_fname = 'pap_traj_' + raw_fname
    return traj_fname


def get_raw_fname(raw_dir, parameters):
    if 'rsc' in raw_dir:
        fname = 'seed_0_pidx_{}.pkl'.format(parameters.pidx)
    else:
        fname = 'sampling_strategy_uniformpidx_{}_planner_seed_0_gnn_seed_0.pkl'.format(parameters.pidx)
    return fname


def quit_if_already_done(fpath, config):
    if os.path.isfile(fpath) and not config.f:
        print "Already done"
        sys.exit(-1)


def main():
    parameters = parse_parameters()
    raw_dir = get_raw_dir(parameters)
    raw_fname = get_raw_fname(raw_dir, parameters)
    save_dir = get_save_dir(parameters)
    processed_fname = get_processed_fname(raw_fname)
    print "Raw fname", raw_dir + raw_fname
    print "Processed fname ", save_dir + processed_fname
    quit_if_already_done(save_dir + processed_fname, parameters)

    fname = raw_dir + raw_fname
    plan_data = pickle.load(open(fname, 'r'))

    traj = get_sampler_traj_instance(fname, parameters.pidx, plan_data)
    if parameters.planner == 'greedy':
        nodes = plan_data['nodes']
        positive_data, neutral_data = traj.get_greedy_data(nodes, parameters)
    else:
        plan = plan_data['plan']
        positive_data = traj.get_rsc_data(plan, parameters)
        neutral_data = None

    data = {'neutral_data': neutral_data, 'positive_data': positive_data}
    pickle.dump(data, open(save_dir + processed_fname, 'wb'))


if __name__ == '__main__':
    main()
