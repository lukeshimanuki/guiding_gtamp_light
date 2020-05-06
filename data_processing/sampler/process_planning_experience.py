from trajectory_representation.sampler_trajectory import SamplerTrajectory, SAHSSamplerTrajectory

import pickle
import os
import argparse
import socket
import sys

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def get_save_dir():
    save_dir = ROOTDIR + '/planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/irsc/sampler_trajectory_data/'
    save_dir = ROOTDIR + '/planning_experience/processed/domain_two_arm_mover/n_objs_pack_1/sahs/uses_rrt/sampler_trajectory_data/includes_n_in_way/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_raw_dir():
    raw_dir = ROOTDIR + '/planning_experience/raw/two_arm_mover/n_objs_pack_1//'
    raw_dir = ROOTDIR + 'planning_experience/raw/two_arm_mover/n_objs_pack_4/' \
                        'qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/'
    raw_dir = ROOTDIR + 'planning_experience/raw/two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/'
    raw_dir = ROOTDIR + 'planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    return raw_dir


def get_p_idx(fname):
    return int(fname.split('.pkl')[0])


def save_traj(traj, save_fname):
    # making picklable
    for state in traj.states:
        state.problem_env = None
    traj.problem_env = None
    pickle.dump(traj, open(save_fname, 'wb'))


def process_plan_file(filename, pidx):
    print "Plan file name", filename
    plan_data = pickle.load(open(filename, 'r'))
    plan = plan_data['plan']
    hvals = plan_data['hvalues']
    if not plan_data['success']:
        return None
    # traj = SamplerTrajectory(pidx)
    traj = SAHSSamplerTrajectory(pidx, plan_data['n_objs_pack'])
    traj.add_trajectory(plan, hvals)
    return traj


def parse_parameters():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-pidx', type=int, default=0)
    parser.add_argument('-f', action='store_true', default=False)
    parameters = parser.parse_args()

    return parameters


def get_processed_fname(raw_fname):
    traj_fname = 'pap_traj_' + raw_fname
    return traj_fname


def get_raw_fname(parameters):
    fname = 'pidx_%d_planner_seed_0_train_seed_0_domain_two_arm_mover.pkl' % parameters.pidx
    #fname = 'seed_0_pidx_' + str(parameters.pidx) + '.pkl'
    fname = 'pidx_%d_planner_seed_0_gnn_seed_0.pkl' %parameters.pidx
    return fname


def quit_if_already_done(fpath, config):
    if os.path.isfile(fpath) and not config.f:
        print "Already done"
        sys.exit(-1)


def main():
    parameters = parse_parameters()

    raw_dir = get_raw_dir()
    raw_fname = get_raw_fname(parameters)
    save_dir = get_save_dir()
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
