import os
import numpy as np
import pickle
import re

from plotters.compare_n_nodes import get_n_nodes


def get_weight_dir(atype, region, num_episode, train_type, sampler_seed_idx, domain):
    if atype == 'pick':
        sampler_weight_path = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/fc/'.format(domain,
                                                                                                          num_episode,
                                                                                                          atype,
                                                                                                          train_type)
    else:
        sampler_weight_path = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/{}/fc/'.format(
            domain,
            num_episode,
            atype,
            region,
            train_type)

    seed_dirs = os.listdir(sampler_weight_path)
    candidate_seeds = []
    for sd_dir in seed_dirs:
        weight_files = [f for f in os.listdir(sampler_weight_path + sd_dir) if 'epoch' in f and '.pt' in f]
        if len(weight_files) > 1:
            seed = int(sd_dir.split('_')[1])
            candidate_seeds.append(seed)
    candidate_seeds = np.sort(candidate_seeds)
    seed = int(candidate_seeds[sampler_seed_idx])
    target_weight_dir = sampler_weight_path + '/seed_{}'.format(seed) + '/'
    return target_weight_dir


def get_top_k_epochs(weight_dir, k):
    if os.path.isfile(weight_dir + 'top_k_epochs.pkl'):
        top_k_epochs = pickle.load(open(weight_dir + 'top_k_epochs.pkl', 'r'))
    else:
        saved_epochs = [f.split('_')[-1].split('.pt')[0] for f in os.listdir(weight_dir) if
                        '.pt' in f and 'best' not in f]
        logfiles = [f for f in os.listdir(weight_dir) if '.pt' not in f]

        saved_epoch_kdes = []
        for epoch in saved_epochs:
            epoch_logfile = [f for f in logfiles if 'epoch_{}'.format(epoch) in f][0]
            kde = float(epoch_logfile.split('kde_')[1].split('_')[0])
            saved_epoch_kdes.append(kde)

        sorted_idxs_wrt_kdes = np.argsort(saved_epoch_kdes)[::-1]
        sorted_epochs_wrt_kdes = np.array(saved_epochs)[sorted_idxs_wrt_kdes]
        top_k_epochs = sorted_epochs_wrt_kdes[0:k]
        pickle.dump(top_k_epochs, open(weight_dir + 'top_k_epochs.pkl', 'wb'))
    return top_k_epochs


def evaluate_on_valid_pidxs(atype, region, sampler_seed_idx, top_k_epochs):
    if atype == 'pick':
        learned_sampler_atype = 'pick'
    elif atype == 'place':
        if 'home' in region:
            learned_sampler_atype = 'place_home'
        elif 'loading' in region:
            learned_sampler_atype = 'place_loading'
        else:
            # implement the ones for one arm domain
            raise NotImplementedError
    top_k_epoch_cmd = ''
    for epoch in top_k_epochs:
        top_k_epoch_cmd += epoch + ' '
    cmd = 'python test_scripts/multithreaded/run_sahs.py ' \
          '-use_learning ' \
          '-learned_sampler_atype {} ' \
          '-sampler_seed_idx {} ' \
          '-epochs {}'.format(learned_sampler_atype, sampler_seed_idx, top_k_epoch_cmd)
    print cmd
    os.system(cmd)


def get_action_seed_dirs(atype, region, test_dir):
    seed_dirs = os.listdir(test_dir)
    seed_dirs = [sd_dir.split('sampler_seed_')[1] for sd_dir in seed_dirs]
    if atype == 'pick':
        pattern = '[0-9]+_-1_-1'
    elif atype == 'place':
        if 'home' in region:
            pattern = '-1_[0-9]+_-1'
        elif 'loading' in region:
            pattern = '-1_-1_[0-9]+'
        else:
            raise NotImplementedError
    action_sd_dirs = ['sampler_seed_' + sd_dir for sd_dir in seed_dirs if re.match(pattern, sd_dir)]
    return action_sd_dirs


def get_epoch_dir(atype, region, target_dir):
    seed_dirs = os.listdir(target_dir)
    seed_dirs = [sd_dir.split('sampler_epoch_')[1] for sd_dir in seed_dirs]
    if atype == 'pick':
        pattern = '[0-9]+_-1_-1'
    elif atype == 'place':
        if 'home' in region:
            pattern = '-1_[0-9]+_-1'
        elif 'loading' in region:
            pattern = '-1_-1_[0-9]+'
        else:
            raise NotImplementedError
    action_epoch_dirs = ['sampler_epoch_' + sd_dir for sd_dir in seed_dirs if re.match(pattern, sd_dir)]
    return action_epoch_dirs


def get_top_epoch(atype, region, sampler_seed_idx):
    n_objs = 1
    n_train = 1000
    algo_name = 'wgandi'
    test_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_{}/' \
               'qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True' \
               '/using_learned_sampler/{}/'.format(n_objs, n_train)
    seed_dirs = get_action_seed_dirs(atype, region, test_dir)
    seeds = [np.max([int(i) for i in sd_dir.split('sampler_seed_')[1].split('_')]) for sd_dir in seed_dirs]
    sorted_seed_idxs = np.argsort(seeds)
    target_seed_dir = np.array(seed_dirs)[sorted_seed_idxs][sampler_seed_idx]
    target_sd_dir = test_dir + target_seed_dir + '/'

    epoch_dirs = get_epoch_dir(atype, region, target_sd_dir)
    best_epoch_dir = None
    min_n_nodes = np.inf
    for epoch_dir in epoch_dirs:
        target_dir = target_sd_dir + epoch_dir + '/' + algo_name + '/n_mp_limit_5_n_iter_limit_2000/'
        pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir, is_valid_idxs=True)
        condition = np.median(n_nodes) < np.median(min_n_nodes) and np.mean(successes) == 1
        if condition:
            best_epoch_dir = epoch_dir
            min_n_nodes = np.median(n_nodes)
        break
    best_epoch = np.max([int(i) for i in best_epoch_dir.split('sampler_epoch_')[1].split('_')])
    return best_epoch


def evaluate_on_test_pidxs(atype, region, sampler_seed_idx, top_epoch):
    if atype == 'pick':
        learned_sampler_atype = 'pick'
    elif atype == 'place':
        if 'home' in region:
            learned_sampler_atype = 'place_home'
        elif 'loading' in region:
            learned_sampler_atype = 'place_loading'
        else:
            # implement the ones for one arm domain
            raise NotImplementedError
    top_k_epoch_cmd = top_epoch
    planner_seeds = '0 1 2 3'
    cmd = 'python test_scripts/multithreaded/run_sahs.py ' \
          '-use_learning ' \
          '-learned_sampler_atype {} ' \
          '-sampler_seed_idx {} ' \
          '-epochs {} ' \
          '-use_test_pidxs ' \
          '-planner_seeds_to_run {}'.format(learned_sampler_atype, sampler_seed_idx, top_k_epoch_cmd, planner_seeds)
    print cmd
    os.system(cmd)


def upload_test_results():
    pass


def main():
    atype = 'place'
    region = 'loading_region'
    num_episode = 1000
    train_type = 'wgangp'
    domain = 'two_arm_mover'
    sampler_seed_idxs = [0, 1, 2, 3]
    k = 100
    for sampler_seed_idx in sampler_seed_idxs:
        weight_dir = get_weight_dir(atype, region, num_episode, train_type, sampler_seed_idx, domain)
        top_k_epochs = get_top_k_epochs(weight_dir, k)
        evaluate_on_valid_pidxs(atype, region, sampler_seed_idx, top_k_epochs)
        top_epoch = get_top_epoch(atype, region, sampler_seed_idx)
        evaluate_on_test_pidxs(atype, region, sampler_seed_idx, top_epoch)

    upload_test_results()


if __name__ == '__main__':
    main()
