import os
import multiprocessing
import numpy as np
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
from test_scripts.run_greedy import parse_arguments
from plotters.print_sampler_epoch_tests import get_n_nodes, get_sampler_dir, get_target_epoch_dir


def get_top_epoch(algo_name, learned_sampler_atype, sampler_seed_idx):
    seed_dirs = get_sampler_dir(algo_name, learned_sampler_atype)
    seed_dir = seed_dirs[sampler_seed_idx]
    target_dirs = get_target_epoch_dir(seed_dir, is_valid_idxs=False)
    top_n_nodes = np.inf
    top_epoch = None
    for target_dir in target_dirs:
        pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir, is_valid_idxs=False)
        n_nodes = np.median(n_nodes)
        print 'n_data {} successes {} n nodes median {} mean {} std {} n_iks {}'.format(n_data,
                                                                                        np.mean(successes),
                                                                                        np.median(n_nodes),
                                                                                        np.mean(n_nodes),
                                                                                        np.std(
                                                                                            n_nodes) * 1.96 / np.sqrt(
                                                                                            n_data),
                                                                                        np.mean(np.hstack(
                                                                                            pidx_iks.values())))

        if n_nodes < top_n_nodes:
            top_n_nodes = n_nodes
            top_epoch = np.max([int(i) for i in target_dir.split('sampler_epoch_')[1].split('/')[0].split('_')])
    print top_epoch
    return top_epoch


def worker_p(config):
    command = 'python ./test_scripts/run_greedy.py'

    for key, value in zip(config.keys(), config.values()):
        option = ' -' + str(key) + ' ' + str(value)
        command += option
    print command
    print '\n'
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    return worker_p(multi_args)


def convert_seed_epoch_idxs_to_seed_and_epoch(atype, region, config):
    if atype == 'pick':
        sampler_weight_path = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/fc/'.format(config.domain,
                                                                                                          config.num_episode,
                                                                                                          atype,
                                                                                                          config.train_type)
    else:
        sampler_weight_path = './generators/learning/learned_weights/{}/num_episodes_{}/{}/{}/{}/fc/'.format(
            config.domain,
            config.num_episode,
            atype,
            region,
            config.train_type)

    seed_dirs = os.listdir(sampler_weight_path)
    candidate_seeds = []
    for sd_dir in seed_dirs:
        weight_files = [f for f in os.listdir(sampler_weight_path + sd_dir) if 'epoch' in f and '.pt' in f]
        if len(weight_files) > 1:
            seed = int(sd_dir.split('_')[1])
            candidate_seeds.append(seed)
    # todo sort the candidate seeds in order
    candidate_seeds = np.sort(candidate_seeds)
    seed = int(candidate_seeds[config.sampler_seed_idx])
    epochs = [f for f in os.listdir(sampler_weight_path + 'seed_{}'.format(seed)) if 'epoch' in f and '.pt' in f]
    epoch = int(epochs[config.sampler_epoch_idx].split('_')[-1].split('.pt')[0])
    print sampler_weight_path
    print "Candidate seeds {}".format(candidate_seeds)
    print "Selected seed {} epoch {}".format(seed, epoch)

    return seed, epoch, epochs


def determine_action_name(config):
    action_name = ''
    if 'pick' in config.learned_sampler_atype:
        action_name += 'pick'
    if 'loading' in config.learned_sampler_atype:
        action_name += '_place_obj_region'
    if 'home' in config.learned_sampler_atype:
        action_name += '_place_goal_region'
    return action_name


def get_all_configs(target_pidx_idxs, setup):
    configs = []
    epochs_to_run = setup.epochs_to_evaluate
    if epochs_to_run is None:
        epochs_to_run = [0]
    print "Total number of epochs", len(epochs_to_run)
    sampler_seed_idx_to_run = setup.sampler_seed_idx
    planner_seeds_to_run = [int(i) for i in setup.planner_seeds_to_run]
    if setup.num_trains_to_run is None:
        num_trains_to_run = [5000]
    else:
        num_trains_to_run = [int(i) for i in setup.num_trains_to_run]
    # I need to set pick epoch, place obj region epoch, and place goal region epoch separately

    if setup.use_best_epochs:
        pick_action_epoch = get_top_epoch(setup.train_type, 'pick', setup.sampler_seed_idx)
        place_obj_region_epoch = get_top_epoch(setup.train_type, 'place_loading', setup.sampler_seed_idx)
        place_goal_region_epoch = get_top_epoch(setup.train_type, 'place_home', setup.sampler_seed_idx)

    for num_train in num_trains_to_run:
        for epoch in epochs_to_run:
            for planner_seed in planner_seeds_to_run:
                for idx in target_pidx_idxs:
                    config = {}
                    for k, v in setup._get_kwargs():
                        if type(v) is bool and v is True:
                            config[k] = ''
                        elif type(v) is not bool:
                            config[k] = v

                    if setup.use_best_epochs:
                        config['pick_epoch'] = pick_action_epoch
                        config['place_obj_region_epoch'] = place_obj_region_epoch
                        config['place_goal_region_epoch'] = place_goal_region_epoch
                    else:
                        action_name = determine_action_name(setup)
                        config[action_name + '_epoch'] = int(epoch)
                    config['sampler_seed_idx'] = sampler_seed_idx_to_run
                    config['pidx'] = idx
                    config['planner_seed'] = planner_seed
                    config['num_train'] = num_train
                    configs.append(config)
    return configs


def main():
    # specify configs.sampler_seed_idx and configs.planner_seed and test_multiple_epochs for testing across epochs
    # specify a particular epoch, or use use_best_kde_sampler option for choosing an epoch to run across problems
    cmd = 'python upload_greedy_results.py &'
    os.system(cmd)
    cmd = 'python delete_openrave_tmp_files.py &'
    os.system(cmd)

    setup = parse_arguments()
    setup.use_region_agnostic = True

    setup.timelimit = np.inf
    setup.num_node_limit = 100

    if setup.use_test_pidxs:
        if setup.n_objs_pack == 1:
            pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018, 40020,
                     40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
        else:
            pidxs = [40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016, 40017,
                     40019, 40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036]
    else:
        pidxs = [40200, 40201, 40202, 40204, 40205, 40206, 40207, 40208, 40209]

    configs = get_all_configs(pidxs, setup)

    n_workers = multiprocessing.cpu_count()
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
