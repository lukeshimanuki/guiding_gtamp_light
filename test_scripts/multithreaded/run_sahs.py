import os
import multiprocessing
import numpy as np
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import argparse
from test_scripts.run_greedy import parse_arguments


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
    if 'pick' in config.learned_sampler_atype:
        return 'pick'
    elif 'place' in config.learned_sampler_atype:
        if 'loading' in config.learned_sampler_atype:
            return 'place_obj_region'
        elif 'home' in config.learned_sampler_atype:
            return 'place_goal_region'
        else:
            raise NotImplementedError


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
                    action_name = determine_action_name(setup)
                    config[action_name+'_epoch'] = int(epoch)
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
