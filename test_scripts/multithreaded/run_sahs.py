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


def get_all_configs(target_pidx_idxs, setup):
    if setup.use_learning and setup.test_multiple_epochs:
        if 'pick' in setup.learned_sampler_atype:
            _, _, total_epochs = convert_seed_epoch_idxs_to_seed_and_epoch('pick', '', setup)
        else:
            region = setup.learned_sampler_atype.split('_')[1]
            _, _, total_epochs = convert_seed_epoch_idxs_to_seed_and_epoch('place', region + '_region', setup)
        total_epochs = range(len(total_epochs))
    else:
        total_epochs = [0]

    configs = []
    print "Total number of epochs", len(total_epochs)
    for epoch in total_epochs:
        for idx in target_pidx_idxs:
            config = {}
            for k, v in setup._get_kwargs():
                if type(v) is bool and v is True:
                    config[k] = ''
                elif type(v) is not bool:
                    config[k] = v
            config['sampler_epoch_idx'] = epoch
            config['pidx'] = idx
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
    setup.absq_seed = 2

    setup.timelimit = np.inf
    setup.num_node_limit = 100

    if setup.use_test_pidxs:
        if setup.n_objs_pack == 1:
            pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018, 40020,
                     40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
        else:
            target_pidxs = [40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016, 40017,
                            40019, 40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036]
            target_pidxs = target_pidxs[0:25]
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
