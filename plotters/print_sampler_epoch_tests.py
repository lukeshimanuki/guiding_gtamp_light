import numpy as np
from compare_n_nodes import get_n_nodes
import os


def get_n_nodes(target_dir, is_valid_idxs):
    test_files = os.listdir(target_dir)
    n_iks = []
    n_nodes = []
    n_mps = []

    if 'rsc' in target_dir:
        test_file_pidxs = [int(filename.split('pidx_')[1].split('.pkl')[0]) for filename in test_files if
                           'pkl' in filename]
    else:
        test_file_pidxs = [int(filename.split('pidx_')[1].split('_')[0]) for filename in test_files if
                           'pkl' in filename]
    test_files = np.array(test_files)[np.argsort(test_file_pidxs)]
    if 'two_arm' in target_dir:
        if is_valid_idxs:
            target_pidxs = [40200, 40201, 40202, 40204, 40205, 40206, 40207, 40208, 40209]
        else:
            if 'n_objs_pack_1' in target_dir:
                target_pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012,
                                40018,
                                40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
            else:
                target_pidxs = [40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016,
                                40017,
                                40019, 40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036]
    else:
        target_pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20015, 20019, 20021, 20023, 20024, 20035,
                        20046, 20047, 20051, 20053, 20056, 20057, 20061, 20063, 20066, 20067, 20069, 20072, 20075,
                        20080, 20083, 20084, 20086, 20093, 20094, 20095]
        target_pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20047, 20051,
                        20053, 20057, 20061, 20063, 20066, 20069, 20072, 20075, 20084, 20086, 20093, 20094, 20095]
        target_pidxs = range(20000, 20100)

        # target_pidxs = [20002, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20051, 20061, 20063, 20066, 20069, 20072, 20075,
        # 20093]

        # target_pidxs.remove(20055)
        # target_pidxs.remove(20090)
        # target_pidxs.remove(20093)
        # target_pidxs = [20000, 20001, 20005, 20009, 20011, 20023, 20027, 20030, 20035, 20046, 20060, 20061, 20076]

    targets = []
    for pidx in target_pidxs:
        for i in range(5):
            targets.append((pidx, i))

    # print "number of target pidxs", len(target_pidxs)
    successes = []
    times = []
    pidx_times = {}
    pidx_nodes = {}
    pidx_iks = {}
    for filename in test_files:
        if 'pkl' not in filename:
            print 'File skipped', filename
            continue
        if 'gnn_seed' in filename:
            absqseed = int(filename.split('gnn_seed_')[-1].split('.pkl')[0])

            # target_absqseed = 1 if 'a179000' in target_dir else 0
            target_absqseed = 0
            # if absqseed != target_absqseed:
            # print filename
            #    continue

        if 'rsc' in target_dir:
            pidx = int(filename.split('pidx_')[1].split('.pkl')[0])
        else:
            pidx = int(filename.split('pidx_')[1].split('_')[0])

        seed = int(filename.split('seed_')[1].split('_')[0])
        if not pidx in target_pidxs:
            continue
        fin = pickle.load(open(target_dir + filename, 'r'))
        # targets.remove((pidx, seed))

        if 'num_nodes' in fin:
            n_node = fin['num_nodes']
        else:
            n_node = fin['n_nodes']

        n_nodes.append(n_node)

        if 'two_arm' in target_dir:
            if 'n_objs_pack_4' in target_dir:
                timelimit = 2000
            else:
                timelimit = 2000
        else:
            timelimit = 1000

        """
        if fin['tottime'] >= timelimit:
            successes.append(False)
            timetaken = timelimit
        else:
            successes.append(fin['success'])
            timetaken = fin['tottime']
        """
        successes.append(fin['success'])
        timetaken = fin['tottime']


        times.append(timetaken)
        if pidx in pidx_times:
            pidx_nodes[pidx].append(n_node)
            pidx_times[pidx].append(timetaken)
            pidx_iks[pidx].append(fin['n_feasibility_checks']['ik'])
        else:
            pidx_nodes[pidx] = [n_node]
            pidx_times[pidx] = [timetaken]
            pidx_iks[pidx] = [fin['n_feasibility_checks']['ik']]

    n_data = len(n_nodes)
    # print "*****REMAINING****", len(targets)
    # for pidx in targets:
    #    pidx_times[pidx_times.keys()[0]].append(timelimit)
    #    successes.append(False)

    return pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks

def get_sampler_dir(algo_name, learned_sampler_atype):
    # place loading
    if algo_name == 'wgandi':
        if learned_sampler_atype == 'place_loading':
            # place loading wgangp candidate seeds: [0,2,4,5,6]
            seed = '-1_-1_3'  # best epoch: 11386 - median 17, mean 19, success rate 1; evaluated 340 epochs
            seed = '-1_-1_4'  # best epoch: 7492 - median 18.5, mean 28.55, success rate 0.94; evaluated 437 epochs
            seed = '-1_-1_5'  # best epoch 10901? evaluated 313 epochs
            seed = '-1_-1_12'  # best epoch: 9232 - median 13, mean 19, success rate 1; evaluated 487 epochs; I may have to evaluate more
            seeds = ['-1_-1_3', '-1_-1_4', '-1_-1_5', '-1_-1_12']
        elif learned_sampler_atype == 'pick':
            seed = '2_-1_-1'  # best epoch 26613
            seed = '11_-1_-1'  # best_epoch 9840 evaluated 290 epochs; smpler seed idx 1
            seed = '15_-1_-1'  # best epoch 7935
            seed = '20_-1_-1'  # best epoch 8854
            seeds = ['2_-1_-1', '11_-1_-1', '15_-1_-1', '20_-1_-1']
        elif learned_sampler_atype == 'place_home':
            # uniform 0.87, 39, 47.23
            seed = '-1_4_-1'  # 8421 0.93 34 40.1; 6842 0.89 32.5 41.85; 8565 0.87 35.5 45.43
            seed = '-1_6_-1'  # 8395 0.93 34 40.19; 8389 0.93 34 42.09
            seed = '-1_10_-1'  # 6013 0.9 35.5 44.51
            seed = '-1_2_-1'  # 11138 0.83 39 48.83; 10945 0.9375 31.5 38.5
            seeds = ['-1_4_-1', '-1_6_-1', '-1_10_-1', '-1_2_-1']
    else:
        if learned_sampler_atype == 'place_loading':
            seeds = ['-1_-1_4', '-1_-1_2', '-1_-1_0', '-1_-1_5']
        elif learned_sampler_atype == 'pick':
            seeds = ['31_-1_-1', '12_-1_-1']
        elif learned_sampler_atype == 'place_home':
            seeds = ['-1_6_-1', '-1_7_-1', '-1_2_-1', '-1_-1_0', '-1_1_-1']

    if learned_sampler_atype == 'place_loading' or learned_sampler_atype == 'pick':
        n_objs = 1
    else:
        n_objs = 4

    seed_dir_path = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_{}/' \
                    'qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/' \
                    'using_learned_sampler/1000/{}/'.format(n_objs, algo_name)

    seed_dirs = [seed_dir_path + 'sampler_seed_{}'.format(sd) for sd in seeds]
    return seed_dirs


def get_target_epoch_dir(seed_dir, is_valid_idxs):
    target_dirs = []
    for epoch_dir in os.listdir(seed_dir):
        if 'wgandi' in seed_dir:
            try:
                epoch_dir_try = seed_dir + '/' + epoch_dir + '/wgandi/n_mp_limit_5_n_iter_limit_2000/'
                n_data = len(os.listdir(epoch_dir_try))
                epoch_dir = epoch_dir_try
            except OSError:
                epoch_dir = seed_dir + '/' + epoch_dir + '/n_mp_limit_5_n_iter_limit_2000/'
                n_data = len(os.listdir(epoch_dir))
        else:
            epoch_dir = seed_dir + '/' + epoch_dir + '/' + 'n_mp_limit_5_n_iter_limit_2000/'
            n_data = len(os.listdir(epoch_dir))
        if is_valid_idxs:
            condition = n_data >= 9
        else:
            condition = n_data > 36
        #print epoch_dir, n_data
        if condition:
            target_dirs.append(epoch_dir)
    print len(target_dirs)
    return target_dirs


def print_epoch_test_results():
    # print all the epochs that have more than 36 problems?
    algo_name = 'wgandi'
    learned_sampler_atype = 'pick'
    seed_dirs = get_sampler_dir(algo_name, learned_sampler_atype)
    is_valid_idxs = False
    print_uniform = False
    if print_uniform:
        print "Uniform"
        if learned_sampler_atype == 'place_loading' or learned_sampler_atype == 'pick':
            n_objs = 1
        else:
            n_objs = 4
        target_dir = 'test_results/sahs_results/domain_two_arm_mover/' \
                     'n_objs_pack_{}/qlearned_hcount_old_number_in_goal/' \
                     'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/' \
                     'n_mp_limit_5_n_iter_limit_2000/'.format(n_objs)
        pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir, is_valid_idxs)
        print 'n_data {} successes {} n nodes median {} mean {} std {} n_iks {}'.format(n_data, np.mean(successes),
                                                                                        np.median(n_nodes),
                                                                                        np.mean(n_nodes),
                                                                                        np.std(
                                                                                            n_nodes) * 1.96 / np.sqrt(
                                                                                            n_data),
                                                                                        np.mean(
                                                                                            np.hstack(
                                                                                                pidx_iks.values())))
    else:
        print "Algo name {} atype {}".format(algo_name, learned_sampler_atype)
        for sd_dir in seed_dirs:
            print "****New seed dir****"
            target_dirs = get_target_epoch_dir(sd_dir, is_valid_idxs)
            for target_dir in target_dirs:
                pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir, is_valid_idxs)
                print 'n_data {} successes {} n nodes median {} mean {} std {} n_iks {}'.format(n_data, np.mean(successes),
                                                                                                np.median(n_nodes),
                                                                                                np.mean(n_nodes),
                                                                                                np.std(n_nodes) * 1.96 / np.sqrt(n_data),
                                                                                                np.mean(np.hstack(pidx_iks.values())))



if __name__ == '__main__':
    print_epoch_test_results()
