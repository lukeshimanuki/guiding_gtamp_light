import pickle
import os
import numpy as np
#from matplotlib import pyplot as plt
from print_sampler_epoch_tests import get_n_nodes

def get_n_nodes(target_dir):
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
        #target_pidxs = range(20000, 20100)

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
    #target_pidxs = range(40200, 40210)
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

        if fin['tottime'] >= timelimit:
            successes.append(False)
            timetaken = timelimit
        else:
            successes.append(fin['success'])
            timetaken = fin['tottime']
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
    print n_data
    return pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks


def get_target_idxs(pidx_nodes_1, pidx_nodes_2, n_objs_pack, domain, threshold):
    if 'one_arm' in domain:
        target_idxs = [pidx for pidx in pidx_nodes_1 if
                       abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) >= 150]
    else:
        if n_objs_pack == 1:
            target_idxs = [pidx for pidx in pidx_nodes_1 if
                           abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) >= 20]
        else:
            target_idxs = [pidx for pidx in pidx_nodes_1 if
                           pidx in pidx_nodes_2 and abs(
                               np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) > threshold]
    print 'n target idxs', len(target_idxs)
    return target_idxs


def plot_one_arm():
    target_dir = 'cloud_results/240c6f4/sahs_results/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_3, pidx_times_3 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)

    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_1, pidx_times_1 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_3, pidx_times_3 = get_n_nodes(target_dir)
    target_dir = 'cloud_results//3dbf9ca/pure_learning/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_5, pidx_times_5 = get_n_nodes(target_dir)

    target_dir = 'cloud_results/3c193cf/irsc/one_arm_mover/n_objs_pack_1/'
    pidx_nodes_4, pidx_times_4 = get_n_nodes(target_dir)

    plt.figure()
    plt.boxplot([np.hstack(pidx_times_1.values()), np.hstack(pidx_times_2.values()), np.hstack(pidx_times_3.values()),
                 np.hstack(pidx_times_4.values()), np.hstack(pidx_times_5.values())],
                labels=['SAGS\nRankSampler', 'SAGS\nRank', 'SAGS\nHCount', 'IRSC', 'Pure\nLearning'],
                positions=[0, 1, 2, 3, 4],
                whis=(10, 90))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig("./plotters/plots/{}_arm_{}_obj.eps".format('one_arm_mover', 1))
    plt.savefig("../IJRR_GTAMP/figures/{}_arm_{}_obj.eps".format('one_arm_mover', 1))


def plot_two_arm():
    n_objss = [1]
    target_dir = 'cloud_results/ac2e048/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print np.mean(np.hstack(pidx_times.values())), np.mean(np.hstack(pidx_times.values())), np.median(
        np.hstack(pidx_times.values())), np.median(np.hstack(pidx_times.values())), np.mean(n_nodes), np.mean(successes)

    print  "****Ranking function****"
    target_dir = 'cloud_results/1533b3c/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    target_dir = 'cloud_results/187aad7/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes2, pidx_times2, successes2, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print np.mean(np.hstack(pidx_times.values())), np.mean(np.hstack(pidx_times2.values())), np.median(
        np.hstack(pidx_times.values())), np.median(np.hstack(pidx_times2.values()))
    print np.mean(np.hstack(pidx_nodes.values())), np.mean(np.hstack(pidx_nodes2.values())), np.median(
        np.hstack(pidx_nodes.values())), np.median(np.hstack(pidx_nodes2.values()))
    print np.mean(successes), np.mean(successes)

    print  "****WGANGP 1000 data points****"
    target_dir = 'cloud_results/8db0c37/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/1000/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    print  "****WGANGP 4000 data points****"
    target_dir = 'cloud_results/8db0c37/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/4000/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    print  "****WGANGP 1 packing obj training 4900 data points****"
    target_dir = 'cloud_results/92d0985/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/4900/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    import pdb;
    pdb.set_trace()

    print '****n objs pack 4****'
    print '****n objs pack 4****'
    print '****n objs pack 4****'
    print '****n objs pack 4****'
    print  "****Ranking function****"
    target_dir = 'cloud_results/1533b3c/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    print  "****WGANGP 4000 data points****"
    target_dir = 'cloud_results/8db0c37/sahs_results/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/4000/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    print  "****WGANGP 1000 data points****"
    target_dir = 'cloud_results/8db0c37/sahs_results/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/1000/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    print  "****WGANGP 1 packing obj training 4900 data points****"
    target_dir = 'cloud_results/92d0985/sahs_results/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/4900/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    import pdb;
    pdb.set_trace()

    target_dir = 'cloud_results/1533b3c/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/92d0985/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/4900/sampler_seed_0/wgandi/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/92d0985/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/using_learned_sampler/4900/sampler_seed_0/wgangp/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    import pdb;
    pdb.set_trace()

    # target_dir = 'cloud_results/1533b3c/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    # pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    # import pdb;pdb.set_trace()

    target_dir = 'cloud_results/16653e7/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)

    """
    target_dir = 'cloud_results/c40c7ab/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/f3138bf/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    import pdb;pdb.set_trace()
    print '***Using CORL weight'
    target_dir = 'cloud_results/9123ee0/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    """
    target_dir = 'cloud_results/934adde/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)

    print "***Using RSC data, shortest statetype"
    target_dir = 'cloud_results/0559f0c/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    import pdb;
    pdb.set_trace()
    print "***Using RSC data, mc statetype"
    target_dir = 'cloud_results/7fb3872/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    print "***Using HCountGreedy data, shortest statetype"
    target_dir = 'cloud_results/a179000/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5002_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    import pdb;
    pdb.set_trace()

    target_dir = 'cloud_results/2353673/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/9226036/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    _, pidx_times_3 = get_n_nodes(target_dir)

    import pdb;
    pdb.set_trace()

    for n_objs in n_objss:
        print  "****Hcount****"
        if n_objs == 1:
            target_dir = 'cloud_results/934adde//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        else:
            target_dir = 'cloud_results/9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        _, pidx_times_3 = get_n_nodes(target_dir)

        print  "****Ranking function****"
        if n_objs == 1:
            # target_dir = 'cloud_results/c8c5552/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
            target_dir = 'cloud_results/8437aa8/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5001_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
            target_dir = 'cloud_results/a179000/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5002_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
            target_dir = 'cloud_results/934adde/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
            target_dir = 'cloud_results/a0866e8/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5002_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
            target_dir = 'cloud_results/934adde_two_arm_n_objs_pack_1_results/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
            target_dir = 'cloud_results/0559f0c/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        else:
            target_dir = 'cloud_results/9226036/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
        import pdb;
        pdb.set_trace()

        print  "****Ranking+sampler****"
        if n_objs == 1:
            target_dir = 'cloud_results/934adde_two_arm_n_objs_pack_1_results//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
        else:
            target_dir = 'cloud_results/9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
        pidx_nodes_1, pidx_times_1 = get_n_nodes(target_dir)
        # Target idxs determined by comparing RSC and SAGSRANKSAMPLER
        # target_idxs = get_target_idxs(pidx_times_4, pidx_times_1, 4, 'two_arm', 1500)

        print  "****Pure learning****"
        if n_objs == 1:
            target_dir = 'cloud_results/067e376/pure_learning/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
        else:
            target_dir = 'cloud_results/067e376/pure_learning/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
        _, pidx_times_5 = get_n_nodes(target_dir)

        print  "****RSC****"
        if n_objs == 1:
            target_dir = 'cloud_results/934adde_two_arm_n_objs_pack_1_results//irsc/two_arm_mover/n_objs_pack_1/'
        else:
            target_dir = 'cloud_results/9226036/irsc/two_arm_mover/n_objs_pack_4/'
        _, pidx_times_4 = get_n_nodes(target_dir)

        plt.figure()
        plt.boxplot(
            [np.hstack(pidx_times_1.values()), np.hstack(pidx_times_2.values()), np.hstack(pidx_times_3.values()),
             np.hstack(pidx_times_4.values()), np.hstack(pidx_times_5.values())],
            labels=['SAGS\nRankSampler', 'SAGS\nRank', 'SAGS\nHCount', 'IRSC', 'Pure\nLearning'],
            positions=[0, 1, 2, 3, 4],
            whis=(10, 90))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig("./plotters/plots/{}_arm_{}_obj.eps".format('two_arm_mover', n_objs))
        plt.savefig("../IJRR_GTAMP/figures/{}_arm_{}_obj.eps".format('two_arm_mover', n_objs))


def get_target_dirs(root_dir):
    target_dirs = []
    for seed_dir in os.listdir(root_dir):
        target_dir = root_dir + seed_dir
        for epoch_dir in os.listdir(target_dir):
            eval_dir = target_dir + '/' + epoch_dir + '/n_mp_limit_5_n_iter_limit_2000/'
            n_evaled_files = len(os.listdir(eval_dir))
            if n_evaled_files >= 25:
                target_dirs.append(eval_dir)
    return target_dirs


def get_results(target_dirs):
    successes = []
    num_nodes = []
    pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018, 40020,
             40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
    for target_dir in target_dirs:
        if '-1' in target_dir:
            continue
        eval_files = os.listdir(target_dir)
        print target_dir
        seed_n_nodes=[]
        for eval_file in eval_files:
            #if 'gnn_seed_2' not in eval_file:
            #    continue
            eval_file_pidx = int(eval_file.split('pidx_')[1].split('_')[0])
            if eval_file_pidx in pidxs:
                result = pickle.load(open(target_dir + eval_file, 'r'))
                successes.append(result['success'])
                num_nodes.append(result['num_nodes'])
                seed_n_nodes.append(result['num_nodes'])
        #print np.median(num_nodes)
        #print len(seed_n_nodes)

    print len(successes)
    print np.mean(successes)
    print np.median(num_nodes)
    print np.mean(num_nodes)
    import pdb;pdb.set_trace()

def wgangp_vs_wgandi():
    uniform_target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/n_mp_limit_5_n_iter_limit_2000/'
    get_results([uniform_target_dir])

    root_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/using_learned_sampler/1000/'
    wgangp_dir = root_dir + '/wgangp/'
    wgangp_dirs = get_target_dirs(wgangp_dir)
    get_results(wgangp_dirs)

    wgandi_dir = root_dir + '/wgandi/'
    wgandi_dirs = get_target_dirs(wgandi_dir)
    get_results(wgandi_dirs)

def compare_task_guidance():
    qlearned_lm = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/n_mp_limit_5_n_iter_limit_2000/'
    qlearned_mse = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/n_mp_limit_5_n_iter_limit_2000/'
    get_results([qlearned_lm])
    get_results([qlearned_mse])

    qleanred_hcount_mse = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_mse/n_mp_limit_5_n_iter_limit_2000/'
    get_results([qleanred_hcount_mse])
    qleanred_hcount_lm = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/n_mp_limit_5_n_iter_limit_2000/'
    get_results([qleanred_hcount_lm])

    import pdb;
    pdb.set_trace()


def print_results(target_dir):
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print np.mean(successes), np.mean(n_nodes), np.median(n_nodes)


def main():
    n_objss = [1]
    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/n_mp_limit_5_n_iter_limit_2000/'
    print_results(target_dir)

    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/h_without_predicates/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/n_mp_limit_5_n_iter_limit_2000/'
    print_results(target_dir)

    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/hcount_old_number_in_goal_hand_desigend_action_values/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/n_mp_limit_5_n_iter_limit_2000/'
    print_results(target_dir)

    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/n_mp_limit_5_n_iter_limit_2000/'
    print_results(target_dir)

    # wgangp_vs_wgandi()
    # compare_task_guidance()
    # plot_one_arm()
    # plot_two_arm()


if __name__ == '__main__':
    main()
