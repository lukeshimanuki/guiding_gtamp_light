import pickle
import os
import numpy as np
from matplotlib import pyplot as plt


def get_n_nodes(target_dir):
    print target_dir
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
            if 'pure_learning' not in target_dir:
                assert '934adde' in target_dir, 'n objs pack for two arm must use commit 934adde'
            target_pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                            40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
        else:
            target_pidxs = [40321, 40203, 40338, 40089, 40220, 40223, 40352, 40357, 40380, 40253, 40331, 40260, 40353,
                            40393, 40272, 40148, 40149, 40283, 40162, 40292, 40295, 40185, 40314, 40060]
            target_pidxs = range(40000,40300)
            target_pidxs = [40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016, 40017, 40019, 40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036, 40037, 40038, 40039, 40040, 40044, 40045, 40046, 40049, 40050, 40053, 40054, 40055, 40056, 40057, 40060, 40061, 40062, 40064, 40066, 40069, 40073, 40074, 40076, 40078, 40083, 40084, 40085, 40086, 40089, 40093, 40094, 40095, 40097, 40099, 40100, 40104, 40105, 40108, 40109, 40110, 40112, 40117, 40119, 40120, 40124, 40128, 40133, 40135, 40138, 40141, 40142, 40143, 40144, 40146, 40147, 40148, 40150, 40151, 40154, 40159, 40163, 40165, 40166, 40167, 40170, 40174, 40175, 40176, 40177, 40178, 40180, 40182, 40184, 40185, 40186, 40187, 40190, 40191, 40196, 40198, 40202, 40205, 40206, 40208, 40209, 40210, 40212, 40217, 40218, 40220, 40221, 40223, 40227, 40228, 40230, 40232, 40233, 40234, 40238, 40240, 40243, 40244, 40246, 40247, 40249, 40251, 40252, 40258, 40262, 40264, 40265, 40267, 40268, 40269, 40271, 40274, 40275, 40276, 40281, 40283, 40284, 40285, 40286, 40287, 40288, 40290, 40291, 40292, 40294, 40295, 40296, 40297, 40298]
            target_pidxs =[40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016, 40017, 40019, 40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036, 40037, 40038, 40039, 40040, 40044, 40045, 40046, 40049, 40050, 40053, 40054, 40055, 40056, 40057, 40060, 40061, 40062, 40064, 40066, 40069, 40073, 40074, 40076, 40078, 40083, 40084, 40085, 40086, 40089, 40093, 40094, 40095, 40097, 40099, 40100, 40104, 40105, 40108, 40109, 40110, 40112, 40117, 40119, 40120, 40124, 40128, 40133, 40135, 40138, 40141, 40142, 40143, 40144, 40146, 40147, 40148, 40150, 40151, 40154, 40159, 40163, 40165, 40166, 40167, 40170, 40174, 40175, 40176, 40177, 40178, 40180, 40182, 40184, 40185, 40186, 40187, 40190, 40191, 40196, 40198, 40202, 40205, 40206, 40208, 40209, 40210, 40212, 40217, 40218, 40220, 40221, 40223, 40227, 40228, 40230, 40232, 40233, 40234, 40238, 40240, 40243, 40244, 40246, 40247, 40249, 40251, 40252, 40258, 40262, 40264, 40265, 40267, 40268, 40269, 40271, 40274, 40275, 40276, 40281, 40283, 40284, 40285, 40286, 40287, 40288, 40290, 40291, 40292, 40294, 40295, 40296, 40297, 40298]
            #target_pidxs = target_pidxs[0:25]
    else:
        target_pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20015, 20019, 20021, 20023, 20024, 20035,
                        20046, 20047, 20051, 20053, 20056, 20057, 20061, 20063, 20066, 20067, 20069, 20072, 20075,
                        20080, 20083, 20084, 20086, 20093, 20094, 20095]
        target_pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20047, 20051,
                        20053, 20057, 20061, 20063, 20066, 20069, 20072, 20075, 20084, 20086, 20093, 20094, 20095]


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

    print "number of target pidxs", len(target_pidxs)
    successes = []
    times = []
    pidx_times = {}
    pidx_nodes = {}
    for filename in test_files:
        if 'pkl' not in filename:
            print 'File skipped', filename
            continue
        if 'gnn_seed' in filename:
            absqseed = int(filename.split('gnn_seed_')[-1].split('.pkl')[0])
            if absqseed != 0:
                continue

        if 'rsc' in target_dir:
            pidx = int(filename.split('pidx_')[1].split('.pkl')[0])
        else:
            pidx = int(filename.split('pidx_')[1].split('_')[0])

        seed = int(filename.split('seed_')[1].split('_')[0])
        if not pidx in target_pidxs:
            continue
        fin = pickle.load(open(target_dir + filename, 'r'))
        targets.remove((pidx, seed))

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

        times.append(timetaken)
        if pidx in pidx_times:
            pidx_nodes[pidx].append(n_node)
            pidx_times[pidx].append(timetaken)
        else:
            pidx_nodes[pidx] = [n_node]
            pidx_times[pidx] = [timetaken]

    n_data = len(n_nodes)
    print "*****REMAINING****", len(targets)
    for pidx in targets:
        pidx_times[pidx_times.keys()[0]].append(timelimit)
        successes.append(False)

    print 'n_data', n_data
    print 'success', np.mean(successes), np.sum(successes)
    print 'n nodes', np.mean(n_nodes), np.std(n_nodes) * 1.96 / np.sqrt(n_data)
    print 'times', np.mean(times), np.std(times) * 1.96 / np.sqrt(n_data)
    return pidx_nodes, pidx_times


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
                           pidx in pidx_nodes_2 and abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) > threshold]
    print 'n target idxs', len(target_idxs)
    return target_idxs


def get_target_dir(algo, n_objs_pack, domain):
    if 'one_arm' in domain:
        if 'greedy' in algo:
            target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/' \
                         'domain_{}/' \
                         'n_objs_pack_1/' \
                         '{}' \
                         'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                         '{}/' \
                         'n_mp_limit_5_n_iter_limit_50/'.format(domain, heuristic_fcn, learned_sampler_folder)


def plot_one_arm():
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
    n_objss = [1, 4]
    for n_objs in n_objss:
        print  "****RSC****"
        if n_objs == 1:
            target_dir = 'cloud_results/934adde_two_arm_n_objs_pack_1_results//irsc/two_arm_mover/n_objs_pack_1/'
        else:
            target_dir = 'cloud_results/9226036/irsc/two_arm_mover/n_objs_pack_4/'
        _, pidx_times_4 = get_n_nodes(target_dir)

        print  "****Ranking function****"
        if n_objs == 1:
            target_dir = 'cloud_results/934adde_two_arm_n_objs_pack_1_results/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        else:
            target_dir = 'cloud_results/9226036/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)

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

        print  "****Hcount****"
        if n_objs == 1:
            target_dir = 'cloud_results/934adde_two_arm_n_objs_pack_1_results//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        else:
            target_dir = 'cloud_results/9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
        _, pidx_times_3 = get_n_nodes(target_dir)

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


def main():
    plot_one_arm()
    #plot_two_arm()


if __name__ == '__main__':
    main()
