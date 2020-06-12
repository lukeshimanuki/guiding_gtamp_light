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
    else:
        target_pidxs = range(20000, 20100)
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
        if 'n_objs_pack_4' in target_dir:
            timelimit = 8000
        else:
            timelimit = 2000

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
    print "remaining", len(targets), targets
    # for pidx in targets:
    #    times.append(timelimit)

    print 'n_data', n_data
    print 'success', np.mean(successes), np.sum(successes)
    print 'n nodes', np.mean(n_nodes), np.std(n_nodes) * 1.96 / np.sqrt(n_data)
    print 'times', np.mean(times), np.std(times) * 1.96 / np.sqrt(n_data)
    return pidx_nodes, pidx_times


def get_target_idxs(pidx_nodes_1, pidx_nodes_2, n_objs_pack, domain):
    if 'one_arm' in domain:
        target_idxs = [pidx for pidx in pidx_nodes_1 if
                       abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) >= 150]
    else:
        if n_objs_pack == 1:
            target_idxs = [pidx for pidx in pidx_nodes_1 if
                           abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) >= 20]
        else:
            target_idxs = [pidx for pidx in pidx_nodes_1 if
                           pidx in pidx_nodes_2 and abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) > 60]
    print 'n target idxs', len(target_idxs)
    return target_idxs


def plot_one_arm():
    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_1, pidx_times_1 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)
    target_dir = 'cloud_results/ea42d4e/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_3, pidx_times_3 = get_n_nodes(target_dir)
    target_dir = 'cloud_results//ea42d4e/pure_learning/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_50/'
    pidx_nodes_3, pidx_times_3 = get_n_nodes(target_dir)

    target_dir = 'cloud_results/3c193cf/irsc/one_arm_mover/n_objs_pack_1/'
    pidx_nodes_4, pidx_times_4 = get_n_nodes(target_dir)

    plt.boxplot([np.hstack(pidx_times_1.values()), np.hstack(pidx_times_2.values()), np.hstack(pidx_times_3.values()),
                 np.hstack(pidx_times_4.values())],
                labels=['Rank+Sampler', 'Rank', 'HCount', 'IRSC'], positions=[0, 1, 2, 3])
    target_idxs = get_target_idxs(pidx_times_1, pidx_times_2, 1, 'one_arm')
    import pdb;
    pdb.set_trace()


def plot_two_arm():
    n_objs = 4
    print  "****RSC****"
    if n_objs == 1:
        target_dir = 'test_results/934adde_two_arm_n_objs_pack_1_results//irsc/two_arm_mover/n_objs_pack_1/'
    else:
        target_dir = 'cloud_results/9226036/irsc/two_arm_mover/n_objs_pack_4/'
    _, pidx_times_4 = get_n_nodes(target_dir)

    print  "****Pure learning****"
    if n_objs == 1:
        target_dir = 'cloud_results/067e376/pure_learning/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    else:
        target_dir = 'cloud_results/067e376/pure_learning/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    _, pidx_times_5 = get_n_nodes(target_dir)

    print  "****Ranking function****"
    if n_objs == 1:
        target_dir = 'test_results/934adde_two_arm_n_objs_pack_1_results/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    else:
        target_dir = 'cloud_results/9226036/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)

    print  "****Ranking+sampler****"
    if n_objs == 1:
        target_dir = 'test_results/934adde_two_arm_n_objs_pack_1_results//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    else:
        target_dir = 'cloud_results/9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_1, pidx_times_1 = get_n_nodes(target_dir)

    print  "****Hcount****"
    if n_objs == 1:
        target_dir = 'test_results/934adde_two_arm_n_objs_pack_1_results//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    else:
        target_dir = 'cloud_results/9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    _, pidx_times_3 = get_n_nodes(target_dir)

    plt.boxplot([np.hstack(pidx_times_1.values()), np.hstack(pidx_times_2.values()), np.hstack(pidx_times_3.values()),
                 np.hstack(pidx_times_4.values()), np.hstack(pidx_times_5.values())],
                labels=['Rank+Sampler', 'Rank', 'HCount', 'IRSC', 'PureLearning'], positions=[0, 1, 2, 3, 4])
    import pdb;pdb.set_trace()

def main():
    plot_one_arm()
    #plot_two_arm()



    import pdb;
    pdb.set_trace()



if __name__ == '__main__':
    main()
