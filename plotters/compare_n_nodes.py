import pickle
import os
import numpy as np


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
            target_pidxs = range(40000, 40400)
            target_pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                            40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
        else:
            target_pidxs = range(40000, 40400)
            target_pidxs.remove(40034)
            target_pidxs.remove(40079)
            target_pidxs.remove(40060)  # running this one runs out of memory - dunno why
            # target_pidxs = [40067, 40068, 40072, 40073, 40074, 40076, 40087, 40088, 40089, 40092, 40093, 40095, 40097,
            #                40001, 40002,
            #                40006, 40007, 40009, 40011, 40012, 40013, 40015, 40017, 40019, 40020, 40021, 40025, 40029,
            #                40033, 40039, 40042, 40045, 40052, 40053, 40056, 40059, 40062]
            target_pidxs = range(40000, 40400)
            target_pidxs = [40001, 40002, 40006, 40007, 40009, 40011, 40012, 40013, 40015, 40017, 40019, 40020, 40021,
                            40025, 40029, 40033, 40039, 40042, 40045, 40052, 40053, 40056, 40059, 40060, 40062, 40067,
                            40068, 40072, 40073, 40074, 40076, 40087, 40088, 40089, 40092, 40093, 40095, 40097, 40104,
                            40105, 40106, 40108, 40110, 40113, 40114, 40115, 40116, 40119, 40121, 40123, 40124, 40126,
                            40130, 40132, 40134, 40136, 40139, 40141, 40144, 40148, 40149, 40150, 40151, 40154, 40159,
                            40160, 40162, 40167, 40170, 40172, 40173, 40174, 40176, 40177, 40179, 40180, 40181, 40183,
                            40184, 40185, 40187, 40188, 40189, 40193, 40195, 40197, 40198, 40200, 40201, 40203, 40204,
                            40205, 40207, 40210, 40212, 40213, 40216, 40217, 40218, 40220, 40223, 40238, 40240, 40243,
                            40245, 40253, 40256, 40258, 40259, 40260, 40263, 40266, 40267, 40271, 40272, 40274, 40275,
                            40277, 40281, 40283, 40285, 40292, 40293, 40295, 40297, 40298, 40301, 40302, 40303, 40304,
                            40306, 40308, 40311, 40312, 40313, 40314, 40320, 40321, 40331, 40335, 40338, 40339, 40343,
                            40344, 40349, 40352, 40353, 40356, 40357, 40363, 40367, 40368, 40374, 40376, 40378, 40380,
                            40381, 40384, 40386, 40390, 40391, 40392, 40393]
            target_idxs = [40001, 40006, 40007, 40009, 40011, 40013, 40020, 40021, 40029, 40033, 40039, 40042, 40045,
                           40052, 40059,
                           40060, 40067, 40068, 40073, 40074, 40076, 40087, 40089, 40092, 40093, 40095, 40097, 40106,
                           40108, 40110,
                           40114, 40115, 40116, 40119, 40121, 40123, 40124, 40126, 40130, 40132, 40134, 40136, 40139,
                           40144, 40148,
                           40149, 40150, 40154, 40159, 40162, 40170, 40172, 40173, 40174, 40176, 40177, 40179, 40180,
                           40183, 40184,
                           40185, 40187, 40189, 40193, 40195, 40197, 40198, 40200, 40201, 40203, 40204, 40207, 40212,
                           40216, 40217,
                           40218, 40220, 40223, 40238, 40243, 40245, 40253, 40256, 40259, 40260, 40267, 40271, 40272,
                           40274, 40275,
                           40281, 40283, 40292, 40293, 40295, 40301, 40303, 40304, 40308, 40311, 40312, 40313, 40314,
                           40320, 40321,
                           40331, 40335, 40338, 40343, 40344, 40349, 40352, 40353, 40357, 40367, 40368, 40374, 40380,
                           40381, 40386,
                           40390, 40391, 40392, 40393]
            target_pidxs = [40007, 40013, 40060, 40068, 40089, 40092, 40097, 40106, 40114, 40132, 40139, 40148, 40149, 40150, 40159, 40162, 40174, 40177, 40185, 40198, 40203, 40204, 40218, 40220, 40223, 40253, 40260, 40267, 40272, 40281, 40283, 40292, 40295, 40314, 40320, 40321, 40331, 40338, 40349, 40352, 40353, 40357, 40380, 40392, 40393]
            target_pidxs = [40320, 40321, 40068, 40198, 40203, 40338, 40089, 40349, 40223, 40352, 40097, 40357, 40220, 40114, 40331, 40253, 40392, 40393, 40139, 40272, 40353, 40148, 40149, 40283, 40162, 40292, 40295, 40260, 40380, 40174, 40185, 40314, 40060]
            target_pidxs = [40321, 40203, 40338, 40089, 40220, 40223, 40352, 40357, 40380, 40253, 40331, 40260, 40353, 40393, 40272, 40148, 40149, 40283, 40162, 40292, 40295, 40185, 40314, 40060]
    else:
        target_pidxs = range(20000, 20030)

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
        if 'n_feasibility_checks' in fin:
            n_ik = fin['n_feasibility_checks']['ik']
            n_mp = fin['n_feasibility_checks']['mp']
            # n_infeasible_mp = fin['n_feasibility_checks']['infeasible_mp']
        else:
            n_ik = fin['search_time_to_reward'][-1][2]
        if 'search_time_to_reward' in fin:
            if True:
                where_is_three = np.where(np.array(fin['search_time_to_reward'])[:, -1] == 3)[0][0]
                n_steps_after_three = len(np.array(fin['search_time_to_reward'])[where_is_three:, -1])
        # print filename, n_ik
        n_iks.append(n_ik)
        n_nodes.append(n_node)
        n_mps.append(n_mp)
        if 'n_objs_pack_4' in target_dir:
            timelimit = 8000
        else:
            timelimit = 2000

        if fin['tottime'] >= timelimit:
            successes.append(False)
            times.append(timelimit)
        else:
            successes.append(fin['success'])
            times.append(fin['tottime'])

        if pidx in pidx_times:
            pidx_nodes[pidx].append(n_node)
            pidx_times[pidx].append(fin['tottime'])
        else:
            pidx_nodes[pidx] = [n_node]
            pidx_times[pidx] = [fin['tottime']]

    n_data = len(n_nodes)
    print "remaining", len(targets)
    print 'n_data', n_data
    print 'success', np.mean(successes)
    print 'n nodes', np.mean(n_nodes), np.std(n_nodes) * 1.96 / np.sqrt(n_data)
    print 'times', np.mean(times), np.std(times) * 1.96 / np.sqrt(n_data)
    return pidx_nodes, pidx_times


def main():
    target_dir = '934adde_two_arm_n_objs_pack_1_results/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    target_dir = '9226036/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_1, pidx_times_1 = get_n_nodes(target_dir)

    target_dir = '934adde_two_arm_n_objs_pack_1_results//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    target_dir = '9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    pidx_nodes_2, pidx_times_2 = get_n_nodes(target_dir)

    get_target_idxs = True
    if get_target_idxs:
        if 'n_objs_pack_1' in target_dir:
            target_idxs = [pidx for pidx in pidx_times_1 if
                           abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_1[pidx])) >= 20]
        else:
            target_idxs = [pidx for pidx in pidx_nodes_1 if pidx in pidx_nodes_2 and abs(np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx])) > 60]
        print len(target_idxs)
    import pdb;
    pdb.set_trace()
    node_diff = [np.mean(pidx_nodes_1[pidx]) - np.mean(pidx_nodes_2[pidx]) for pidx in pidx_times_1]
    time_diff = [np.mean(pidx_times_1[pidx]) - np.mean(pidx_times_2[pidx]) for pidx in pidx_times_1]
    target_dir = '934adde_two_arm_n_objs_pack_1_results//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_1/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    target_dir = '9226036//sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    pidx_times_3 = get_n_nodes(target_dir)

    target_dir = '934adde_two_arm_n_objs_pack_1_results//irsc/two_arm_mover/n_objs_pack_1/'
    target_dir = '9226036//irsc/two_arm_mover/n_objs_pack_4/'
    pidx_times_4 = get_n_nodes(target_dir)


if __name__ == '__main__':
    main()
