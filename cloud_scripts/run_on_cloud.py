import os
import subprocess
import re
import time


def filter_runs_already_done(undone, dir):
    # filter things already one
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            seed = int(file.split('_')[1])
            pidx = int(file.split('_')[3].split('.pkl')[0])
            undone.remove((seed, pidx))
    return undone


def get_yaml_file_name(algorithm, domain):
    if 'rsc' in algorithm:
        if 'one' in domain:
            yaml_file = 'run_one_arm_rsc.yaml'
        else:
            yaml_file = 'run_rsc_two_arm.yaml'
    else:
        if 'one' in domain:
            if 'greedy-learn' in algorithm:
                yaml_file = 'run_learned_one_arm.yaml'
            elif 'greedy' == algorithm or 'greedy-hcount' == algorithm:
                yaml_file = 'run_one_arm.yaml'
            elif 'pure-learning' == algorithm:
                yaml_file = 'run_pure_learning_one_arm.yaml'
        else:
            if 'greedy-learn' in algorithm:
                yaml_file = 'run_learned_two_arm.yaml'
            elif 'pure-learning' in algorithm:
                yaml_file = 'run_pure_learning_two_arm.yaml'
            else:
                yaml_file = 'run_uniform_two_arm.yaml'

    return yaml_file


def get_s3_path(domain, algorithm, n_objs_pack, commithash, n_iter_limit):
    if 'rsc' in algorithm:
        if 'one' in domain:
            assert n_objs_pack == 1
            s3_path = 'csail/bkim/guiding-gtamp/test_results/{}/irsc/one_arm_mover/n_objs_pack_1'.format(
                commithash[0:7])
        else:
            s3_path = 'csail/bkim/guiding-gtamp/test_results/{}/irsc/two_arm_mover/n_objs_pack_{}'.format(
                commithash[:7], n_objs_pack)
    else:
        if 'one' in domain:
            if 'greedy-learn' == algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                          '{}/' \
                          'sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'using_learned_sampler/' \
                          'n_mp_limit_5_n_iter_limit_{}/'.format(commithash[0:7], n_iter_limit)
            elif 'greedy' == algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                          '{}/' \
                          'sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5002_mse_weight_0.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'n_mp_limit_5_n_iter_limit_{}/'.format(commithash[0:7], n_iter_limit)
            elif 'greedy-hcount' == algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                          '{}/' \
                          'sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/' \
                          'hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'n_mp_limit_5_n_iter_limit_{}/'.format(commithash[0:7], n_iter_limit)
            elif 'pure-learning' in algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                          '{}/' \
                          'pure_learning/' \
                          'domain_one_arm_mover/' \
                          'n_objs_pack_1/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'using_learned_sampler/n_mp_limit_5_n_iter_limit_{}/'.format(commithash[0:7], n_iter_limit)
        else:
            if 'greedy-learn' in algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                          '9226036/' \
                          'sahs_results/uses_rrt/' \
                          'domain_two_arm_mover/' \
                          'n_objs_pack_{}/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'.format(n_objs_pack)
            elif 'pure-learning' in algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                          '{}/' \
                          'pure_learning/' \
                          'domain_two_arm_mover/' \
                          'n_objs_pack_{}/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'.format(commithash[0:7], n_objs_pack)
            elif 'greedy' == algorithm:
                if 'hcount' in algorithm:
                    s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                              '9226036/' \
                              'sahs_results/uses_rrt/domain_two_arm_mover/' \
                              'n_objs_pack_{}/' \
                              'hcount_old_number_in_goal/' \
                              'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                              'n_mp_limit_5_n_iter_limit_2000/'.format(n_objs_pack)
                else:
                    s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                              '{}/' \
                              'sahs_results/uses_rrt/domain_two_arm_mover/' \
                              'n_objs_pack_{}/' \
                              'qlearned_hcount_old_number_in_goal/' \
                              'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                              'n_mp_limit_5_n_iter_limit_2000/'.format(commithash[0:7], n_objs_pack)
            else:
                raise NotImplementedError
    return s3_path


def get_target_pidxs(domain, n_objs_pack):
    if 'one-arm' in domain:
        pidxs = range(20000, 20100)
        # pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20047, 20051, 20053,
        #         20057, 20061, 20063, 20066, 20069, 20072, 20075, 20084, 20086, 20093, 20094, 20095]
        # pidxs = [20000, 20001, 20005, 20009, 20011, 20023, 20027, 20030, 20035, 20046, 20060, 20061, 20076]
        pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20047, 20051,
                 20053, 20057, 20061, 20063, 20066, 20069, 20072, 20075, 20084, 20086, 20093, 20094, 20095]
    else:
        if n_objs_pack == 1:
            pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                     40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
            # pidxs = [p for p in range(40000, 40100) if p not in pidxs]
        else:
            pidxs = [40321, 40203, 40338, 40089, 40220, 40223, 40352, 40357, 40380, 40253, 40331, 40260, 40353,
                     40393, 40272, 40148, 40149, 40283, 40162, 40292, 40295, 40185, 40314, 40060]
            pidxs = [40321, 40203, 40338, 40089, 40220, 40223, 40352, 40357, 40380, 40253, 40331, 40260, 40353,
                     40393, 40272, 40148, 40149, 40283, 40162, 40292, 40295, 40185, 40314, 40060]
            pidxs = [40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016, 40017, 40019,
                     40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036, 40037, 40038, 40039,
                     40040, 40044, 40045, 40046, 40049, 40050, 40053, 40054, 40055, 40056, 40057, 40060, 40061, 40062,
                     40064, 40066, 40069, 40073, 40074, 40076, 40078, 40083, 40084, 40085, 40086, 40089, 40093, 40094,
                     40095, 40097, 40099, 40100, 40104, 40105, 40108, 40109, 40110, 40112, 40117, 40119, 40120, 40124,
                     40128, 40133, 40135, 40138, 40141, 40142, 40143, 40144, 40146, 40147, 40148, 40150, 40151, 40154,
                     40159, 40163, 40165, 40166, 40167, 40170, 40174, 40175, 40176, 40177, 40178, 40180, 40182, 40184,
                     40185, 40186, 40187, 40190, 40191, 40196, 40198, 40202, 40205, 40206, 40208, 40209, 40210, 40212,
                     40217, 40218, 40220, 40221, 40223, 40227, 40228, 40230, 40232, 40233, 40234, 40238, 40240, 40243,
                     40244, 40246, 40247, 40249, 40251, 40252, 40258, 40262, 40264, 40265, 40267, 40268, 40269, 40271,
                     40274, 40275, 40276, 40281, 40283, 40284, 40285, 40286, 40287, 40288, 40290, 40291, 40292, 40294,
                     40295, 40296, 40297, 40298]
            target_pidxs = [40000, 40002, 40003, 40004, 40005, 40007, 40008, 40010, 40012, 40014, 40015, 40016, 40017,
                            40019, 40021, 40023, 40024, 40025, 40026, 40028, 40030, 40031, 40033, 40035, 40036, 40037,
                            40038, 40039, 40040, 40044, 40045, 40046, 40049, 40050, 40053, 40054, 40055, 40056, 40057,
                            40060, 40061, 40062, 40064, 40066, 40069, 40073, 40074, 40076, 40078, 40083, 40084, 40085,
                            40086, 40089, 40093, 40094, 40095, 40097, 40099, 40100, 40104, 40105, 40108, 40109, 40110,
                            40112, 40117, 40119, 40120, 40124, 40128, 40133, 40135, 40138, 40141, 40142, 40143, 40144,
                            40146, 40147, 40148, 40150, 40151, 40154, 40159, 40163, 40165, 40166, 40167, 40170, 40174,
                            40175, 40176, 40177, 40178, 40180, 40182, 40184, 40185, 40186, 40187, 40190, 40191, 40196,
                            40198, 40202, 40205, 40206, 40208, 40209, 40210, 40212, 40217, 40218, 40220, 40221, 40223,
                            40227, 40228, 40230, 40232, 40233, 40234, 40238, 40240, 40243, 40244, 40246, 40247, 40249,
                            40251, 40252, 40258, 40262, 40264, 40265, 40267, 40268, 40269, 40271, 40274, 40275, 40276,
                            40281, 40283, 40284, 40285, 40286, 40287, 40288, 40290, 40291, 40292, 40294, 40295, 40296,
                            40297, 40298]
            pidxs = target_pidxs[0:25]
    return pidxs


def get_done_seed_and_pidx_pairs(domain, algorithm, n_objs_pack, commithash, n_iter_limit):
    s3_path = get_s3_path(domain, algorithm, n_objs_pack, commithash, n_iter_limit)
    try:
        result = subprocess.check_output('mc ls {}'.format(s3_path), shell=True)
    except subprocess.CalledProcessError:
        return []
    if 'rsc' in algorithm:
        runs_finished = re.findall('seed_[0-9]*_pidx_[0-9]*', result)
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[-1], 'seed': fin.split('_')[1]} for fin in runs_finished]
    else:
        runs_finished = re.findall('pidx_[0-9]*_planner_seed_[0-9]*_gnn_seed_[0-9]*', result)
        if 'sampling_strategy' in runs_finished[0]:
            import pdb;
            pdb.set_trace()
            seed_pidx_pairs_finished = [{'pidx': fin.split('_')[1], 'seed': fin.split('_')[-1]} for fin in
                                        runs_finished]
        else:
            seed_pidx_pairs_finished = [{'pidx': fin.split('_')[1], 'seed': fin.split('_')[4]} for fin in runs_finished]

    return seed_pidx_pairs_finished


def get_seed_and_pidx_pairs_that_needs_to_run(pidxs, seed_pidx_pairs_finished):
    undone = [(seed, pidx) for seed in range(5) for pidx in pidxs]
    done = [(int(k['seed']), int(k['pidx'])) for k in seed_pidx_pairs_finished]
    undone = [un for un in undone if un not in done]
    return undone


def get_running_seed_and_pidx_pairs(algorithm, domain):
    cmd = 'kubectl get jobs $(kubectl get jobs -o=jsonpath=\'{.items[?(@.status.running>0)].metadata.name}\') -n beomjoon'
    results = subprocess.check_output(cmd, shell=True).split('\n')
    running = []
    for result in results:
        if domain not in result:
            continue
        if algorithm not in result:
            continue
        pidx = int(result.split('-')[5])
        seed = int(result.split('-')[6].split(' ')[0])
        running.append({'pidx': pidx, 'seed': seed})
    return running


def get_commithash(domain, n_objs_pack, algorithm):
    if domain == 'two-arm-mover':
        if n_objs_pack == 1:
            if algorithm == 'pure-learning':
                commithash = '067e37659b0642bbdb7736ba0ec21151756daddc'
            else:
                commithash = 'c8c5552beedb9eff538ce615c0fc972eea033b7a'
                commithash = '719916fb8bd41db4275e6b61c0de2a2e43bb92cc'
                commithash = '8437aa88a5377bb662b8d3ea6e9bb7766ff924a2'
                commithash = 'a51467dfa18adb205cef2663a992a127500b7ca9'
                # commithash = 'a0866e889f51628af202006d17435edc487b5e72'
                commithash = 'a179000bc1d4901246e55f1a25b735e04bf3a513'
                commithash = '0559f0c889998dca9bf0eb23e1f2a9d4fadca05d'
                # commithash = '7fb3872302918060efe5c42ed0e049ca02a6f483'
                # commithash = '3cd9ad084ca61ca7c7540ce530ccb35c300ef6a4'
                commithash = '9123ee00418fb112e47086a6c2f16c578c3fe57a'
                commithash = '1ac72ff240dd41f14786db797517157aba40a974'
                commithash = '1ed923194f611ff79196c92da99a7e4b1eb7b694'
                commithash = 'f3138bfe8bc003df7f60801449dbc2ca08a7b0e8'
                commithash = 'c40c7abe2b0f13a1c4f0af76d555f151c31e907a'
                commithash = '058d10290cde087d2d355557ebe2c2d1549013d9'
                commithash = '16653e723158d294bd9cfb37ab7dec84bb6e04a4'
                commithash = 'ebbac7b60a74d03d92f6086c327638f9ff829b3e'
                commithash = '1533b3cdb3c77631128662a605c5ced62759ef08'
                commithash = 'c4d77b309ebc1f3105376ae29e2e336980b0790c'
        elif n_objs_pack == 4:
            if algorithm == 'pure-learning':
                commithash = '067e37659b0642bbdb7736ba0ec21151756daddc'
            elif algorithm == 'rsc':
                commithash = '9226036991cce39a9315f7d9f06ff3d76d47339b'
            else:
                commithash = '9226036991cce39a9315f7d9f06ff3d76d47339b'
                commithash = '2353673088ac3b34458cc5c8a802c0336b8c6473'
                commithash = '6838dd3f58640d96a00ab5835e2125f3b1072bf5'
        else:
            raise NotImplementedError
    else:
        if 'rsc' == algorithm:
            commithash = '3c193cf4e4e9fbb60c45a9ec3e78290cd07e4548'
        elif 'pure-learning' == algorithm:
            commithash = '3dbf9ca1073de489d7b64e198cd53c6b156e3136'
        else:
            # commithash = '2306c1823e4c197806bb948f5934c043fde7ff05'
            commithash = 'ea42d4ee62c93857d6a2ed0962420f4088344832'
            commithash = '0559f0c889998dca9bf0eb23e1f2a9d4fadca05d'
            commithash = '2353673088ac3b34458cc5c8a802c0336b8c6473'
    print "Commit hash", commithash
    return commithash


def main():
    algos = ['greedy-learn', 'greedy', 'rsc', 'pure-learning']
    sampler_seeds = [0]
    sampler_types = ['wgandi', 'wgangp']
    n_objs_packs = [1]
    domain = 'two-arm-mover'
    for algorithm in algos:
        if algorithm == 'rsc':
            absqseeds = [0]
        else:
            absqseeds = [0]
        for sampler_type in sampler_types:
            for sampler_seed in sampler_seeds:
                for absq_seed in absqseeds:
                    for n_objs_pack in n_objs_packs:
                        if 'hcount' in algorithm:
                            hoption = 'hcount_old_number_in_goal'
                        else:
                            hoption = 'qlearned_hcount_old_number_in_goal'

                        if domain == 'one-arm-mover':
                            timelimit = 1000 * n_objs_pack
                            n_iter_limit = 50
                        else:
                            timelimit = 2000 * n_objs_pack
                            n_iter_limit = 2000

                        target_pidxs = get_target_pidxs(domain, n_objs_pack)
                        yaml_file = get_yaml_file_name(algorithm, domain)
                        commithash = get_commithash(domain, n_objs_pack, algorithm)
                        if algorithm == 'greedy-learn':
                            seed_pidx_pairs_running = []  # get_running_seed_and_pidx_pairs(domain, algorithm)
                        else:
                            seed_pidx_pairs_running = []
                        seed_pidx_pairs_finished = []  # get_done_seed_and_pidx_pairs(domain, algorithm, n_objs_pack, commithash, n_iter_limit)
                        undone = get_seed_and_pidx_pairs_that_needs_to_run(target_pidxs,
                                                                           seed_pidx_pairs_finished + seed_pidx_pairs_running)
                        print "Remaining runs", len(undone)
                        consecutive_runs = 0
                        for idx, un in enumerate(undone):
                            pidx = un[1]
                            seed = un[0]
                            cmd = 'cat cloud_scripts/{} | ' \
                                  'sed \"s/NAME/{}-{}-{}-{}-{}-n-objs-{}-absqseed-{}/\" | ' \
                                  'sed \"s/PIDX/{}/\" | sed \"s/PLANSEED/{}/\" |  ' \
                                  'sed \"s/HOPTION/{}/\" |  ' \
                                  'sed \"s/TIMELIMIT/{}/\" |  ' \
                                  'sed \"s/NOBJS/{}/\" |  ' \
                                  'sed \"s/COMMITHASH/{}/\" |  ' \
                                  'sed \"s/NITERLIMIT/{}/\" |  ' \
                                  'sed \"s/ABSQSEED/{}/\" |  ' \
                                  'sed \"s/SAMPLERSEED/{}/\" |  ' \
                                  'sed \"s/TRAINTYPE/{}/\" |  ' \
                                  'kubectl apply -f - -n beomjoon;'.format(yaml_file, commithash[0:7],
                                                                           algorithm, domain, pidx, seed, n_objs_pack,
                                                                           absq_seed,
                                                                           pidx, seed,
                                                                           hoption, timelimit, n_objs_pack, commithash,
                                                                           n_iter_limit, absq_seed, sampler_seed,
                                                                           sampler_type)
                            print idx, cmd
                            os.system(cmd)

                            time.sleep(2)
                            consecutive_runs += 1
                            if consecutive_runs % 100 == 0:
                                print "Long break"
                                time.sleep(30)


if __name__ == '__main__':
    main()
