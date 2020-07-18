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


def get_s3_path(domain, algorithm, n_objs_pack, commithash, n_iter_limit, sampler_seed, sampler_algo,
                sampler_train_data):
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
                          '{}/' \
                          'sahs_results/' \
                          'domain_two_arm_mover/' \
                          'n_objs_pack_{}/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True/' \
                          'using_learned_sampler/{}/sampler_seed_{}/{}/n_mp_limit_5_n_iter_limit_2000/'.format(
                    commithash[0:7],
                    n_objs_pack,
                    sampler_train_data,
                    sampler_seed,
                    sampler_algo)
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
        # pidxs = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20047, 20051, 20053,
        #         20057, 20061, 20063, 20066, 20069, 20072, 20075, 20084, 20086, 20093, 20094, 20095]
        # pidxs = [20000, 20001, 20005, 20009, 20011, 20023, 20027, 20030, 20035, 20046, 20060, 20061, 20076]
        pidxs1 = [20001, 20002, 20003, 20004, 20008, 20009, 20011, 20019, 20021, 20024, 20035, 20047, 20051,
                  20053, 20057, 20061, 20063, 20066, 20069, 20072, 20075, 20084, 20086, 20093, 20094, 20095]
        pidxs = range(20000, 20100)
        pidxs = [p for p in pidxs if p not in pidxs1]
    else:
        if n_objs_pack == 1:
            pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                     40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
        else:
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


def get_done_seed_and_pidx_pairs(domain, algorithm, n_objs_pack, commithash, n_iter_limit, sampler_seed, sampler_type,
                                 sampler_train_data):
    s3_path = get_s3_path(domain, algorithm, n_objs_pack, commithash, n_iter_limit, sampler_seed, sampler_type,
                          sampler_train_data)
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


def get_running_seed_and_pidx_pairs(algorithm, domain, commithash, n_objs_pack):
    cmd = 'kubectl get jobs $(kubectl get jobs -o=jsonpath=\'{.items[?(@.status.running>0)].metadata.name}\') -n beomjoon'
    results = subprocess.check_output(cmd, shell=True).split('\n')
    running = []
    for result in results:
        if domain not in result:
            continue
        if algorithm not in result:
            continue
        pidx = int(result.split('-')[6])
        seed = int(result.split('-')[7])
        result_commithash = result.split('-')[0]
        result_n_objs_pack = int(result.split('-')[8])
        if result_commithash == commithash[0:7] and result_n_objs_pack == n_objs_pack:
            running.append({'pidx': pidx, 'seed': seed})
    return running


def get_commithash(domain, n_objs_pack, algorithm):
    if domain == 'two-arm-mover':
        if n_objs_pack == 1:
            if algorithm == 'pure-learning':
                commithash = '067e37659b0642bbdb7736ba0ec21151756daddc'
            elif algorithm == 'greedy-learn':
                commithash = '8db0c370a4c8fb4b85d6884f9ce367793f7b7f86'
                commithash = '5474a248214538464b33e70e264e3260cec73a12'
                commithash = 'dd3a4d85b8482cb1c3a06d515f435872da2dc1b4'
                commithash = '9766e58e443182469885723fc251960eb6a7a2ca'
                commithash = '9687b7dd7f692f700bdba438c98147c68b31bb3e'
                commithash = '3c599d703a26fd7ad9ee48fa09f6e0071af7e300'
                commithash = '3e5ce70c9ada5599e40289af7df6398247ccf4db'
                commithash = '6377a4c1bab85c480c5d49c4245f00a7287f7155'
                commithash = '6ad90368c210cf2faaea593d664a2bbaf97f1605'
            else:
                commithash = '1533b3cdb3c77631128662a605c5ced62759ef08'
        elif n_objs_pack == 4:
            if algorithm == 'pure-learning':
                commithash = '067e37659b0642bbdb7736ba0ec21151756daddc'
            elif algorithm == 'rsc':
                commithash = '9226036991cce39a9315f7d9f06ff3d76d47339b'
            elif algorithm == 'greedy-learn':
                commithash = 'c4d77b309ebc1f3105376ae29e2e336980b0790c'
                commithash = '8db0c370a4c8fb4b85d6884f9ce367793f7b7f86'
                commithash = '5474a248214538464b33e70e264e3260cec73a12'
                commithash = 'dd3a4d85b8482cb1c3a06d515f435872da2dc1b4'
            else:
                commithash = '9226036991cce39a9315f7d9f06ff3d76d47339b'
                commithash = '3c961d82f1a3ae9fbe185abc64e45ad92c6c9f90'
        else:
            raise NotImplementedError
    else:
        if 'rsc' == algorithm:
            commithash = '3c193cf4e4e9fbb60c45a9ec3e78290cd07e4548'
        elif 'pure-learning' == algorithm:
            commithash = '3dbf9ca1073de489d7b64e198cd53c6b156e3136'
        else:
            commithash = '240c6f4f00c7e530d442fe4a3d344324eff0d1ea'
            commithash = 'dd3a4d85b8482cb1c3a06d515f435872da2dc1b4'
    print "Commit hash", commithash
    return commithash


def main():
    algos = ['greedy-learn', 'greedy', 'rsc', 'pure-learning']
    algos = ['greedy-learn']
    sampler_seeds = [0,1,2,3]
    sampler_types = ['wgandi']
    n_objs_packs = [1]
    sampler_train_data = 100
    domain = 'two-arm-mover'
    for sampler_train_data in [200]:  # [10, 50, 200]:
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
                                seed_pidx_pairs_running = []  # get_running_seed_and_pidx_pairs(domain, algorithm, commithash, n_objs_pack)
                            else:
                                seed_pidx_pairs_running = []
                            """
                            seed_pidx_pairs_finished = get_done_seed_and_pidx_pairs(domain, algorithm, n_objs_pack,
                                                                                    commithash, n_iter_limit,
                                                                                    sampler_seed, sampler_type,
                                                                                    sampler_train_data)
                            """
                            seed_pidx_pairs_finished = []
                            undone = get_seed_and_pidx_pairs_that_needs_to_run(target_pidxs,
                                                                               seed_pidx_pairs_finished + seed_pidx_pairs_running)
                            print "Remaining runs", len(undone)
                            consecutive_runs = 0
                            for idx, un in enumerate(undone):
                                pidx = un[1]
                                seed = un[0]
                                cmd = 'cat cloud_scripts/{} | ' \
                                      'sed \"s/NAME/{}-{}-{}-{}-{}-{}-{}-{}-{}-{}/\" | ' \
                                      'sed \"s/PIDX/{}/\" | sed \"s/PLANSEED/{}/\" |  ' \
                                      'sed \"s/HOPTION/{}/\" |  ' \
                                      'sed \"s/TIMELIMIT/{}/\" |  ' \
                                      'sed \"s/NOBJS/{}/\" |  ' \
                                      'sed \"s/COMMITHASH/{}/\" |  ' \
                                      'sed \"s/NITERLIMIT/{}/\" |  ' \
                                      'sed \"s/ABSQSEED/{}/\" |  ' \
                                      'sed \"s/SAMPLERSEED/{}/\" |  ' \
                                      'sed \"s/TRAINTYPE/{}/\" |  ' \
                                      'sed \"s/NUMEPISODE/{}/\" |  ' \
                                      'kubectl apply -f - -n beomjoon;'.format(yaml_file, commithash[0:7],
                                                                               algorithm, domain, pidx, seed,
                                                                               n_objs_pack,
                                                                               absq_seed, sampler_type,
                                                                               sampler_train_data,
                                                                               sampler_seed,
                                                                               pidx, seed,
                                                                               hoption, timelimit, n_objs_pack,
                                                                               commithash,
                                                                               n_iter_limit, absq_seed, sampler_seed,
                                                                               sampler_type, sampler_train_data)
                                print idx, cmd
                                os.system(cmd)

                                time.sleep(2)
                                consecutive_runs += 1
                                if consecutive_runs % 100 == 0:
                                    print "Long break"
                                    time.sleep(30)


if __name__ == '__main__':
    main()
