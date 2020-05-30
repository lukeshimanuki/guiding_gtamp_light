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
            yaml_file = 'run_discretized_rsc_one_arm.yaml'
        else:
            yaml_file = 'run_rsc_two_arm.yaml'
    else:
        if 'one' in domain:
            yaml_file = 'run_discretized_uniform_one_arm.yaml'
        else:
            if 'greedy-learn' in algorithm:
                yaml_file = 'run_learned_two_arm.yaml'
            elif 'pure-learning' in algorithm:
                yaml_file = 'run_pure_learning_two_arm.yaml'
            else:
                yaml_file = 'run_uniform_two_arm.yaml'

    return yaml_file


def get_s3_path(domain, algorithm, n_objs_pack):
    if 'rsc' in algorithm:
        if 'one' in domain:
            assert n_objs_pack == 1
            s3_path = 'csail/bkim/guiding-gtamp/test_results/irsc/one_arm_mover/n_objs_pack_1'
        else:
            s3_path = 'csail/bkim/guiding-gtamp/test_results/9226036/irsc/two_arm_mover/n_objs_pack_{}'.format(
                n_objs_pack)
    else:
        if 'one' in domain:
            s3_path = 'csail/bkim/guiding-gtamp/test_results/' \
                      'f0cf459f5cb177eaacd17a0fa9b6b89caa96dbe3/' \
                      'sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/' \
                      'qlearned_hcount_old_number_in_goal/' \
                      'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                      'n_mp_limit_5_n_iter_limit_500/'
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
                          '9226036/' \
                          'pure_learning/' \
                          'domain_two_arm_mover/' \
                          'n_objs_pack_{}/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'.format(n_objs_pack)
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
                              '9226036/' \
                              'sahs_results/uses_rrt/domain_two_arm_mover/' \
                              'n_objs_pack_{}/' \
                              'qlearned_hcount_old_number_in_goal/' \
                              'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                              'n_mp_limit_5_n_iter_limit_2000/'.format(n_objs_pack)
            else:
                raise NotImplementedError
    return s3_path


def get_target_pidxs(domain):
    if 'one' in domain:
        pidxs = range(20000, 20030)
    else:
        # previous run
        pidxs = range(40000, 40100)
        pidxs.remove(40034)
        pidxs.remove(40079)
        pidxs.remove(40060)  # running this one runs out of memory - dunno why

        # new runs
        pidxs = range(40100, 40400)
    return pidxs


def get_done_seed_and_pidx_pairs(domain, algorithm, n_objs_pack):
    s3_path = get_s3_path(domain, algorithm, n_objs_pack)
    try:
        result = subprocess.check_output('mc ls {}'.format(s3_path), shell=True)
    except subprocess.CalledProcessError:
        return []
    if 'rsc' in algorithm:
        runs_finished = re.findall('seed_[0-9]*_pidx_[0-9]*', result)
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[-1], 'seed': fin.split('_')[1]} for fin in runs_finished]
    else:
        runs_finished = re.findall('pidx_[0-9]*_planner_seed_[0-9]*', result)
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[1], 'seed': fin.split('_')[-1]} for fin in runs_finished]
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

        pidx = int(result.split('-')[-2])
        seed = int(result.split('-')[-1].split(' ')[0])
        running.append({'pidx': pidx, 'seed': seed})
    return running


def main():
    for algorithm in ['pure-learning']:
        for n_objs_pack in [1,4]:
            domain = 'two-arm-mover'
            timelimit = 2000 * n_objs_pack

            if 'hcount' in algorithm:
                hoption = 'hcount_old_number_in_goal'
            else:
                hoption = 'qlearned_hcount_old_number_in_goal'

            target_pidxs = get_target_pidxs(domain)
            yaml_file = get_yaml_file_name(algorithm, domain)

            seed_pidx_pairs_finished = get_done_seed_and_pidx_pairs(domain, algorithm, n_objs_pack)
            seed_pidx_pairs_running = get_running_seed_and_pidx_pairs(domain, algorithm)
            # seed_pidx_pairs_finished = []
            undone = get_seed_and_pidx_pairs_that_needs_to_run(target_pidxs,
                                                               seed_pidx_pairs_finished + seed_pidx_pairs_running)

            print "Remaining runs", len(undone)
            consecutive_runs = 0
            for idx, un in enumerate(undone):
                pidx = un[1]
                seed = un[0]

                cmd = 'cat cloud_scripts/{} | ' \
                      'sed \"s/NAME/{}-{}-{}-{}/\" | ' \
                      'sed \"s/PIDX/{}/\" | sed \"s/PLANSEED/{}/\" |  ' \
                      'sed \"s/HOPTION/{}/\" |  ' \
                      'sed \"s/TIMELIMIT/{}/\" |  ' \
                      'sed \"s/NOBJS/{}/\" |  ' \
                      'kubectl apply -f - -n beomjoon;'.format(yaml_file, algorithm, domain, pidx, seed, pidx, seed,
                                                               hoption, timelimit, n_objs_pack)
                print idx, cmd
                os.system(cmd)
                time.sleep(2)
                consecutive_runs += 1
                if consecutive_runs % 100 == 0:
                    print "Long break"
                    time.sleep(30)


if __name__ == '__main__':
    main()
