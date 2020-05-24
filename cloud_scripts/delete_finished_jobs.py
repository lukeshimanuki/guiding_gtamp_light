import os
import subprocess
import re


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
    if algorithm == 'irsc':
        if 'one' in domain:
            yaml_file = 'run_discretized_rsc_one_arm.yaml'
        else:
            raise NotImplementedError
    else:
        if 'one' in domain:
            yaml_file = 'run_discretized_uniform_one_arm.yaml'
        else:
            if 'learn' in algorithm:
                yaml_file = 'run_learned_two_arm.yaml'
            else:
                yaml_file = 'run_uniform_two_arm.yaml'
    return yaml_file


def main():
    algorithm = 'greedy'
    domain = 'two-arm-mover'

    if 'rsc' in algorithm:
        if 'one' in domain:
            s3_path = 'csail/bkim/guiding-gtamp/test_results/irsc/one_arm_mover/n_objs_pack_1'
        else:
            raise NotImplementedError
    else:
        if 'one' in domain:
            s3_path = 'csail/bkim/guiding-gtamp/test_results/f0cf459f5cb177eaacd17a0fa9b6b89caa96dbe3/sahs_results/uses_rrt/domain_one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_500/'
        else:
            if 'learn' in algorithm:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/f0cf459f5cb177eaacd17a0fa9b6b89caa96dbe3/' \
                          'sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
            else:
                s3_path = 'csail/bkim/guiding-gtamp/test_results/f0cf459f5cb177eaacd17a0fa9b6b89caa96dbe3/' \
                          'sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/' \
                          'qlearned_hcount_old_number_in_goal/' \
                          'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
                          'n_mp_limit_5_n_iter_limit_2000/'
    result = subprocess.check_output('mc ls {}'.format(s3_path), shell=True)

    if 'rsc' in algorithm:
        runs_finished = re.findall('seed_[0-9]*_pidx_[0-9]*', result)
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[-1], 'seed': fin.split('_')[1]} for fin in runs_finished]
    else:
        runs_finished = re.findall('pidx_[0-9]*_planner_seed_[0-9]*', result)
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[1], 'seed': fin.split('_')[-1]} for fin in runs_finished]
    # 307, 71
    print len(seed_pidx_pairs_finished)
    import pdb;pdb.set_trace()
    for seed_pidx in seed_pidx_pairs_finished:
        cmd = "kubectl delete job.batch/{}-{}-{}-{} -n beomjoon".format(algorithm, domain, seed_pidx['pidx'],
                                                                        seed_pidx['seed'])
        """
        cmd = "kubectl delete job.batch/discretized-rsc-{}-{} -n beomjoon".format(seed_pidx['pidx'],
                                                                                  seed_pidx['seed'])
        """
        print cmd
        os.system(cmd)


if __name__ == '__main__':
    main()
