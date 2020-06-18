import os
import subprocess
import re
import time


def get_yaml_file_name(algorithm, domain):
    yaml_file = 'run_gather_planning_exp.yaml'
    return yaml_file


def get_s3_path(commithash):
    s3_path = 'csail/bkim/guiding-gtamp/planning_experience/' \
              '{}/' \
              'raw/two_arm_mover' \
              'n_objs_pack_1/' \
              'hcount_old_number_in_goal/' \
              'q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/' \
              'n_mp_limit_5_n_iter_limit_2000/'.format(commithash[0:7])
    return s3_path


def main():
    yaml_file = 'run_process_planning_exp.yaml'

    proecssing_type = 'sampler' # 'abstract_q'

    consecutive_runs = 0
    time.sleep(100)
    for pidx in range(5000):
        cmd = 'cat cloud_scripts/{} | ' \
              'sed \"s/NAME/process-{}/\" | ' \
              'sed \"s/PIDX/{}/\" |  ' \
              'sed \"s/TYPE/{}/\" |  ' \
              'kubectl apply -f - -n beomjoon;'.format(yaml_file, pidx, pidx, 'sampler')
        print cmd
        os.system(cmd)

        time.sleep(2)
        consecutive_runs += 1
        if consecutive_runs % 100 == 0:
            print "Long break"
            time.sleep(30)


if __name__ == '__main__':
    main()
