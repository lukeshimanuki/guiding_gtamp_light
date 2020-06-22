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


def get_done_seed_and_pidx_pairs(commithash):
    s3_path = get_s3_path(commithash)
    try:
        result = subprocess.check_output('mc ls {}'.format(s3_path), shell=True)
    except subprocess.CalledProcessError:
        return []

    runs_finished = re.findall('pidx_[0-9]*_planner_seed_[0-9]*_gnn_seed_[0-9]*', result)
    if 'sampling_strategy' in runs_finished[0]:
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[1], 'seed': fin.split('_')[-1]} for fin in runs_finished]
    else:
        seed_pidx_pairs_finished = [{'pidx': fin.split('_')[1], 'seed': fin.split('_')[4]} for fin in runs_finished]

    return seed_pidx_pairs_finished


def get_seed_and_pidx_pairs_that_needs_to_run(pidxs, seed_pidx_pairs_finished):
    undone = [(seed, pidx) for seed in range(1) for pidx in pidxs]
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


def main():
    domain = 'two-arm-mover'
    hoption = 'hcount_old_number_in_goal'
    algorithm = 'greedy-hcount'

    timelimit = 2000
    n_iter_limit = 2000
    n_objs_pack = 1
    absq_seed = 0

    target_pidxs = range(1501, 3000)
    yaml_file = get_yaml_file_name(algorithm, domain)
    commithash = '9399178e4d84dd6756d45556e2865e89d89a923d'

    seed_pidx_pairs_running = []# get_running_seed_and_pidx_pairs(domain, algorithm)
    seed_pidx_pairs_finished = []# get_done_seed_and_pidx_pairs(commithash)
    undone = get_seed_and_pidx_pairs_that_needs_to_run(target_pidxs, seed_pidx_pairs_finished + seed_pidx_pairs_running)
    print "Remaining runs", len(undone)
    consecutive_runs = 0
    for idx, un in enumerate(undone):
        pidx = un[1]
        seed = un[0]
        cmd = 'cat cloud_scripts/{} | ' \
              'sed \"s/NAME/plan-exp-{}-{}-{}-{}-n-objs-{}-absqseed-{}/\" | ' \
              'sed \"s/PIDX/{}/\" | sed \"s/PLANSEED/{}/\" |  ' \
              'sed \"s/HOPTION/{}/\" |  ' \
              'sed \"s/TIMELIMIT/{}/\" |  ' \
              'sed \"s/NOBJS/{}/\" |  ' \
              'sed \"s/COMMITHASH/{}/\" |  ' \
              'sed \"s/NITERLIMIT/{}/\" |  ' \
              'sed \"s/ABSQSEED/{}/\" |  ' \
              'kubectl apply -f - -n beomjoon;'.format(yaml_file,
                                                       algorithm, domain, pidx, seed, n_objs_pack,
                                                       absq_seed,
                                                       pidx, seed,
                                                       hoption, timelimit, n_objs_pack, commithash,
                                                       n_iter_limit, absq_seed)
        print idx, cmd
        os.system(cmd)
        import pdb;pdb.set_trace()
        time.sleep(2)
        consecutive_runs += 1
        if consecutive_runs % 100 == 0:
            print "Long break"
            time.sleep(30)


if __name__ == '__main__':
    main()
