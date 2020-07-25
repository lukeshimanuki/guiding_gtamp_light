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
    yaml_file = 'run_generator.yaml'
    return yaml_file


def main():
    yaml_file = 'run_epoch_validation.yaml'
    cmd = 'cp ~/.kube/lis_config ~/.kube/config'
    os.system(cmd)

    n_objs_pack = 1
    is_test_single_epoch = False
    if not is_test_single_epoch:
        atype = 'place_loading'
        test_multiple_epochs = '-test_multiple_epochs'
        epoch = ''
        planner_seeds = [0]
        sampler_seed_idxs = [0]
        cpus = '58'
        memory = '100Gi'
        use_test_pidxs = ''
    else:
        test_multiple_epochs = ''
        atype = 'place_home'
        sampler_seed_idxs = [0]
        if 0 in sampler_seed_idxs and atype == 'place_home':
            epoch = 10945
        elif 1 in sampler_seed_idxs and atype == 'place_home':
            epoch = 6842
        elif 2 in sampler_seed_idxs and atype == 'place_home':
            epoch = 8389
        elif 3 in sampler_seed_idxs and atype == 'place_home':
            epoch = 6013
        else:
            raise NotImplementedError

        planner_seeds = [0, 1, 2, 3]
        if atype == 'place_home':
            epoch = '-place_goal_region_epoch {}'.format(epoch)
        elif atype == 'place_loading':
            epoch = '-place_goal_region_epoch {}'.format(epoch)
        elif atype == 'pick':
            epoch = '-pick_epoch {}'.format(epoch)
        cpus = '15'
        memory = '70Gi'
        use_test_pidxs = '-use_test_pidxs'

    for planner_seed in planner_seeds:
        for sampler_seed_idx in sampler_seed_idxs:
            cmd = 'cat cloud_scripts/{} | ' \
                  'sed \"s/NAME/run-valid-plan-seed-{}-smpl-seed-{}-nobjs-{}/\" |  ' \
                  'sed \"s/ATYPE/{}/\" |  ' \
                  'sed \"s/SAMPLERSEEDIDX/{}/\" |  ' \
                  'sed \"s/PLANNERSEED/{}/\" |  ' \
                  'sed \"s/NOBJS/{}/\" |  ' \
                  'sed \"s/TESTMULTIPLEEPOCHS/{}/\" |  ' \
                  'sed \"s/EPOCH/{}/\" |  ' \
                  'sed \"s/CPUS/{}/\" |  ' \
                  'sed \"s/MEMORY/{}/\" |  ' \
                  'sed \"s/USETESTPIDXS/{}/\" |  ' \
                  'kubectl apply -f - -n beomjoon;'.format(yaml_file, planner_seed, sampler_seed_idx, n_objs_pack,
                                                           atype, sampler_seed_idx, planner_seed, n_objs_pack,
                                                           test_multiple_epochs, epoch, cpus, memory, use_test_pidxs)
            print cmd
            os.system(cmd)


if __name__ == '__main__':
    main()
