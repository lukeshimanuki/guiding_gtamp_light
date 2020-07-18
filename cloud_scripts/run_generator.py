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
    yaml_file = 'run_generator.yaml'
    atype = 'place_loading'

    for planner_seed in [1, 2, 3]:
        for sampler_seed in [0, 1, 2, 3]:
            cmd = 'cat cloud_scripts/{} | ' \
                  'sed \"s/NAME/run-gen-plan-seed-{}-smpl-seed-{}/\" |  ' \
                  'sed \"s/ATYPE/{}/\" |  ' \
                  'sed \"s/SAMPLERSEED/{}/\" |  ' \
                  'sed \"s/PLANNERSEED/{}/\" |  ' \
                  'kubectl apply -f - -n beomjoon;'.format(yaml_file, planner_seed, sampler_seed,
                                                         atype, sampler_seed, planner_seed)
            print cmd
            os.system(cmd)


if __name__ == '__main__':
    main()
