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
    yaml_file = 'run_sampler_epoch.yaml'
    cmd = 'cp ~/.kube/lis_config ~/.kube/config'
    os.system(cmd)

    cpus = '60'
    memory = '300Gi'

    atype = 'place'
    region_name = 'home-region'
    region = 'home_region'
    cmd = 'cat cloud_scripts/{} | ' \
          'sed \"s/NAME/select-epoch-atype-{}-region-{}/\" |  ' \
          'sed \"s/ATYPE/{}/\" |  ' \
          'sed \"s/REGION/{}/\" |  ' \
          'sed \"s/CPUS/{}/\" |  ' \
          'sed \"s/MEMORY/{}/\" |  ' \
          'kubectl apply -f - -n beomjoon;'.format(yaml_file, atype, region_name, atype, region, cpus, memory)
    print cmd
    os.system(cmd)


if __name__ == '__main__':
    main()
