import os
import subprocess
import re
import time


def main():
    yaml_file = 'run_process_planning_exp.yaml'

    proecssing_type = 'abstract_q'

    consecutive_runs = 0
    for pidx in range(1600):
        cmd = 'cat cloud_scripts/{} | ' \
              'sed \"s/NAME/process-{}/\" | ' \
              'sed \"s/PIDX/{}/\" |  ' \
              'sed \"s/TYPE/{}/\" |  ' \
              'kubectl apply -f - -n beomjoon;'.format(yaml_file, pidx, pidx, proecssing_type)
        print cmd
        os.system(cmd)
        time.sleep(2)
        consecutive_runs += 1
        if consecutive_runs % 100 == 0:
            print "Long break"
            time.sleep(30)


if __name__ == '__main__':
    main()
