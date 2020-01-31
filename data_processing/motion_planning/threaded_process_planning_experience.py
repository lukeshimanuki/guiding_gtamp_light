import os
import sys
from multiprocessing.pool import ThreadPool  # dummy is nothing but multiprocessing but wrapper around threading
import multiprocessing
import argparse
import time

import socket

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def worker_p(config):
    command = 'python ./data_processing/motion_planning/process_planning_experience.py ' + str(config)

    print command
    os.system(command)


def worker_wrapper_multi_input(multi_args):
    time.sleep(1)
    return worker_p(multi_args)


def main():
    flist = os.listdir(ROOTDIR + '/planning_experience/motion_planning_experience/')
    configs = range(len(flist))

    n_workers = multiprocessing.cpu_count()
    print configs
    pool = ThreadPool(n_workers)
    results = pool.map(worker_wrapper_multi_input, configs)


if __name__ == '__main__':
    main()
