import socket
import os
import pickle
import numpy as np

from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_utils import utils

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560' or hostname == 'phaedra' or hostname == 'shakey' or hostname == 'lab':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def get_save_dir():
    save_dir = ROOTDIR + '/planning_experience/processed/motion_plans/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_raw_dir():
    raw_dir = ROOTDIR + '/planning_experience/motion_planning_experience/'
    return raw_dir


def process_single_data(fname, problem_env, key_configs, save_file):
    mp_results = pickle.load(open(fname, 'r'))
    first_obj_poses = mp_results[0]['object_poses']

    [utils.set_obj_xytheta(first_obj_poses[obj.GetName()], obj.GetName()) for obj in problem_env.objects]
    q0s = []
    qgs = []
    labels = []
    collisions = []

    collision_vector = utils.compute_occ_vec(key_configs)
    for mp_result in mp_results:
        object_poses = mp_result['object_poses']
        assert object_poses == first_obj_poses
        q0s.append(mp_result['q0'])
        qgs.append(mp_result['qg'])
        labels.append(mp_result['label'])
        collisions.append(collision_vector)

    q0s = np.vstack(q0s)
    qgs = np.vstack(qgs)
    collisions = np.vstack(collisions)
    labels = np.vstack(labels)

    pickle.dump({'q0s': q0s, 'qgs': qgs, 'collisions': collisions, 'labels': labels}, open(save_file, 'wb'))
    print "Done with file", fname


def process_data(raw_dir, save_dir):
    problem_env = PaPMoverEnv(0)
    key_configs, edges = pickle.load(open('prm.pkl', 'r'))
    for raw_file in os.listdir(raw_dir):
        save_file = save_dir + raw_file
        process_single_data(raw_dir + raw_file, problem_env, key_configs, save_file)


def main():
    raw_dir = get_raw_dir()
    save_dir = get_save_dir()
    process_data(raw_dir, save_dir)


if __name__ == '__main__':
    main()
