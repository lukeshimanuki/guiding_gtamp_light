import socket
import os
import pickle
import numpy as np
import sys

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
    pick_q0s = []
    pick_qgs = []
    pick_labels = []
    pick_collisions = []

    place_q0s = []
    place_qgs = []
    place_labels = []
    place_collisions = []

    collision_vector = utils.compute_occ_vec(key_configs)
    for mp_result in mp_results:
        object_poses = mp_result['object_poses']
        assert object_poses == first_obj_poses
        if mp_result['held_obj'] is None:
            pick_q0s.append(mp_result['q0'])
            pick_qgs.append(mp_result['qg'])
            pick_labels.append(mp_result['label'])
            pick_collisions.append(collision_vector)
        else:
            place_q0s.append(mp_result['q0'])
            place_qgs.append(mp_result['qg'])
            place_labels.append(mp_result['label'])
            utils.two_arm_pick_object(mp_result['held_obj'], {'q_goal': mp_result['q0']})
            place_collision = utils.compute_occ_vec(key_configs)
            utils.two_arm_place_object({'q_goal': mp_result['q0']})
            place_collisions.append(place_collision)

    pickle.dump({'pick_q0s': pick_q0s, 'pick_qgs': pick_qgs, 'pick_collisions': pick_collisions,
                 'pick_labels': pick_labels,
                 'place_q0s': place_q0s, 'place_qgs': place_qgs, 'place_collisions': place_collisions,
                 'place_labels': place_labels}, open(save_file, 'wb'))

    print "Done with file", fname


def process_data(raw_dir, save_dir, idx):
    problem_env = PaPMoverEnv(0)
    key_configs, edges = pickle.load(open('prm.pkl', 'r'))
    raw_files = os.listdir(raw_dir)
    raw_file = raw_files[idx]
    save_file = save_dir + raw_file
    process_single_data(raw_dir + raw_file, problem_env, key_configs, save_file)


def main():
    raw_dir = get_raw_dir()
    save_dir = get_save_dir()
    idx = int(sys.argv[1])
    process_data(raw_dir, save_dir, idx)


if __name__ == '__main__':
    main()
