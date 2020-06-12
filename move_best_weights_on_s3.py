import os
import sys
import numpy as np
import pickle


def get_best_epoch(weight_dir):
    result_dir = weight_dir + '/result_summary/'
    assert os.path.isdir(result_dir), "Did you run plotters/evaluate_generators.py?"
    result_files = os.listdir(result_dir)
    assert len(result_files) > 0, "Did you run plotters/evaluate_generators.py?"
    iters = [int(f.split('_')[-1].split('.')[0]) for f in result_files]
    results = np.array([pickle.load(open(result_dir + result_file, 'r')) for result_file in result_files])

    max_kde_idx = np.argsort(results[:, 1])[::-1][0]
    max_kde_iteration = iters[max_kde_idx]
    weight_file = '/gen_iter_%d.pt' % max_kde_iteration
    return weight_file


def send_to_s3(weight_file):
    command = 'mc cp {} {}'.format(weight_file, 'csail/bkim')
    os.system(command)


def download_from_s3(weight_file, weight_dir):
    command = 'mc cp csail/bkim/{} generators/learning/{}'.format(weight_dir+weight_file, weight_dir)
    print command
    os.system(command)


def main():
    if 'two_arm' in sys.argv[2]:
        pick_weight_dir = 'learned_weights/two_arm_mover/pick/wgangp/fc/seed_2/'
        pick_weight_file = 'gen_iter_106700.pt'  # get_best_epoch(pick_weight_dir)
        place_goal_weight_dir = 'learned_weights/two_arm_mover/place/home_region/wgangp/fc/seed_3/'
        place_goal_weight_file = 'gen_iter_35900.pt'  # '/gen_iter_92000.pt' #
        place_obj_weight_dir = 'learned_weights/two_arm_mover/place/loading_region/wgangp/fc/seed_3/'
        place_obj_weight_file = 'gen_iter_16500.pt'  # '/gen_iter_13400.pt' #
    else:
        pick_weight_dir = 'learned_weights/one_arm_mover/pick/wgangp/fc/seed_0/'
        pick_weight_file = 'gen_iter_26900.pt'
        place_goal_weight_dir = 'None'
        place_goal_weight_file = ''
        place_obj_weight_dir = 'learned_weights/one_arm_mover/place/center_shelf_region/wgangp/fc/seed_0/'
        place_obj_weight_file = 'gen_iter_13600.pt'

    if sys.argv[1] == 'upload':
        send_to_s3(pick_weight_dir + pick_weight_file)
        send_to_s3(place_goal_weight_dir + place_goal_weight_file)
        send_to_s3(place_obj_weight_dir + place_obj_weight_file)
    elif sys.argv[1] == 'download':
        download_from_s3(pick_weight_file, pick_weight_dir)
        download_from_s3(place_goal_weight_file, place_goal_weight_dir)
        download_from_s3(place_obj_weight_file, place_obj_weight_dir)


if __name__ == '__main__':
    main()
