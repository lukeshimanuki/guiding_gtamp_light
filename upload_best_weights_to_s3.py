import os
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
    weight_file = weight_dir + '/gen_iter_%d.pt' % max_kde_iteration
    return weight_file


def send_to_s3(weight_file):
    command = 'mc cp {} {}'.format(weight_file, 'csail/bkim')
    os.system(command)


def main():
    pick_weight_dir = 'generators/learning/learned_weights/pick/wgangp/fc/seed_2/'
    pick_weight_file = get_best_epoch(pick_weight_dir)
    place_home_weight_dir = 'generators/learning/learned_weights/place/home_region/wgangp/fc/seed_0/'
    place_home_weight_file = get_best_epoch(place_home_weight_dir)
    place_loading_weight_dir = 'generators/learning/learned_weights/place/loading_region/wgangp/fc/seed_1/'
    place_loading_weight_file = get_best_epoch(place_loading_weight_dir)

    send_to_s3(pick_weight_dir + pick_weight_file)
    send_to_s3(place_home_weight_dir + place_home_weight_file)
    send_to_s3(place_loading_weight_dir + place_loading_weight_file)


if __name__ == '__main__':
    main()
