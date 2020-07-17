import numpy as np
import os
import socket
import pickle
import collections
from test_scripts.run_greedy import get_seed_and_epochs

fdir = './generators/sampler_performances/'


def get_results(fin_list):
    result = {'iks': [], 'actions': [], 'pick_mps': [], 'pick_infeasible_mps': [], 'place_mps': [],
              'place_infeasible_mps': [], 'success': []}
    target_pidxs = [4, 3, 14, 2, 17, 7, 23, 11, 9, 19, 21, 16, 1, 10, 12, 18, 24, 15]
    target_pidxs = [4, 3, 14, 2, 17, 7, 23, 10, 12, 18, 24, 15]
    target_pidxs = [40064, 40071, 40077, 40078, 40080, 40083, 40088, 40097, 40098, 40003, 40007, 40012, 40018,
                    40020, 40023, 40030, 40032, 40033, 40036, 40038, 40047, 40055, 40059, 40060, 40062]
    target_pidxs = range(10)


    for fin in fin_list:
        data = open(fdir + fin, 'r').read().splitlines()

        for l in data:
            pidx = int(l.split(',')[0])
  
            if pidx not in target_pidxs:
                continue
            result['iks'].append(int(l.split(',')[2]))
            result['pick_mps'].append(int(l.split(',')[3]))
            result['pick_infeasible_mps'].append(int(l.split(',')[4]))
            result['place_mps'].append(int(l.split(',')[5]))
            result['place_infeasible_mps'].append(int(l.split(',')[6]))
            result['actions'].append(int(l.split(',')[9]))
            result['success'].append(int(l.split(',')[-1]))

    return result


def print_results(results, result_file):
    iks = results['iks']
    n_data = len(iks)
    if n_data < 10:
        return
    print 'n data',n_data
    #print "****Summary****"
    keys = ['iks', 'actions', 'pick_mps', 'pick_infeasible_mps', 'place_mps', 'place_infeasible_mps', 'success']
    keys = ['actions']
    for key in keys:
        print result_file, "%s %.3f %.3f" % (key, np.mean(results[key]), np.std(results[key]) * 1.96 / np.sqrt(n_data))


def average_over_problems(fin_list):
    raw_dir = './planning_experience/for_testing_generators/'

    result = {}
    for fin in fin_list:
        data = open(fdir + fin, 'r').read().splitlines()

        for l in data:
            pidx = int(l.split(',')[0])
            if pidx in result:
                result[pidx].append(int(l.split(',')[9]))
            else:
                result[pidx] = [int(l.split(',')[9])]
    return result


def main():
    unif_file = 'phaedra//uniform.txt'.format(socket.gethostname())
    unif_results = get_results([unif_file])
    print_results(unif_results, unif_file)

    old_file = 'phaedra//using_200_data_points/wgandi_sampler_seed_0.txt'.format(socket.gethostname())
    old_results = get_results([old_file])
    print_results(old_results, old_file)

    best_file = 'phaedra/place_loading/wgandi_sampler_seed_0_epoch_439.txt'.format(socket.gethostname())
    best_results = get_results([best_file])
    print_results(best_results, best_file)
    import pdb;pdb.set_trace()

    print '============================================================'
    # place loading seed dir order: [4, 3, 12, 5]

    config_type = collections.namedtuple('type', 'domain num_episode train_type sampler_seed sampler_epoch')

    file_list = ['/phaedra/place_loading/' + f for f in os.listdir('generators/sampler_performances/phaedra/place_loading/')]
    epoch_n_actions =  []

    target_seed = 0
    config = config_type(
          domain = 'two_arm_mover',
          num_episode = 1000,
          train_type = 'wgandi',
          sampler_seed = target_seed,
          sampler_epoch = 0)
    _, _, seeds, epochs = get_seed_and_epochs('place', 'loading_region', config)

    for f in file_list:
      epoch = int(f.split('epoch_')[1].split('.txt')[0])
      seed = int(f.split('seed_')[1].split('_')[0])
      if seed != 0:
          continue
      results = get_results([f])
      iks = results['iks']
      n_data = len(iks)
      if n_data < 10:
          continue
      print_results(results, [f])
      
      true_seed = seeds[seed]
      true_epoch = epochs[epoch]
      epoch_n_actions.append([true_epoch, np.mean(results['actions']), epoch])

    epoch_n_actions = np.array(epoch_n_actions)
    epoch_n_actions = epoch_n_actions[np.argsort(epoch_n_actions[:, 1]),:]
    best_epoch = epoch_n_actions[np.argmin(epoch_n_actions[:, 1]), 0]
    epoch_idx = epoch_n_actions[np.argmin(epoch_n_actions[:, 1]), 2]
    n_actions = epoch_n_actions[np.argmin(epoch_n_actions[:,1]),1]
    print "Best epoch for seed {} is {} with {} number of actions".format(true_seed, best_epoch, n_actions)
    print "Uniform {} number of actions".format(np.mean(unif_results['actions']))
    import pdb;pdb.set_trace()

    smpler_avg = average_over_problems(file_list)
    unif_avg = average_over_problems([unif_file])
  
    diff_sorted_idxs = np.argsort([np.mean(smpler_avg[k]) - np.mean(unif_avg[k]) for k in unif_avg])
    sorted_keys = np.array(unif_avg.keys())[diff_sorted_idxs]
    print 'former:',unif_file
    print 'latter:',file_list
    hard_keys = []
    print "{} {} {} {}".format( "PIDX", "Smpler", "unif", "Smpler-unif")
    for k in sorted_keys:
        if abs(np.mean(smpler_avg[k]) - np.mean(unif_avg[k])) > 10:
            #unif_lb = np.mean(smpler_avg[k]) - 1.96 / np.sqrt(len(smpler_avg[k])) * np.std(smpler_avg[k])
            #learned_ub = np.mean(unif_avg[k]) + np.std(unif_avg[k]) * 1.96 / np.sqrt(len(smpler_avg[k]))
            #unif_ub = np.mean(smpler_avg[k]) + 1.96 / np.sqrt(len(smpler_avg[k])) * np.std(smpler_avg[k])
            #learned_lb = np.mean(unif_avg[k]) - np.std(unif_avg[k]) * 1.96 / np.sqrt(len(smpler_avg[k]))
            print "%10d %15.2f+-%.4f %15.2f+-%.4f %15.4f " % (
            k, 
            np.mean(smpler_avg[k]), 1.96 / np.sqrt(len(smpler_avg[k])) * np.std(smpler_avg[k]), 
            np.mean(unif_avg[k]), np.std(unif_avg[k]) * 1.96 / np.sqrt(len(smpler_avg[k])), 
            np.mean(smpler_avg[k]) - np.mean(unif_avg[k]),
           # unif_lb, learned_ub, unif_lb > learned_ub, learned_lb > unif_ub
            )
            hard_keys.append(k)
    print hard_keys
          


if __name__ == '__main__':
    main()
