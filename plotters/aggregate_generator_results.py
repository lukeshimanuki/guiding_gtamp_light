import numpy as np
import os
import socket
import pickle
import collections
#from test_scripts.run_generator import convert_seed_epoch_idxs_to_seed_and_epoch

fdir = './generators/sampler_performances/'


def get_results(fin_list):
    result = {'iks': [], 'actions': [], 'pick_mps': [], 'pick_infeasible_mps': [], 'place_mps': [],
              'place_infeasible_mps': [], 'success': []}
    target_pidxs = range(9)


    for fin in fin_list:
        data = open(fdir + fin, 'r').read().splitlines()

        for l in data:
            pidx = int(l.split(',')[0])
  
            #if pidx not in target_pidxs:
            #    continue
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
    if n_data < 9:
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
    unif_file = 'phaedra//pick_place_home_place_loading//uniform.txt'.format(socket.gethostname())
    unif_results = get_results([unif_file])
    print_results(unif_results, unif_file)

    print '============================================================'


    res_path = '//37bf3df09590//place_loading/sampler_seed_5/wgandi/'
    file_list = [res_path+ f for f in os.listdir('generators/sampler_performances/'+res_path)]
    #file_list = ['/phaedra/place_loading/sampler_seed_4/wgandi/' + f for f in os.listdir('generators/sampler_performances/phaedra/place_loading/sampler_seed_4/wgandi/')]
    epoch_n_actions =  []


    for f in file_list:
      epoch = int(f.split('epoch_')[1].split('.txt')[0])
      results = get_results([f])
      iks = results['iks']
      n_data = len(iks)
      if n_data < 9:
          continue
      print_results(results, [f])
      epoch_n_actions.append([epoch, np.mean(results['actions'])])

    epoch_n_actions = np.array(epoch_n_actions)
    epoch_n_actions = epoch_n_actions[np.argsort(epoch_n_actions[:, 1]),:]
    best_epoch = epoch_n_actions[np.argmin(epoch_n_actions[:, 1]), 0]
    n_actions = epoch_n_actions[np.argmin(epoch_n_actions[:,1]),1]
    print "Best epoch is {} with {} number of actions".format(best_epoch, n_actions)
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
