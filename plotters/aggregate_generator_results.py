import numpy as np
import os
import socket
import pickle

fdir = './generators/sampler_performances/'


def get_results(fin_list):
    result = {'iks': [], 'actions': [], 'pick_mps': [], 'pick_infeasible_mps': [], 'place_mps': [],
              'place_infeasible_mps': [], 'success': []}
    target_pidxs = [4, 3, 14, 2, 17, 7, 23, 11, 9, 19, 21, 16, 1, 10, 12, 18, 24, 15]
    target_pidxs = [4, 3, 14, 2, 17, 7, 23, 10, 12, 18, 24, 15]


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
    print 'n data',n_data
    print "****Summary****"
    print result_file
    keys = ['iks', 'actions', 'pick_mps', 'pick_infeasible_mps', 'place_mps', 'place_infeasible_mps', 'success']
    for key in keys:
        print "%s %.3f %.3f" % (key, np.mean(results[key]), np.std(results[key]) * 1.96 / np.sqrt(n_data))


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

    print '============================================================'
    file_list = ['phaedra/wgandi_sampler_seed_0.txt', 
                 'phaedra/wgandi_sampler_seed_1.txt',\
                 'phaedra/wgandi_sampler_seed_2.txt',\
                # 'phaedra/wgandi_sampler_seed_3.txt',
                ]
    results = get_results(file_list)
    print_results(results, file_list)

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
