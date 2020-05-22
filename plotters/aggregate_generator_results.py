import numpy as np
import os
import socket
import pickle

fdir = './generators/sampler_performances/'


def get_results(fin):
    data = open(fdir + fin, 'r').read().splitlines()
    result = {'iks': [], 'actions': [], 'pick_mps': [], 'pick_infeasible_mps': [], 'place_mps': [],
              'place_infeasible_mps': [], 'success': []}

    pidxs = [60053, 60023, 60081, 60001, 60021, 60008, 60062, 60079, 60033, 60044, 60031, 60018, 60075, 60050, 60030, 60020, 60098, 60016, 60067, 60061, 60024, 60096, 60005, 60088, 60091, 60010, 60011, 60045, 60006, 60099, 60038, 60083, 60058, 60046, 60029, 60032, 60097, 60039]
    pidxs = [60089, 60061, 60094, 60075, 60074, 60050, 60096, 60057, 60008, 60088, 60026, 60003, 60010, 60067, 60091,
             60031, 60006, 60024, 60030, 60062, 60099, 60018, 60011, 60029, 60098, 60083, 60079, 60016, 60045, 60038,
             60046, 60032, 60058, 60097, 60039]

    for l in data:
        pidx = int(l.split(',')[0])
        #if True:  # int(l.split(',')[-1]) == 1:
        #if 60000<= pidx < 60100:
        if pidx in pidxs:
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


def average_over_problems(fin):
    raw_dir = './planning_experience/for_testing_generators/'
    data = open(fdir + fin, 'r').read().splitlines()

    result = {}
    for l in data:
        pidx = int(l.split(',')[0])
        if pidx in result:
            result[pidx].append(int(l.split(',')[9]))
        else:
            result[pidx] = [int(l.split(',')[9])]
    return result


def main():
    file2 = 'phaedra//uniform_sqrt_pap_mps_n_mp_limit_5.txt'.format(socket.gethostname())
    #file2 = 'phaedra//uniform_sqrt_pap_mps_n_mp_limit_5.txt'
    results2 = get_results(file2)
    print_results(results2, file2)
    file2_avg = average_over_problems(file2)

    print '============================================================'
    file1 = 'phaedra/pap_pick_fc_place_fc.txt'.format(socket.gethostname())
    results1 = get_results(file1)
    print_results(results1, file1)
    file1_avg = average_over_problems(file1)

    diff_sorted_idxs = np.argsort([np.mean(file2_avg[k]) - np.mean(file1_avg[k]) for k in file1_avg])
    sorted_keys = np.array(file1_avg.keys())[diff_sorted_idxs]
    print 'former:',file2
    print 'latter:',file1
    hard_keys = []
    for k in sorted_keys:
        """
        raw_dir = './planning_experience/for_testing_generators/'
        fname = 'pidx_%d_planner_seed_0_gnn_seed_0.pkl' % k
        try:
            plan_data = pickle.load(open(raw_dir + fname, 'r'))
        except:
            plan_data = pickle.load(open(raw_dir+'sampling_strategy_uniform'+fname,'r'))
        plan = plan_data['plan']
        plan_length = len(plan)
        """
        if np.mean(file2_avg[k]) - np.mean(file1_avg[k]) > 5:
            unif_lb = np.mean(file2_avg[k]) - 1.96 / np.sqrt(len(file2_avg[k])) * np.std(file2_avg[k])
            learned_ub = np.mean(file1_avg[k]) + np.std(file1_avg[k]) * 1.96 / np.sqrt(len(file2_avg[k]))
            unif_ub = np.mean(file2_avg[k]) + 1.96 / np.sqrt(len(file2_avg[k])) * np.std(file2_avg[k])
            learned_lb = np.mean(file1_avg[k]) - np.std(file1_avg[k]) * 1.96 / np.sqrt(len(file2_avg[k]))
            print "%10d %15.2f+-%.4f %15.2f+-%.4f %15.4f %15.4f %15.4f  better? %2d worse? %2d" % (
            k, 
            np.mean(file2_avg[k]), 1.96 / np.sqrt(len(file2_avg[k])) * np.std(file2_avg[k]), 
            np.mean(file1_avg[k]), np.std(file1_avg[k]) * 1.96 / np.sqrt(len(file2_avg[k])), 
            np.mean(file2_avg[k]) - np.mean(file1_avg[k]),
            unif_lb, learned_ub, unif_lb > learned_ub, learned_lb > unif_ub)
            hard_keys.append(k)
    print hard_keys
          


if __name__ == '__main__':
    main()
