import numpy as np
import os
import socket
import pickle

fdir = './generators/sampler_performances/'


def get_results(fin):
    data = open(fdir + fin, 'r').read().splitlines()
    result = {'iks': [], 'actions': [], 'pick_mps': [], 'pick_infeasible_mps': [], 'place_mps': [], 'place_infeasible_mps': [], 'success': []}

    raw_dir = './planning_experience/for_testing_generators/'
    for l in data:
        pidx = int(l.split(',')[0])
        """
        result_log = "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n" % (
            config.pidx, config.seed, total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible,
            total_place_mp_checks, total_place_mp_infeasible, total_mp_checks, total_infeasible_mp,
            n_total_actions, goal_reached
        )
        """
        if True: #int(l.split(',')[-1]) == 1:
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
    print "****Summary****"
    print result_file
    keys = ['iks', 'actions', 'pick_mps', 'pick_infeasible_mps', 'place_mps', 'place_infeasible_mps', 'success']
    for key in keys:
        print "%s %.3f %.3f" % (key, np.mean(results[key]), np.std(results[key])*1.96/np.sqrt(n_data))

def average_over_problems(fin):
    raw_dir = './planning_experience/for_testing_generators/'
    all_plan_exp_files = os.listdir(raw_dir)
    data = open(fdir + fin, 'r').read().splitlines()

    result = {'iks': [], 'actions': [], 'mps': [], 'infeasible_mps': [], 'success': []}
    result = {}
    for l in data:
        pidx = int(l.split(',')[0])
        if pidx in result:
            result[pidx].append(int(l.split(',')[9]))
        else:
            result[pidx] = [int(l.split(',')[9])]
    return result

def main():
    file2 = 'phaedra//unif_sqrt_pap_mps_n_mp_limit_5.txt'.format(socket.gethostname())
    results2 = get_results(file2)
    print_results(results2, file2)
    file2_avg = average_over_problems(file2)

    print '============================================================'
    file1 = 'phaedra/place_fc.txt'.format(socket.gethostname())
    results1 = get_results(file1)
    print_results(results1, file1)
    file1_avg = average_over_problems(file1)

    diff_sorted_idxs = np.argsort([np.mean(file2_avg[k])-np.mean(file1_avg[k]) for k in file1_avg])
    sorted_keys = np.array(file1_avg.keys())[diff_sorted_idxs]
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
        unif_lb = np.mean(file2_avg[k]) - 1.96/np.sqrt(len(file2_avg[k]))*np.std(file2_avg[k])
        learned_ub = np.mean(file1_avg[k]) + np.std(file1_avg[k])*1.96/np.sqrt(len(file2_avg[k]))
        unif_ub = np.mean(file2_avg[k]) + 1.96/np.sqrt(len(file2_avg[k]))*np.std(file2_avg[k])
        learned_lb = np.mean(file1_avg[k]) - np.std(file1_avg[k])*1.96/np.sqrt(len(file2_avg[k]))
        print "%10d %15.2f+-%.4f %15.2f+-%.4f %15.4f %15.4f %15.4f  better? %2d worse? %2d" % (k, np.mean(file2_avg[k]), 1.96/np.sqrt(len(file2_avg[k]))*np.std(file2_avg[k]), np.mean(file1_avg[k]), np.std(file1_avg[k])*1.96/np.sqrt(len(file2_avg[k])), np.mean(file2_avg[k])-np.mean(file1_avg[k]), unif_lb, learned_ub, unif_lb > learned_ub, learned_lb > unif_ub)


if __name__ == '__main__':
    main()
