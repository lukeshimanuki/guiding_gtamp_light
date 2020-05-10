import numpy as np
import os
import socket

fdir = './generators/sampler_performances/'


def get_results(fin):
    data = open(fdir + fin, 'r').read().splitlines()
    result = {'iks': [], 'actions': [], 'mps': [], 'infeasible_mps': [], 'success': []}

    """
    config.pidx, config.seed, total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible,
        total_place_mp_checks,
        total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached
    """
    raw_dir = './planning_experience/for_testing_generators/'
    all_plan_exp_files = os.listdir(raw_dir)

    for l in data:
        pidx = int(l.split(',')[0])
        # if pidx > 9:
        #    continue
        print l
        result['iks'].append(int(l.split(',')[2]))
        result['mps'].append(int(l.split(',')[3]))
        result['infeasible_mps'].append(int(l.split(',')[4]) + int(l.split(',')[6]))
        result['actions'].append(int(l.split(',')[9]))
        result['success'].append(int(l.split(',')[-1]))

    return result


def print_results(results, result_file):
    iks = results['iks']
    mps = results['mps']
    actions = results['actions']
    infeasible_mps = results['infeasible_mps']
    successes = results['success']
    n_data = len(iks)
    print "****Summary****"
    print result_file
    print "n_data", n_data
    print 'total iks %.3f %.3f' % (np.mean(iks), np.std(iks) * 1.96 / np.sqrt(n_data))
    print 'total mps %.3f %.3f' % (np.mean(mps), np.std(mps) * 1.96 / np.sqrt(n_data))
    print 'total infeasible mps %.3f %.3f' % (np.mean(infeasible_mps), np.std(infeasible_mps) * 1.96 / np.sqrt(n_data))
    print 'resets %.3f %.3f' % (np.mean(actions), np.std(actions) * 1.96 / np.sqrt(n_data))
    print 'success rate %.3f %.3f' % (np.mean(successes), np.std(successes) * 1.96 / np.sqrt(n_data))


def main():
    file1 = '{}/epoch_home_98100_epoch_loading_9700.txt'.format(socket.gethostname())
    results1 = get_results(file1)
    file2 = '{}/unif_sqrt_pap_mps_n_mp_limit_5.txt'.format(socket.gethostname())
    results2 = get_results(file2)

    print_results(results1, file1)
    print_results(results2, file2)


if __name__ == '__main__':
    main()
