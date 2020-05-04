import csv
import numpy as np

# result_log = "%d,%d,%d,%d,%d,%d,%d\n" % (
# config.pidx, config.seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions)

fdir = './generators/sampler_performances/'

learned_file = 'voo_log_n_mp_limit_5.txt'
learned_file = 'voo_gauss_n_mp_limit_5.txt'
learned_file = 'epoch_home_98400_epoch_loading_41900.txt'


def total_result():
    # pidx, seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions
    learned_total_iks = []
    learned_total_resets = []
    for l in learned:
        learned_total_iks.append(int(l.split(',')[3]))  # total n of infeasible mp calls
        learned_total_resets.append(int(l.split(',')[-1]))

    unif_total_iks = []
    unif_total_resets = []
    for l in uniform:
        unif_total_iks.append(int(l.split(',')[3]))
        unif_total_resets.append(int(l.split(',')[-1]))

    print np.mean(learned_total_iks), np.std(learned_total_iks) * 1.96 / np.sqrt(len(learned_total_iks))
    print np.mean(unif_total_iks), np.std(unif_total_iks) * 1.96 / np.sqrt(len(unif_total_iks))

    print np.mean(learned_total_resets), np.std(learned_total_resets) * 1.96 / np.sqrt(len(learned_total_resets))
    print np.mean(unif_total_resets), np.std(unif_total_resets) * 1.96 / np.sqrt(len(unif_total_resets))


def get_results(fin):
    data = open(fdir + fin, 'r').read().splitlines()
    result = {'iks': [], 'actions': [], 'mps': [], 'infeasible_mps': []}

    """
    config.pidx, config.seed, total_ik_checks, total_pick_mp_checks, total_pick_mp_infeasible,
        total_place_mp_checks,
        total_place_mp_infeasible, total_mp_checks, total_infeasible_mp, n_total_actions, goal_reached
    """
    for l in data:
        result['iks'].append(int(l.split(',')[2]))
        result['mps'].append(int(l.split(',')[3]))
        result['infeasible_mps'].append(int(l.split(',')[4]) + int(l.split(',')[6]))
        result['actions'].append(int(l.split(',')[9]))
    return result


def print_results(results, result_file):
    iks = results['iks']
    mps = results['mps']
    actions = results['actions']
    infeasible_mps = results['infeasible_mps']
    n_data = len(iks)
    print "****Summary****"
    print result_file
    print "n_data", n_data
    print 'total iks %.3f %.3f' % (np.mean(iks), np.std(iks) * 1.96 / np.sqrt(n_data))
    print 'total mps %.3f %.3f' % (np.mean(mps), np.std(mps) * 1.96 / np.sqrt(n_data))
    print 'total infeasible mps %.3f %.3f' % (np.mean(infeasible_mps), np.std(infeasible_mps) * 1.96 / np.sqrt(n_data))
    print 'resets %.3f %.3f' % (np.mean(actions), np.std(actions) * 1.96 / np.sqrt(n_data))


def main():
    learned_total_iks = []
    learned_total_resets = []
    target_pidx = 20006

    # pidx, seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions

    file1 = 'voo_sqrt_pap_mps_n_mp_limit_5.txt'
    results1 = get_results(file1)
    file2 = 'unif_sqrt_pap_mps_n_mp_limit_5.txt'
    results2 = get_results(file2)

    print_results(results1, file1)
    print_results(results2, file2)

    # print float(np.sum(learned_n_infeasible_mp)) / np.sum(learned_n_mp)
    # print float(np.sum(unif_n_infeasible_mp)) / np.sum(unif_n_mp)


if __name__ == '__main__':
    main()
