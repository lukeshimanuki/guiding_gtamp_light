import csv
import numpy as np

# result_log = "%d,%d,%d,%d,%d,%d,%d\n" % (
# config.pidx, config.seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions)

fdir = './generators/sampler_performances/'

learned_file = 'voo_log_n_mp_limit_5.txt'
learned_file = 'voo_gauss_n_mp_limit_5.txt'
learned_file = 'voo_sqrt_n_mp_limit_5.txt'
#learned_file = 'epoch_home_98400_epoch_loading_41900.txt'
uniform_file = 'unif_gauss_n_mp_limit_5.txt'

learned = open(fdir + learned_file, 'r').read().splitlines()
uniform = open(fdir + uniform_file, 'r').read().splitlines()


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


def main():
    learned_total_iks = []
    learned_total_resets = []
    target_pidx = 20006

    # pidx, seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions
    learned_n_mp = []
    learned_n_infeasible_mp = []
    for l in learned:
        learned_total_iks.append(int(l.split(',')[2]))  # total n of infeasible mp calls
        learned_total_resets.append(int(l.split(',')[-1]))
        learned_n_mp.append(int(l.split(',')[3]))
        learned_n_infeasible_mp.append(int(l.split(',')[4]))

    unif_total_iks = []
    unif_total_resets = []
    unif_n_mp = []
    unif_n_infeasible_mp = []
    for l in uniform:
        unif_total_iks.append(int(l.split(',')[2]))
        unif_total_resets.append(int(l.split(',')[-1]))
        unif_n_mp.append(int(l.split(',')[3]))
        unif_n_infeasible_mp.append(int(l.split(',')[4]))

    print "****Summary****"
    print 'ndata %25s %d %25s %d' % (learned_file, len(learned_total_iks), uniform_file, len(unif_total_iks))
    print 'total iks %50s %.3f %.3f' % (
        learned_file, np.mean(learned_total_iks), np.std(learned_total_iks) * 1.96 / np.sqrt(len(learned_total_iks)))
    print 'total iks %50s %.3f %.3f' % (
        uniform_file, np.mean(unif_total_iks), np.std(unif_total_iks) * 1.96 / np.sqrt(len(unif_total_iks)))

    print 'total mps  %50s %.3f %.3f' % (learned_file, np.mean(learned_n_mp), np.std(learned_n_mp) * 1.96 / np.sqrt(
        len(learned_n_mp)))
    print 'total mps  %50s %.3f %.3f' % (uniform_file, np.mean(unif_n_mp), np.std(unif_n_mp) * 1.96 / np.sqrt(
        len(unif_n_mp)))

    print 'infeasible mps %50s %.3f %.3f' % (learned_file, np.mean(learned_n_infeasible_mp), np.std(
        learned_n_infeasible_mp) * 1.96 / np.sqrt(
        len(learned_n_infeasible_mp)))
    print 'infeasible mps %50s %.3f %.3f' % (uniform_file, np.mean(unif_n_infeasible_mp), np.std(
        unif_n_infeasible_mp) * 1.96 / np.sqrt(
        len(unif_n_infeasible_mp)))

    print 'total resets %50s %.3f %.3f' % (
    learned_file, np.mean(learned_total_resets), np.std(learned_total_resets) * 1.96 / np.sqrt(
        len(learned_total_resets)))
    print 'total resets %50s %.3f %.3f' % (
    uniform_file, np.mean(unif_total_resets), np.std(unif_total_resets) * 1.96 / np.sqrt(
        len(unif_total_resets)))

    # print float(np.sum(learned_n_infeasible_mp)) / np.sum(learned_n_mp)
    # print float(np.sum(unif_n_infeasible_mp)) / np.sum(unif_n_mp)


if __name__ == '__main__':
    main()
