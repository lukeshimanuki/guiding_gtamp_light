import csv
import numpy as np

# result_log = "%d,%d,%d,%d,%d,%d,%d\n" % (
# config.pidx, config.seed, total_ik_checks, total_mp_checks, total_infeasible_mp, goal_reached, n_total_actions)

fdir = './generators/sampler_performances/'

learned_file = 'voo_sqrt_n_mp_limit_5.txt'
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
        pidx = int(l.split(',')[0])
        #if pidx != target_pidx:
        #    continue
        #print l
        learned_total_iks.append(int(l.split(',')[2]))  # total n of infeasible mp calls
        learned_total_resets.append(int(l.split(',')[-1]))
        learned_n_mp.append(int(l.split(',')[3]))
        learned_n_infeasible_mp.append(int(l.split(',')[4]))

    unif_total_iks = []
    unif_total_resets = []
    unif_n_mp = []
    unif_n_infeasible_mp = []
    print "****Uniform****"
    for l in uniform:
        pidx = int(l.split(',')[0])
        #if pidx != target_pidx:
        #    continue
        #print l
        unif_total_iks.append(int(l.split(',')[2]))
        unif_total_resets.append(int(l.split(',')[-1]))
        unif_n_mp.append(int(l.split(',')[3]))
        unif_n_infeasible_mp.append(int(l.split(',')[4]))
    print "****Summary****"
    print len(learned_total_iks), len(unif_total_iks)
    print np.mean(learned_total_iks), np.std(learned_total_iks) * 1.96 / np.sqrt(len(learned_total_iks))
    print np.mean(unif_total_iks), np.std(unif_total_iks) * 1.96 / np.sqrt(len(unif_total_iks))

    print np.mean(learned_total_resets), np.std(learned_total_resets) * 1.96 / np.sqrt(len(learned_total_resets))
    print np.mean(unif_total_resets), np.std(unif_total_resets) * 1.96 / np.sqrt(len(unif_total_resets))
    print len(unif_total_resets)

    print np.mean(learned_n_mp)
    print np.mean(unif_n_mp)

    #print float(np.sum(learned_n_infeasible_mp)) / np.sum(learned_n_mp)
    #print float(np.sum(unif_n_infeasible_mp)) / np.sum(unif_n_mp)

if __name__ == '__main__':
    main()