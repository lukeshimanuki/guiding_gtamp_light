import csv
import numpy as np

fdir = './generators/sampler_performances/'

learned_file = 'epoch_home_98400_epoch_loading_41900.txt'
uniform_file = 'uniform.txt'

learned = open(fdir + learned_file, 'r').read().splitlines()
uniform = open(fdir + uniform_file, 'r').read().splitlines()

learned_total_iks = []
learned_total_resets = []
for l in learned:
    learned_total_iks.append(int(l.split(',')[4])) # total n of infeasible mp calls
    learned_total_resets.append(int(l.split(',')[-1]))

unif_total_iks = []
unif_total_resets = []
for l in uniform:
    unif_total_iks.append(int(l.split(',')[4]))
    unif_total_resets.append(int(l.split(',')[-1]))

print np.mean(learned_total_iks), np.std(learned_total_iks)*1.96/np.sqrt(len(learned_total_iks))
print np.mean(unif_total_iks), np.std(unif_total_iks)*1.96/np.sqrt(len(unif_total_iks))

print np.mean(learned_total_resets), np.std(learned_total_resets)*1.96/np.sqrt(len(learned_total_resets))
print np.mean(unif_total_resets), np.std(unif_total_resets)*1.96/np.sqrt(len(unif_total_resets))


