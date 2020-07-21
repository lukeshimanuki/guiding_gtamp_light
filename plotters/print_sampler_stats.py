import os
import numpy as np
from matplotlib import pyplot as plt

target_dir = 'generators/learning/learned_weights/two_arm_mover/num_episodes_1000/place/loading_region/wgandi/fc'
is_pick = 'pick' in target_dir
# do these values data-dependent?
if 'two_arm_mover' in target_dir:
    if is_pick:
        target_kde = -150
        target_entropy = 3.8
    else:
        if 'home_region' in target_dir:
            target_kde = -40
            target_entropy = 3.53
        else:
            target_kde = -70
            target_entropy = 3.15

seeds = os.listdir(target_dir)
n_sat = 0
sat_seeds = []
for seed in seeds:
    file_dir = target_dir + '/' + seed + '/'
    stat_files = [f for f in os.listdir(file_dir) if '.pt' not in f]
    stats = []
    epochs = []
    print "Seed ",seed
    for stat_file in stat_files:
        mse = float(stat_file.split('_')[1])
        kde = float(stat_file.split('_')[3])
        entropy = float(stat_file.split('_')[5])
        epoch = int(stat_file.split('_')[-1])

        stats.append([mse, kde,entropy])
        epochs.append(epoch)

    stats = np.array(stats)[np.argsort(epochs)]
    epochs = np.array(epochs)[np.argsort(epochs)]

    """
    for kde, entropy, epoch in zip(stats[:, 1], entropies, epochs):
        seed = int(sd_dir.split('_')[1])
        if kde > target_kde and (entropy > target_entropy and entropy != np.inf):
            print 'best kde, entropies, seed', kde, entropy, int(sd_dir.split('_')[1])
            ones_that_satisfy.append([epoch, kde, entropy])
        if len(ones_that_satisfy) >= 1:
            one_with_highest_kde = np.argmax(np.array(ones_that_satisfy)[:, 1])
            epoch = np.array(ones_that_satisfy)[one_with_highest_kde, 0]
            candidate_seeds.append([seed, epoch])
            candidate_seed_kdes.append(kde)

    n_sat+=satisfy_targets
    if satisfy_targets:
        sat_seeds.append(seed)
    """

    plt.figure()
    plt.plot(stats[:, 1])
    plt.ylim(bottom=-300)
    plt.savefig('./plotters/{}.png'.format(seed))

    plt.close('all')

print n_sat
import pdb;pdb.set_trace()
