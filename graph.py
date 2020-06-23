import pickle
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import collections

def main():
	dirname = sys.argv[1]
	domain = sys.argv[2]
	n_objs_pack = int(sys.argv[3])
	paths = {
		'IRSC': "irsc/{}_arm_mover/n_objs_pack_{}",
		'SAHS w/ just H': "sahs_results/uses_prm/domain_{}_arm_mover/n_objs_pack_{}/hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000",
		'SAHS w/ H and Q': "sahs_results/uses_prm/domain_{}_arm_mover/n_objs_pack_{}/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000",
	}

	if domain == 'one':
		timelimit = 1000
	elif domain == 'two':
		timelimit = 500 * n_objs_pack
	else:
		raise NotImplementedError

	num_trials = 500

	stats = collections.defaultdict(lambda: collections.defaultdict(list))

	for name, path in paths.items():
		fullpath = dirname + '/' + path.format(domain, n_objs_pack)
		filenames = os.listdir(fullpath)

		for filename in filenames:
			if 20000 <= int(filename.split('pidx_')[1].split('_')[0].split('.')[0]) < 20100 and 0 <= int(filename.split('seed_')[1].split('_')[0].split('.')[0]) < 5:
				results = pickle.load(open(fullpath + '/' + filename, 'rb'))
				stats[name]['time'].append(min(timelimit, results['tottime']) if results['success'] else timelimit)
				stats[name]['success'].append(results['success'] and results['tottime'] <= timelimit)
				if name != 'IRSC':
					stats[name]['nodes'].append(results['num_nodes'])
				if results['n_feasibility_checks']['ik'] > 0:
					stats[name]['ik'].append(results['n_feasibility_checks']['ik'])
				if domain != 'one':
					stats[name]['mp'].append(results['n_feasibility_checks']['mp'])
		stats[name]['time'] += [timelimit] * (num_trials - len(stats[name]['time']))

	statistics = {
		'time': 'Time (s)',
		'nodes': 'Number of Nodes',
		'ik': 'Number of Inverse Kinematics Calls',
		'mp': 'Number of Motion Planner Calls',
	}

	for key, label in statistics.items():
		if not any(key in s for s in stats.values()):
			continue

		plt.figure()


		plt.boxplot([s[key] for s in stats.values() if key in s], whis=(10,90), labels=[k for k,s in stats.items() if key in s], sym='', positions=[i for i,s in enumerate(stats.values()) if key in s])
		for i, (name, s) in enumerate(stats.items()):
			if key in s:
				x = s[key]
				plt.plot(np.ones(len(x)) * i, x, 'k.', label=name)

		#plt.legend()
		plt.title("{} Arm, {} Object{}".format(domain[0].upper() + domain[1:], n_objs_pack, 's' if n_objs_pack > 1 else ''))
		#plt.xlabel('Algorithm')
		plt.ylabel(label)

		plt.savefig("{}_arm_{}_obj_{}.pdf".format(domain, n_objs_pack, key))

if __name__ == '__main__':
	main()

