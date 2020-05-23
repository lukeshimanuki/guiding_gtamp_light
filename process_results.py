import pickle
import os
import sys
import numpy as np

def compute_dir(dirname):
	filenames = os.listdir(dirname)

	num_trials = 500

	if 'n_objs_pack_1' in dirname:
		timelimit = 500
	elif 'n_objs_pack_4' in dirname:
		timelimit = 800
	else:
		raise NotImplementedError

	tottime = []
	success = 0
	num_nodes = []
	n_ik = []
	n_mp = []

	for filename in filenames:
		if 20000 <= int(filename.split('pidx_')[1].split('_')[0].split('.')[0]) < 20100 and 0 <= int(filename.split('seed_')[1].split('_')[0].split('.')[0]) < 5:
			results = pickle.load(open(dirname + '/' + filename, 'rb'))
			tottime.append(results['tottime'] if results['success'] else timelimit)
			success += 1 if results['success'] and results['tottime'] <= timelimit else 0
			num_nodes.append(results['num_nodes'])
			n_ik.append(results['n_feasibility_checks']['ik'])
			n_mp.append(results['n_feasibility_checks']['mp'])

	p = float(success) / num_trials

	tottime = [
		min(t, timelimit)
		for t in tottime
	] + [timelimit] * (num_trials - len(tottime))

	return (
		np.mean(tottime), 1.96*np.std(tottime)/np.sqrt(len(tottime)),
		p, 1.96*np.sqrt(p*(1-p)/num_trials),
		np.mean(num_nodes), 1.96*np.std(num_nodes)/np.sqrt(len(num_nodes)),
		np.mean(n_ik), 1.96*np.std(n_ik)/np.sqrt(len(n_ik)),
		np.mean(n_mp), 1.96*np.std(n_mp)/np.sqrt(len(n_mp)),
	)

def main():
	dirname = sys.argv[1]
	print("{}: {}".format(dirname, "{0:.1f}\\pm{1:.1f}&{2:.2f}\\pm{3:.2f}&{4:.1f}\\pm{5:.1f}&{6:.1f}\\pm{7:.1f}&{8:.1f}\\pm{9:.1f}".format(
		*compute_dir(dirname)
	)))

if __name__ == '__main__':
	main()

