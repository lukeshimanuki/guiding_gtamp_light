import pickle
import os
import numpy as np

def get_n_nodes(target_dir):
    test_files = os.listdir(target_dir)
    n_iks = [] 
    n_nodes = []
    n_mps = []


    test_file_pidxs = [int(filename.split('pidx_')[1].split('_')[0]) for filename in test_files if 'pkl' in filename]
    test_files = np.array(test_files)[np.argsort(test_file_pidxs)]

    target_pidxs = [60053, 60023, 60081, 60001, 60021, 60008, 60062, 60079, 60033, 60044, 60031, 60018, 60075, 60050, 60030, 60020, 60098, 60016, 60067, 60061, 60024, 60096, 60005, 60088, 60091, 60010, 60011, 60045, 60006, 60099, 60038, 60083, 60058, 60046, 60029, 60032, 60097, 60039]

    for filename in test_files:
        if 'pkl' not in filename:
            print filename
            continue
        pidx = int(filename.split('pidx_')[1].split('_')[0])
        if not pidx in target_pidxs:
            continue
        fin = pickle.load(open(target_dir+filename,'r'))
        if 'num_nodes' in fin:
            n_node = fin['num_nodes'] 
        else:
            n_node = fin['n_nodes'] 
        if 'n_feasibility_checks' in fin:
            n_ik = fin['n_feasibility_checks']['ik']
            n_mp = fin['n_feasibility_checks']['mp']
            #n_infeasible_mp = fin['n_feasibility_checks']['infeasible_mp']
        else:
            n_ik = fin['search_time_to_reward'][-1][2]  
        if 'search_time_to_reward' in fin:
            if True:
              where_is_three = np.where(np.array(fin['search_time_to_reward'])[:,-1] == 3)[0][0]
              n_steps_after_three = len(np.array(fin['search_time_to_reward'])[where_is_three:,-1])
        #print filename, n_ik
        n_iks.append(n_ik)
        n_nodes.append(n_node)
        n_mps.append(n_mp)
        #print filename, n_node

    del n_iks[np.argmax(n_nodes)]
    del n_mps[np.argmax(n_nodes)]
    del n_nodes[np.argmax(n_nodes)]
    print 'n nodes',np.mean(n_nodes), np.std(n_nodes)
    print 'iks', np.mean(n_iks), np.std(n_iks)
    print 'mps', np.mean(n_mps), np.std(n_mps)
    print len(n_nodes)

def main():
    print "****Learned****"
    target_dir = 'test_results/100fcbc9afffe3c3b54b761c628496e2daee830f/sahs_results/uses_rrt/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/using_learned_sampler/n_mp_limit_5_n_iter_limit_2000/'
    n_nodes = get_n_nodes(target_dir)

    print "****UNIFORM****"
    target_dir = 'planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    n_nodes = get_n_nodes(target_dir)
    


if __name__ == '__main__':
    main()
