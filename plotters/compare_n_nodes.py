import pickle
import os
import numpy as np

def get_n_nodes(target_dir):
    test_files = os.listdir(target_dir)
    n_nodes = [] 
    for filename in test_files:
        fin = pickle.load(open(target_dir+filename,'r'))
        if 'n_feasibility_checks' in fin:
            n_node = fin['n_feasibility_checks']['ik']
        else:
            n_node = fin['search_time_to_reward'][-1][2]
        n_nodes.append(n_node)
    print len(n_nodes)
    print np.mean(n_nodes), np.std(n_nodes)

def main():
    print "****SAHS****"
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/' \
                 'n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_' \
                 '1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    n_nodes = get_n_nodes(target_dir)
     
    print "****MCTS****"
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/' \
                 'sampling_strategy_uniform/n_mp_trials_3/widening_5.0/uct_0.1/' \
                 'switch_frequency_50/reward_shaping_True/learned_q_True/'
    n_nodes = get_n_nodes(target_dir)

if __name__ == '__main__':
    main()
