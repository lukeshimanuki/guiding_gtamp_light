import pickle
import os
import numpy as np

def get_n_nodes(target_dir):
    test_files = os.listdir(target_dir)
    n_iks = [] 
    n_nodes = []
    for filename in test_files:
        pidx = int(filename.split('_')[1])
        if pidx > 9:
            continue
        fin = pickle.load(open(target_dir+filename,'r'))
        if 'num_nodes' in fin:
            n_node = fin['num_nodes'] 
        else:
            n_node = fin['n_nodes'] 
        if 'n_feasibility_checks' in fin:
            n_ik = fin['n_feasibility_checks']['ik']
        else:
            n_ik = fin['search_time_to_reward'][-1][2]  
        if 'search_time_to_reward' in fin:
            if True:
              where_is_three = np.where(np.array(fin['search_time_to_reward'])[:,-1] == 3)[0][0]
              n_steps_after_three = len(np.array(fin['search_time_to_reward'])[where_is_three:,-1])
              #print filename, where_is_three, n_steps_after_three, n_ik
        n_iks.append(n_ik)
        n_nodes.append(n_node)
        #print filename, n_node
    print len(n_nodes)
    print np.mean(n_nodes), np.std(n_nodes)
    print len(n_iks)
    print np.mean(n_iks), np.std(n_iks)

def main():
    print "****SAHS****"
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/' \
                 'n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_' \
                 '1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    n_nodes = get_n_nodes(target_dir)

    print "****MCTS****"
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/' \
                 'sampling_strategy_uniform/n_mp_trials_3/widening_10.0/uct_0.1/' \
                 'switch_frequency_50/reward_shaping_True/learned_q_True/'
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_10/widening_0.2/uct_0.1/switch_frequency_100/reward_shaping_True/learned_q_True/use_pwTrue/'
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_10/widening_0.2/uct_0.1/switch_frequency_10/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
    n_nodes = get_n_nodes(target_dir)

if __name__ == '__main__':
    main()
