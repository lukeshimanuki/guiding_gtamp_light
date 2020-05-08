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
    for filename in test_files:
        if 'pkl' not in filename:
            print filename
            continue
        pidx = int(filename.split('pidx_')[1].split('_')[0])
        #if pidx > 9:
        #    continue
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
    print "****SAHS_VOO****"
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/' \
                 'n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_' \
                 '1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    #target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/sampling_strategy_voo/'
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/sampling_strategy_voo_sqrt_counter2_less_than_10/'
    #target_dir = 'test_results/ab199e8bb7a25168a074419d96ad58c5eca0da65/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    target_dir = 'test_results/c72025431875642fba217e4ac99becac0ee471a8/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    n_nodes = get_n_nodes(target_dir)

    print "****SAHS_UNIFORM****"
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/' \
                 'sampling_strategy_uniform/n_mp_trials_3/widening_10.0/uct_0.1/' \
                 'switch_frequency_50/reward_shaping_True/learned_q_True/'
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_10/widening_0.2/uct_0.1/switch_frequency_100/reward_shaping_True/learned_q_True/use_pwTrue/'
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_10/widening_0.2/uct_0.1/switch_frequency_10/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
    target_dir = 'test_results/a42ecdeef726ec8339a8615dcde0cc9d4c1d5506/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
#    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_5/n_feasibility_checks_2000/widening_-5.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/' \
                 'n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_' \
                 '1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/sampling_strategy_uniform/'
    n_nodes = get_n_nodes(target_dir)
    
    print "****MCTS_VOO****"
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_voo/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/explr_p_0.3/'
    target_dir = 'test_results/a75516e20d6bb86e8bc1282aaaf373981af17633/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_voo/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/explr_p_0.3/'
    target_dir = 'test_results/e06ee96edacf4e7cd0f71ba61c383218d18833d5/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_voo/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_9999/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/explr_p_0.3/sqrt_counter2_less_than_2/'
    target_dir = 'test_results/46a564c4d4e1d22f9f91d1f3c0889dd00b777223/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_voo/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_9999/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/explr_p_0.3/'

    #n_nodes = get_n_nodes(target_dir)

    print "****MCTS_UNIF****"
    #target_dir = 'test_results/db91924a4116957040d9ab1c9ee941f20700504b/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
    #n_nodes = get_n_nodes(target_dir)
    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_5/n_feasibility_checks_2000/widening_-4.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
    target_dir = 'test_results/a75516e20d6bb86e8bc1282aaaf373981af17633/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/sampling_strategy_uniform/n_mp_trials_5/n_feasibility_checks_2000/widening_-1.0/uct_0.0/switch_frequency_50/reward_shaping_True/learned_q_True/use_pw_True/use_ucb_at_cont_nodes_True/'
    #n_nodes = get_n_nodes(target_dir)

if __name__ == '__main__':
    main()
