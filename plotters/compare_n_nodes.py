import pickle
import os
def get_n_nodes(target_dir):
    test_files = os.listdir(target_dir)
    n_nodes =0
    for filename in test_files:
        fin = pickle.load(open(target_dir+filename,'r'))
        if 'n_feasibility_checks' in fin:
            n_nodes += fin['n_feasibility_checks']['ik']
        else:
            n_nodes += fin['search_time_to_reward'][-1][2]

    print n_nodes / float(len(test_files))

def main():
    target_dir = 'test_results/sahs_results/uses_rrt/uses_reachability_clf_False/domain_two_arm_mover/' \
                 'n_objs_pack_4/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_' \
                 '1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    n_nodes = get_n_nodes(target_dir)

    target_dir = 'test_results/mcts_results_with_q_bonus/domain_two_arm_mover/n_objs_pack_1/' \
                 'sampling_strategy_uniform/n_mp_trials_3/widening_10/uct_0.1/' \
                 'switch_frequency_50/reward_shaping_True/learned_q_True/'
    n_nodes = get_n_nodes(target_dir)

if __name__ == '__main__':
    main()