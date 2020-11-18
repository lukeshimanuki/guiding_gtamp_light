from matplotlib import pyplot as plt
import numpy as np
from compare_n_nodes import print_results, get_n_nodes, get_target_dirs, get_sampler_results


def compare_abstract_q_representations():
    # pose-based abstract q with large margin loss
    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/pose_qlearned_hcount_old_number_in_goal/' \
                 'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_loss_largemargin/' \
                 'n_mp_limit_5_n_iter_limit_2000/'
    print "Pose based abstract Q"
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print_results(target_dir)
    n_nodes_pose = np.hstack(pidx_nodes.values())

    # GNN-based abstract q with large margin loss
    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
                 'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
                 'n_mp_limit_5_n_iter_limit_2000/'
    print "GNN based abstract Q"
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print_results(target_dir)
    n_nodes_gnn = np.hstack(pidx_nodes.values())

    plt.figure()
    plt.boxplot(
        [n_nodes_gnn, n_nodes_pose],
        labels=['Abstract\nState', 'PoseBased\nState'],
        positions=[0, 1],
        whis=(10, 90), medianprops={'linewidth': 4.})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig("./plotters/plots/poses_vs_abstractstate.eps")
    plt.savefig("../IJRR_GTAMP/figures/poses_vs_abstractstate.eps")


def compare_losses():
    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
                 'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_mse/' \
                 'n_mp_limit_5_n_iter_limit_2000/'
    print "***Abstract Q MSE loss***"
    # print_results(target_dir)
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    n_nodes_mse = np.hstack(pidx_nodes.values())

    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
                 'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
                 'n_mp_limit_5_n_iter_limit_2000/'
    print "***Abstract Q pessimistic loss***"
    # print_results(target_dir)
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    n_nodes_lm = np.hstack(pidx_nodes.values())

    root_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
               'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
               'using_learned_sampler/1500/'
    wgangp_dir = root_dir + '/actorcritic/'
    print "***Sampler actor critic loss***"
    wgangp_dirs = get_target_dirs(wgangp_dir)
    n_nodes_actorcritic = get_sampler_results(wgangp_dirs)

    root_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
               'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
               'using_learned_sampler/1000/'
    wgangp_dir = root_dir + '/wgangp/'
    print "***Sampler GAN loss***"
    wgangp_dirs = get_target_dirs(wgangp_dir)
    n_nodes_gan = get_sampler_results(wgangp_dirs)

    plt.figure()
    plt.boxplot(
        [n_nodes_mse, n_nodes_actorcritic, n_nodes_lm, n_nodes_gan, ],
        labels=['SAHS\nMSE', 'SAHS\nRank\nActorCritic', 'SAHS\nRank', 'SAHS\nRank\nWGANGP'],
        positions=[0, 1, 2, 3],
        whis=(10, 90), medianprops={'linewidth': 4.})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig("./plotters/plots/loss_comparison.eps")
    plt.savefig("../IJRR_GTAMP/figures/loss_comparison.eps")


def compare_representations():
    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/pose_qlearned_hcount_old_number_in_goal/' \
                 'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_False_loss_largemargin/' \
                 'n_mp_limit_5_n_iter_limit_2000/'
    print "****Pose based abstract Q***"
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print_results(target_dir)
    n_nodes_pose_heuristic = np.hstack(pidx_nodes.values())

    # GNN-based abstract q with large margin loss
    target_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
                 'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
                 'n_mp_limit_5_n_iter_limit_2000/'
    print "***GNN based abstract Q***"
    pidx_nodes, pidx_times, successes, n_nodes, n_data, pidx_iks = get_n_nodes(target_dir)
    print_results(target_dir)
    n_nodes_gnn_heuristic = np.hstack(pidx_nodes.values())

    print "***Pose based sampler***"
    root_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
               'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
               'using_learned_sampler/state_mode_pose/1500/'
    wgangp_dir = root_dir + '/wgangp/'
    wgangp_dirs = get_target_dirs(wgangp_dir)
    n_nodes_pose_sampler = get_sampler_results(wgangp_dirs)

    print "***Konf based sampler***"
    root_dir = 'test_results/sahs_results/domain_two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/' \
               'q_config_num_train_5000_mse_weight_0.0_use_region_agnostic_True_loss_largemargin/' \
               'using_learned_sampler/1000/'
    wgangp_dir = root_dir + '/wgangp/'
    wgangp_dirs = get_target_dirs(wgangp_dir)
    n_nodes_konf = get_sampler_results(wgangp_dirs)

    plt.figure()
    plt.boxplot(
        [n_nodes_pose_heuristic, n_nodes_pose_sampler, n_nodes_gnn_heuristic, n_nodes_konf],
        labels=['SAHS\nPoseRank', 'SAHS\nRank\nPoseSampler', 'SAHS\nRank', 'SAHS\nRank\nWGANGP'],
        positions=[0, 1, 2, 3],
        whis=(10, 90), medianprops={'linewidth': 4.})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig("./plotters/plots/poses_vs_relaxed.eps")
    plt.savefig("../IJRR_GTAMP/figures/poses_vs_relaxed.eps")


def main():
    compare_losses()
    compare_representations()
    #compare_abstract_q_representations()
    #compare_abstract_q_losses()
    #compare_sampler_representations()
    #compare_sampler_losses()


if __name__ == '__main__':
    main()
