from reachability_classification.train_gnn import get_test_acc
from gtamp_utils import utils
from gtamp_problem_environments.mover_env import Mover
from planners.subplanners.motion_planner import BaseMotionPlanner
from generators.feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityCheckerWithoutSavingFeasiblePick
from trajectory_representation.operator import Operator

from generators.sampler import UniformSampler

from reachability_classification.datasets.dataset import GNNReachabilityDataset
from reachability_classification.classifiers.encoded_q_gnn import \
    EncodedQGNNReachabilityNet as ReachabilityNet

from reachability_classification.classifiers.separate_q0_qg_qk_ck_gnn_multiple_passes import \
    Separateq0qgqkckMultiplePassGNNReachabilityNet as ReachabilityNet

from reachability_classification.classifiers.relative_qgqk_gnn import \
    RelativeQgQkGNN as ReachabilityNet

import sys
import torch
import pickle
import numpy as np
import random
import time


def create_environment(problem_idx):
    problem_env = Mover(problem_idx)
    openrave_env = problem_env.env
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    return problem_env, openrave_env


def load_weights(net, epoch, action_type, seed, n_msg_passing, device):
    gnn_name = net.__class__._get_name(net)
    PATH = './reachability_classification/weights/atype_%s_seed_%d_%s_epoch_%d.pt' % (
        action_type, seed, gnn_name, epoch)
    # PATH = './reachability_classification/weights/atype_%s_%s_epoch_%d.pt' % (action_type, gnn_name, epoch)
    PATH = './reachability_classification/weights/atype_%s_n_msgs_%d_seed_%d_%s_epoch_%d.pt' \
           % (action_type, n_msg_passing, seed, gnn_name, epoch)
    print PATH
    net.load_state_dict(torch.load(PATH, device))
    net.eval()


def compute_relative_config(src_config, end_config):
    src_config = np.array(src_config)
    end_config = np.array(end_config)

    assert len(src_config.shape) == 2, \
        'Put configs in shapes (n_config,dim_config)'

    rel_config = end_config - src_config
    neg_idxs_to_fix = rel_config[:, -1] < -np.pi
    pos_idxs_to_fix = rel_config[:, -1] > np.pi

    # making unique rel angles; keep the range to [-pi,pi]
    rel_config[neg_idxs_to_fix, -1] = rel_config[neg_idxs_to_fix, -1] + 2 * np.pi
    rel_config[pos_idxs_to_fix, -1] = rel_config[pos_idxs_to_fix, -1] - 2 * np.pi

    return rel_config


def make_vertices(qg, key_configs, collisions, net):
    q0 = utils.get_robot_xytheta().squeeze()
    if key_configs.shape[-1] == 4:
        q0 = utils.encode_pose_with_sin_and_cos_angle(q0)
        qg = utils.encode_pose_with_sin_and_cos_angle(qg)
        repeat_q0 = np.repeat(np.array(q0)[None, :], 618, axis=0)
        repeat_qg = np.repeat(np.array(qg)[None, :], 618, axis=0)
        v = np.hstack([key_configs, repeat_q0, repeat_qg, collisions])

    if 'Relative' in net.__class__.__name__:
        rel_qg = compute_relative_config(q0[None, :], qg[None, :])
        rel_qk = compute_relative_config(q0[None, :], key_configs)
        repeat_qg = np.repeat(np.array(rel_qg), 618, axis=0)
        v = np.hstack([rel_qk, repeat_qg, collisions])
    else:
        repeat_q0 = np.repeat(np.array(q0)[None, :], 618, axis=0)
        repeat_qg = np.repeat(np.array(qg)[None, :], 618, axis=0)
        v = np.hstack([key_configs, repeat_q0, repeat_qg, collisions])

    # v = np.hstack([prm_vertices, repeat_q0, repeat_qg, self.collisions[idx]])
    v = v[None, :]
    v = torch.from_numpy(v).float()
    return v


def sample_qg(problem_env, region_name):
    q0 = utils.get_robot_xytheta()
    openrave_env = problem_env.env
    place_domain = utils.get_place_domain(region=problem_env.regions[region_name])
    dim_parameters = place_domain.shape[-1]
    domain_min = place_domain[0]
    domain_max = place_domain[1]
    qgs = np.random.uniform(domain_min, domain_max, (500, dim_parameters)).squeeze()
    for qg in qgs:
        utils.set_robot_config(qg)
        if not openrave_env.CheckCollision(problem_env.robot):
            break
    utils.set_robot_config(q0)
    return qg


def compute_collision_and_make_predictions(qg, key_configs, net):
    collisions = utils.compute_occ_vec(key_configs)
    col = utils.convert_binary_vec_to_one_hot(collisions.squeeze())
    vertex = make_vertices(qg, key_configs, col, net)
    output = net(vertex)
    # print output, (output > 0.5).numpy()[0, 0]
    # print output
    return (output > 0.5).numpy()[0, 0]


def check_motion_plan(qg, problem_env):
    problem_env.motion_planner.algorithm = 'rrt'
    motion, status = problem_env.motion_planner.get_motion_plan(qg, source='sampler',
                                                                n_iterations=[20, 50, 100, 500, 1000])
    # print status
    return motion, status


def get_clf_accuracy(problem_env, key_configs, net, region_name):
    misclf = 0

    false_positives = 0
    false_negatives = 0
    n_negatives = 0
    n_positives = 0
    qg = sample_qg(problem_env, region_name=region_name)
    pred = compute_collision_and_make_predictions(qg, key_configs, net)
    path, status = check_motion_plan(qg, problem_env)
    if status == "HasSolution":
        n_positives += 1
    else:
        n_negatives += 1

    if status == "HasSolution" and pred == False:
        misclf += 1
        false_negatives += 1
    elif status == "NoSolution" and pred == True:
        misclf += 1
        false_positives += 1

    return [misclf, false_negatives, false_positives, n_positives, n_negatives]


def compute_clf_rates_from_diff_pinstances(problem_env, key_configs, net, action_type):
    clf_rates = []
    for problem_seed in range(400):
        np.random.seed(problem_seed)
        random.seed(problem_seed)
        [utils.randomly_place_region(obj, problem_env.regions['loading_region']) for obj in problem_env.objects]
        utils.randomly_place_region(problem_env.robot, problem_env.regions['loading_region'])
        if action_type == 'place':
            sampler = UniformSampler(problem_env.regions['home_region'])
            checker = TwoArmPaPFeasibilityCheckerWithoutSavingFeasiblePick(problem_env)
            status = "NoSolution"
            while status != "HasSolution":
                pick_param = sampler.sample()
                target_obj = problem_env.objects[np.random.randint(8)]
                abstract_action = Operator(operator_type='two_arm_pick_two_arm_place',
                                           discrete_parameters={'object': target_obj, 'place_region':'home_region'})
                op_parameters, status = checker.check_feasibility(abstract_action, pick_param)
            utils.two_arm_pick_object(target_obj, op_parameters['pick'])
            region_name = 'loading_region'
        else:
            region_name = 'loading_region'

        clf_rate = get_clf_accuracy(problem_env, key_configs, net, region_name)
        if action_type == 'place':
            utils.two_arm_place_object(op_parameters['pick'])
        clf_rates.append(clf_rate)
        # print "pidx %d Clf rate %.2f " % (problem_seed, clf_rate)
        print np.array(clf_rates).sum(axis=0), problem_seed


def main():
    device = torch.device("cpu")
    edges = pickle.load(open('prm_edges_for_reachability_gnn.pkl', 'r'))
    n_msg_passing = 0
    net = ReachabilityNet(edges, n_key_configs=618, device=device, n_msg_passing=n_msg_passing)
    action_type = 'place'
    load_weights(net, 35, action_type, 1, n_msg_passing, device)

    get_data_test_acc = False
    if get_data_test_acc:
        seed = 0
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset = GNNReachabilityDataset(action_type)
        n_train = int(len(dataset) * 0.9)
        trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
        testloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=20, pin_memory=True)
        print "Getting test accuracy for n_test %d..." % len(testset)
        test_acc = get_test_acc(testloader, net, device, len(testset))
        print test_acc
        import pdb;
        pdb.set_trace()

    problem_seed = 1
    problem_env, openrave_env = create_environment(problem_seed)

    if 'Encoded' in net.__class__.__name__:
        tmp, _ = pickle.load(open('prm.pkl', 'r'))
        key_configs = np.zeros((618, 4))
        for pidx, p in enumerate(tmp):
            key_configs[pidx] = utils.encode_pose_with_sin_and_cos_angle(p)
    else:
        key_configs, _ = pickle.load(open('prm.pkl', 'r'))

    compute_clf_acc = False
    if compute_clf_acc:
        compute_clf_rates_from_diff_pinstances(problem_env, key_configs, net, action_type)

    utils.viewer()
    if action_type == 'place':
        status = "NoSolution"
        while status != "HasSolution":
            sampler = UniformSampler(problem_env.regions['home_region'])
            checker = TwoArmPaPFeasibilityCheckerWithoutSavingFeasiblePick(problem_env)
            pick_param = sampler.sample()
            target_obj = problem_env.objects[np.random.randint(8)]
            abstract_action = Operator(operator_type='two_arm_pick_two_arm_place',
                                       discrete_parameters={'object': target_obj, 'place_region': 'home_region'})
            op_parameters, status = checker.check_feasibility(abstract_action, pick_param)
        utils.two_arm_pick_object(target_obj, op_parameters['pick'])

    collisions = utils.compute_occ_vec(key_configs)
    col = utils.convert_binary_vec_to_one_hot(collisions.squeeze())
    qg = sample_qg(problem_env, 'home_region')
    utils.visualize_path([qg])

    vertex = make_vertices(qg, key_configs, col, net)
    vertex_outputs = net.get_vertex_activations(vertex).data.numpy().squeeze()
    """
    top_activations = np.max(vertex_outputs, axis=0)
    top_k_args = np.argsort(abs(top_activations))[-10:]
    """
    top_k_args = np.argsort(abs(vertex_outputs))[-30:]
    top_k_key_configs = key_configs[top_k_args, :]

    import pdb;
    pdb.set_trace()
    utils.visualize_path(top_k_key_configs)
    # todo visualize the activations

    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
