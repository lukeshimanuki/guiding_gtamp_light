from reachability_classification.classifiers.separate_q0_qg_qk_ck_gnn_multiple_passes import \
    Separateq0qgqkckMultiplePassGNNReachabilityNet as ReachabilityNet
from reachability_classification.datasets.dataset import GNNReachabilityDataset
from reachability_classification.train_gnn import get_test_acc

from visualize_learned_sampler import create_environment
from gtamp_utils import utils

import torch
import pickle
import numpy as np
import random
import time


def load_weights(net, epoch):
    gnn_name = net.__class__._get_name(net)
    PATH = './reachability_classification/weights/%s_epoch_%d.pt' % (gnn_name, epoch)
    net.load_state_dict(torch.load(PATH))
    net.eval()


def make_vertices(qg, key_configs, collisions):
    q0 = utils.get_robot_xytheta().squeeze()
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
    vertex = make_vertices(qg, key_configs, col)
    output = net(vertex)
    print output, (output > 0.5).numpy()[0, 0]
    return (output > 0.5).numpy()[0, 0]


def check_motion_plan(qg, problem_env):
    problem_env.motion_planner.algorithm = 'rrt'
    motion, status = problem_env.motion_planner.get_motion_plan(qg, source='sampler')
    print status
    return motion, status


def get_clf_accuracy(problem_env, key_configs, net):
    misclf = 0

    for _ in range(100):
        qg = sample_qg(problem_env, region_name='loading_region')
        pred = compute_collision_and_make_predictions(qg, key_configs, net)
        stime = time.time()
        path, status = check_motion_plan(qg, problem_env)
        print time.time() - stime
        if status == "HasSolution" and pred == False:
            misclf += 1
        elif status == "NoSolution" and pred == True:
            misclf += 1
    return (100.0 - misclf) / 100.0


def main():
    device = torch.device("cpu")
    edges = pickle.load(open('prm_edges_for_reachability_gnn.pkl', 'r'))
    net = ReachabilityNet(edges, n_key_configs=618, device=device)
    load_weights(net, 98)
    """
    dataset = GNNReachabilityDataset()
    n_train = int(len(dataset) * 0.9)
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    testloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=20, pin_memory=True)
    test_acc = get_test_acc(testloader, net, device, 5000)
    import pdb;pdb.set_trace()
    """

    problem_seed = 6
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)
    utils.viewer()

    key_configs, _ = pickle.load(open('prm.pkl', 'r'))

    # qg = np.array([0.0534, -7.87, 0])
    # utils.visualize_path([qg])

    clf_rate = get_clf_accuracy(problem_env, key_configs, net)
    print "Clf rate %.2f " % clf_rate
    import pdb;
    pdb.set_trace()

    vertex_outputs = net.get_vertex_activations(vertex).data.numpy().squeeze()
    top_k_args = np.argsort(abs(vertex_outputs))[-60:]
    top_k_key_configs = key_configs[top_k_args, :]
    utils.visualize_path(top_k_key_configs)
    # todo visualize the activations

    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
