from trajectory_representation.concrete_node_state import ConcreteNodeState
from gtamp_utils import utils
from generators.TwoArmPaPGeneratory import TwoArmPaPGenerator
from generators.sampler import UniformSampler
from gtamp_utils.utils import get_pick_domain, get_place_domain
from generators.learning.utils.sampler_utils import make_input_for_place

from gtamp_problem_environments.mover_env import Mover
from planners.subplanners.motion_planner import BaseMotionPlanner

from generators.learning.pytorch_implementations.SuggestionNetwork import SuggestionNetwork
from generators.feasibility_checkers import two_arm_pick_feasibility_checker
from trajectory_representation.operator import Operator
from generators.feasibility_checkers.two_arm_pick_feasibility_checker import TwoArmPickFeasibilityChecker

from generators.learning.train_sampler import parse_args
from generators.learning.utils.model_creation_utils import create_policy

import numpy as np
import random
import pickle
from generators.learning.AdversarialVOO import AdversarialVOO


def create_environment(problem_idx):
    problem_env = Mover(problem_idx)
    openrave_env = problem_env.env
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    return problem_env, openrave_env


def compute_state(obj, region, problem_env):
    goal_entities = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4', 'home_region']
    return ConcreteNodeState(problem_env, obj, region, goal_entities)


def visualize_samples(samples, problem_env, target_obj_name):
    target_obj = problem_env.env.GetKinBody(target_obj_name)

    orig_color = utils.get_color_of(target_obj)
    utils.set_color(target_obj, [1, 0, 0])

    utils.visualize_placements(samples, target_obj_name)
    utils.set_color(target_obj, orig_color)


def get_feasible_pick(problem_env, target_obj):
    pick_domain = utils.get_pick_domain()
    dim_parameters = pick_domain.shape[-1]
    domain_min = pick_domain[0]
    domain_max = pick_domain[1]
    smpls = np.random.uniform(domain_min, domain_max, (500, dim_parameters)).squeeze()

    feasibility_checker = two_arm_pick_feasibility_checker.TwoArmPickFeasibilityChecker(problem_env)
    op = Operator('two_arm_pick', {"object": target_obj})

    for smpl in smpls:
        pick_param, status = feasibility_checker.check_feasibility(op, smpl, parameter_mode='ir_params')
        if status == 'HasSolution':
            op.continuous_parameters = pick_param
            return op


def generate_smpls(problem_env, sampler, plan):
    # make a prediction
    # Make a feasible pick sample for the target object

    idx = 0
    plan_action = plan[0]
    while True:
        if plan_action.discrete_parameters['place_region'] == 'home_region':
            utils.set_obj_xytheta(plan_action.continuous_parameters['place']['object_pose'], plan_action.discrete_parameters['object'])
        else:
            break
        idx+=1
        plan_action = plan[idx]

    target_obj_name = plan_action.discrete_parameters['object']
    place_region = 'loading_region'
    abstract_action = Operator('two_arm_pick_two_arm_place', {'object': target_obj_name, 'place_region': place_region})
    abstract_action.continuous_parameters = plan_action.continuous_parameters
    pick_base_pose = plan_action.continuous_parameters['pick']['q_goal']
    abstract_action.execute_pick()
    utils.set_robot_config(plan_action.continuous_parameters['place']['q_goal'])
    import pdb;pdb.set_trace()

    goal_entities = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4', 'home_region']
    sampler_state = ConcreteNodeState(problem_env, target_obj_name, place_region, goal_entities)
    inp = make_input_for_place(sampler_state, pick_base_pose)

    key_configs = pickle.load(open('prm.pkl', 'r'))[0]

    cols = inp['collisions'].squeeze()
    colliding_idxs = np.where(cols[:, 1] == 0)[0]
    colliding_key_configs = key_configs[colliding_idxs, :]

    samples = []
    values = []
    for _ in range(20):
        sample = sampler.sample_from_voo(inp['collisions'], inp['poses'],
                                         voo_iter=50,
                                         colliding_key_configs=None,
                                         tried_samples=np.array([]))
        values.append(sampler.value_network.predict([sample[None,:], inp['collisions'], inp['poses']])[0,0])
        sample = utils.decode_pose_with_sin_and_cos_angle(sample)
        samples.append(sample)
    utils.visualize_path(samples)
    print values
    # visualize_samples(samples, problem_env, target_obj_name)

    # visualize_samples([samples[np.argmax(values)]], problem_env, target_obj_name)
    return samples


def main():
    problem_seed = 0
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)
    # load up the plan data

    raw_dir = './planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    fname = 'pidx_%d_planner_seed_0_gnn_seed_0.pkl' %problem_seed
    plan_data = pickle.load(open(raw_dir+fname, 'r'))
    plan = plan_data['plan']

    config = parse_args()
    n_key_configs = 618
    n_collisions = 618
    sampler = create_policy(config, n_collisions, n_key_configs)
    sampler.load_weights(additional_name='epoch_' + str(200))

    utils.viewer()
    smpls = generate_smpls(problem_env, sampler, plan)
    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
