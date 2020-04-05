import argparse
import numpy as np
import random
import pickle

from manipulation.primitives.savers import DynamicEnvironmentStateSaver
from gtamp_problem_environments.mover_env import PaPMoverEnv
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from planners.subplanners.motion_planner import BaseMotionPlanner
from generators.learning.learning_algorithms.WGANGP import WGANgp
from gtamp_problem_environments.mover_env import Mover

from generators.sampler import UniformSampler, PlaceOnlyLearnedSampler
from generators.TwoArmPaPGeneratory import TwoArmPaPGenerator
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.operator import Operator

from gtamp_utils import utils


def get_problem_env(config, goal_region, goal_objs):
    problem_env = PaPMoverEnv(config.pidx)
    goal = [goal_region] + goal_objs
    [utils.set_color(o, [0, 0, 0.8]) for o in goal_objs]
    problem_env.set_goal(goal)
    return problem_env


def parse_arguments():
    parser = argparse.ArgumentParser(description='Greedy planner')
    parser.add_argument('-v', action='store_true', default=False)
    parser.add_argument('-pidx', type=int, default=0)  # used for threaded runs
    parser.add_argument('-epoch', type=int, default=10000)  # used for threaded runs
    config = parser.parse_args()
    return config


def create_environment(problem_idx):
    problem_env = Mover(problem_idx)
    problem_env.set_motion_planner(BaseMotionPlanner(problem_env, 'prm'))
    return problem_env


def get_learned_smpler(epoch):
    action_type = 'place'
    region = 'loading_region'
    model = WGANgp(action_type ,region)
    model.load_weights(epoch)
    return model


def load_planning_experience_data(problem_seed):
    raw_dir = './planning_experience/raw/uses_rrt/two_arm_mover/n_objs_pack_1/' \
              'qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_' \
              'use_region_agnostic_False_mix_rate_1.0/n_mp_limit_10_n_iter_limit_200/'
    fname = 'pidx_%d_planner_seed_0_gnn_seed_0.pkl' % problem_seed

    plan_data = pickle.load(open(raw_dir + fname, 'r'))
    plan = plan_data['plan']

    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env = create_environment(problem_seed)

    return plan, problem_env


class DummyAbstractState:
    def __init__(self, problem_env, goal_entities):
        self.prm_vertices, self.prm_edges = pickle.load(open('./prm.pkl', 'rb'))
        self.problem_env = problem_env
        self.goal_entities = goal_entities


def execute_plan_with_sampler(plan, sampler_model, problem_env, goal_entities):
    # abstract_state = ShortestPathPaPState(problem_env, goal_entities) # todo I actually don't need the whole thing
    abstract_state = DummyAbstractState(problem_env, goal_entities)
    for action in plan:
        abstract_action = action
        if abstract_action.discrete_parameters['place_region'] == 'loading_region':
            sampler = PlaceOnlyLearnedSampler(sampler_model, abstract_state, abstract_action)
            samples = sampler.samples

            generator = TwoArmPaPGenerator(abstract_state, abstract_action, sampler,
                                           n_parameters_to_try_motion_planning=100,
                                           n_iter_limit=200, problem_env=problem_env)
            next_pt = generator.sample_next_point()
            import pdb;pdb.set_trace()
        else:
            action.execute()


def main():
    # Load the planning experience data, and see how long it takes to solve the problem using the learned sampler

    config = parse_arguments()
    np.random.seed(config.pidx)
    random.seed(config.pidx)

    goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3', 'rectangular_packing_box4']
    goal_region = 'home_region'
    plan, problem_env = load_planning_experience_data(config.pidx)
    if config.v:
        utils.viewer()

    smpler = get_learned_smpler(config.epoch)
    execute_plan_with_sampler(plan, smpler, problem_env, goal_objs + [goal_region])

    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    main()
