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


def generate_smpls(problem_env, sampler, target_obj_name):
    # make a prediction
    # Make a feasible pick sample for the target object

    SAMPLE_NEW_PICK = False
    place_region = 'loading_region'
    abstract_action = Operator('two_arm_pick_two_arm_place', {'object': target_obj_name, 'place_region': place_region})
    if SAMPLE_NEW_PICK:
        sampler = UniformSampler(problem_env.regions[place_region])
        generator = TwoArmPaPGenerator(None, abstract_action, sampler,
                                       n_parameters_to_try_motion_planning=10,
                                       n_iter_limit=200, problem_env=problem_env,
                                       reachability_clf=None)
        samples = generator.sample_next_point()
        pick_base_pose = samples['pick']['q_goal']
    else:
        pick_base_pose = np.array([1.38876479, -6.72431372, 2.13087774])
        abstract_action.continuous_parameters = {'pick':{'q_goal':pick_base_pose}}

    goal_entities = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4', 'home_region']
    sampler_state = ConcreteNodeState(problem_env, target_obj_name, place_region, goal_entities)
    inp = make_input_for_place(sampler_state, pick_base_pose)

    samples = [sampler.sample_from_voo(inp['collisions'], inp['goal_flags'], inp['poses'], inp['key_configs'], voo_iter=10) for _ in range(100)]
    values = [sampler.value_network.predict([s[None, :], inp['goal_flags'], inp['key_configs'], inp['collisions'], inp['poses']])[0,0] for s in samples]
    # Why does it ignore weights on the actions?

    print values
    import pdb;pdb.set_trace()

    samples = [utils.decode_pose_with_sin_and_cos_angle(s) for s in samples]
    utils.viewer()
    utils.visualize_placements(samples, target_obj_name)
    return samples



def main():
    problem_seed = 0
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_env, openrave_env = create_environment(problem_seed)

    config = parse_args()
    n_key_configs = 618
    n_collisions = 618
    sampler = create_policy(config, n_collisions, n_key_configs)
    sampler.load_weights(additional_name='epoch_' + str(5))

    obj_to_visualize = 'rectangular_packing_box1'
    smpls = generate_smpls(problem_env, sampler, target_obj_name=obj_to_visualize)
    visualize_samples(smpls, problem_env, obj_to_visualize)


if __name__ == '__main__':
    main()
