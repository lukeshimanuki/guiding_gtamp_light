from gtamp_problem_environments.mover_env import PaPMoverEnv
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from generators.samplers.pick_place_learned_sampler import compute_v_manip
from trajectory_representation.concrete_node_state import TwoArmConcreteNodeState
from gtamp_utils import utils
import numpy as np

import os
import pickle

problem_env = PaPMoverEnv(0)
goal_objs = ['square_packing_box1', 'rectangular_packing_box3']
goal_region = 'home_region'
goal_entities = goal_objs + [goal_region]
problem_env.set_goal(goal_objs, goal_region)
abs_state_file = './test_scripts/paper_figure_generators/abs_state.pkl'
if os.path.isfile(abs_state_file):
    abs_state = pickle.load(open(abs_state_file, 'r'))
else:
    abs_state = ShortestPathPaPState(problem_env, goal_entities)
    abs_state.make_pklable()
    pickle.dump(abs_state, open(abs_state_file, 'wb'))

abs_actions = problem_env.get_applicable_ops()
abs_state.problem_env = problem_env
concrete_state = TwoArmConcreteNodeState(abs_state, abs_actions[0])

collision_vec = concrete_state.pick_collision_vector.squeeze()
key_configs = concrete_state.key_configs
reduced_key_configs = key_configs[np.random.choice(range(len(key_configs)), 40)]

problem_env.env.SetViewer('qtcoin')
viewer = problem_env.env.GetViewer()
cam_trans = np.array([[0.99990954, -0.00714068, -0.01139814, 1.86213541],
                      [-0.00205184, -0.9184984, 0.39541936, -6.90927124],
                      [-0.01329274, -0.3953602, -0.91842997, 9.73728466],
                      [0., 0., 0., 1.]])
viewer.SetCamera(cam_trans)
vmanip = compute_v_manip(abs_state, goal_objs).squeeze()
konf_vmanip = key_configs[vmanip == 1, :]

reduced_key_configs = np.vstack([konf_vmanip, key_configs[np.random.choice(range(len(key_configs)), 50)]])


colliding_konfs = []
default = utils.get_robot_xytheta()
for k in reduced_key_configs:
    utils.set_robot_config(k)
    if problem_env.env.CheckCollision(problem_env.robot):
        colliding_konfs.append(k)
utils.set_robot_config(default)

utils.visualize_path(colliding_konfs)
utils.visualize_path(reduced_key_configs)
utils.visualize_path(konf_vmanip)

