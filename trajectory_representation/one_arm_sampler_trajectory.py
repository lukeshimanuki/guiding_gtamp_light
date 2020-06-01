from sampler_trajectory import SamplerTrajectory
from trajectory_representation.one_arm_pap_state import OneArmPaPState
from gtamp_problem_environments.one_arm_mover_env import PaPOneArmMoverEnv
from gtamp_utils import utils
from openravepy import DOFAffine, RaveCreateKinBody, RaveCreateRobot
from trajectory_representation.concrete_node_state import OneArmConcreteNodeState

import os
import pickle
import numpy as np
import copy
import time


def rightarm_torso_base_distance(c1, c2, xmax_diff, ymax_diff, arm_max_diff):
    c1_base = c1[-3:]
    c2_base = c2[-3:]
    base_dist = utils.base_pose_distance(c1_base, c2_base, xmax_diff, ymax_diff)
    arm_config_1 = c1[:-3]
    arm_config_2 = c2[:-3]

    arm_config_1_ = copy.deepcopy(arm_config_1)
    clean_arm_config(arm_config_1_)
    arm_config_2_ = copy.deepcopy(arm_config_2)
    clean_arm_config(arm_config_2_)

    arm_diff = abs(arm_config_2_ - arm_config_1_)
    arm_dist = np.dot(arm_diff, 1.0 / arm_max_diff)
    return base_dist, arm_dist


def compute_v_manip(abs_state, goal_objs, key_configs):
    v_manip = np.zeros((len(key_configs), 1))
    stime = time.time()
    goal_obj = goal_objs[0]

    xmax_diff = np.max(np.array(key_configs)[:, -3:], axis=0)[0] - np.min(np.array(key_configs)[:, -3:], axis=0)[0]
    ymax_diff = np.max(np.array(key_configs)[:, -3:], axis=0)[1] - np.min(np.array(key_configs)[:, -3:], axis=0)[1]
    arm_max_diff = np.max(np.array(key_configs)[:, :-3], axis=0) - np.min(np.array(key_configs)[:, :-3], axis=0)

    if goal_obj in abs_state.nocollision_pick_op:
        pick_op = abs_state.nocollision_pick_op[goal_obj]
    else:
        pick_op, objs_in_way = abs_state.collision_pick_op[goal_obj]
    pick_op_config = pick_op.continuous_parameters['q_goal']

    best_arm_dist = np.inf
    best_base_dist = np.inf
    minidx = 0
    base_dist_threshold = 0.1
    while best_arm_dist == np.inf:
        for idx, k in enumerate(key_configs):
            base_dist, arm_dist = rightarm_torso_base_distance(pick_op_config, k, xmax_diff, ymax_diff, arm_max_diff)
            if base_dist < base_dist_threshold:
                if arm_dist < best_arm_dist:
                    minidx = idx
                    best_arm_dist = arm_dist
                    best_base_dist = base_dist
        if best_arm_dist == np.inf:
            base_dist_threshold += 0.01

    v_manip[minidx] = 1
    print 'v_manip creation time', time.time() - stime
    return v_manip


def clean_arm_config(arm_config):
    if arm_config[5] < 0:
        arm_config[5] += 2 * np.pi

    if arm_config[-1] < 0:
        arm_config[-1] += 2 * np.pi

    if arm_config[5] > 2 * np.pi:
        arm_config[5] -= 2 * np.pi

    if arm_config[-1] > 2 * np.pi:
        arm_config[-1] -= 2 * np.pi


class OneArmSAHSSamplerTrajectory(SamplerTrajectory):
    def __init__(self, problem_idx, n_objs_pack):
        SamplerTrajectory.__init__(self, problem_idx, n_objs_pack)
        key_configs_exist = os.path.isfile('one_arm_key_configs.pkl')
        if key_configs_exist:
            self.key_configs = pickle.load(open('one_arm_key_configs.pkl', 'r'))['konfs']
        else:
            self.key_configs = self.make_key_configs()

    def create_environment(self):
        problem_env = PaPOneArmMoverEnv(self.problem_idx)
        goal = ['rectangular_packing_box1_region'] + ['c_obst1']
        problem_env.set_goal(goal)
        return problem_env, problem_env.env

    def compute_state(self, abs_state, action):
        return

    @staticmethod
    def make_key_configs():
        raw_dir = './planning_experience/raw/one_arm_mover/n_objs_pack_1/qlearned_hcount_old_number_in_goal/q_config_num_train_5000_mse_weight_1.0_use_region_agnostic_False_mix_rate_1.0/n_mp_limit_5_n_iter_limit_2000/'

        # collect all the key configs
        exp_files = os.listdir(raw_dir)
        key_configs = []
        for exp_file in exp_files:
            plan = pickle.load(open(raw_dir + exp_file, 'r'))['plan']
            if plan is None:
                continue
            for p in plan:
                pick_config = p.continuous_parameters['pick']['q_goal']
                key_configs.append(pick_config)
            print 'Number of unreduced key configs', len(key_configs)
        xmax_diff = np.max(np.array(key_configs)[:, -3:], axis=0)[0] - np.min(np.array(key_configs)[:, -3:], axis=0)[0]
        ymax_diff = np.max(np.array(key_configs)[:, -3:], axis=0)[1] - np.min(np.array(key_configs)[:, -3:], axis=0)[1]
        arm_max_diff = np.max(np.array(key_configs)[:, :-3], axis=0) - np.min(np.array(key_configs)[:, :-3], axis=0)

        base_dist_threshold = 0.1
        arm_dist_threshold = 0.53
        reduced_key_configs = []
        for k in key_configs:
            if len(reduced_key_configs) == 0:
                reduced_key_configs.append(k)
                continue

            base_dists = []
            arm_dists = []
            for k2 in reduced_key_configs:
                base_dist, arm_dist = rightarm_torso_base_distance(k, k2, xmax_diff, ymax_diff, arm_max_diff)
                base_dists.append(base_dist)
                arm_dists.append(arm_dist)

            # either min base dist is bigger than 0.2 or arm dist is bigger than 0.53
            if np.min(base_dists) > base_dist_threshold or np.min(arm_dists) > arm_dist_threshold:
                reduced_key_configs.append(k)
            print "Key configs so far", len(reduced_key_configs)
        pickle.dump({'all_konfs': key_configs, 'konfs': reduced_key_configs}, open('one_arm_key_configs.pkl', 'wb'))
        return reduced_key_configs

    def compute_n_in_way_for_object_moved(self, object_moved, abs_state, goal_objs):
        assert len(goal_objs) == 1
        goal_obj = goal_objs[0]

        # Pick in way counts
        if goal_obj in abs_state.nocollision_pick_op:
            n_in_way = 0
        else:
            pick_op, objs_in_way = abs_state.collision_pick_op[goal_obj]
            n_in_way = int(object_moved in objs_in_way)
        return n_in_way

    def add_trajectory(self, plan):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()
        goal_objs = ['c_obst1']
        goal_region = ['rectangular_packing_box1_region']
        goal_entities = goal_region + goal_objs
        [utils.set_color(g, [1, 0, 0]) for g in goal_objs]

        self.problem_env = problem_env
        abs_state = OneArmPaPState(problem_env, goal_entities)
        abs_state.make_pklable()
        pickle.dump(abs_state, open('temp.pkl', 'wb'))
        #abs_state = pickle.load(open('temp.pkl', 'r'))
        abs_state.make_plannable(self.problem_env)
        for action in plan:
            assert action.type == 'one_arm_pick_one_arm_place'
            state = OneArmConcreteNodeState(abs_state, action, self.key_configs)
            action_info = self.get_action_info(action)

            ## Sanity check
            """
            abs_pick_pose = action_info['pick_abs_base_pose']
            portion, base_angle, facing_angle_offset \
                = utils.get_ir_parameters_from_robot_obj_poses(abs_pick_pose, state.abs_obj_pose)
            grasp_params = action_info['pick_action_parameters'][0:3]
            pick_params = np.hstack([grasp_params, portion, base_angle, facing_angle_offset])[None, :]
            print pick_params, action_info['pick_action_parameters']
            print np.all(np.isclose(pick_params, action_info['pick_action_parameters']))
            """
            ## end of sanity check

            object_moved = action.discrete_parameters['object']
            prev_n_in_way = self.compute_n_in_way_for_object_moved(object_moved, abs_state, goal_objs)
            prev_v_manip = compute_v_manip(abs_state, goal_objs, self.key_configs)
            action.execute_pick()
            action.execute()

            print "Computing state..."
            abs_state = OneArmPaPState(self.problem_env, goal_entities, parent_state=abs_state, parent_action=action)
            n_in_way = self.compute_n_in_way_for_object_moved(object_moved, abs_state, goal_objs)

            print action.discrete_parameters['object'], action.discrete_parameters['place_region']
            print 'Prev n in way {} curr n in way {}'.format(prev_n_in_way, n_in_way)

            self.add_sah_tuples(state, action_info, prev_n_in_way, n_in_way, prev_v_manip, None)

        self.add_state_prime()
        print "Done!"
        openrave_env.Destroy()
