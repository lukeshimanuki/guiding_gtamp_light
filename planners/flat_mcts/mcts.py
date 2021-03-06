from mcts_tree_continuous_node import ContinuousTreeNode
# from discrete_node_with_psa import DiscreteTreeNodeWithPsa
# from mcts_tree_discrete_pap_node import PaPDiscreteTreeNodeWithPriorQ
from discrete_node_with_prior_q import DiscreteTreeNodeWithPriorQ
from mcts_tree import MCTSTree
from generators.samplers.uniform_sampler import UniformSampler
from generators.samplers.voo_sampler import VOOSampler
from generators.TwoArmPaPGenerator import TwoArmPaPGenerator
from generators.voo import TwoArmVOOGenerator

# from generators.uniform import UniformPaPGenerator
# from generators.voo import PaPVOOGenerator

from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from trajectory_representation.state import StateWithoutCspacePredicates
from trajectory_representation.one_arm_pap_state import OneArmPaPState

## openrave helper libraries
from gtamp_utils import utils

import numpy as np
import sys
import socket
import pickle
import time
import os

sys.setrecursionlimit(15000)
DEBUG = False

hostname = socket.gethostname()
if hostname == 'dell-XPS-15-9560':
    from mcts_graphics import write_dot_file


# todo
#   create MCTS for each environment. Each one will have different compute_state functions
class MCTS:
    def __init__(self, parameters, problem_env, goal_entities, v_fcn, learned_q):
        # MCTS parameters
        self.widening_parameter = parameters.widening_parameter
        self.ucb_parameter = parameters.ucb_parameter
        self.time_limit = parameters.timelimit
        self.n_motion_plan_trials = parameters.n_motion_plan_trials
        self.use_ucb = parameters.use_ucb
        self.use_progressive_widening = parameters.pw
        self.n_feasibility_checks = parameters.n_feasibility_checks
        self.use_v_fcn = parameters.use_learned_q
        self.use_shaped_reward = parameters.use_shaped_reward
        self.planning_horizon = parameters.planning_horizon
        self.sampling_strategy = parameters.sampling_strategy
        self.explr_p = parameters.explr_p
        self.switch_frequency = parameters.switch_frequency
        self.parameters = parameters

        self.v_fcn = v_fcn
        self.learned_q = learned_q
        # Hard-coded params
        self.check_reachability = True
        self.discount_rate = 1.0

        # Environment setup
        self.problem_env = problem_env
        self.env = self.problem_env.env
        self.robot = self.problem_env.robot

        # MCTS initialization
        self.s0_node = None
        self.tree = MCTSTree(self.ucb_parameter)
        self.best_leaf_node = None
        self.goal_entities = goal_entities

        # Logging purpose
        self.search_time_to_reward = []
        self.reward_lists = []
        self.progress_list = []

        self.found_solution = False
        self.swept_volume_constraint = None

    def load_pickled_tree(self, fname=None):
        if fname is None:
            fname = 'tmp_tree.pkl'
        self.tree = pickle.load(open(fname, 'r'))

    def visit_all_nodes(self, curr_node):
        children = curr_node.children.values()
        print curr_node in self.tree.nodes
        for c in children:
            self.visit_all_nodes(c)

    def save_tree(self, fname=None):
        if fname is None:
            fname = 'tmp_tree.pkl'
        self.tree.make_tree_picklable()
        pickle.dump(self.tree, open(fname, 'wb'))

    def load_tree(self, fname=None):
        if fname is None:
            fname = 'tmp_tree.pkl'
        self.tree = pickle.load(open(fname, 'r'))

    def get_node_at_idx(self, idx):
        for n in self.tree.nodes:
            if n.idx == idx:
                return n
        return None

    def create_sampling_agent(self, node):
        operator_skeleton = node.operator_skeleton
        if 'one_arm' in self.problem_env.name:
            dont_check_motion_existence = True
        else:
            dont_check_motion_existence = False

        abstract_state = node.state
        abstract_action = node.operator_skeleton
        place_region = self.problem_env.regions[abstract_action.discrete_parameters['place_region']]
        if self.sampling_strategy == 'uniform':
            sampler = UniformSampler(place_region)
            generator = TwoArmPaPGenerator(abstract_state, abstract_action, sampler,
                                           n_parameters_to_try_motion_planning=self.n_motion_plan_trials,
                                           n_iter_limit=self.n_feasibility_checks, problem_env=self.problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
        elif self.sampling_strategy == 'voo':
            target_obj = abstract_action.discrete_parameters['object']
            sampler = VOOSampler(target_obj, place_region, self.explr_p,
                                 self.problem_env.reward_function.worst_potential_value)
            generator = TwoArmVOOGenerator(abstract_state, abstract_action, sampler,
                                           n_parameters_to_try_motion_planning=self.n_motion_plan_trials,
                                           n_iter_limit=self.n_feasibility_checks, problem_env=self.problem_env,
                                           pick_action_mode='ir_parameters',
                                           place_action_mode='object_pose')
        else:
            raise NotImplementedError
        return generator

    def compute_state(self, parent_node, parent_action):
        print "Computing state..."
        if self.problem_env.is_goal_reached():
            state = parent_node.state
        else:
            if parent_node is None:
                parent_state = None
            else:
                parent_state = parent_node.state

            if self.problem_env.name.find('one_arm') != -1:
                state = OneArmPaPState(self.problem_env,
                                       parent_state=parent_state,
                                       parent_action=parent_action,
                                       goal_entities=self.goal_entities)
            else:
                cache_file_name_for_debugging = './planners/mcts_cache_for_debug/pidx_{}_seed_{}.pkl'.format(
                    self.parameters.pidx, self.parameters.planner_seed)
                is_root_node = parent_state is None
                cache_file_exists = os.path.isfile(cache_file_name_for_debugging)
                is_beomjoon_local_machine = False #socket.gethostname() == 'lab'
                if cache_file_exists and is_root_node and is_beomjoon_local_machine:
                    state = pickle.load(open(cache_file_name_for_debugging, 'r'))
                    state.make_plannable(self.problem_env)
                else:
                    state = ShortestPathPaPState(self.problem_env,
                                                 parent_state=parent_state,
                                                 parent_action=parent_action,
                                                 goal_entities=self.goal_entities, planner='mcts')
                    if is_root_node and is_beomjoon_local_machine:
                        state.make_pklable()
                        pickle.dump(state, open(cache_file_name_for_debugging, 'wb'))
                        state.make_plannable(self.problem_env)
        return state

    def get_current_state(self, parent_node, parent_action, is_parent_action_infeasible):
        # this needs to be factored
        # why do I need a parent node? Can I just get away with parent state?
        is_operator_skeleton_node = (parent_node is None) or (not parent_node.is_operator_skeleton_node)
        if is_parent_action_infeasible:
            state = None
        elif is_operator_skeleton_node:
            state = self.compute_state(parent_node, parent_action)
        else:
            state = parent_node.state

        return state

    def create_node(self, parent_action, depth, parent_node, is_parent_action_infeasible, is_init_node=False):
        state_saver = utils.CustomStateSaver(self.problem_env.env)
        is_operator_skeleton_node = (parent_node is None) or (not parent_node.is_operator_skeleton_node)
        state = self.get_current_state(parent_node, parent_action, is_parent_action_infeasible)

        if is_operator_skeleton_node:
            applicable_op_skeletons = self.problem_env.get_applicable_ops(parent_action)
            node = DiscreteTreeNodeWithPriorQ(state, self.ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                              is_init_node, applicable_op_skeletons, self.learned_q)
        else:
            node = ContinuousTreeNode(state, parent_action, self.ucb_parameter, depth, state_saver,
                                      is_operator_skeleton_node, is_init_node)
            node.sampling_agent = self.create_sampling_agent(node)

        node.parent = parent_node
        node.parent_action = parent_action
        return node

    @staticmethod
    def get_best_child_node(node):
        if len(node.children) == 0:
            return None
        else:
            best_child_action_idx = np.argmax(node.Q.values())
            best_child_action = node.Q.keys()[best_child_action_idx]
            return node.children[best_child_action]

    def retrace_best_plan(self):
        plan = []
        _, _, best_leaf_node = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        curr_node = best_leaf_node

        while not curr_node.parent is None:
            plan.append(curr_node.parent_action)
            curr_node = curr_node.parent

        plan = plan[::-1]
        return plan, best_leaf_node

    def get_best_goal_node(self):
        leaves = self.tree.get_leaf_nodes()
        goal_nodes = [leaf for leaf in leaves if leaf.is_goal_node]
        if len(goal_nodes) > 1:
            best_traj_reward, curr_node, _ = self.tree.get_best_trajectory_sum_rewards_and_node(self.discount_rate)
        else:
            curr_node = goal_nodes[0]
        return curr_node

    def switch_init_node(self, node):
        self.s0_node.is_init_node = False
        self.s0_node = node
        self.s0_node.is_init_node = True
        self.problem_env.reset_to_init_state(node)
        self.found_solution = False

    @staticmethod
    def choose_child_node_to_descend_to(node):
        # todo: implement the one with highest visitation
        if node.is_operator_skeleton_node and len(node.A) == 1:
            # descend to grand-child
            only_child_node = node.children.values()[0]
            best_action = only_child_node.Q.keys()[np.argmax(only_child_node.Q.values())]
            best_node = only_child_node.children[best_action]
        else:
            best_action = node.Q.keys()[np.argmax(node.Q.values())]
            best_node = node.children[best_action]
        return best_node

    def log_current_tree_to_dot_file(self, iteration, node_to_search_from):
        if socket.gethostname() == 'dell-XPS-15-9560':
            write_dot_file(self.tree, iteration, '', node_to_search_from)

    def log_performance(self, elapsed_time, history_n_objs_in_goal, n_feasibility_checks, iteration):
        self.search_time_to_reward.append([elapsed_time, iteration,
                                           n_feasibility_checks['ik'],
                                           n_feasibility_checks['mp'],
                                           max(history_n_objs_in_goal)])

    def get_total_n_feasibility_checks(self):
        total_ik_checks = 0
        total_mp_checks = 0
        for n in self.tree.nodes:
            if n.sampling_agent is not None:
                total_ik_checks += n.sampling_agent.n_ik_checks
                total_mp_checks += n.sampling_agent.n_mp_checks
        return {'mp': total_mp_checks, 'ik': total_ik_checks}

    def is_time_to_switch(self, root_node):
        reached_frequency_limit = max(root_node.N.values()) >= self.switch_frequency
        has_enough_actions = True #root_node.is_operator_skeleton_node #or len(root_node.A) > 3
        return reached_frequency_limit and has_enough_actions

    def get_node_to_switch_to(self, node_to_search_from):
        is_time_to_switch_node = self.is_time_to_switch(node_to_search_from)
        if not is_time_to_switch_node:
            return node_to_search_from

        if node_to_search_from.is_operator_skeleton_node:
            node_to_search_from = node_to_search_from.get_child_with_max_value()
        else:
            max_child = node_to_search_from.get_child_with_max_value()
            if max_child.parent_action.continuous_parameters['is_feasible']:
                node_to_search_from = node_to_search_from.get_child_with_max_value()

        return self.get_node_to_switch_to(node_to_search_from)

    def search(self, n_iter=np.inf, iteration_for_tree_logging=0, node_to_search_from=None, max_time=np.inf):
        depth = 0
        elapsed_time = 0

        if node_to_search_from is None:
            self.s0_node = self.create_node(None,
                                            depth=0,
                                            parent_node=None,
                                            is_parent_action_infeasible=False,
                                            is_init_node=True)
            self.tree.set_root_node(self.s0_node)
            node_to_search_from = self.s0_node

        if n_iter == np.inf:
            n_iter = 999999

        new_trajs = []
        plan = []
        history_of_n_objs_in_goal = []
        for iteration in range(1, n_iter):
            print '*****SIMULATION ITERATION %d' % iteration
            self.problem_env.reset_to_init_state(node_to_search_from)

            new_traj = []
            stime = time.time()
            self.simulate(node_to_search_from, node_to_search_from, depth, new_traj)
            elapsed_time += time.time() - stime

            n_feasibility_checks = self.get_total_n_feasibility_checks()
            n_objs_in_goal = len(self.problem_env.get_objs_in_region('home_region'))
            history_of_n_objs_in_goal.append(n_objs_in_goal)
            self.log_performance(elapsed_time, history_of_n_objs_in_goal, n_feasibility_checks, iteration)
            print "Time {} n_feasible_checks {} max progress {}".format(elapsed_time,
                                                                        n_feasibility_checks['ik'],
                                                                        max(history_of_n_objs_in_goal))

            #is_time_to_switch_node = self.is_time_to_switch(node_to_search_from)
            #if is_time_to_switch_node:
            #    node_to_search_from = self.get_node_to_switch_to(node_to_search_from)

            if self.found_solution:
                print "Optimal score found"
                plan, _ = self.retrace_best_plan()
                break

            if elapsed_time > max_time:
                print "Time is up"
                break

        self.problem_env.reset_to_init_state(node_to_search_from)
        return self.search_time_to_reward, n_feasibility_checks, plan

    def get_best_trajectory(self, node_to_search_from, trajectories):
        traj_rewards = []
        curr_node = node_to_search_from
        for trj in trajectories:
            traj_sum_reward = 0
            for aidx, a in enumerate(trj):
                traj_sum_reward += np.power(self.discount_rate, aidx) * curr_node.reward_history[a][0]
                curr_node = curr_node.children[a]
            traj_rewards.append(traj_sum_reward)
        return trajectories[np.argmax(traj_rewards)], curr_node

    def choose_action(self, curr_node, depth):
        if curr_node.is_operator_skeleton_node:
            print "Skeleton node"
            # here, perform psa with the learned q
            action = curr_node.perform_ucb_over_actions()
        else:
            print 'Instance node'
            if curr_node.sampling_agent is None:  # this happens if the tree has been pickled
                curr_node.sampling_agent = self.create_sampling_agent(curr_node)

            if not self.use_progressive_widening:
                w_param = self.widening_parameter
            else:
                w_param = self.widening_parameter * np.power(0.9, depth / 2)
            if not curr_node.is_reevaluation_step(w_param,
                                                  self.problem_env.reward_function.infeasible_reward,
                                                  self.use_progressive_widening,
                                                  self.use_ucb):
                print "Sampling new action"
                new_continuous_parameters = self.sample_continuous_parameters(curr_node)
                curr_node.add_actions(new_continuous_parameters)
                action = curr_node.A[-1]
            else:
                print "Re-evaluation of actions"
                # todo I want UCB here
                if self.use_ucb:
                    action = curr_node.perform_ucb_over_actions()
                else:
                    action = curr_node.choose_new_arm()
        return action

    @staticmethod
    def update_goal_node_statistics(curr_node, reward):
        # todo rewrite this function
        curr_node.Nvisited += 1
        curr_node.reward = reward

    def simulate(self, curr_node, node_to_search_from, depth, new_traj):
        if self.problem_env.reward_function.is_goal_reached():
            if not curr_node.is_goal_and_already_visited:
                self.found_solution = True
                curr_node.is_goal_node = True
                print "Solution found, returning the goal reward", self.problem_env.reward_function.goal_reward
                self.update_goal_node_statistics(curr_node, self.problem_env.reward_function.goal_reward)
            return self.problem_env.reward_function.goal_reward

        if depth == self.planning_horizon:
            print "Depth limit reached"
            return 0

        if DEBUG:
            print "At depth ", depth
            print "Is it time to pick?", self.problem_env.is_pick_time()

        action = self.choose_action(curr_node, depth)
        is_action_feasible = self.apply_action(curr_node, action)

        if not curr_node.is_action_tried(action):
            next_node = self.create_node(action, depth + 1, curr_node, not is_action_feasible)
            self.tree.add_node(next_node, action, curr_node)
            reward = self.problem_env.reward_function(curr_node.state, next_node.state, action, depth)
            next_node.parent_action_reward = reward
            next_node.sum_ancestor_action_rewards = next_node.parent.sum_ancestor_action_rewards + reward
        else:
            next_node = curr_node.children[action]
            reward = next_node.parent_action_reward

        print "Reward", reward

        if not is_action_feasible:
            # this (s,a) is a dead-end
            print "Infeasible action"
            if self.use_v_fcn:
                sum_rewards = reward + curr_node.parent.v_fcn[curr_node.parent_action]
                print sum_rewards
            else:
                sum_rewards = reward
        else:
            sum_rewards = reward + self.discount_rate * self.simulate(next_node, node_to_search_from,
                                                                      depth + 1, new_traj)

        curr_node.update_node_statistics(action, sum_rewards, reward)
        if curr_node == node_to_search_from and curr_node.parent is not None:
            self.update_ancestor_node_statistics(curr_node.parent, curr_node.parent_action, sum_rewards)

        return sum_rewards

    def update_ancestor_node_statistics(self, node, action, child_sum_rewards):
        if node is None:
            return
        parent_reward_to_node = node.reward_history[action][0]
        parent_sum_rewards = parent_reward_to_node + child_sum_rewards  # rwd up to parent + rwd from child to leaf
        node.update_node_statistics(action, parent_sum_rewards, parent_reward_to_node)
        self.update_ancestor_node_statistics(node.parent, node.parent_action, parent_sum_rewards)

    def apply_action(self, node, action):
        if node.is_operator_skeleton_node:
            print "Applying skeleton", action.type, action.discrete_parameters['object'], \
                action.discrete_parameters['place_region']
            is_feasible = self.problem_env.apply_operator_skeleton(node.state, action)
        else:
            is_feasible = self.problem_env.apply_operator_instance(node.state, action, self.check_reachability)
            if is_feasible:
                print "Applying instance", action.discrete_parameters['object'], action.discrete_parameters[
                    'place_region'], action.continuous_parameters['place']['q_goal']
            else:
                print "Applying infeasible instance", action.discrete_parameters['object'], action.discrete_parameters[
                    'place_region']

        return is_feasible

    def sample_continuous_parameters(self, node):
        if self.problem_env.name.find('one_arm') != -1:
            raise NotImplementedError
        else:
            if self.sampling_strategy == 'voo':
                action_parameters = [np.hstack([a.continuous_parameters['pick']['action_parameters'],
                                                a.continuous_parameters['place']['action_parameters']])
                                     for a in node.Q.keys()]
                q_values = node.Q.values()
                # todo
                #   save all the mp values that did not work out
                smpled_param = node.sampling_agent.sample_next_point(action_parameters, q_values)
                if not smpled_param['is_feasible'] and self.sampling_strategy=='voo':
                    node.sampling_agent.update_mp_infeasible_samples(smpled_param['samples'])
            else:
                smpled_param = node.sampling_agent.sample_next_point()
                node.needs_to_sample = False
        return smpled_param
