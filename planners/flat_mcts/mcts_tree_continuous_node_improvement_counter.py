import numpy as np
from trajectory_representation.operator import Operator
from planners.flat_mcts.mcts_tree_node import TreeNode


def upper_confidence_bound(n, n_sa):
    return 2 * np.sqrt(np.log(n) / float(n_sa))


class ContinuousTreeNode(TreeNode):
    def __init__(self, state, operator_skeleton, ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                 is_init_node):
        TreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node)
        self.operator_skeleton = operator_skeleton
        self.max_sum_rewards = {}
        self.needs_to_sample = False
        self.improvement_counter_limit = 5
        self.improvement_counter = self.improvement_counter_limit

    def add_actions(self, action):
        new_action = Operator(operator_type=self.operator_skeleton.type,
                              discrete_parameters=self.operator_skeleton.discrete_parameters,
                              continuous_parameters=action)
        self.A.append(new_action)
        self.N[new_action] = 0

    def is_reevaluation_step(self, widening_parameter, infeasible_rwd, use_progressive_widening, use_ucb):
        n_arms = len(self.A)
        if n_arms < 1:
            print "There is zero arm"
            return False

        feasible_actions = [a for a in self.A if a.continuous_parameters['is_feasible']]
        n_feasible_actions = len(feasible_actions)
        if n_feasible_actions < 1:
            print "There are fewer than one feasible actions"
            return False

        there_is_new_action = np.any(np.array(self.N.values()) < 2)
        if not there_is_new_action:
            past_node_value = np.array(self.prevQ.values())
            curr_node_value = np.array(self.Q.values())
            q_value_improved = np.any(curr_node_value - past_node_value >= 0)
        else:
            q_value_improved = True

        if there_is_new_action or q_value_improved:
            if there_is_new_action:
                print "There is new action. Re-evaluating"
            else:
                print "Curr node value {} Past node value {} Q improved. Re-evaluating".format(curr_node_value,
                                                                                               past_node_value)
            self.improvement_counter = 5
            is_reevaluation = True
        else:
            print "Q value decreased:", curr_node_value, past_node_value, curr_node_value - past_node_value
            self.improvement_counter -= 1
            print "Improvement counter", self.improvement_counter
            if self.improvement_counter > 0:
                is_reevaluation = True
            else:
                is_reevaluation = False

        print "Is reevaluation?", is_reevaluation
        return is_reevaluation

    def get_never_evaluated_action(self):
        # get list of actions that do not have an associated Q values
        no_evaled = [a for a in self.A if a not in self.Q.keys()]
        no_evaled_feasible = [a for a in no_evaled if a.continuous_parameters['base_pose'] is not None]
        if len(no_evaled_feasible) == 0:
            return np.random.choice(no_evaled)
        else:
            return np.random.choice(no_evaled_feasible)

    def perform_ucb_over_actions(self):
        for action in self.N:
            if action.continuous_parameters['is_feasible']:
                print action.continuous_parameters['place']['q_goal'], self.Q[action], self.N[action]
            else:
                print "infeasible action", self.N[action]

        there_is_new_action = np.any(self.N.values() < 2)
        if there_is_new_action:
            return self.N.keys()[np.argmin(self.N.values())]
        else:
            past_node_value = np.array(self.prevQ.values())
            curr_node_value = np.array(self.Q.values())
            q_value_improved = np.any(curr_node_value - past_node_value >= 0)
            if not q_value_improved:
                assert self.improvement_counter > 0
                return self.Q.keys()[np.argmax(self.Q.values())]
            else:
                return self.Q.keys()[np.argmax(q_value_improved)]

        return action

    def update_node_statistics(self, action, sum_rewards, reward):
        self.Nvisited += 1

        is_action_never_tried = self.N[action] == 0
        if is_action_never_tried:
            self.reward_history[action] = [reward]
            self.Q[action] = sum_rewards
            self.N[action] += 1
        elif self.children[action].is_goal_node:
            self.reward_history[action] = [reward]
            self.Q[action] = sum_rewards
            self.N[action] += 1
        else:
            self.reward_history[action].append(reward)
            self.N[action] += 1
            children_action_values = [self.children[action].Q[child_action] for child_action in self.children[action].Q]
            self.prevQ[action] = self.Q[action]
            self.Q[action] = reward + np.max(children_action_values)
            print 'Rwd %.5f, max child val %.5f' % (reward, np.max(children_action_values))
            print "Updated. Current Q %.5f Prev Q %.5f" % (self.Q[action], self.prevQ[action])

            if len(self.prevQ.values()) == len(self.Q.values()):
                parent_node_value = np.array(self.prevQ.values())
                curr_node_value = np.array(self.Q.values())
                q_value_improved = np.any(curr_node_value - parent_node_value >= 0)
                if q_value_improved:
                    self.needs_to_sample = False
                    self.improvement_counter = self.improvement_counter_limit
                else:
                    self.needs_to_sample = True
            else:
                self.needs_to_sample = False
