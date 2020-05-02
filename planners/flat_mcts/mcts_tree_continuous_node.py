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



    def add_actions(self, action):
        new_action = Operator(operator_type=self.operator_skeleton.type,
                              discrete_parameters=self.operator_skeleton.discrete_parameters,
                              continuous_parameters=action)
        self.A.append(new_action)
        self.N[new_action] = 0

    def is_reevaluation_step(self, widening_parameter, infeasible_rwd, use_progressive_widening, use_ucb):
        n_arms = len(self.A)
        if n_arms < 1:
            return False

        feasible_actions = [a for a in self.A if a.continuous_parameters['is_feasible']]
        n_feasible_actions = len(feasible_actions)

        if n_feasible_actions < 1:
            return False

        there_is_new_action = np.any(np.array(self.N.values()) == 0)
        parent_node_value = np.max(self.parent.Q.values())
        curr_node_value = np.max(self.Q.values())
        q_value_improved = curr_node_value - parent_node_value == 0
        # if not, I am trying this action because of UCB parameter
        if there_is_new_action or q_value_improved:
            if there_is_new_action:
                print "There is new action. Re-evaluating"
            else:
                print "Curr node value {} Parent node value {} Q improved. Re-evaluating".format(curr_node_value, parent_node_value)
            return True
        else:
            print "Q value decreased:", curr_node_value - parent_node_value
            return False

        if not use_ucb:
            new_action = self.A[-1]
            is_new_action_infeasible = not new_action.continuous_parameters['is_feasible']
            if is_new_action_infeasible:
                return False

        if use_progressive_widening:
            n_actions = len(self.A)
            is_time_to_sample = n_actions <= widening_parameter * self.Nvisited
            print "PW %d / %.2f" % (n_actions, widening_parameter*self.Nvisited)
            return not is_time_to_sample
        else:
            if self.n_ucb_iterations < widening_parameter:
                self.n_ucb_iterations += 1
                print "Re-evaluation iter: %d / %d" % (self.n_ucb_iterations, widening_parameter)
                return True
            else:
                self.n_ucb_iterations = 0
                return False

    def get_never_evaluated_action(self):
        # get list of actions that do not have an associated Q values
        no_evaled = [a for a in self.A if a not in self.Q.keys()]
        no_evaled_feasible = [a for a in no_evaled if a.continuous_parameters['base_pose'] is not None]
        if len(no_evaled_feasible) == 0:
            return np.random.choice(no_evaled)
        else:
            return np.random.choice(no_evaled_feasible)

    def get_action_with_highest_ucb_value(self, actions, q_values):
        best_value = -np.inf
        best_action = None
        for action, value in zip(actions, q_values):
            ucb_value = self.compute_ucb_value(action)
            action_evaluation = value + ucb_value
            if action_evaluation > best_value:
                best_value = action_evaluation
                best_action = action
            #print "Placement {} Qval {}".format(action['place']['q_goal'], value)
            if action.continuous_parameters['is_feasible']:
                print action.continuous_parameters['place']['q_goal'], value, ucb_value, self.N[action]
            else:
                print "infeasible action", value
        # from gtamp_utils import utils
        # utils.visualize_placements(np.array( [actions[0].continuous_parameters['place']['q_goal']]), 'square_packing_box1')
        return best_action

    def perform_ucb_over_actions(self):
        assert not self.is_operator_skeleton_node
        actions = self.A
        q_values = [self.Q[a] for a in self.A]
        if len(q_values) == 1:
            best_action = self.A[-1]
        else:
            best_action = self.get_action_with_highest_ucb_value(actions, q_values)
        return best_action

        """
        there_is_new_action = np.any(np.array(self.N.values()) == 0)
        if there_is_new_action:
            for action in self.N:
                if action.continuous_parameters['is_feasible']:
                    print action.continuous_parameters['place']['q_goal'], self.N[action]
                else:
                    print "infeasible action", self.N[action]
            for action in self.N:
                if self.N[action] == 0:
                    return action
        else:
            parent_node_value = np.max(self.Q.values())  # potential_function(self.parent)
            curr_node_value = np.max(self.Q.values())
            q_value_improved = curr_node_value - parent_node_value == 0
            assert q_value_improved
            print 'Nsa', self.N.values()[np.argmax(self.Q.values())]
            #idx = np.where(np.array(self.Q.values()) - np.array(self.prevQ.values()) > 0)[0][0]
            return self.Q.keys()[np.argmax(self.Q.values())]

        """
