from mcts_tree_discrete_node import DiscreteTreeNode
from planners.heuristics import get_objects_to_move
from planners.sahs.helper import compute_heuristic

import numpy as np


class DiscreteTreeNodeWithPriorQ(DiscreteTreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions,
                 learned_q):
        # psa is based on the number of objs to move
        DiscreteTreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node,
                                  is_init_node, actions, learned_q)
        is_infeasible_state = self.state is None
        if is_infeasible_state:
            for a in self.A:
                self.Q[a] = 0
        else:
            for a in actions:
                self.Q[a] = -compute_heuristic(state, a, learned_q, 'qlearned_hcount_old_number_in_goal', mixrate=1.0)
            self.learned_q_values = {a: self.learned_q.predict(self.state, a) for a in actions}

    def perform_ucb_over_actions(self):
        # todo this function is to be deleted once everything has been implemented
        assert self.is_operator_skeleton_node
        actions = self.A
        q_values = [self.Q[a] for a in self.A]

        best_action = self.get_action_with_highest_ucb_value(actions, q_values)
        return best_action

    def print_q_value(self, action):
        value = self.Q[action]
        ucb_value = self.compute_ucb_value(action)
        action_evaluation = value + ucb_value
        exp_sum = np.sum([np.exp(q) for q in self.learned_q_values.values()])
        q_bonus = np.exp(self.learned_q_values[action]) / float(exp_sum + 1e-5)

        obj_name = action.discrete_parameters['object']
        region_name = action.discrete_parameters['place_region']
        print "%25s %15s Reachable? %d  ManipFree? %d IsGoal? %d Q? %.5f QBonus? %.5f UCB? %.5f Q+UCB? %.5f Nsa %d" \
              % (obj_name, region_name, self.state.is_entity_reachable(obj_name),
                 self.state.binary_edges[(obj_name, region_name)][-1],
                 obj_name in self.state.goal_entities, self.Q[action], q_bonus,
                 ucb_value, action_evaluation, self.N[action])

    def get_action_with_highest_ucb_value(self, actions, q_values):
        best_value = -np.inf
        action_evaluation_values = []

        for action, value in zip(actions, q_values):
            ucb_value = self.compute_ucb_value(action)
            action_evaluation = value + ucb_value
            action_evaluation_values.append(action_evaluation)
            self.print_q_value(action)
            if action_evaluation > best_value:
                best_value = action_evaluation

        boolean_idxs_with_highest_ucb = (np.max(action_evaluation_values) == action_evaluation_values).squeeze()
        best_action_idx = np.random.randint(np.sum(boolean_idxs_with_highest_ucb))
        best_action = np.array(actions)[boolean_idxs_with_highest_ucb][best_action_idx]
        return best_action
