from planners.flat_mcts.mcts_tree_node import TreeNode


class DiscreteTreeNode(TreeNode):
    def __init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node, actions, learned_q):
        self.learned_q = learned_q
        TreeNode.__init__(self, state, ucb_parameter, depth, state_saver, is_operator_skeleton_node, is_init_node)
        self.add_actions(actions)

    def add_actions(self, actions):
        if self.is_operator_skeleton_node:
            for action in actions:
                self.A.append(action)
                self.N[action] = 0

    def perform_ucb_over_actions(self):
        assert self.is_operator_skeleton_node
        actions = self.A
        q_values = [self.Q[a] for a in self.A]
        best_action = self.get_action_with_highest_ucb_value(actions, q_values)
        return best_action



