from gtamp_problem_environments.reward_functions.reward_function import GenericRewardFunction
from planners.heuristics import compute_hcount
from planners.sahs.helper import compute_heuristic


class ShapedRewardFunction(GenericRewardFunction):
    def __init__(self, problem_env, goal_objects, goal_region, planning_horizon):
        GenericRewardFunction.__init__(self, problem_env, goal_objects, goal_region, planning_horizon)
        self.worst_potential_value = -8 # must move all objects
        # potential_function is minus of the number of objects to move (smaller the n_objs_to_move, the better)

    def potential_function(self, node):
        # potential function?  Max of value of (s,a)
        is_cont_node = node.__class__.__name__ == 'ContinuousTreeNode'
        if is_cont_node:
            discrete_node = node.parent
        else:
            discrete_node = node
        potential_vals = []
        for a in discrete_node.A:
            val = -compute_heuristic(discrete_node.state, a, discrete_node.learned_q,
                                    'qlearned_hcount_old_number_in_goal', mixrate=1.0)
            potential_vals.append(val)
        return max(potential_vals)

    def __call__(self, curr_node, next_node, action, time_step):
        if action.is_skeleton:
            return 0
        else:
            next_state = next_node.state
            if self.is_goal_reached():
                return 1
            elif next_state is None:
                return self.worst_potential_value
            else:
                assert curr_node.__class__.__name__ == 'ContinuousTreeNode'
                print "Current node potential values"
                potential_curr = self.potential_function(curr_node)
                print "Next node potential values"
                potential_next = self.potential_function(next_node)
                curr_state = curr_node.state
                true_reward = GenericRewardFunction.__call__(self, curr_state, next_state, action, time_step)

                print "(Hcount,Hcount_prime) = %.5f, %.5f" % (potential_curr, potential_next)
                shaping_val = potential_next - potential_curr
                return true_reward + shaping_val


