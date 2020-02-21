
class Node(object):
    def __init__(self, parent, parent_action, state, reward=0):
        self.parent = parent  # parent.state is initial state
        self.parent_action = parent_action
        self.state = state  # resulting state
        self.reward = reward  # resulting reward
        self.heuristic_vals = {}
        self.generators = {}
        self.h_for_sampler_training = None  # for planning experience purpose
        self.is_goal_traj = False
        self.goal_action = None
        if parent is None:
            self.depth = 1
        else:
            self.depth = parent.depth + 1

    def set_heuristic(self, action, val):
        self.heuristic_vals[action] = val

    def backtrack(self):
        node = self
        while node is not None:
            yield node
            node = node.parent