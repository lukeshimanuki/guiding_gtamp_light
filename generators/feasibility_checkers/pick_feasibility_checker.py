from gtamp_utils.utils import get_pick_base_pose_and_grasp_from_pick_parameters
import time


class PickFeasibilityChecker(object):
    def __init__(self, problem_env, action_mode):
        self.problem_env = problem_env
        self.env = problem_env.env
        self.robot = self.env.GetRobots()[0]
        self.objects_to_check_collision = []
        self.action_mode = action_mode

    def check_feasibility(self, operator_skeleton, pick_parameters, swept_volume_to_avoid=None):
        # This function checks if the base pose is not in collision and if there is a feasible pick
        obj = operator_skeleton.discrete_parameters['object']
        if type(obj) == str or type(obj) == unicode:
            obj = self.problem_env.env.GetKinBody(obj)

        if self.action_mode == 'ir_parameters':
            grasp_params, pick_base_pose = get_pick_base_pose_and_grasp_from_pick_parameters(obj, pick_parameters)
        elif self.action_mode == 'robot_base_pose':
            grasp_params = pick_parameters[0:3]
            pick_base_pose = pick_parameters[3:]
        else:
            raise NotImplementedError
        g_config = self.compute_feasible_grasp_config(obj, pick_base_pose, grasp_params)

        if g_config is not None:
            pick_action = {'operator_name': operator_skeleton.type, 'q_goal': pick_base_pose,
                           'grasp_params': grasp_params, 'g_config': g_config, 'action_parameters': pick_parameters}
            return pick_action, 'HasSolution'
        else:
            pick_action = {'operator_name': operator_skeleton.type, 'q_goal': None, 'grasp_params': None,
                           'g_config': None, 'action_parameters': pick_parameters}
            return pick_action, "NoSolution"

    def compute_feasible_grasp_config(self, obj, pick_base_pose, grasp_params):
        #with self.robot:
        grasp_config = self.compute_grasp_config(obj, pick_base_pose, grasp_params)
        if grasp_config is not None:
            grasp_is_feasible = self.is_grasp_config_feasible(obj, pick_base_pose, grasp_params, grasp_config)
            if grasp_is_feasible:
                return grasp_config
        else:
            return None

    def is_grasp_config_feasible(self, obj, pick_base_pose, grasp_params, grasp_config):
        raise NotImplementedError

    def compute_grasp_config(self, obj, pick_base_pose, grasp_params):
        raise NotImplementedError
