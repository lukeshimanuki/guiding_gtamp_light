from generators.feasibility_checkers.two_arm_pick_feasibility_checker import TwoArmPickFeasibilityChecker
from generators.feasibility_checkers.two_arm_place_feasibility_checker import TwoArmPlaceFeasibilityChecker
from trajectory_representation.operator import Operator
from mover_library import utils


# class TwoArmPaPFeasibilityChecker(TwoArmPickFeasibilityChecker, TwoArmPlaceFeasibilityChecker):
class TwoArmPaPFeasibilityChecker:
    def __init__(self, problem_env, pick_action_mode, place_action_mode):
        assert place_action_mode == 'object_pose' or place_action_mode == 'robot_base_pose', 'Invalid place action mode'
        assert pick_action_mode == 'ir_parameters' or pick_action_mode == 'robot_base_pose', 'Invalid pick action mode'
        self.problem_env = problem_env
        self.pick_feasibility_checker = TwoArmPickFeasibilityChecker(problem_env, action_mode=pick_action_mode)
        self.place_feasibility_checker = TwoArmPlaceFeasibilityChecker(problem_env, action_mode=place_action_mode)
        self.feasible_pick = []

    def check_place_feasible(self, pick_parameters, place_parameters, operator_skeleton):
        pick_op = Operator('two_arm_pick', operator_skeleton.discrete_parameters)
        pick_op.continuous_parameters = pick_parameters

        # todo remove the CustomStateSaver
        # saver = utils.CustomStateSaver(self.problem_env.env)
        original_config = utils.get_body_xytheta(self.problem_env.robot).squeeze()
        pick_op.execute()
        place_op = Operator('two_arm_place', operator_skeleton.discrete_parameters)

        place_cont_params, place_status = self.place_feasibility_checker.check_feasibility(place_op, place_parameters)
        utils.two_arm_place_object(pick_op.continuous_parameters)
        utils.set_robot_config(original_config)

        # saver.Restore()
        return place_cont_params, place_status

    def check_pick_feasible(self, pick_parameters, operator_skeleton):
        pick_op = Operator('two_arm_pick', operator_skeleton.discrete_parameters)
        params, status = self.pick_feasibility_checker.check_feasibility(pick_op, pick_parameters)
        return params, status

    def check_feasibility(self, operator_skeleton, parameters, swept_volume_to_avoid=None):
        # todo make this parameter mode explicit in the constructor
        pick_parameters = parameters[:6]
        place_parameters = parameters[-3:]

        # We are disabling this to make it easier to implement getting place-samples on-demand from learned sampler.
        we_already_have_feasible_pick = len(self.feasible_pick) > 0
        if we_already_have_feasible_pick:
            pick_parameters = self.feasible_pick[0]
        else:
            pick_parameters, pick_status = self.check_pick_feasible(pick_parameters, operator_skeleton)

            if pick_status != 'HasSolution':
                return {'pick': None, 'place': None}, "PickFailed"
            else:
                self.feasible_pick.append(pick_parameters)

        place_parameters, place_status = self.check_place_feasible(pick_parameters, place_parameters, operator_skeleton)

        if place_status != 'HasSolution':
            return {'pick': pick_parameters, 'place': None}, "PlaceFailed"

        pap_continuous_parameters = {'pick': pick_parameters, 'place': place_parameters}
        self.feasible_pick = []
        return pap_continuous_parameters, 'HasSolution'


class TwoArmPaPFeasibilityCheckerWithoutSavingFeasiblePick(TwoArmPaPFeasibilityChecker):
    def __init__(self, problem_env):
        TwoArmPaPFeasibilityChecker.__init__(self, problem_env)

    def check_feasibility(self, operator_skeleton, parameters, swept_volume_to_avoid=None, parameter_mode='obj_pose'):
        pick_parameters = parameters[:6]
        place_parameters = parameters[-3:]

        pick_parameters, pick_status = self.check_pick_feasible(pick_parameters, operator_skeleton)
        if pick_status != 'HasSolution':
            return None, "PickFailed"

        place_parameters, place_status = self.check_place_feasible(pick_parameters, place_parameters, operator_skeleton,
                                                                   parameter_mode=parameter_mode)

        if place_status != 'HasSolution':
            return None, "PlaceFailed"

        pap_continuous_parameters = {'pick': pick_parameters, 'place': place_parameters}
        self.feasible_pick = []
        return pap_continuous_parameters, 'HasSolution'
