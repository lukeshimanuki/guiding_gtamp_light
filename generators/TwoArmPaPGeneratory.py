from generator import Generator
from gtamp_utils.utils import get_pick_domain, get_place_domain
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker


class TwoArmPaPGenerator(Generator):
    def __init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning, n_iter_limit, problem_env,
                 pick_action_mode, place_action_mode, reachability_clf=None):
        self.pick_action_mode = pick_action_mode
        self.place_action_mode = place_action_mode
        Generator.__init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning, n_iter_limit,
                           problem_env, reachability_clf)

    def get_feasibility_checker(self):
        return TwoArmPaPFeasibilityChecker(self.problem_env, pick_action_mode=self.pick_action_mode,
                                           place_action_mode=self.place_action_mode)


