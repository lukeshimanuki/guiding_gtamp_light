from generator import Generator
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker
import time


class TwoArmPaPGenerator(Generator):
    def __init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning, n_iter_limit,
                 problem_env,
                 pick_action_mode, place_action_mode):
        self.pick_action_mode = pick_action_mode
        self.place_action_mode = place_action_mode
        Generator.__init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning,
                           n_iter_limit,
                           problem_env)

    def get_feasibility_checker(self):
        return TwoArmPaPFeasibilityChecker(self.problem_env, pick_action_mode=self.pick_action_mode,
                                           place_action_mode=self.place_action_mode)

    def sample_ik_feasible_and_collision_free_op_parameters(self, actions=None, q_values=None):
        assert self.n_iter_limit > 0
        feasible_op_parameters = []
        feasibility_check_time = 0
        stime = time.time()
        orig_ik_checks = self.n_ik_checks
        self.feasibility_checker.feasible_pick = []
        for _ in range(self.n_iter_limit):
            self.n_ik_checks += 1
            sampled_op_parameters = self.sampler.sample()

            stime2 = time.time()
            op_parameters, status = self.feasibility_checker.check_feasibility(self.abstract_action,
                                                                               sampled_op_parameters)
            feasibility_check_time += time.time() - stime2

            if status == 'HasSolution':
                feasible_op_parameters.append(op_parameters)
                self.feasibility_checker.feasible_pick = []
                if len(feasible_op_parameters) >= self.n_parameters_to_try_motion_planning:
                    break
        smpling_time = time.time() - stime
        print "IK time {:.5f} Total IK checks {} Status {}".format(smpling_time, self.n_ik_checks-orig_ik_checks, status)
        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status
