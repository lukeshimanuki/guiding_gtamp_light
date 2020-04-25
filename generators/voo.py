from generator import Generator
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker
import numpy as np
import time

IK_FEASIBLE_AND_BASEPOSE_COLLISION_FREE = 0
INFEASIBLE = -1


class VOOGenerator(Generator):
    def __init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning, n_iter_limit,
                 problem_env, pick_action_mode, place_action_mode):
        self.pick_action_mode = pick_action_mode
        self.place_action_mode = place_action_mode
        self.tried_samples = []
        self.tried_sample_labels = []

        Generator.__init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning,
                           n_iter_limit, problem_env)

    def get_feasibility_checker(self):
        return TwoArmPaPFeasibilityChecker(self.problem_env, pick_action_mode=self.pick_action_mode,
                                           place_action_mode=self.place_action_mode)

    def sample_next_point(self, actions, q_values, dont_check_motion_existence=False):
        target_obj = self.abstract_action.discrete_parameters['object']
        if target_obj in self.feasible_pick_params:
            self.feasibility_checker.feasible_pick = self.feasible_pick_params[target_obj]

        feasible_op_parameters, status = self.sample_ik_feasible_and_collision_free_op_parameters(actions, q_values)
        if status == "NoSolution":
            return {'is_feasible': False}

        # We would have to move these to the loop in order to be fair
        if dont_check_motion_existence:
            chosen_op_param = self.choose_one_of_params(feasible_op_parameters, status)
        else:
            chosen_op_param = self.check_existence_of_feasible_motion_plan(feasible_op_parameters)
        return chosen_op_param

    def sample_ik_feasible_and_collision_free_op_parameters(self, actions, q_values):
        assert self.n_iter_limit > 0
        feasible_op_parameters = []
        feasibility_check_time = 0
        stime = time.time()
        # note that this assumes you are doing two arm pap
        for _ in range(self.n_iter_limit):
            self.n_ik_checks += 1
            sampled_op_parameters = self.sampler.sample(actions, q_values)

            stime2 = time.time()
            op_parameters, status = self.feasibility_checker.check_feasibility(self.abstract_action,
                                                                               sampled_op_parameters)
            feasibility_check_time += time.time() - stime2

            if status == 'HasSolution':
                self.tried_samples.append(np.hstack([op_parameters['pick']['action_parameters'],
                                                     op_parameters['place']['action_parameters']]))
                self.tried_sample_labels.append(IK_FEASIBLE_AND_BASEPOSE_COLLISION_FREE)
                feasible_op_parameters.append(op_parameters)

                if len(feasible_op_parameters) >= self.n_parameters_to_try_motion_planning:
                    break
            else:
                self.tried_samples.append(sampled_op_parameters)
                self.tried_sample_labels.append(INFEASIBLE)

        smpling_time = time.time() - stime
        print "IK time {:.5f}".format(smpling_time)
        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status
