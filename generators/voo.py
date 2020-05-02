from generator import Generator
from feasibility_checkers.two_arm_pap_feasiblity_checker import TwoArmPaPFeasibilityChecker
import numpy as np
import time
from gtamp_utils import utils

IK_FEASIBLE_AND_BASEPOSE_COLLISION_FREE = 0
INFEASIBLE = -1


class TwoArmVOOGenerator(Generator):
    def __init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning, n_iter_limit,
                 problem_env, pick_action_mode, place_action_mode):
        self.pick_action_mode = pick_action_mode
        self.place_action_mode = place_action_mode
        self.basic_tested_samples = []
        self.basic_tested_sample_values = []

        Generator.__init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning,
                           n_iter_limit, problem_env)

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
        if actions is None:
            actions = []
            q_values = []
        print 'n tried samples', len(q_values)

        evaled_values = q_values + self.basic_tested_sample_values
        if len(evaled_values) > 0 and np.max(evaled_values) > self.sampler.infeasible_action_value:
            evaled_x = actions + self.basic_tested_samples
            best_evaled_x = evaled_x[np.argmax(evaled_values)]
            distances = [self.sampler.distance_fn(best_evaled_x, x) for x in evaled_x]
            n_smpls = 10 if len(distances) > 10 else len(distances)
            k_nearest_neighbor_to_best_point = evaled_x[np.argsort(distances)][1:n_smpls]
            self.sampler.k_nearest_neighbor_to_best_point = k_nearest_neighbor_to_best_point

        basic_feasible_sample_label = 0
        for _ in range(self.n_iter_limit):
            self.n_ik_checks += 1
            evaled_actions = actions + self.basic_tested_samples
            evaled_values = q_values + self.basic_tested_sample_values
            sampled_op_parameters = self.sampler.sample(evaled_actions, evaled_values)

            stime2 = time.time()
            op_parameters, status = self.feasibility_checker.check_feasibility(self.abstract_action,
                                                                               sampled_op_parameters)
            feasibility_check_time += time.time() - stime2

            if status == 'HasSolution':
                # I should keep these. But their value needs to be updated if we ever get to try them.
                # Let's not do it for now.
                # But if I do this, I believe I can improve the first-encountered states too.
                # What value should this take?
                #   If there is no feasible action, it should have the highest value
                #   If there are feasible actions, then don't do this. You can follow them.
                if len(evaled_values) == 0 or \
                        np.max(evaled_values) == self.sampler.infeasible_action_value or \
                        np.max(evaled_values) == basic_feasible_sample_label:
                    sampled_op_parameters[0:6] = op_parameters['pick']['action_parameters']
                    self.basic_tested_samples.append(sampled_op_parameters)
                    self.basic_tested_sample_values.append(basic_feasible_sample_label)

                feasible_op_parameters.append(op_parameters)
                self.feasibility_checker.feasible_pick = []
                if len(feasible_op_parameters) >= self.n_parameters_to_try_motion_planning:
                    break
            else:
                # do this because we re-use picks that were feasible in feasibility checker
                if status == 'PickFailed':
                    self.basic_tested_samples.append(sampled_op_parameters)
                    self.basic_tested_sample_values.append(self.sampler.infeasible_action_value)
                elif status == 'PlaceFailed':
                    sampled_op_parameters[0:6] = op_parameters['pick']['action_parameters']
                    self.basic_tested_samples.append(sampled_op_parameters)
                    self.basic_tested_sample_values.append(self.sampler.infeasible_action_value)

        smpling_time = time.time() - stime
        print "IK time {:.5f} Total IK checks {}".format(smpling_time, self.n_ik_checks - orig_ik_checks)
        """
        if len(feasible_op_parameters) > 0:
            utils.viewer()
            pick_configs = [op['pick']['q_goal'] for op in feasible_op_parameters]
            place_configs = [op['place']['q_goal'] for op in feasible_op_parameters]
            import pdb;pdb.set_trace()
            utils.visualize_path(pick_configs)
            utils.visualize_path(place_configs)
            import pdb;pdb.set_trace()
        """

        # Remove all the temporarily added ones
        idxs_to_remove = np.where(self.basic_tested_sample_values == basic_feasible_sample_label)[0]
        for i in idxs_to_remove:
            self.basic_tested_samples.pop(i)
            self.basic_tested_sample_values.pop(i)

        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status
