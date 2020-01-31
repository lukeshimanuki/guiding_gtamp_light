import time
import numpy as np
import pickle
import uuid
import torch

from gtamp_utils import utils


class Generator:
    def __init__(self, abstract_state, abstract_action, sampler, n_parameters_to_try_motion_planning, n_iter_limit,
                 problem_env, reachability_clf=None):
        self.abstract_state = abstract_state
        self.abstract_action = abstract_action
        self.sampler = sampler
        self.n_parameters_to_try_motion_planning = n_parameters_to_try_motion_planning
        self.n_iter_limit = n_iter_limit
        self.problem_env = problem_env
        self.feasible_pick_params = {}
        self.feasibility_checker = self.get_feasibility_checker()
        self.tried_samples = []
        self.reachability_clf = reachability_clf

    def get_feasibility_checker(self):
        raise NotImplementedError

    def sample_next_point(self, dont_check_motion_existence=False):
        target_obj = self.abstract_action.discrete_parameters['object']
        if target_obj in self.feasible_pick_params:
            self.feasibility_checker.feasible_pick = self.feasible_pick_params[target_obj]

        feasible_op_parameters, status = self.sample_feasible_op_parameters()
        if status == "NoSolution":
            return {'is_feasible': False}

        # We would have to move these to the loop in order to be fair
        if dont_check_motion_existence:
            chosen_op_param = self.choose_one_of_params(feasible_op_parameters, status)
        else:
            chosen_op_param = self.get_param_with_feasible_motion_plan(feasible_op_parameters)
        return chosen_op_param

    def sample_feasible_op_parameters(self):
        assert self.n_iter_limit > 0
        feasible_op_parameters = []
        feasibility_check_time = 0
        stime = time.time()

        for i in range(self.n_iter_limit):
            op_parameters = self.sampler.sample()

            self.tried_samples.append(op_parameters)
            stime2 = time.time()
            op_parameters, status = self.feasibility_checker.check_feasibility(self.abstract_action, op_parameters)
            feasibility_check_time += time.time() - stime2

            if status == 'HasSolution':
                if self.reachability_clf is not None:
                    # make a vertex
                    #pred = self.reachability_clf.predict(op_parameters['pick']['q_goal'], self.abstract_state)
                    is_reachable = self.reachability_clf.predict(op_parameters, self.abstract_state, self.abstract_action)
                    if not is_reachable:
                        continue

                feasible_op_parameters.append(op_parameters)
                if len(feasible_op_parameters) >= self.n_parameters_to_try_motion_planning:
                    break

        smpling_time = time.time() - stime
        print "Sampling time", smpling_time
        print "Feasibilty time", feasibility_check_time
        if len(feasible_op_parameters) == 0:
            feasible_op_parameters.append(op_parameters)  # place holder
            status = "NoSolution"
        else:
            status = "HasSolution"

        return feasible_op_parameters, status

    @staticmethod
    def choose_one_of_params(candidate_parameters, status):
        sampled_feasible_parameters = status == "HasSolution"
        if sampled_feasible_parameters:
            chosen_op_param = candidate_parameters[0]
            chosen_op_param['motion'] = [chosen_op_param['q_goal']]
            chosen_op_param['is_feasible'] = True
        else:
            chosen_op_param = {'is_feasible': False}

        return chosen_op_param

    def get_param_with_feasible_motion_plan(self, candidate_parameters):
        n_feasible = len(candidate_parameters)
        n_mp_tried = 0

        obj_poses = {o.GetName(): utils.get_body_xytheta(o) for o in self.problem_env.objects}
        prepick_q0 = utils.get_body_xytheta(self.problem_env.robot)

        all_mp_data = []
        for op in candidate_parameters:
            print "n_mp_tried / n_feasible_params = %d / %d" % (n_mp_tried, n_feasible)
            chosen_pick_param = self.get_motion_plan([op['pick']])
            n_mp_tried += 1

            mp_data = {'q0': prepick_q0, 'qg': op['pick']['q_goal'], 'object_poses': obj_poses, 'held_obj': None}
            if not chosen_pick_param['is_feasible']:
                print "Pick motion does not exist"
                mp_data['label'] = False
                all_mp_data.append(mp_data)
                continue
            else:
                mp_data['label'] = True
                mp_data['motion'] = chosen_pick_param['motion']
                all_mp_data.append(mp_data)

            original_config = utils.get_body_xytheta(self.problem_env.robot).squeeze()
            utils.two_arm_pick_object(self.abstract_action.discrete_parameters['object'], chosen_pick_param)
            mp_data = {'q0': op['pick']['q_goal'], 'qg': op['place']['q_goal'], 'object_poses': obj_poses,
                       'held_obj': self.abstract_action.discrete_parameters['object'],
                       'region': self.abstract_action.discrete_parameters['place_region']}
            chosen_place_param = self.get_motion_plan([op['place']])  # calls MP
            utils.two_arm_place_object(chosen_pick_param)
            utils.set_robot_config(original_config)

            if chosen_place_param['is_feasible']:
                mp_data['label'] = True
                mp_data['motion'] = chosen_place_param['motion']
                all_mp_data.append(mp_data)
                print 'Motion plan exists'
                break
            else:
                mp_data['label'] = False
                all_mp_data.append(mp_data)
                print "Place motion does not exist"

        pickle.dump(all_mp_data,
                    open('./planning_experience/motion_planning_experience/' + str(uuid.uuid4()) + '.pkl', 'wb'))

        if not chosen_pick_param['is_feasible']:
            print "Motion plan does not exist"
            return {'is_feasible': False}

        if not chosen_place_param['is_feasible']:
            print "Motion plan does not exist"
            return {'is_feasible': False}

        chosen_pap_param = {'pick': chosen_pick_param, 'place': chosen_place_param, 'is_feasible': True}
        return chosen_pap_param

    def get_motion_plan(self, candidate_parameters):
        motion_plan_goals = [op['q_goal'] for op in candidate_parameters]
        self.problem_env.motion_planner.algorithm = 'rrt'
        motion, status = self.problem_env.motion_planner.get_motion_plan(motion_plan_goals[0],
                                                                         source='sampler',
                                                                         n_iterations=[20, 50, 100, 500,
                                                                                       1000, 5000, 10000])
        self.problem_env.motion_planner.algorithm = 'prm'
        found_feasible_motion_plan = status == "HasSolution"

        if found_feasible_motion_plan:
            which_op_param = np.argmin(np.linalg.norm(motion[-1] - motion_plan_goals, axis=-1))
            chosen_op_param = candidate_parameters[which_op_param]
            chosen_op_param['motion'] = motion
            chosen_op_param['is_feasible'] = True
        else:
            chosen_op_param = candidate_parameters[0]
            chosen_op_param['is_feasible'] = False

        return chosen_op_param
