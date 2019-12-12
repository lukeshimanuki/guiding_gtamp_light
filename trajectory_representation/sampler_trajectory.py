from gtamp_problem_environments.mover_env import Mover
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState

import numpy as np
import random
import sys


def get_pick_base_poses(action, smples):
    pick_base_poses = []
    for smpl in smples:
        smpl = smpl[0:4]
        sin_cos_encoding = smpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        smpl = np.hstack([smpl[0:2], decoded_angle])
        abs_base_pose = utils.get_absolute_pick_base_pose_from_ir_parameters(smpl, action.discrete_parameters['object'])
        pick_base_poses.append(abs_base_pose)
    return pick_base_poses


def get_place_base_poses(action, smples, mover):
    place_base_poses = smples[:, 4:]
    to_return = []
    for bsmpl in place_base_poses:
        sin_cos_encoding = bsmpl[-2:]
        decoded_angle = utils.decode_sin_and_cos_to_angle(sin_cos_encoding)
        bsmpl = np.hstack([bsmpl[0:2], decoded_angle])
        to_return.append(bsmpl)
    to_return = np.array(to_return)
    to_return[:, 0:2] += mover.regions[action.discrete_parameters['region']].box[0]
    return to_return


class SamplerTrajectory:
    def __init__(self, problem_idx, n_objs_pack):
        self.problem_idx = problem_idx
        self.paps_used = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_prime = []
        self.seed = None  # this defines the initial state
        self.problem_env = None
        self.n_objs_pack = n_objs_pack

    def compute_state(self, obj, region, goal_entities):
        if not 'two_arm_mover' in self.problem_env.name:
            raise NotImplementedError
        return ConcreteNodeState(self.problem_env, obj, region, goal_entities)

    def add_state_prime(self):
        self.state_prime = self.states[1:]

    def add_sar_tuples(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def create_environment(self):
        problem_env = Mover(self.problem_idx)
        openrave_env = problem_env.env
        return problem_env, openrave_env

    @staticmethod
    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    def add_trajectory(self, plan):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()
        self.problem_env = problem_env
        if 'two_arm' in problem_env.name:
            goal_entities = ['home_region'] + [obj.GetName() for obj in problem_env.objects[:self.n_objs_pack]]
        else:
            raise NotImplementedError

        state = None
        utils.viewer()
        for action_idx, action in enumerate(plan):
            if 'pick' in action.type:
                associated_place = plan[action_idx + 1]
                state = self.compute_state(action.discrete_parameters['object'],
                                           associated_place.discrete_parameters['region'],
                                           goal_entities)

                ## Visualization purpose
                """
                obj = action.discrete_parameters['object']
                region = associated_place.discrete_parameters['region']
                collision_vec = np.delete(state.state_vec, [415, 586, 615, 618, 619], axis=1)
                smpls = generate_smpls(obj, collision_vec, state, admon)
                utils.visualize_path(smpls)
                utils.visualize_path([associated_place.continuous_parameters['q_goal']])
                import pdb; pdb.set_trace()
                """
                ##############################################################################

                action.execute()
                obj_pose = utils.get_body_xytheta(action.discrete_parameters['object'])
                robot_pose = utils.get_body_xytheta(self.problem_env.robot)
                pick_parameters = utils.get_ir_parameters_from_robot_obj_poses(robot_pose, obj_pose)
                recovered = utils.get_absolute_pick_base_pose_from_ir_parameters(pick_parameters, obj_pose)
                pick_base_pose = action.continuous_parameters['q_goal']
                pick_base_pose = utils.clean_pose_data(pick_base_pose)
                recovered = utils.clean_pose_data(recovered)
                if pick_parameters[0] > 1:
                    sys.exit(-1)
                assert np.all(np.isclose(pick_base_pose, recovered))
            else:
                if action == plan[-1]:
                    reward = 0
                else:
                    reward = -1
                action.execute()
                place_base_pose = action.continuous_parameters['q_goal']

                action_info = {
                    'object_name': action.discrete_parameters['object'],
                    'region_name': action.discrete_parameters['region'],
                    'pick_base_ir_parameters': pick_parameters,
                    'place_abs_base_pose': place_base_pose,
                    'pick_abs_base_pose': pick_base_pose,

                }
                self.add_sar_tuples(state, action_info, reward)

        self.add_state_prime()
        print "Done!"
        openrave_env.Destroy()


class SAHSSamplerTrajectory(SamplerTrajectory):
    def __init__(self, problem_idx, n_objs_pack):
        SamplerTrajectory.__init__(self, problem_idx, n_objs_pack)

    def add_trajectory(self, plan):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()

        if 'two_arm' in problem_env.name:
            goal_entities = ['home_region'] + [obj.GetName() for obj in problem_env.objects[:self.n_objs_pack]]
        else:
            raise NotImplementedError

        self.problem_env = problem_env

        # utils.viewer()
        for action_idx, action in enumerate(plan):
            assert action.type == 'two_arm_pick_two_arm_place'
            target_obj = problem_env.env.GetKinBody(action.discrete_parameters['object'])
            target_obj.Enable(False)
            state = self.compute_state(action.discrete_parameters['object'],
                                       action.discrete_parameters['region'],
                                       goal_entities)
            target_obj.Enable(True)

            pick_action_info = action.continuous_parameters['pick']
            place_action_info = action.continuous_parameters['place']

            pick_parameters = pick_action_info['action_parameters']

            pick_base_pose = utils.clean_pose_data(pick_action_info['q_goal'])
            place_base_pose = utils.clean_pose_data(place_action_info['q_goal'])
            place_obj_abs_pose = utils.clean_pose_data(place_action_info['object_pose'])

            pick_motion = [utils.clean_pose_data(q) for q in pick_action_info['motion']]
            place_motion = [utils.clean_pose_data(q) for q in place_action_info['motion']]
            action_info = {
                'object_name': action.discrete_parameters['object'],
                'region_name': action.discrete_parameters['region'],
                'pick_base_ir_parameters': pick_parameters,
                'place_abs_base_pose': place_base_pose,
                'place_obj_abs_pose': place_obj_abs_pose,
                'pick_abs_base_pose': pick_base_pose,
                'pick_motion': pick_motion,
                'place_motion': place_motion
            }
            if action == plan[-1]:
                reward = 0
            else:
                reward = -1

            print action.discrete_parameters['object'], action.discrete_parameters['region']
            action.execute()
            self.add_sar_tuples(state, action_info, reward)

        self.add_state_prime()
        print "Done!"
        openrave_env.Destroy()

