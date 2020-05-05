from gtamp_problem_environments.mover_env import Mover
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import ConcreteNodeState
from generators.feasibility_checkers import two_arm_place_feasibility_checker
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
# from planners.sahs.helper import compute_hcount, compute_new_number_in_goal, \
#    count_pickable_goal_objs_and_placeable_to_goal_region_not_yet_in_goal_region
from planners.subplanners.motion_planner import BaseMotionPlanner

import numpy as np
import random
import sys
import time


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
        self.hvalues = []
        self.hcounts = []
        self.n_in_way = []
        self.num_in_goal = []
        self.num_papable_to_goal = []
        self.prev_n_in_way = []
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

    def add_sah_tuples(self, s, a, prev_n_in_way, n_in_way):
        self.states.append(s)
        self.actions.append(a)
        self.prev_n_in_way.append(prev_n_in_way)
        self.n_in_way.append(n_in_way)
        """
        self.hvalues.append(hvalue)
        self.hcounts.append(hcount)
        self.num_in_goal.append(num_in_goal)
        self.num_papable_to_goal.append(num_papable_to_goal)
        """

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
        # utils.viewer()
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

    def compute_n_in_way(self, action, abs_state, goal_objs):
        object_moved = action.discrete_parameters['object']
        target_region = action.discrete_parameters['place_region']

        goal_objs_not_in_goal = [goal_obj for goal_obj in goal_objs if
                                 not abs_state.binary_edges[(goal_obj, 'home_region')][0]]
        if target_region == 'home_region':
            goal_objs_whose_pick_paths_blocked_by_object_moved = [goal_obj for goal_obj in goal_objs_not_in_goal
                                                                  if
                                                                  abs_state.binary_edges[(object_moved, goal_obj)][1]]
            n_in_way = len(goal_objs_whose_pick_paths_blocked_by_object_moved)
        else:
            n_in_way = 0
            for goal_obj in goal_objs_not_in_goal:
                pick = abs_state.pick_used[goal_obj]
                pick.execute()
                place_path = abs_state.cached_place_paths[(goal_obj, 'home_region')]
                objects_in_way = [o.GetName() for o in
                                  self.problem_env.get_objs_in_collision(place_path, 'entire_region')]
                n_in_way += object_moved in objects_in_way
                utils.two_arm_place_object(pick.continuous_parameters)
        return n_in_way

    @staticmethod
    def get_action_info(action):
        # Information regarding actions
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
            'region_name': action.discrete_parameters['place_region'],
            'pick_action_parameters': pick_parameters,
            'pick_abs_base_pose': pick_base_pose,
            'place_abs_base_pose': place_base_pose,
            'place_obj_abs_pose': place_obj_abs_pose,
            'pick_motion': pick_motion,
            'place_motion': place_motion
        }
        return action_info

    def add_trajectory(self, plan, hvalues):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()

        goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4']
        goal_entities = ['home_region'] + goal_objs
        [utils.set_color(g, [1, 0, 0]) for g in goal_objs]

        self.problem_env = problem_env

        abs_state = ShortestPathPaPState(problem_env, goal_entities)
        init_saver = abs_state.state_saver
        for action in plan:
            assert action.type == 'two_arm_pick_two_arm_place'
            state = self.compute_state(action.discrete_parameters['object'],
                                       action.discrete_parameters['place_region'],
                                       goal_entities)
            action_info = self.get_action_info(action)
            print "Computing state..."

            prev_n_in_way = self.compute_n_in_way(action, abs_state, goal_objs)
            action.execute_pick()
            state.place_collision_vector = state.get_collison_vector(None)
            action.execute()
            abs_state = ShortestPathPaPState(self.problem_env, goal_entities,
                                             parent_state=abs_state, parent_action=action)
            n_in_way = self.compute_n_in_way(action, abs_state, goal_objs)

            print action.discrete_parameters['object'], action.discrete_parameters['place_region']
            print 'Prev n in way {} curr n in way {}'.format(prev_n_in_way, n_in_way)

            self.add_sah_tuples(state, action_info, prev_n_in_way, n_in_way)

        self.add_state_prime()
        """
        utils.viewer()
        init_saver.Restore()
        for prev_n_in_way, n_in_way, p in zip(self.prev_n_in_way, self.n_in_way, plan):
            print 'Prev n in way {} curr n in way {}'.format(prev_n_in_way, n_in_way)
            import pdb;pdb.set_trace()
            p.execute()
            import pdb;pdb.set_trace()
        import pdb;pdb.set_trace()
        """
        print "Done!"
        openrave_env.Destroy()
