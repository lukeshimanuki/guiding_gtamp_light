from sampler_trajectory import SamplerTrajectory
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from generators.samplers.pick_place_learned_sampler import compute_v_manip
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import TwoArmConcreteNodeState
from test_scripts.run_greedy import get_problem_env, get_goal_obj_and_region
import copy


class TwoArmSAHSSamplerTrajectory(SamplerTrajectory):
    def __init__(self, problem_idx, n_objs_pack):
        SamplerTrajectory.__init__(self, problem_idx, n_objs_pack)

    def compute_n_in_way_for_object_moved(self, object_moved, abs_state, goal_objs):
        goal_objs_not_in_goal = [goal_obj for goal_obj in goal_objs
                                 if not abs_state.binary_edges[(goal_obj, 'home_region')][0]
                                 and goal_obj != object_moved]  # Object can't be in the path to itself
        n_in_way = 0
        original_config = utils.get_robot_xytheta(abs_state.problem_env.robot)
        for goal_obj in goal_objs_not_in_goal:
            Vpre = abs_state.cached_pick_paths[goal_obj]
            objects_in_way = [o.GetName() for o in self.problem_env.get_objs_in_collision(Vpre, 'entire_region')]
            is_object_moved_in_the_pick_path_to_goal_obj = object_moved in objects_in_way
            n_in_way += is_object_moved_in_the_pick_path_to_goal_obj

        for goal_obj in goal_objs_not_in_goal:
            # Don't I include the pick path collisions?
            pick = abs_state.pick_used[goal_obj]
            pick.execute()
            Vmanip = abs_state.cached_place_paths[(goal_obj, 'home_region')]
            objects_in_way = [o.GetName() for o in self.problem_env.get_objs_in_collision(Vmanip, 'entire_region')]
            is_object_moved_in_the_place_path_for_goal_o_to_r = object_moved in objects_in_way
            n_in_way += is_object_moved_in_the_place_path_for_goal_o_to_r
            utils.two_arm_place_object(pick.continuous_parameters)
        utils.set_robot_config(original_config)
        return n_in_way

    # used in RSC data
    def get_pap_used_in_plan(self, plan):
        picks = plan[::2]
        places = plan[1::2]
        obj_to_pick = {p.discrete_parameters['object']: p for p in picks}
        obj_to_place = {(p.discrete_parameters['object'], p.discrete_parameters['place_region']): p for p in places}
        return [obj_to_pick, obj_to_place]

    def compute_state(self, parent_state, parent_action, goal_entities, problem_env):
        if parent_action is not None:
            parent_action.discrete_parameters['two_arm_place_object'] = parent_action.discrete_parameters['object']
        state = ShortestPathPaPState(problem_env, goal_entities, parent_state, parent_action)

        return state

    def delete_moved_objects_from_pap_data(self, pick_used, place_used, moved_obj):
        moved_obj_name = moved_obj.GetName()
        new_pick_used = {key: value for key, value in zip(pick_used.keys(), pick_used.values()) if
                         key != moved_obj_name}

        new_place_used = {}
        for key, value in zip(place_used.keys(), place_used.values()):
            if moved_obj_name == key[0]:
                continue
            new_place_used[key] = value
        return new_pick_used, new_place_used
    # end of used in RSC data

    def get_rsc_data(self, plan, config):
        goal_objs, goal_region = get_goal_obj_and_region(config)
        problem_env = get_problem_env(config, goal_region, goal_objs)

        parent_state = None
        parent_action = None

        paps_used = self.get_pap_used_in_plan(plan)
        pick_used = paps_used[0]
        place_used = paps_used[1]
        goal_entities = goal_objs + [goal_region]
        openrave_env = problem_env.env
        all_data = []
        for action_idx, action in enumerate(plan):
            if 'place' in action.type:
                continue

            target_obj = openrave_env.GetKinBody(action.discrete_parameters['object'])

            pick_used, place_used = self.delete_moved_objects_from_pap_data(pick_used, place_used, target_obj)

            action.is_skeleton = False
            pap_action = copy.deepcopy(action)
            pap_action = pap_action.merge_operators(plan[action_idx + 1])
            pap_action.is_skeleton = False
            pap_action.discrete_parameters['place_region'] = pap_action.discrete_parameters['two_arm_place_place_region'].name
            import pdb;pdb.set_trace()
            action_info = self.get_action_info(pap_action)

            abs_state = self.compute_state(parent_state, parent_action, goal_entities, problem_env)
            v_manip = compute_v_manip(abs_state, goal_objs)
            concrete_state = TwoArmConcreteNodeState(abs_state, pap_action)
            data = {"abs_state": abs_state,
                    "concrete_state": concrete_state,
                    "action_info": action_info,
                    "action": action,
                    "v_manip": v_manip}
            import pdb;pdb.set_trace()

            pap_action.execute()
            parent_state = abs_state
            parent_action = pap_action
            all_data.append(data)
            print "Executed", action.discrete_parameters

        openrave_env.Destroy()
        return data

    def get_greedy_data(self, nodes, config):
        goal_objs, goal_region = get_goal_obj_and_region(config)
        problem_env = get_problem_env(config, goal_region, goal_objs)
        self.problem_env = problem_env

        nodes_ = [n for n in nodes if n.parent is not None]
        for n in nodes:
            if n.state is not None:
                n.state.problem_env = problem_env
        positive_data = []
        neutral_data = []
        for node in nodes_:
            action = node.parent_action
            action_info = self.get_action_info(action)

            # Getting parent state info
            parent_node = node.parent
            parent_abs_state = parent_node.state
            parent_state_saver = parent_node.state.state_saver
            parent_state_saver.Restore()
            parent_concrete_state = TwoArmConcreteNodeState(parent_abs_state, action)
            parent_n_in_way = self.compute_n_in_way_for_object_moved(action.discrete_parameters['object'],
                                                                     parent_abs_state, goal_objs)
            parent_v_manip = compute_v_manip(parent_abs_state, goal_objs)

            abs_state = node.state
            abs_state.state_saver.Restore()
            v_manip = compute_v_manip(abs_state, goal_objs)
            n_in_way = self.compute_n_in_way_for_object_moved(action.discrete_parameters['object'], abs_state,
                                                              goal_objs)
            data = {"abs_state": parent_abs_state,
                    "concrete_state": parent_concrete_state,
                    "action_info": action_info,
                    "action": action,
                    "parent_n_in_way": parent_n_in_way,
                    "n_in_way": n_in_way,
                    "parent_v_manip": parent_v_manip,
                    "v_manip": v_manip}
            if node.is_goal_traj:
                positive_data.append(data)
            else:
                neutral_data.append(data)
        for n in nodes: n.state.problem_env = None
        for d in positive_data: d['concrete_state'].problem_env = None
        for d in neutral_data: d['concrete_state'].problem_env = None

        return positive_data, neutral_data

    def add_trajectory(self, plan):
        print "Problem idx", self.problem_idx
        self.set_seed(self.problem_idx)
        problem_env, openrave_env = self.create_environment()

        goal_objs = ['square_packing_box1', 'square_packing_box2', 'rectangular_packing_box3',
                     'rectangular_packing_box4']
        goal_entities = ['home_region'] + goal_objs
        [utils.set_color(g, [1, 0, 0]) for g in goal_objs]

        self.problem_env = problem_env

        abs_state = ShortestPathPaPState(problem_env, goal_entities)
        for action in plan:
            assert action.type == 'two_arm_pick_two_arm_place'
            state = TwoArmConcreteNodeState(abs_state, action)
            action_info = self.get_action_info(action)
            prev_n_in_way = self.compute_n_in_way_for_object_moved(action.discrete_parameters['object'],
                                                                   abs_state, goal_objs)
            prev_v_manip = compute_v_manip(abs_state, goal_objs)

            action.execute_pick()
            # state.place_collision_vector = state.convert_collision_at_prm_indices_to_col_vec(abs_state.current_collides)
            action.execute()

            print "Computing state..."
            abs_state = ShortestPathPaPState(self.problem_env, goal_entities, parent_state=abs_state,
                                             parent_action=action)
            v_manip = compute_v_manip(abs_state, goal_objs)

            n_in_way = self.compute_n_in_way_for_object_moved(action.discrete_parameters['object'], abs_state,
                                                              goal_objs)

            print action.discrete_parameters['object'], action.discrete_parameters['place_region']
            print 'Prev n in way {} curr n in way {}'.format(prev_n_in_way, n_in_way)
            self.add_sah_tuples(state, action_info, prev_n_in_way, n_in_way, prev_v_manip, v_manip)

        self.add_state_prime()
        print "Done!"
        openrave_env.Destroy()
