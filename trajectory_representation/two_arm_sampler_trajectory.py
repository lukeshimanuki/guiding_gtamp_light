from sampler_trajectory import SamplerTrajectory
from trajectory_representation.shortest_path_pick_and_place_state import ShortestPathPaPState
from generators.samplers.sampler import compute_v_manip
from gtamp_utils import utils
from trajectory_representation.concrete_node_state import TwoArmConcreteNodeState


class TwoArmSAHSSamplerTrajectory(SamplerTrajectory):
    def __init__(self, problem_idx, n_objs_pack):
        SamplerTrajectory.__init__(self, problem_idx, n_objs_pack)

    def compute_n_in_way_for_object_moved(self, object_moved, abs_state, goal_objs):
        goal_objs_not_in_goal = [goal_obj for goal_obj in goal_objs
                                 if not abs_state.binary_edges[(goal_obj, 'home_region')][0]
                                 and goal_obj != object_moved]  # Object can't be in the path to itself
        n_in_way = 0
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
        return n_in_way

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
